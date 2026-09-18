"""Core auditor module for ML model evaluation.

This module contains the main Auditor class that orchestrates model evaluation
across different features and subgroups, with support for bootstrap confidence
intervals.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence as _Sequence
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Literal, Optional, Type

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_curve
from tqdm.auto import tqdm

from model_auditor._evaluation import (
    evaluate_confidence_interval,
    evaluate_error_feature,
    evaluate_feature,
    validate_n_bootstraps,
)
from model_auditor._thresholds import (
    binarize,
    build_threshold_series,
    resolve_threshold,
)
from model_auditor.error_metrics import AuditorErrorMetric, OddsRatio
from model_auditor.metric_inputs import (
    AuditorMetricInput,
    FalseNegatives,
    FalsePositives,
    TrueNegatives,
    TruePositives,
)
from model_auditor.metrics import AuditorMetric
from model_auditor.schemas import (
    AuditorFeature,
    AuditorOutcome,
    AuditorScore,
    CalibrationEvaluation,
    ConditionalThreshold,
    ErrorEvaluation,
    FeatureEvaluation,
    InferenceConfig,
    ScoreEvaluation,
    ThresholdSpec,
)
from model_auditor.utils import collect_metric_inputs

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class Auditor:
    """Evaluate fixed binary predictions overall and within registered subgroups.

    Configure data, scores, an outcome, and metric instances before evaluation.
    Calls are synchronous and use private DataFrame slices. Intervals are pointwise,
    conditional on supplied predictions and policies; they do not account for model
    fitting, threshold selection, or multiple comparisons.

    Args:
        data: Optional DataFrame, copied using pandas ``copy()``. Default None.
            Object-valued cells are not recursively copied. Evaluation requires
            nonempty data, unique columns, finite numeric scores and binary truth.
        features: Optional list of AuditorFeature configurations; default None.
        scores: Optional list of AuditorScore configurations; default None.
        outcome: Optional AuditorOutcome; requires data. Default None.
        metrics: Optional list of metric instances; default None (no metrics).

    Attributes:
        data: Owned DataFrame or None. Prefer add_data() to replace it.
        features: Mutable mapping of registered names to AuditorFeature objects.
        scores: Mutable mapping of registered names to AuditorScore objects.
        outcome: Registered AuditorOutcome, or None before registration.
        metrics: List of configured metric instances. The list is copied, but the
            instances are shared with the caller.

    Raises:
        ValueError: Invalid feature names, duplicate metric names, or an outcome
            supplied without data.
        KeyError: The supplied outcome column is absent from data.

    Example:
        >>> import pandas as pd
        >>> from model_auditor import Auditor
        >>> from model_auditor.metrics import Sensitivity
        >>> auditor = Auditor(pd.DataFrame({"risk": [0.8, 0.2], "y": [1, 0]}))
        >>> auditor.add_score("risk", threshold=0.5)
        >>> auditor.add_outcome("y")
        >>> auditor.set_metrics([Sensitivity()])
        >>> result = auditor.evaluate_metrics("risk", n_bootstraps=None)
        >>> result.features["overall"].levels["Overall"].metrics["sensitivity"].score
        1.0"""

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        features: Optional[list[AuditorFeature]] = None,
        scores: Optional[list[AuditorScore]] = None,
        outcome: Optional[AuditorOutcome] = None,
        metrics: Optional[list[AuditorMetric]] = None,
    ) -> None:
        """Initialize optional registrations; see Auditor for parameter contracts.

        Args:
            data: Optional input DataFrame copied on construction; default None.
            features: Optional feature configurations; default None.
            scores: Optional score configurations; default None.
            outcome: Optional outcome configuration; requires data. Default None.
            metrics: Optional metric instances with distinct names; default None."""
        # initialize data
        self.data: Optional[pd.DataFrame] = None if data is None else data.copy()

        # initialize features
        self.features: dict[str, AuditorFeature] = dict()
        if features is not None:
            for feature in features:
                self.add_feature(**vars(feature))

        # initialize scores
        self.scores: dict[str, AuditorScore] = dict()
        if scores is not None:
            for score in scores:
                self.add_score(**vars(score))

        # initialize outcome
        self.outcome: Optional[AuditorOutcome] = None
        if outcome is not None:
            self.add_outcome(**vars(outcome))

        # initialize metrics
        self.metrics: list[AuditorMetric] = list()
        if metrics is not None:
            self.set_metrics(metrics)

        # initialize attrs for later
        self._inputs: list[Type[AuditorMetricInput]] = list()

    def add_data(self, data: pd.DataFrame) -> None:
        """Replace evaluation data with a pandas copy and clear the outcome registration.

        Args:
            data: Full DataFrame to evaluate. Object-valued cells are not recursively
                copied. The caller's columns are not modified by the Auditor.

        Notes:
            Existing feature, score and metric registrations remain. Call add_outcome()
            again after replacing data; an existing raw ``_truth`` column is retained
            but does not count as outcome registration."""
        self.data = data.copy()
        self.outcome = None

    def add_feature(
        self,
        name: str,
        label: Optional[str] = None,
    ) -> None:
        """Register or replace a subgroup column without modifying the input data.

        Args:
            name: Column name. Existence is checked during evaluation.
            label: Optional display label; None (default) uses the column name.

        Raises:
            ValueError: name is reserved (overall, _truth, _pred, _binary_pred,
                tp, tn, fp or fn).

        Notes:
            Evaluations use declared categorical order, including unobserved levels
            with NaN estimates. Other levels are sorted after conversion to strings.
            Distinct values with the same string label are rejected during evaluation.
            Missing levels follow InferenceConfig.missing; the default excludes them."""
        if name in {
            "overall",
            "_truth",
            "_pred",
            "_binary_pred",
            "tp",
            "tn",
            "fp",
            "fn",
        }:
            raise ValueError(
                f"Feature name {name!r} is reserved for evaluation columns."
            )
        feature = AuditorFeature(
            name=name,
            label=label,
        )
        self.features[feature.name] = feature

    def add_score(
        self,
        name: str,
        label: Optional[str] = None,
        threshold: Optional[ThresholdSpec] = None,
    ) -> None:
        """Register or replace a numeric prediction column and decision threshold.

        Args:
            name: Score column; evaluation requires finite numeric values. Ranking
                metrics accept arbitrary score scales; probability metrics require [0, 1].
            label: Optional display label; None (default) uses name.
            threshold: Finite scalar or ConditionalThreshold. None (default) leaves
                the policy unset. Required only for decision metrics and error analysis.
                A row is positive when its score is greater than or equal to its
                threshold. Evaluation-time thresholds override this configuration.

        Notes:
            Registration defers column and threshold validation until use. It stores
            a conditional threshold object by reference; do not mutate it during use."""
        score = AuditorScore(
            name=name,
            label=label,
            threshold=threshold,
        )
        self.scores[score.name] = score

    def add_outcome(self, name: str, mapping: Optional[dict[Any, int]] = None) -> None:
        """Register truth and populate the owned DataFrame's ``_truth`` column.

        Args:
            name: Existing outcome column.
            mapping: Optional mapping to 0/1, e.g. {"positive": 1, "negative": 0}.
                None (default) uses values directly. The mapping is deep-copied.
                Unmapped values become missing and are rejected during evaluation.

        Raises:
            ValueError: No data has been added.
            KeyError: name is absent from the DataFrame.

        Notes:
            Overwrites the owned ``_truth`` column, never the caller's DataFrame.
            Evaluation rejects missing or nonbinary truth; it does not drop these rows."""
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")

        self.outcome = AuditorOutcome(name, deepcopy(mapping))
        if mapping is not None:
            self.data["_truth"] = self.data[name].map(mapping)
        else:
            self.data["_truth"] = self.data[name]

    def set_metrics(self, metrics: list[AuditorMetric]) -> None:
        """Replace the metrics computed by evaluate_metrics() and comparison methods.

        Args:
            metrics: List of instances implementing AuditorMetric (not metric classes).
                The list is copied, while instances remain shared. An empty list is
                allowed here but metric evaluation and comparisons require a metric.

        Raises:
            ValueError: Metric names are duplicated, including parameterized metrics
                whose names collide."""
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError(
                "Metric names must be unique; duplicate names overwrite results."
            )
        self.metrics = list(metrics)

    def _requires_threshold(self) -> bool:
        return any(
            set(metric.inputs) & {"_binary_pred", "tp", "tn", "fp", "fn"}
            for metric in self.metrics
        )

    def _metadata(
        self,
        score_name: str,
        threshold: Optional[ThresholdSpec],
        n_bootstraps: Optional[int],
        inference: InferenceConfig,
        cohort: Optional[str],
    ) -> dict[str, Any]:
        outcome = getattr(self, "outcome", None)
        return deepcopy(
            {
                "package_version": version("model-auditor"),
                "metrics": [
                    {
                        "name": metric.name,
                        "label": metric.label,
                        "parameters": getattr(metric, "parameters", {}),
                    }
                    for metric in self.metrics
                ],
                "score": score_name,
                "threshold": asdict(threshold)
                if isinstance(threshold, ConditionalThreshold)
                else threshold,
                "outcome": asdict(outcome) if outcome else None,
                "cohort": cohort,
                "n_bootstraps": n_bootstraps,
                "inference": asdict(inference),
                "interval_scope": "pointwise",
                "estimand": "row_weighted_fixed_binary_predictions",
                "selection_adjusted": False,
            }
        )

    def add_intersection(
        self, name: str, features: list[str], label: Optional[str] = None
    ) -> None:
        """Create and register a joint subgroup on the owned DataFrame.

        Args:
            name: New column name; must not already exist or be reserved.
            features: Nonempty list of distinct existing component column names.
                Components need not have been registered individually.
            label: Display label; None (default) uses name.

        Raises:
            ValueError: Data is absent, components are empty/duplicated, name exists
                or is reserved, or component values have ambiguous string labels.
            KeyError: A component column is absent.

        Notes:
            Values are JSON arrays of stringified components, in the supplied order.
            Any missing component makes the intersection missing; evaluation then
            applies InferenceConfig.missing. Only the owned data gains a column.
            Define intersections before inspecting results to avoid selection bias."""
        import json

        from model_auditor._evaluation import _prepare_feature_data

        if self.data is None or not features or len(set(features)) != len(features):
            raise ValueError(
                "Provide data and a nonempty list of distinct feature columns."
            )
        if name in self.data:
            raise ValueError("Intersection name must not overwrite an existing column.")
        for feature in features:
            _prepare_feature_data(self.data, feature)
        self.add_feature(name, label)
        valid = self.data[features].notna().all(axis=1)
        values = (
            self.data[features]
            .astype(str)
            .apply(lambda row: json.dumps(row.tolist()), axis=1)
        )
        self.data[name] = values.where(valid)

    def evaluate_metrics(
        self,
        score_name: str,
        threshold: Optional[ThresholdSpec] = None,
        n_bootstraps: Optional[int] = 1000,
        *,
        inference: Optional[InferenceConfig] = None,
        cohort: Optional[str] = None,
    ) -> ScoreEvaluation:
        """Evaluate model performance for a given score across all features.

        Computes all configured metrics stratified by each feature, with optional
        bootstrap confidence intervals.

        Args:
            score_name: Name of the score column to evaluate.
            threshold: Scalar threshold or conditional threshold specification for
                binarizing scores. If None, uses the threshold defined in the
                AuditorScore object.
            n_bootstraps: Positive resample count, default 1000. None disables all
                intervals, including analytic ones. IID auto rate/OR intervals use
                analytic methods without consuming this resample count.

            inference: None (default) uses InferenceConfig(). Interval method, confidence level, seed, resampling unit,
                and missing-feature policy. Intervals are pointwise and condition
                on fixed predictions and thresholds; no selection correction.
            cohort: Optional evaluation-cohort identifier (default None) saved with provenance.

        Returns:
            ScoreEvaluation object containing metrics for all features and levels.

        Raises:
            ValueError: If no data has been added with .add_data() first.
            ValueError: If no outcome has been defined with .add_outcome() first.
            ValueError: If no metrics have been defined with .set_metrics() first.
            ValueError: If score_name is not found in the registered scores.
            ValueError: If a required decision threshold is not configured.

        Notes:
            Results include synthetic "overall" first, followed by registered features.
            Categorical levels retain declared order and unobserved NaN placeholders;
            other levels are sorted as strings. Missing features default to exclusion,
            recorded per feature. Input data is not mutated. Invalid bootstrap counts,
            empty data, missing columns, nonfinite scores, nonbinary/missing truth,
            invalid thresholds or cluster IDs raise ValueError. Undefined metrics are
            NaN, not zero. See InferenceConfig for bootstrap failure diagnostics.
            A tqdm feature progress bar is displayed during evaluation."""
        if not self.metrics:
            raise ValueError(
                "Please define at least one metric with .set_metrics() first"
            )
        inference = inference or InferenceConfig()
        score, threshold, data_slice, eval_features = self._prepare_evaluation(
            score_name,
            threshold,
            n_bootstraps,
            inference=inference,
            require_threshold=self._requires_threshold(),
        )
        self._collect_inputs()
        data_slice = self._apply_inputs(data_slice)

        score_eval: ScoreEvaluation = ScoreEvaluation(
            name=score.name,
            label=score.label if score.label is not None else score.name,
        )
        score_eval.metadata = self._metadata(
            score_name, threshold, n_bootstraps, inference, cohort
        )
        with tqdm(
            eval_features.values(), position=0, leave=True, desc="Features"
        ) as pbar:
            for feature in pbar:
                pbar.set_postfix({"name": feature.name})

                # e.g. {"f1": {'levelA': 0.2, 'levelB': 0.4}, ... }
                feature_eval: FeatureEvaluation = evaluate_feature(
                    metrics=self.metrics,
                    data=data_slice,
                    feature=feature,
                    n_bootstraps=n_bootstraps,
                    inference=inference,
                )
                score_eval.features[feature.name] = feature_eval

        return score_eval

    def evaluate_errors(
        self,
        score_name: str,
        threshold: Optional[ThresholdSpec] = None,
        n_bootstraps: Optional[int] = 1000,
        *,
        inference: Optional[InferenceConfig] = None,
        cohort: Optional[str] = None,
        error_metric: Optional[AuditorErrorMetric] = None,
    ) -> ErrorEvaluation:
        """Analyse feature-level odds of confusion-matrix group membership.

        For each confusion-matrix group (TP, TN, FP, FN) and each registered
        feature, computes the canonical 2x2 odds ratio for every feature level
        versus all other levels combined:

            OR = (a * d) / (b * c)

        Where a = count(level ∩ group), b = count(level ∩ not-group),
        c = count(not-level ∩ group), d = count(not-level ∩ not-group).

        OR = 1 means the level has the same odds of being in the group as
        all others combined.  OR > 1 means over-represented; OR < 1 means
        under-represented.

        With intervals enabled (n_bootstraps is not None), the stored
        point estimate remains the original-table OR. IID auto inference uses a
        conditional exact interval; other designs use diagnosed resampling.
        Membership enrichment is not a class-conditional error-rate comparison.

        Args:
            score_name: Name of the score column to evaluate.
            threshold: Scalar threshold or conditional threshold specification for
                binarizing scores. If None, uses the threshold defined in the
                AuditorScore object.
            n_bootstraps: Positive resample count, default 1000. None disables all
                intervals, including analytic ones; auto analytic methods need no
                resampling.

            inference: None (default) uses InferenceConfig(). Interval method, confidence level, seed, resampling unit,
                and missing-feature policy. Intervals are pointwise and condition
                on fixed predictions and thresholds; no selection correction.
            cohort: Optional evaluation-cohort identifier (default None) saved with provenance.
            error_metric: Optional custom count-based enrichment metric. Use
                to_numeric_dataframe() for generic exports; the legacy wide
                export is specific to OddsRatio.

        Returns:
            ErrorEvaluation containing one ScoreEvaluation per confusion group.

        Raises:
            ValueError: If no data has been added with .add_data() first.
            ValueError: If no outcome has been defined with .add_outcome() first.
            ValueError: If score_name is not found in the registered scores.
            ValueError: If a required decision threshold is not configured.

        Notes:
            Results include synthetic "overall" first, followed by registered features.
            Categorical levels retain declared order and unobserved NaN placeholders;
            other levels are sorted as strings. Missing features default to exclusion,
            recorded per feature. Input data is not mutated. Invalid bootstrap counts,
            empty data, missing columns, nonfinite scores, nonbinary/missing truth,
            invalid thresholds or cluster IDs raise ValueError. Undefined metrics are
            NaN, not zero. See InferenceConfig for bootstrap failure diagnostics."""
        inference = inference or InferenceConfig()
        score, threshold, data_slice, eval_features = self._prepare_evaluation(
            score_name, threshold, n_bootstraps, inference=inference
        )
        self._add_confusion_columns(data_slice)

        score_label = score.label if score.label is not None else score.name
        error_eval = ErrorEvaluation(
            name=score.name,
            label=score_label,
            threshold=threshold,
        )

        error_eval.metadata = self._metadata(
            score_name, threshold, n_bootstraps, inference, cohort
        )
        error_eval.metadata["estimand"] = "confusion_membership_enrichment_vs_rest"

        # Global dataset size used as the denominator for all % overall calculations.
        # Computed from the full data slice, before any per-feature dropna.
        global_total_n = len(data_slice)
        error_eval.global_total_n = global_total_n

        metric = error_metric if error_metric is not None else OddsRatio()
        error_eval.metadata["error_metric"] = metric.name

        for group_col in ("tp", "tn", "fp", "fn"):
            group_eval = ScoreEvaluation(
                name=group_col,
                label=group_col.upper(),
                metadata=deepcopy(error_eval.metadata),
            )
            for feature in eval_features.values():
                feature_eval, support = evaluate_error_feature(
                    data=data_slice,
                    group_col=group_col,
                    feature=feature,
                    metric=metric,
                    n_bootstraps=n_bootstraps,
                    inference=inference,
                    global_total_n=global_total_n,
                )
                group_eval.features[feature.name] = feature_eval
                # Accumulate per-feature support counts into the error evaluation's sidecar.
                error_eval.support_data.setdefault(group_col, {})[feature.name] = (
                    support
                )
            error_eval.groups[group_col] = group_eval

        return error_eval

    def evaluate_calibration(
        self,
        score_name: str,
        *,
        bins: int = 10,
        n_bootstraps: Optional[int] = 1000,
        inference: Optional[InferenceConfig] = None,
        cohort: Optional[str] = None,
    ) -> CalibrationEvaluation:
        """Evaluate probability accuracy and fixed-bin reliability by subgroup.

        Args:
            score_name: Registered score column with finite probabilities in [0, 1].
            bins: Positive integer number of equal-width bins; default 10. Bins are
                left-closed and right-open except the final bin includes 1.
            n_bootstraps: Positive resample count, default 1000; None disables all
                intervals. IID auto frequency intervals use analytic rate intervals.
            inference: Optional InferenceConfig; None uses its default policies.
            cohort: Optional cohort identifier stored in metadata; default None.

        Returns:
            CalibrationEvaluation with a bins DataFrame and a ScoreEvaluation summary
            of Brier score, log loss, calibration intercept and slope. The configured
            metric list is not replaced. Empty bins have n=0 and NaN estimates; bins
            are produced for observed levels, while summary retains empty categories.

        Raises:
            ValueError: Invalid bins, probabilities, data, truth, score registration,
                inference inputs or bootstrap count. An existing configured threshold
                is still validated even though these metrics do not require one.

        Notes:
            Data is not mutated. Includes overall and registered features. Missing
            features follow inference.missing. Bin intervals concern observed frequency,
            not uncertainty in mean predicted scores or simultaneous calibration bands.
            Intercept/slope fits can return NaN for boundary probabilities, missing
            classes or failed fits; slope also requires variation and no separation."""
        from model_auditor._evaluation import _prepare_feature_data, evaluate_level
        from model_auditor.metrics import (
            BrierScore,
            CalibrationIntercept,
            CalibrationSlope,
            LogLoss,
            Prevalence,
            validate_probabilities,
        )

        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError("bins must be a positive integer.")
        config = inference or InferenceConfig()
        _, threshold, data, features = self._prepare_evaluation(
            score_name, None, n_bootstraps, inference=config, require_threshold=False
        )
        validate_probabilities(data)
        summary = ScoreEvaluation(score_name, score_name)
        metadata = self._metadata(score_name, threshold, n_bootstraps, config, cohort)
        metadata.update(
            binning="equal_width_fixed",
            bins=bins,
            estimand="binary_probability_calibration",
        )
        summary.metadata = deepcopy(metadata)
        metrics = [BrierScore(), LogLoss(), CalibrationIntercept(), CalibrationSlope()]
        metadata["metrics"] = [
            {"name": metric.name, "parameters": getattr(metric, "parameters", {})}
            for metric in metrics
        ]
        summary.metadata = deepcopy(metadata)
        rows = []
        edges = np.linspace(0, 1, bins + 1)
        for feature in features.values():
            summary.features[feature.name] = evaluate_feature(
                metrics, data, feature, n_bootstraps, config
            )
            full, _ = _prepare_feature_data(data, feature.name, config.missing)
            for name, group in full.groupby(feature.name, observed=True):
                indices = np.minimum(
                    np.searchsorted(edges, group["_pred"], side="right") - 1, bins - 1
                )
                for i in range(bins):
                    subset = group.iloc[np.flatnonzero(indices == i)]
                    level = evaluate_level(
                        [Prevalence()], subset, str(i), n_bootstraps, config
                    )
                    metric = level.metrics["prevalence"]
                    rows.append(
                        {
                            "feature": feature.name,
                            "level": str(name),
                            "bin": i,
                            "bin_lower": edges[i],
                            "bin_upper": edges[i + 1],
                            "n": len(subset),
                            "n_pos": level.support["n_pos"],
                            "mean_prediction": subset["_pred"].mean(),
                            "observed_frequency": metric.score,
                            "lower": metric.interval[0] if metric.interval else np.nan,
                            "upper": metric.interval[1] if metric.interval else np.nan,
                            "interval_status": metric.interval_status,
                            "interval_method": metric.interval_method,
                        }
                    )
        frame = pd.DataFrame(rows)
        frame.attrs["metadata"] = deepcopy(metadata)
        return CalibrationEvaluation(frame, summary, metadata)

    def decision_curve(self, score_name: str, thresholds: list[float]) -> pd.DataFrame:
        """Compute descriptive net benefit for probability thresholds on all rows.

        Net benefit is TP/N - FP/N * t/(1-t), in true-positive equivalents per row.
        Threshold probability encodes relative false-positive harm. No confidence
        bands, subgroup estimates or automatic threshold selection are provided.

        Args:
            score_name: Registered column of finite probabilities in [0, 1].
            thresholds: Nonempty sequence of finite probabilities strictly in (0, 1).
                Rows preserve supplied order and duplicates; scores >= t are positive.

        Returns:
            New DataFrame with threshold, net_benefit, act_all, act_none (zero),
            selection_rate (fraction) and n columns. Its interval_scope attribute in
            ``attrs`` is "descriptive_only". Data and configured thresholds are unchanged.

        Raises:
            ValueError: Invalid thresholds, probabilities, data, truth or registrations.
                Any configured score threshold is also validated."""
        from model_auditor.metrics import validate_probabilities

        _, _, data, _ = self._prepare_evaluation(
            score_name, None, None, require_threshold=False
        )
        probabilities = validate_probabilities(data)
        values = np.asarray(thresholds, dtype=float)
        if (
            values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any((values <= 0) | (values >= 1))
        ):
            raise ValueError(
                "Threshold probabilities must be a nonempty sequence in (0, 1)."
            )
        truth = data["_truth"].to_numpy()
        rows = []
        for t in values:
            predictions = probabilities >= t
            tp = np.sum(predictions & (truth == 1))
            fp = np.sum(predictions & (truth == 0))
            rows.append(
                {
                    "threshold": t,
                    "net_benefit": (tp - fp * t / (1 - t)) / len(data),
                    "act_all": truth.mean() - (1 - truth.mean()) * t / (1 - t),
                    "act_none": 0.0,
                    "selection_rate": predictions.mean(),
                    "n": len(data),
                }
            )
        result = pd.DataFrame(rows)
        result.attrs["interval_scope"] = "descriptive_only"
        return result

    def compare_scores(
        self,
        score_name: str,
        reference_score: str,
        *,
        threshold: Optional[ThresholdSpec] = None,
        reference_threshold: Optional[ThresholdSpec] = None,
        contrast: Literal["difference", "ratio"] = "difference",
        n_bootstraps: Optional[int] = 1000,
        inference: Optional[InferenceConfig] = None,
        cohort: Optional[str] = None,
    ) -> ScoreEvaluation:
        """Compare paired scores on the same rows using configured metrics.

        Args:
            score_name: Registered score whose metrics are contrasted.
            reference_score: Registered reference score on the same data. Pairing is
                positional and remains valid with duplicate DataFrame indices.
            reference_threshold: Reference policy override; None uses its registration.
            threshold: Optional override for the score's scalar/conditional policy;
                None uses its registered threshold. Decision metrics require a policy.
            contrast: "difference" (default) subtracts reference; "ratio" divides by it.
                Nonfinite operands or a zero ratio denominator produce NaN.
            n_bootstraps: Positive count, default 1000; None disables intervals.
                Contrasts use percentile resampling even with method="auto".
            inference: Optional InferenceConfig; None uses defaults. Shared resamples
                preserve pairing/overlap. Missing features follow this configuration.
            cohort: Optional cohort identifier in provenance; default None.

        Returns:
            New ScoreEvaluation with original-sample metric contrasts and interval
            diagnostics. No input data or registrations are changed.

        Raises:
            ValueError: Invalid contrast, no metrics, invalid data/truth/scores,
                thresholds, bootstrap count or inference inputs.

        Notes:
            Intervals are pointwise, without selection or multiplicity correction.
            They condition on supplied predictions; evaluate on independent data.
            Includes overall and all registered features, preserving evaluation order.
            Identical paired predictions can have a degenerate but valid interval."""
        from model_auditor._comparisons import contrast_estimate, evaluate_contrast
        from model_auditor._evaluation import _prepare_feature_data, support_counts
        from model_auditor.schemas import LevelEvaluation

        if contrast not in {"difference", "ratio"} or not self.metrics:
            raise ValueError("Choose difference or ratio and configure metrics first.")
        config = inference or InferenceConfig()
        _, threshold, left, features = self._prepare_evaluation(
            score_name,
            threshold,
            n_bootstraps,
            inference=config,
            require_threshold=self._requires_threshold(),
        )
        _, reference_threshold, right, _ = self._prepare_evaluation(
            reference_score,
            reference_threshold,
            n_bootstraps,
            inference=config,
            require_threshold=self._requires_threshold(),
        )
        self._collect_inputs()
        left, right = self._apply_inputs(left), self._apply_inputs(right)
        # Positional pairing remains valid even with duplicate DataFrame indices.
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        paired = left.copy()
        if {"__reference_score", "__reference_binary"} & set(paired.columns):
            raise ValueError(
                "Comparison input columns conflict with reserved reference columns."
            )
        paired["__reference_score"] = right["_pred"]
        paired["__reference_binary"] = right["_binary_pred"]
        result = ScoreEvaluation(score_name, f"{score_name} vs {reference_score}")
        result.metadata = self._metadata(
            score_name, threshold, n_bootstraps, config, cohort
        )
        result.metadata.update(
            reference_score=reference_score,
            reference_threshold=asdict(reference_threshold)
            if isinstance(reference_threshold, ConditionalThreshold)
            else reference_threshold,
            contrast=contrast,
            estimand="paired_model_contrast",
        )
        for feature in features.values():
            data, categories = _prepare_feature_data(
                paired, feature.name, config.missing
            )
            fe = FeatureEvaluation(
                feature.name,
                feature.label or feature.name,
                excluded_n=len(paired) - len(data),
                total_n=len(paired),
            )
            for name, group in data.groupby(feature.name, observed=True):
                level = LevelEvaluation(str(name), support=support_counts(group))
                for metric in self.metrics:

                    def statistic(boot, metric=metric):
                        reference = boot.copy()
                        reference["_pred"] = boot["__reference_score"]
                        reference["_binary_pred"] = boot["__reference_binary"]
                        reference = self._apply_inputs(reference)
                        return contrast_estimate(
                            metric.data_call(boot),
                            metric.data_call(reference),
                            contrast,
                        )

                    identical = group["_pred"].equals(
                        group["__reference_score"]
                    ) and group["_binary_pred"].equals(group["__reference_binary"])
                    level.metrics[metric.name] = evaluate_contrast(
                        group,
                        statistic,
                        metric.name,
                        metric.label,
                        n_bootstraps if metric.ci_eligible else None,
                        config,
                        identical=identical,
                    )
                fe.levels[str(name)] = level
            if categories is not None:
                for name in categories:
                    if name not in fe.levels:
                        level = LevelEvaluation(
                            name, support=support_counts(data.iloc[:0])
                        )
                        for metric in self.metrics:
                            level.update(metric.name, metric.label, float("nan"))
                            level.metrics[metric.name].status = "undefined"
                        fe.levels[name] = level
                fe.levels = {name: fe.levels[name] for name in categories}
            result.features[feature.name] = fe
        return result

    def compare_groups(
        self,
        score_name: str,
        feature: str,
        reference: str,
        *,
        threshold: Optional[ThresholdSpec] = None,
        contrast: Literal["difference", "ratio"] = "difference",
        n_bootstraps: Optional[int] = 1000,
        inference: Optional[InferenceConfig] = None,
        cohort: Optional[str] = None,
    ) -> ScoreEvaluation:
        """Compare each subgroup with an observed reference using configured metrics.

        Args:
            score_name: Registered score to evaluate.
            feature: Registered subgroup feature name.
            reference: Observed level's string label, as shown in result objects.
                An absent reference raises ValueError. The reference itself is omitted
                from returned levels; declared unobserved categories remain as NaN.
            threshold: Optional override for the score's scalar/conditional policy;
                None uses its registered threshold. Decision metrics require a policy.
            contrast: "difference" (default) subtracts reference; "ratio" divides by it.
                Nonfinite operands or a zero ratio denominator produce NaN.
            n_bootstraps: Positive count, default 1000; None disables intervals.
                Contrasts use percentile resampling even with method="auto".
            inference: Optional InferenceConfig; None uses defaults. Shared resamples
                preserve pairing/overlap. Missing features follow this configuration.
            cohort: Optional cohort identifier in provenance; default None.

        Returns:
            New ScoreEvaluation with original-sample metric contrasts and interval
            diagnostics. No input data or registrations are changed.

        Raises:
            ValueError: Invalid contrast, no metrics, invalid data/truth/scores,
                thresholds, bootstrap count or inference inputs.

        Notes:
            Intervals are pointwise, without selection or multiplicity correction.
            They condition on supplied predictions; evaluate on independent data.
            Only the requested feature is returned. FPR/FNR contrasts compare
            class-conditional rates, without implying causal fairness."""
        from model_auditor._comparisons import contrast_estimate, evaluate_contrast
        from model_auditor._evaluation import _prepare_feature_data, support_counts
        from model_auditor.schemas import LevelEvaluation

        if contrast not in {"difference", "ratio"} or not self.metrics:
            raise ValueError("Choose difference or ratio and configure metrics first.")
        config = inference or InferenceConfig()
        _, threshold, data, features = self._prepare_evaluation(
            score_name,
            threshold,
            n_bootstraps,
            inference=config,
            require_threshold=self._requires_threshold(),
        )
        if feature not in self.features:
            raise ValueError("Register the comparison feature first.")
        self._collect_inputs()
        data = self._apply_inputs(data)
        full_n = len(data)
        data, categories = _prepare_feature_data(data, feature, config.missing)
        if reference not in set(data[feature].astype(str)):
            raise ValueError("Reference must be an observed feature level.")
        result = ScoreEvaluation(score_name, f"{feature} vs {reference}")
        result.metadata = self._metadata(
            score_name, threshold, n_bootstraps, config, cohort
        )
        result.metadata.update(
            reference_level=reference,
            contrast=contrast,
            estimand="subgroup_metric_contrast",
        )
        fe = FeatureEvaluation(
            feature,
            features[feature].label or feature,
            excluded_n=full_n - len(data),
            total_n=full_n,
        )
        names = (
            categories
            if categories is not None
            else sorted(data[feature].astype(str).unique())
        )
        for name in names:
            if name == reference:
                continue
            level = LevelEvaluation(
                name,
                support=support_counts(data.loc[data[feature].astype(str) == name]),
            )
            for metric in self.metrics:

                def statistic(boot, metric=metric, name=name):
                    left = boot.loc[boot[feature].astype(str) == name]
                    right = boot.loc[boot[feature].astype(str) == reference]
                    if left.empty or right.empty:
                        return float("nan")
                    return contrast_estimate(
                        metric.data_call(left), metric.data_call(right), contrast
                    )

                level.metrics[metric.name] = evaluate_contrast(
                    data,
                    statistic,
                    metric.name,
                    metric.label,
                    n_bootstraps if metric.ci_eligible else None,
                    config,
                )
            fe.levels[name] = level
        result.features[feature] = fe
        return result

    def optimize_score_threshold(self, score_name: str) -> float:
        """Optimize a score threshold using the Youden index (sensitivity - FPR).

        Args:
            score_name: Name of the target score.

        Raises:
            ValueError: If no scores have been defined with .add_score() first.
            ValueError: If no data has been added with .add_data() first.
            ValueError: If no outcome variable has been defined with .add_outcome() first.

        Returns:
            Finite threshold maximizing the Youden criterion among observed scores.

        Notes:
            Requires finite numeric scores and both truth classes. Emits UserWarning
            identifying selection on tuning data. Returns a scalar without changing
            the registered threshold; pass it to add_score() or an evaluation call.
            Evaluate the selected policy on an independent cohort.
            Ties select the first (largest) finite threshold on the ROC curve."""
        score, fpr, tpr, thresholds = self._prepare_score_roc_curve(
            score_name=score_name
        )

        finite_indices = np.flatnonzero(
            np.isfinite(thresholds) & np.isfinite(tpr - fpr)
        )
        if not finite_indices.size:
            raise ValueError(
                f"No finite Youden threshold is available for score '{score.name}'."
            )
        idx = finite_indices[np.argmax((tpr - fpr)[finite_indices])]
        optimal_threshold: float = float(thresholds[idx])

        warnings.warn(
            f"Optimal threshold for '{score.name}' found at: {optimal_threshold}. "
            "Selected on these data; evaluate on an independent cohort. Youden maximizes balanced accuracy, not general decision utility.",
            stacklevel=2,
        )
        return optimal_threshold

    def optimize_score_threshold_for_target(
        self,
        score_name: str,
        target: float,
        metric: Literal["sensitivity", "specificity"] = "sensitivity",
    ) -> float:
        """Optimize a score threshold to satisfy a target operating metric.

        Args:
            score_name: Name of the target score.
            target: Requested minimum metric value in [0.0, 1.0].
            metric: Metric constraint to satisfy; one of 'sensitivity' or
                'specificity'.

        Raises:
            ValueError: If no scores have been defined with .add_score() first.
            ValueError: If no data has been added with .add_data() first.
            ValueError: If no outcome variable has been defined with .add_outcome() first.
            ValueError: If score_name is not a registered score.
            ValueError: If target is outside [0.0, 1.0].
            ValueError: If metric is not 'sensitivity' or 'specificity'.
            ValueError: If no finite threshold satisfies the requested target.

        Returns:
            Threshold that satisfies the target constraint with the requested
            tie-breaking rule.

        Notes:
            Requires finite numeric scores and both truth classes. Emits UserWarning
            identifying selection on tuning data. Returns a scalar without changing
            the registered threshold; pass it to add_score() or an evaluation call.
            Evaluate the selected policy on an independent cohort.
            With sensitivity (default), chooses the largest feasible threshold; with
            specificity, the smallest feasible threshold. Candidates include observed
            scores and a finite all-negative endpoint above the maximum if representable."""
        if metric not in ("sensitivity", "specificity"):
            raise ValueError(
                "metric must be either 'sensitivity' or 'specificity'. "
                f"Received: {metric}"
            )

        if not np.isfinite(target) or target < 0.0 or target > 1.0:
            raise ValueError(
                f"target must be between 0.0 and 1.0 inclusive. Received: {target}"
            )

        if len(self.scores) == 0:
            raise ValueError("Please define at least one score first")

        if score_name not in self.scores:
            available = ", ".join(self.scores.keys()) or "(none)"
            raise ValueError(
                f"Score '{score_name}' not found. Available scores: {available}"
            )

        score, fpr, tpr, thresholds = self._prepare_score_roc_curve(
            score_name=score_name,
            drop_intermediate=False,
        )

        # sklearn's infinity endpoint represents predicting every case negative.
        with np.errstate(over="ignore"):
            above_max = np.nextafter(float(self.data[score.name].max()), np.inf)
        if np.isfinite(above_max):
            thresholds[0] = above_max
        metric_values = tpr if metric == "sensitivity" else 1.0 - fpr
        finite_threshold_mask = np.isfinite(thresholds)
        feasible_indices = np.flatnonzero(
            (metric_values >= target) & finite_threshold_mask
        )

        if feasible_indices.size == 0:
            finite_metric_values = metric_values[finite_threshold_mask]
            if finite_metric_values.size == 0:
                raise ValueError(
                    f"No finite threshold is available for score '{score.name}'."
                )

            min_achievable = float(np.nanmin(finite_metric_values))
            max_achievable = float(np.nanmax(finite_metric_values))
            raise ValueError(
                f"No finite threshold for score '{score.name}' can satisfy "
                f"{metric} >= {target:.3f}. Achievable {metric} range across "
                f"finite thresholds is [{min_achievable:.3f}, {max_achievable:.3f}]."
            )

        selected_idx = (
            int(feasible_indices[0])
            if metric == "sensitivity"
            else int(feasible_indices[-1])
        )
        optimal_threshold = float(thresholds[selected_idx])

        warnings.warn(
            f"Optimal threshold for '{score.name}' satisfying "
            f"{metric} >= {target:.3f} found at: {optimal_threshold}. "
            "This is an empirical tuning constraint, not a population guarantee; evaluate on an independent cohort."
        )
        return optimal_threshold

    def plot_score_distributions(
        self,
        score_name: str,
        feature_names: Optional[list[str]] = None,
        bins: int | str | _Sequence[float] | NDArray[np.floating] = 30,
        density: bool = True,
        split_classes: bool = False,
    ) -> dict[str, tuple[Figure, NDArray[np.object_]]]:
        """Visualize score distributions stratified by feature levels.

        For each selected feature, produces a figure with vertically stacked
        histograms — one subplot per level — all sharing the same x-axis
        (score values).  Bins are computed once from the full feature slice
        so every level uses an identical bin grid, enabling direct visual
        comparison across subgroups.

        Requires matplotlib.  Does not require ``evaluate_metrics()`` to have
        been called first; it operates directly on the raw data and the
        configured features and score.

        Args:
            score_name: Name of the score column to visualize.  Must have
                been registered with ``add_score()``.
            feature_names: Feature names to plot.  ``None`` (default) plots
                all registered features in insertion order.  An explicit list
                preserves the caller-supplied order.
            bins: Bin specification forwarded to ``numpy.histogram_bin_edges``.
                Accepts an integer (number of bins), a string strategy such as
                ``"auto"`` or ``"fd"``, or a precomputed bin-edge array.
                Default is ``30``.
            density: If ``True`` (default), normalize each histogram so that
                the total area integrates to 1.  Pass ``False`` for raw
                observation counts.
            split_classes: If True, overlay negative and positive class histograms
                using the configured outcome. Each class is normalized separately
                when density=True. Defaults to False (combined distribution).

        Returns:
            Dictionary mapping feature name to a
            ``(matplotlib.figure.Figure, numpy.ndarray)`` tuple.  The ndarray
            is a 1-D array of ``Axes`` objects, one per level, ordered top to
            bottom. Observed categorical levels follow declared order; other
            levels use first appearance (unlike sorted evaluation tables).
            Rows with missing feature or score values are omitted. Each axes
            array has dtype object. Figures are not shown or closed; callers
            own them and should call ``matplotlib.pyplot.close(fig)`` when done.

        Raises:
            ImportError: If matplotlib is not installed.
            ValueError: If no data has been added with ``add_data()``.
            ValueError: If no features have been configured with
                ``add_feature()``.
            ValueError: If ``score_name`` is not a registered score.
            ValueError: If any name in ``feature_names`` is not a registered
                feature.
            ValueError: If a selected feature has no plottable levels after
                null values are filtered out.
        """
        from model_auditor.plotting.distributions import plot_score_distributions

        return plot_score_distributions(
            self, score_name, feature_names, bins, density, split_classes
        )

    def _evaluation_columns(self, threshold: Optional[ThresholdSpec]) -> list[str]:
        """Return base evaluation columns, including conditional-threshold feature."""
        column_list: list[str] = [*self.features.keys(), "_truth"]
        if (
            isinstance(threshold, ConditionalThreshold)
            and threshold.feature not in column_list
        ):
            column_list.append(threshold.feature)
        return column_list

    def _prepare_score_roc_curve(
        self,
        score_name: str,
        drop_intermediate: bool = True,
    ) -> tuple[
        AuditorScore, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        """Validate optimization prerequisites and compute ROC arrays for a score.

        Args:
            score_name: Name of the score to optimize.
            drop_intermediate: Whether to drop suboptimal thresholds while building
                the ROC curve.

        Returns:
            Tuple of (score object, fpr array, tpr array, threshold array).

        Raises:
            ValueError: If no scores have been defined with .add_score() first.
            ValueError: If no data has been added with .add_data() first.
            ValueError: If no outcome variable has been defined with .add_outcome() first.
            KeyError: If score_name is not found in self.scores.
        """
        if len(self.scores) == 0:
            raise ValueError("Please define at least one score first")
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")
        if "_truth" not in self.data.columns.tolist():
            raise ValueError(
                "Please define an outcome variable data with .add_outcome() first"
            )

        # Keep KeyError behaviour for unknown score names in the Youden optimizer.
        score: AuditorScore = self.scores[score_name]
        _, _, data, _ = self._prepare_evaluation(
            score_name, None, None, require_threshold=False
        )
        if data["_truth"].nunique() != 2:
            raise ValueError("Threshold optimization requires both outcome classes.")
        score_list = data["_pred"].to_numpy(dtype=float)
        truth_list = data["_truth"].to_numpy(dtype=float)

        fpr, tpr, thresholds = roc_curve(
            truth_list,
            score_list,
            drop_intermediate=drop_intermediate,
        )
        return score, fpr, tpr, thresholds

    def _collect_inputs(self) -> None:
        """
        Collects the minimum set of metric inputs necessary for evaluation
        (based on the metrics defined in self.metrics with the .set_metrics() method)
        """
        inputs_set: set[str] = set()
        for metric in self.metrics:
            inputs_set.update(metric.inputs)

        inputs_dict: dict[str, Type[AuditorMetricInput]] = collect_metric_inputs()

        # reinit self._inputs and add all necessary inputs to it
        self._inputs: list[Type[AuditorMetricInput]] = list()
        for input_name in sorted(inputs_set):
            if input_name not in {"_truth", "_pred", "_binary_pred"}:
                if (
                    input_name not in inputs_dict
                    and self.data is not None
                    and input_name in self.data
                ):
                    continue
                if input_name not in inputs_dict:
                    raise ValueError(f"Unknown metric input {input_name!r}.")
                self._inputs.append(inputs_dict[input_name])

    def _apply_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method to apply the metric input functions (collected with .collect_inputs())
        to the target data to prepare it for metric calculation

        Args:
            data (pd.DataFrame): Dataframe to add input columns to

        Returns:
            pd.DataFrame: Transformed dataframe with metric input columns
        """
        for input_type in self._inputs:
            metric_input = input_type()
            data: pd.DataFrame = metric_input.data_transform(data)

        return data

    def _add_confusion_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add tp/tn/fp/fn indicator columns to a DataFrame in-place.

        Requires '_truth' and '_binary_pred' columns to already be present.

        Args:
            data: DataFrame to augment.

        Returns:
            The same DataFrame with tp, tn, fp, fn integer columns added.
        """
        for input_type in (
            TruePositives,
            TrueNegatives,
            FalsePositives,
            FalseNegatives,
        ):
            input_type().data_transform(data)
        return data

    def _prepare_evaluation(
        self,
        score_name: str,
        threshold: Optional[ThresholdSpec],
        n_bootstraps: Optional[int],
        *,
        inference: Optional[InferenceConfig] = None,
        require_threshold: bool = True,
    ) -> tuple[
        AuditorScore, Optional[ThresholdSpec], pd.DataFrame, dict[str, AuditorFeature]
    ]:
        """Validate evaluation inputs and prepare a private slice for either analysis."""
        validate_n_bootstraps(n_bootstraps)
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")
        if self.outcome is None or "_truth" not in self.data.columns:
            raise ValueError(
                "Please define an outcome variable with .add_outcome() first"
            )
        if score_name not in self.scores:
            available = ", ".join(self.scores) or "(none)"
            raise ValueError(
                f"Score '{score_name}' not found. Available scores: {available}"
            )
        if not self.data.columns.is_unique:
            raise ValueError("Evaluation data must have unique column names.")
        score = self.scores[score_name]
        if require_threshold or threshold is not None or score.threshold is not None:
            threshold = resolve_threshold(score, threshold)
        columns = self._evaluation_columns(threshold)
        if inference is not None and inference.cluster is not None:
            if inference.cluster in {
                "_truth",
                "_pred",
                "_binary_pred",
                "overall",
                "tp",
                "tn",
                "fp",
                "fn",
            }:
                raise ValueError(
                    "Cluster column cannot be a reserved evaluation column."
                )
            if inference.cluster not in columns:
                columns.append(inference.cluster)
        for metric in self.metrics:
            for name in metric.inputs:
                if (
                    name in self.data
                    and name
                    not in {"_truth", "_pred", "_binary_pred", "tp", "tn", "fp", "fn"}
                    and name not in columns
                ):
                    columns.append(name)
        missing = [name for name in [*columns, score.name] if name not in self.data]
        if missing:
            raise ValueError(f"Evaluation columns not found in data: {missing!r}")
        if self.data.empty:
            raise ValueError("Evaluation data must contain at least one row.")
        if not self.data["_truth"].isin([0, 1]).all():
            raise ValueError(
                "Outcome values must be binary (0 or 1) with no missing values; check the outcome mapping."
            )
        scores = self.data[score.name]
        if (
            not pd.api.types.is_numeric_dtype(scores)
            or not np.isfinite(scores).all()
            or scores.isna().any()
        ):
            raise ValueError(
                f"Score '{score.name}' must contain finite numeric values with no missing values."
            )
        data = self.data.loc[:, columns].copy()
        data["_pred"] = scores
        data["_binary_pred"] = (
            binarize(data["_pred"], build_threshold_series(data, threshold))
            if threshold is not None
            else np.nan
        )
        if (
            inference is not None
            and inference.cluster is not None
            and data[inference.cluster].isna().any()
        ):
            raise ValueError("Cluster IDs must have no missing values.")
        data["overall"] = "Overall"
        features = {
            "overall": AuditorFeature(name="overall", label="Overall"),
            **self.features,
        }
        return score, threshold, data, features

    def _evaluate_confidence_interval(
        self, data: pd.DataFrame, n_bootstraps: int
    ) -> dict[str, tuple[float, float]]:
        """Calculate bootstrap intervals for the configured eligible metrics."""
        return evaluate_confidence_interval(self.metrics, data, n_bootstraps)
