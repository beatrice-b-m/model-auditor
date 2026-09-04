"""Core auditor module for ML model evaluation.

This module contains the main Auditor class that orchestrates model evaluation
across different features and subgroups, with support for bootstrap confidence
intervals.
"""

import warnings
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import version
from typing import Any, Literal, Optional, Type, Union

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
    ConditionalThreshold,
    ErrorEvaluation,
    FeatureEvaluation,
    InferenceConfig,
    ScoreEvaluation,
    ThresholdSpec,
)
from model_auditor.utils import collect_metric_inputs


class Auditor:
    """Main class for auditing ML model performance across subgroups.

    The Auditor class provides a flexible interface for evaluating model
    predictions stratified by features (subgroups), supporting multiple
    metrics and bootstrap confidence interval calculation.

    Attributes:
        data: DataFrame containing the evaluation data.
        features: Dictionary mapping feature names to AuditorFeature objects.
        scores: Dictionary mapping score names to AuditorScore objects.
        metrics: List of metrics to compute during evaluation.

    Example:
        >>> auditor = Auditor()
        >>> auditor.add_data(df)
        >>> auditor.add_feature(name="age_group")
        >>> auditor.add_score(name="risk_score", threshold=0.5)
        >>> auditor.add_outcome(name="outcome")
        >>> auditor.set_metrics([Sensitivity(), Specificity()])
        >>> results = auditor.evaluate_metrics(score_name="risk_score")
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        features: Optional[list[AuditorFeature]] = None,
        scores: Optional[list[AuditorScore]] = None,
        outcome: Optional[AuditorOutcome] = None,
        metrics: Optional[list[AuditorMetric]] = None,
    ) -> None:
        """Initialize the Auditor.

        Args:
            data: DataFrame containing the data for evaluation.
            features: List of AuditorFeature objects defining stratification variables.
            scores: List of AuditorScore objects defining prediction columns.
            outcome: AuditorOutcome object defining the ground truth column.
            metrics: List of AuditorMetric objects to compute during evaluation.
        """
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
        """
        Method to add a dataframe to the auditor

        Args:
            data (pd.DataFrame): Full dataframe which will be subset for subgroup evaluation
        """
        self.data = data.copy()
        self.outcome = None

    def add_feature(
        self,
        name: str,
        label: Optional[str] = None,
    ) -> None:
        """
        Method to add a feature to the auditor. Equivalent to a grouping variable in
        packages like tableone, the score variable will be stratified by this feature

        Args:
            name (str): Column name for the feature.
            label (Optional[str], optional): Optional label for the feature. Defaults to None.
        """
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
        """
        Method to add a score to the auditor. Expects a continuous feature which will
        be used to calculate metrics and confidence intervals

        Args:
            name (str): Column name for the score.
            label (Optional[str], optional): Optional label for the score. Defaults to None.
            threshold (Optional[ThresholdSpec], optional): Scalar threshold or
            conditional threshold specification used to binarize the score column.
        """
        score = AuditorScore(
            name=name,
            label=label,
            threshold=threshold,
        )
        self.scores[score.name] = score

    def add_outcome(self, name: str, mapping: Optional[dict[Any, int]] = None) -> None:
        """Add an outcome (ground truth) variable to the auditor.

        Args:
            name: Column name for the outcome variable.
            mapping: Optional dictionary to map outcome values to binary (0/1).
                For example, {"positive": 1, "negative": 0}.

        Raises:
            ValueError: If no data has been added with .add_data() first.
        """
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")

        self.outcome = AuditorOutcome(name, deepcopy(mapping))
        if mapping is not None:
            self.data["_truth"] = self.data[name].map(mapping)
        else:
            self.data["_truth"] = self.data[name]

    def set_metrics(self, metrics: list[AuditorMetric]) -> None:
        """
        Method to define the metrics the auditor will use during evaluation of score variables.

        Args:
            metrics (list[AuditorMetric]): A list of metrics classes following the AuditorMetric
            protocol (pre-made metrics listed in model_auditor.metrics)
        """
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
            n_bootstraps: Number of bootstrap samples for confidence interval
                calculation. Set to None to disable CI calculation.

            inference: Interval method, confidence level, seed, resampling unit,
                and missing-feature policy. Intervals are pointwise and condition
                on fixed predictions and thresholds; no selection correction.
            cohort: Optional evaluation-cohort identifier saved with provenance.

        Returns:
            ScoreEvaluation object containing metrics for all features and levels.

        Raises:
            ValueError: If no data has been added with .add_data() first.
            ValueError: If no outcome has been defined with .add_outcome() first.
            ValueError: If no metrics have been defined with .set_metrics() first.
            ValueError: If score_name is not found in the registered scores.
            ValueError: If a required decision threshold is not configured.
        """
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
            n_bootstraps: Number of bootstrap samples for confidence intervals.
                Set to None to disable CI calculation.

            inference: Interval method, confidence level, seed, resampling unit,
                and missing-feature policy. Intervals are pointwise and condition
                on fixed predictions and thresholds; no selection correction.
            cohort: Optional evaluation-cohort identifier saved with provenance.
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
        """
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
        """
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
        """
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
        bins: Union[int, str] = 30,
        density: bool = True,
        split_classes: bool = False,
    ) -> dict:
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
            bottom to match the level order used in ``to_dataframe()``.

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
