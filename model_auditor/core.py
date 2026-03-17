"""Core auditor module for ML model evaluation.

This module contains the main Auditor class that orchestrates model evaluation
across different features and subgroups, with support for bootstrap confidence
intervals.
"""

from typing import Any, Optional, Type, Union
import warnings
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_curve
from tqdm.auto import tqdm

from model_auditor.error_metrics import AuditorErrorMetric, OddsRatio
from model_auditor.metric_inputs import AuditorMetricInput
from model_auditor.metrics import AuditorMetric
from model_auditor.schemas import (
    AuditorFeature,
    AuditorScore,
    AuditorOutcome,
    ErrorEvaluation,
    FeatureEvaluation,
    LevelEvaluation,
    ScoreEvaluation,
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
        if outcome is not None:
            self.add_outcome(**vars(outcome))

        # initialize metrics
        self.metrics: list[AuditorMetric] = list()
        if metrics is not None:
            self.metrics = metrics

        # initialize attrs for later
        self._inputs: list[Type[AuditorMetricInput]] = list()

    def add_data(self, data: pd.DataFrame) -> None:
        """
        Method to add a dataframe to the auditor

        Args:
            data (pd.DataFrame): Full dataframe which will be subset for subgroup evaluation
        """
        self.data = data.copy()

    def add_feature(
        self, name: str, label: Optional[str] = None,
    ) -> None:
        """
        Method to add a feature to the auditor. Equivalent to a grouping variable in
        packages like tableone, the score variable will be stratified by this feature

        Args:
            name (str): Column name for the feature.
            label (Optional[str], optional): Optional label for the feature. Defaults to None.
        """
        feature = AuditorFeature(
            name=name,
            label=label,
        )
        self.features[feature.name] = feature

    def add_score(
        self, name: str, label: Optional[str] = None, threshold: Optional[float] = None
    ) -> None:
        """
        Method to add a score to the auditor. Expects a continuous feature which will
        be used to calculate metrics and confidence intervals

        Args:
            name (str): Column name for the score.
            label (Optional[str], optional): Optional label for the score. Defaults to None.
            threshold (Optional[float], optional): Threshold used to binarize the score column.
            Defaults to None and can be optimized using the Youden index or updated separately later.
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

        if mapping is not None:
            self.data["_truth"] = self.data[name].map(mapping)
        else:
            self.data["_truth"] = self.data[name]

    def optimize_score_threshold(self, score_name: str) -> float:
        """
        Method to optimize the decision threshold for a score based on the Youden index.

        Args:
            score_name (str): Name of the target score

        Raises:
            ValueError: If no scores have been defined with .add_score() first
            ValueError: If no data has been added with .add_data() first
            ValueError: If no outcome variable has been defined with .add_outcome() first

        Returns:
            float: Optimal threshold identified
        """
        if len(self.scores) == 0:
            raise ValueError("Please define at least one score first")
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")
        elif "_truth" not in self.data.columns.tolist():
            raise ValueError(
                "Please define an outcome variable data with .add_outcome() first"
            )

        # throws an error if the score has not been defined
        score: AuditorScore = self.scores[score_name]
        score_list: list[float] = self.data[score.name].astype(float).tolist()  # type: ignore

        # otherwise the target score will be the single item in the list
        truth_list: list[float] = self.data["_truth"].astype(float).tolist()  # type: ignore

        # calculate optimal threshold
        fpr, tpr, thresholds = roc_curve(truth_list, score_list)
        idx: int = np.argmax(tpr - fpr).astype(int)
        optimal_threshold: float = thresholds[idx]

        warnings.warn(f"Optimal threshold for '{score.name}' found at: {optimal_threshold}")
        return optimal_threshold

    def set_metrics(self, metrics: list[AuditorMetric]) -> None:
        """
        Method to define the metrics the auditor will use during evaluation of score variables.

        Args:
            metrics (list[AuditorMetric]): A list of metrics classes following the AuditorMetric
            protocol (pre-made metrics listed in model_auditor.metrics)
        """
        self.metrics: list[AuditorMetric] = metrics

    def plot_score_distributions(
        self,
        score_name: str,
        feature_names: Optional[list[str]] = None,
        bins: Union[int, str] = 30,
        density: bool = True,
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
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for score distribution plots. "
                "Install it with: pip install matplotlib"
            ) from exc

        if self.data is None:
            raise ValueError("Please add data with .add_data() first")

        if not self.features:
            raise ValueError(
                "No features configured. Add features with .add_feature() first"
            )

        if score_name not in self.scores:
            available = ", ".join(self.scores.keys()) or "(none)"
            raise ValueError(
                f"Score '{score_name}' not found. Available scores: {available}"
            )

        score: AuditorScore = self.scores[score_name]
        score_label = score.label if score.label is not None else score.name

        # Resolve feature selection.
        if feature_names is None:
            selected_features = list(self.features.keys())
        else:
            unknown = [f for f in feature_names if f not in self.features]
            if unknown:
                raise ValueError(
                    f"Unknown feature(s): {unknown!r}. "
                    f"Available features: {list(self.features.keys())!r}"
                )
            selected_features = list(feature_names)

        plots: dict[str, tuple] = {}

        for fname in selected_features:
            feature = self.features[fname]
            feature_label = feature.label if feature.label is not None else feature.name
            feature_col = feature.name

            # Drop rows where the feature value or score value is null.
            feature_slice = self.data.dropna(subset=[feature_col, score.name]).copy()

            # Determine level order, honouring categorical dtype when present.
            is_categorical = isinstance(
                feature_slice[feature_col].dtype, pd.CategoricalDtype
            )
            if is_categorical:
                declared = [
                    str(c) for c in feature_slice[feature_col].cat.categories.tolist()
                ]
                observed = set(feature_slice[feature_col].astype(str).unique())
                levels = [c for c in declared if c in observed]
            else:
                # Preserve first-appearance order; coerce values to str so level
                # keys are consistent across dtype variations.
                seen: list[str] = []
                seen_set: set[str] = set()
                for val in feature_slice[feature_col].astype(str):
                    if val not in seen_set:
                        seen.append(val)
                        seen_set.add(val)
                levels = seen

            if not levels:
                raise ValueError(
                    f"Feature '{fname}' has no plottable levels after filtering "
                    "null values from the data."
                )

            # Shared bin edges — computed once from the full feature slice so
            # every level subplot uses an identical grid.
            bin_edges = np.histogram_bin_edges(
                feature_slice[score.name].values, bins=bins
            )

            n_levels = len(levels)
            fig, axes_raw = plt.subplots(
                n_levels,
                1,
                sharex=True,
                figsize=(8, max(2.0, 1.5 * n_levels)),
            )
            # plt.subplots returns a bare Axes when nrows=1; normalise to ndarray.
            axes: np.ndarray = np.atleast_1d(axes_raw)

            str_col = feature_slice[feature_col].astype(str)
            for ax, level_name in zip(axes, levels):
                level_data = feature_slice.loc[
                    str_col == level_name, score.name
                ].values

                ax.hist(level_data, bins=bin_edges, density=density, zorder=2)

                # Style: grid behind bars, no y-ticks, level label at left,
                # minimal spine set.
                ax.xaxis.grid(True, zorder=0)
                ax.set_axisbelow(True)
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_ylabel(level_name, rotation=0, ha="right", va="center")

            axes[-1].set_xlabel(score_label)
            fig.suptitle(f"{feature_label}: {score_label}")
            fig.tight_layout()

            plots[fname] = (fig, axes)

        return plots

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _resolve_threshold(
        self, score: AuditorScore, threshold: Optional[float]
    ) -> float:
        """Resolve the effective threshold for a score.

        Args:
            score: The AuditorScore containing the optional default threshold.
            threshold: Override value from the caller; None to use the score default.

        Returns:
            The resolved threshold value.

        Raises:
            ValueError: If both the caller argument and the score default are None.
        """
        if threshold is not None:
            return threshold
        if score.threshold is not None:
            return score.threshold
        raise ValueError(
            f"Threshold for score '{score.name}' must be defined via "
            "add_score(threshold=...) or passed to the evaluation method."
        )

    def _collect_inputs(self) -> None:
        """
        Collects the minimum set of metric inputs necessary for evaluation
        (based on the metrics defined in self.metrics with the .define_metrics() method)
        """
        inputs_set: set[str] = set()
        for metric in self.metrics:
            inputs_set.update(metric.inputs)

        inputs_dict: dict[str, Type[AuditorMetricInput]] = collect_metric_inputs()

        # reinit self._inputs and add all necessary inputs to it
        self._inputs: list[Type[AuditorMetricInput]] = list()
        for input_name in list(inputs_set):
            if input_name not in ["_truth", "_pred"]:
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

    def _binarize(self, score_data: pd.Series, threshold: float) -> pd.Series:
        """Convert continuous scores to binary predictions using a threshold.

        Args:
            score_data: Series of continuous score values.
            threshold: Threshold value; scores >= threshold become 1, else 0.

        Returns:
            Series of binary predictions (0 or 1).
        """
        return (score_data >= threshold).astype(int)

    def _add_confusion_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add tp/tn/fp/fn indicator columns to a DataFrame in-place.

        Requires '_truth' and '_binary_pred' columns to already be present.

        Args:
            data: DataFrame to augment.

        Returns:
            The same DataFrame with tp, tn, fp, fn integer columns added.
        """
        data["tp"] = ((data["_truth"] == 1.0) & (data["_binary_pred"] == 1.0)).astype(int)
        data["tn"] = ((data["_truth"] == 0.0) & (data["_binary_pred"] == 0.0)).astype(int)
        data["fp"] = ((data["_truth"] == 0.0) & (data["_binary_pred"] == 1.0)).astype(int)
        data["fn"] = ((data["_truth"] == 1.0) & (data["_binary_pred"] == 0.0)).astype(int)
        return data

    # ------------------------------------------------------------------ #
    # Public evaluation methods                                            #
    # ------------------------------------------------------------------ #

    def evaluate_metrics(
        self,
        score_name: str,
        threshold: Optional[float] = None,
        n_bootstraps: Optional[int] = 1000,
    ):
        """Evaluate model performance for a given score across all features.

        Computes all configured metrics stratified by each feature, with optional
        bootstrap confidence intervals.

        Args:
            score_name: Name of the score column to evaluate.
            threshold: Decision threshold for binarizing scores. If None, uses
                the threshold defined in the AuditorScore object.
            n_bootstraps: Number of bootstrap samples for confidence interval
                calculation. Set to None to disable CI calculation.

        Returns:
            ScoreEvaluation object containing metrics for all features and levels.

        Raises:
            ValueError: If no data has been added with .add_data() first.
            ValueError: If no outcome has been defined with .add_outcome() first.
            ValueError: If no metrics have been defined with .set_metrics() first.
            ValueError: If score_name is not found in the registered scores.
            ValueError: If threshold is None and not defined in the score object.
        """
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")

        if "_truth" not in self.data.columns:
            raise ValueError(
                "Please define an outcome variable with .add_outcome() first"
            )

        if len(self.metrics) == 0:
            raise ValueError(
                "Please define at least one metric with .set_metrics() first"
            )

        # get score
        if score_name not in self.scores:
            available = ", ".join(self.scores.keys()) or "(none)"
            raise ValueError(
                f"Score '{score_name}' not found. Available scores: {available}"
            )

        score: AuditorScore = self.scores[score_name]
        threshold = self._resolve_threshold(score, threshold)

        # collect metric inputs to prep for evaluation
        self._collect_inputs()

        # get the list of columns to retain in the data
        column_list: list[str] = [*self.features.keys(), "_truth"]

        # copy a slice of the dataframe
        data_slice: pd.DataFrame = self.data.loc[:, column_list]  # type: ignore
        data_slice["_pred"] = self.data[score.name]
        data_slice["_binary_pred"] = self._binarize(score_data=data_slice["_pred"], threshold=threshold)  # type: ignore
        data_slice = self._apply_inputs(data=data_slice)

        # create an 'Overall' feature which will be used to calculate metrics on the full data
        data_slice["overall"] = "Overall"
        eval_features: dict[str, AuditorFeature] = {
            "overall": AuditorFeature(
                name="overall",
                label="Overall",
            )
        }
        eval_features.update(**self.features)

        score_eval: ScoreEvaluation = ScoreEvaluation(
            name=score.name,
            label=score.label if score.label is not None else score.name,
        )
        with tqdm(
            eval_features.values(), position=0, leave=True, desc="Features"
        ) as pbar:
            for feature in pbar:
                pbar.set_postfix({"name": feature.name})

                # e.g. {"f1": {'levelA': 0.2, 'levelB': 0.4}, ... }
                feature_eval: FeatureEvaluation = self._evaluate_feature(
                    data=data_slice, feature=feature, n_bootstraps=n_bootstraps
                )
                score_eval.features[feature.name] = feature_eval

        return score_eval

    def evaluate_errors(
        self,
        score_name: str,
        threshold: Optional[float] = None,
        n_bootstraps: Optional[int] = 1000,
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

        With bootstrap resampling enabled (n_bootstraps is not None), the stored
        point estimate is the bootstrap mean OR and the confidence interval
        spans the 2.5th–97.5th percentiles of the bootstrap distribution.

        Args:
            score_name: Name of the score column to evaluate.
            threshold: Decision threshold for binarizing scores. If None, uses
                the threshold defined in the AuditorScore object.
            n_bootstraps: Number of bootstrap samples for confidence intervals.
                Set to None to disable CI calculation.

        Returns:
            ErrorEvaluation containing one ScoreEvaluation per confusion group.

        Raises:
            ValueError: If no data has been added with .add_data() first.
            ValueError: If no outcome has been defined with .add_outcome() first.
            ValueError: If score_name is not found in the registered scores.
            ValueError: If threshold is None and not defined in the score object.
        """
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")

        if "_truth" not in self.data.columns:
            raise ValueError(
                "Please define an outcome variable with .add_outcome() first"
            )

        if score_name not in self.scores:
            available = ", ".join(self.scores.keys()) or "(none)"
            raise ValueError(
                f"Score '{score_name}' not found. Available scores: {available}"
            )

        score: AuditorScore = self.scores[score_name]
        threshold = self._resolve_threshold(score, threshold)

        # Build the analysis slice: feature columns + truth + binarised predictions
        # + all four confusion-matrix indicator columns.
        column_list: list[str] = [*self.features.keys(), "_truth"]
        data_slice: pd.DataFrame = self.data.loc[:, column_list].copy()  # type: ignore
        data_slice["_pred"] = self.data[score.name]
        data_slice["_binary_pred"] = self._binarize(
            score_data=data_slice["_pred"], threshold=threshold
        )
        self._add_confusion_columns(data_slice)

        # Synthetic 'Overall' feature (same pattern as evaluate_metrics).
        data_slice["overall"] = "Overall"
        eval_features: dict[str, AuditorFeature] = {
            "overall": AuditorFeature(name="overall", label="Overall")
        }
        eval_features.update(**self.features)

        score_label = score.label if score.label is not None else score.name
        error_eval = ErrorEvaluation(
            name=score.name,
            label=score_label,
            threshold=threshold,
        )

        # Global dataset size used as the denominator for all % overall calculations.
        # Computed from the full data slice, before any per-feature dropna.
        global_total_n = len(data_slice)
        error_eval.global_total_n = global_total_n

        metric = OddsRatio()

        for group_col in ("tp", "tn", "fp", "fn"):
            group_eval = ScoreEvaluation(
                name=group_col,
                label=group_col.upper(),
            )
            for feature in eval_features.values():
                feature_eval, support = self._evaluate_error_feature(
                    data=data_slice,
                    group_col=group_col,
                    feature=feature,
                    metric=metric,
                    n_bootstraps=n_bootstraps,
                    global_total_n=global_total_n,
                )
                group_eval.features[feature.name] = feature_eval
                # Accumulate per-feature support counts into the error evaluation's sidecar.
                error_eval.support_data.setdefault(group_col, {})[feature.name] = support
            error_eval.groups[group_col] = group_eval

        return error_eval

    # ------------------------------------------------------------------ #
    # Private evaluation helpers                                           #
    # ------------------------------------------------------------------ #

    def _evaluate_feature(
        self, data: pd.DataFrame, feature: AuditorFeature, n_bootstraps: Optional[int]
    ) -> FeatureEvaluation:
        """Evaluate all metrics for a single feature across its levels.

        When the feature column carries a categorical dtype, levels are ordered
        according to the declared category order.  Categories present in the
        declaration but absent in the data appear as placeholder rows whose
        metric scores are NaN.

        Args:
            data: DataFrame containing the evaluation data with metric input columns.
            feature: The feature to stratify evaluation by.
            n_bootstraps: Number of bootstrap samples for CI calculation, or None.

        Returns:
            FeatureEvaluation containing metrics for each level of the feature.
        """
        feature_col = feature.name

        # Detect categorical dtype *before* any transformation so we capture the
        # user-declared category order.  dropna preserves the dtype, but astype(str)
        # would destroy it.
        is_categorical = isinstance(data[feature_col].dtype, pd.CategoricalDtype)
        declared_categories: list[str] = []
        if is_categorical:
            declared_categories = [
                str(c) for c in data[feature_col].cat.categories.tolist()
            ]

        # Drop rows where the feature value is NaN; they don't belong to any level.
        feature_data = data.dropna(subset=[feature_col]).copy()

        if is_categorical:
            # Keep the categorical dtype so the groupby key type is consistent.
            # observed=True restricts grouping to categories that actually appear in
            # this slice; unobserved categories are handled as placeholders below.
            feature_groups = feature_data.groupby(feature_col, observed=True)
        else:
            # Non-categorical path: coerce to string (existing behaviour).
            feature_data[feature_col] = feature_data[feature_col].astype(str)
            feature_groups = feature_data.groupby(feature_col)

        feature_eval: FeatureEvaluation = FeatureEvaluation(
            name=feature.name,
            label=feature.label if feature.label is not None else feature.name,
        )
        for metric in self.metrics:
            # gets a dict with the current metric calculated for levels of the feature
            # e.g. {levelA: 0.5, levelB: 0.5}
            level_eval_dict = feature_groups.apply(metric.data_call).to_dict()
            # Normalise keys to strings; categorical keys are the category values,
            # which may already be strings but we guarantee it here.
            level_eval_dict = {str(k): v for k, v in level_eval_dict.items()}

            feature_eval.update(
                metric_name=metric.name,
                metric_label=metric.label,
                data=level_eval_dict,  # type: ignore
            )

        # if calculating confidence intervals, do that here
        if n_bootstraps is not None:
            for level_name, level_data in feature_groups:
                # calculate confidence intervals for eligible metrics for the current feature level
                level_metric_intervals: dict[str, tuple[float, float]] = (
                    self._evaluate_confidence_interval(
                        data=level_data, n_bootstraps=n_bootstraps
                    )
                )
                # register the calculated intervals
                feature_eval.update_intervals(
                    level_name=str(level_name),
                    metric_intervals=level_metric_intervals,
                )

        if is_categorical:
            # Rebuild levels dict in declared category order.  For each declared
            # category: use the computed LevelEvaluation if the category was
            # observed, otherwise insert a placeholder with NaN metric scores so
            # that to_dataframe() / style_dataframe() produce a complete row.
            ordered_levels: dict[str, LevelEvaluation] = {}
            for cat_str in declared_categories:
                if cat_str in feature_eval.levels:
                    ordered_levels[cat_str] = feature_eval.levels[cat_str]
                else:
                    placeholder = LevelEvaluation(name=cat_str)
                    for metric in self.metrics:
                        placeholder.update(
                            metric_name=metric.name,
                            metric_label=metric.label,
                            metric_score=float("nan"),
                        )
                    ordered_levels[cat_str] = placeholder
            feature_eval.levels = ordered_levels

        return feature_eval

    def _evaluate_error_feature(
        self,
        data: pd.DataFrame,
        group_col: str,
        feature: AuditorFeature,
        metric: AuditorErrorMetric,
        n_bootstraps: Optional[int],
        global_total_n: int,
    ) -> tuple[FeatureEvaluation, dict[str, dict[str, float]]]:
        """Compute odds ratios for one feature within one confusion-matrix group.

        Calculates the canonical 2x2 odds ratio for each feature level versus all
        other levels combined, then optionally runs bootstrap resampling to derive
        confidence intervals and replace the point estimate with the bootstrap mean.

        Categorical dtype is honoured: declared-but-unobserved categories appear
        as NaN placeholder rows (same behaviour as _evaluate_feature).

        Args:
            data: Full data slice including confusion indicator columns.
            group_col: Column name of the confusion indicator ('tp', 'tn', etc.).
            feature: The feature whose levels are being analysed.
            metric: Error metric to compute (e.g. OddsRatio).
            n_bootstraps: Bootstrap iterations, or None to skip.
            global_total_n: Total rows in the full data slice; used as the
                denominator for pct_overall in the returned support counts.

        Returns:
            Two-tuple (feature_eval, support) where support maps each level name to
            {"n": int, "pct_overall": float, "pct_group": float}.
        """
        feature_col = feature.name

        is_categorical = isinstance(data[feature_col].dtype, pd.CategoricalDtype)
        declared_categories: list[str] = []
        if is_categorical:
            declared_categories = [
                str(c) for c in data[feature_col].cat.categories.tolist()
            ]

        # Drop rows where the feature is NaN; they don't belong to any level.
        full_data = data.dropna(subset=[feature_col]).copy()
        if not is_categorical:
            full_data[feature_col] = full_data[feature_col].astype(str)

        full_total = len(full_data)

        # Per-level counts over the full dataset.
        if is_categorical:
            full_gs = full_data.groupby(feature_col, observed=True).size()
        else:
            full_gs = full_data.groupby(feature_col).size()
        full_counts: dict[str, int] = {str(k): int(v) for k, v in full_gs.items()}

        # Per-level counts within the confusion group.
        group_data = full_data[full_data[group_col] == 1]
        group_total = len(group_data)

        if is_categorical:
            group_gs = group_data.groupby(feature_col, observed=True).size() if not group_data.empty else pd.Series(dtype=int)
        else:
            group_gs = group_data.groupby(feature_col).size() if not group_data.empty else pd.Series(dtype=int)
        group_counts: dict[str, int] = {str(k): int(v) for k, v in group_gs.items()}

        # All levels to evaluate: declared order for categorical, observed order otherwise.
        all_levels: list[str] = (
            declared_categories if is_categorical else list(full_counts.keys())
        )

        feature_eval = FeatureEvaluation(
            name=feature.name,
            label=feature.label if feature.label is not None else feature.name,
        )

        # Point estimates (raw ratio, or bootstrap mean if bootstraps requested).
        level_scores: dict[str, float] = {}
        for level_name in all_levels:
            full_count = full_counts.get(level_name, 0)
            group_count = group_counts.get(level_name, 0)
            level_scores[level_name] = metric.compute(
                group_count=group_count,
                group_total=group_total,
                full_count=full_count,
                full_total=full_total,
            )

        for level_name in all_levels:
            feature_eval.update(
                metric_name=metric.name,
                metric_label=metric.label,
                data={level_name: level_scores[level_name]},
            )

        # Bootstrap: replaces point estimates with bootstrap mean and adds CI.
        # Levels with an undefined baseline (NaN score) are excluded — they cannot
        # yield a meaningful CI and the NaN placeholder should be preserved.
        if n_bootstraps is not None and metric.ci_eligible:
            valid_levels = [l for l in all_levels if not np.isnan(level_scores[l])]

            if valid_levels:
                bootstrap_results: dict[str, NDArray[np.float64]] = {
                    l: np.empty(n_bootstraps, dtype=np.float64) for l in valid_levels
                }
                n = len(data)  # resample from the full slice (all confusion groups)

                for i in range(n_bootstraps):
                    boot = data.sample(n, replace=True)
                    boot_full = boot.dropna(subset=[feature_col]).copy()
                    if not is_categorical:
                        boot_full[feature_col] = boot_full[feature_col].astype(str)

                    boot_full_total = len(boot_full)
                    boot_group = boot_full[boot_full[group_col] == 1]
                    boot_group_total = len(boot_group)

                    if is_categorical:
                        bfgs = boot_full.groupby(feature_col, observed=True).size()
                        bgGs = boot_group.groupby(feature_col, observed=True).size() if not boot_group.empty else pd.Series(dtype=int)
                    else:
                        bfgs = boot_full.groupby(feature_col).size()
                        bgGs = boot_group.groupby(feature_col).size() if not boot_group.empty else pd.Series(dtype=int)

                    bfgs_dict: dict[str, int] = {str(k): int(v) for k, v in bfgs.items()}
                    bgGs_dict: dict[str, int] = {str(k): int(v) for k, v in bgGs.items()}

                    for level_name in valid_levels:
                        bootstrap_results[level_name][i] = metric.compute(
                            group_count=bgGs_dict.get(level_name, 0),
                            group_total=boot_group_total,
                            full_count=bfgs_dict.get(level_name, 0),
                            full_total=boot_full_total,
                        )

                for level_name in valid_levels:
                    bs = bootstrap_results[level_name]
                    point_estimate = float(np.nanmean(bs))
                    lower, upper = np.nanpercentile(bs, [2.5, 97.5])
                    # np.nanpercentile returns NaN when interpolating between
                    # two inf values (inf − inf = NaN).  Recover the correct
                    # bound: if the bootstrap tail contains +inf OR values
                    # (b*c == 0 in that sample), the true CI bound is +inf.
                    if np.isnan(upper) and np.any(np.isposinf(bs)):
                        upper = float("inf")
                    if np.isnan(lower) and np.any(np.isneginf(bs)):
                        lower = float("-inf")
                    lm = feature_eval.levels[level_name].metrics[metric.name]
                    lm.score = point_estimate
                    lm.interval = (float(lower), float(upper))

        # Categorical ordering and placeholders (mirrors _evaluate_feature).
        if is_categorical:
            ordered_levels: dict[str, LevelEvaluation] = {}
            for cat_str in declared_categories:
                if cat_str in feature_eval.levels:
                    ordered_levels[cat_str] = feature_eval.levels[cat_str]
                else:
                    placeholder = LevelEvaluation(name=cat_str)
                    placeholder.update(
                        metric_name=metric.name,
                        metric_label=metric.label,
                        metric_score=float("nan"),
                    )
                    ordered_levels[cat_str] = placeholder
            feature_eval.levels = ordered_levels

        # Build sidecar support counts per level for the wide-format DataFrame export.
        # These are derived from the same full_data / group_data counts already computed above.
        # For categorical placeholders (levels in declared_categories but absent from full_counts),
        # all counts are zero.
        support: dict[str, dict[str, float]] = {}
        for level_name in all_levels:
            g_n = group_counts.get(level_name, 0)
            g_pct_overall = g_n / global_total_n if global_total_n > 0 else float("nan")
            g_pct_group = g_n / group_total if group_total > 0 else 0.0
            support[level_name] = {
                "n": g_n,
                "pct_overall": g_pct_overall,
                "pct_group": g_pct_group,
            }
        if is_categorical:
            for cat_str in declared_categories:
                if cat_str not in support:
                    support[cat_str] = {"n": 0, "pct_overall": 0.0, "pct_group": 0.0}

        return feature_eval, support

    def _evaluate_confidence_interval(
        self, data: pd.DataFrame, n_bootstraps: int
    ) -> dict[str, tuple[float, float]]:
        """Calculate bootstrap confidence intervals for all CI-eligible metrics.

        Uses bootstrap resampling to estimate 95% confidence intervals for
        metrics that have ci_eligible=True.

        Args:
            data: DataFrame containing the data for a single feature level.
            n_bootstraps: Number of bootstrap samples to draw.

        Returns:
            Dictionary mapping metric names to (lower, upper) confidence bounds.
        """
        n: int = len(data)

        bootstrap_results: dict[str, NDArray[np.float64]] = dict()
        for metric in self.metrics:
            if metric.ci_eligible:
                bootstrap_results[metric.name] = np.empty(
                    shape=(n_bootstraps), dtype=np.float64
                )

        # sample n_bootstrap times with replacement
        for i in range(n_bootstraps):
            boot_data: pd.DataFrame = data.sample(n, replace=True)

            # calculate metrics on current bootstrap data
            for metric in self.metrics:
                if metric.ci_eligible:
                    bootstrap_results[metric.name][i] = metric.data_call(boot_data)

        metric_intervals: dict[str, tuple[float, float]] = dict()
        for metric_name, bootstrap_array in bootstrap_results.items():
            # get 95% confidence bounds for metric
            lower, upper = np.nanpercentile(bootstrap_array, [2.5, 97.5])
            metric_intervals[metric_name] = (lower, upper)

        return metric_intervals
