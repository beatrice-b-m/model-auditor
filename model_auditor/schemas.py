"""Data schemas for model auditor evaluation results.

This module defines the data structures used to store and organize
evaluation results, including features, scores, outcomes, and their
associated metrics at various levels of aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from model_auditor._styling import _apply_tier_styling


@dataclass(frozen=True)
class InferenceConfig:
    """Inference for fixed binary predictions, with pointwise intervals.

    ``auto`` uses Wilson intervals for IID binomial rates and conditional exact
    intervals for IID odds ratios; other statistics use percentile resampling.
    Set rate_interval="exact" for conservative Clopper-Pearson binomial intervals
    instead of approximate Wilson coverage, especially with very small counts.
    Cluster resampling preserves whole subjects, with row-weighted estimates.
    Stratification conditions on observed class counts. Neither design refits
    models or corrects data-driven threshold/subgroup selection. A seed creates
    a local generator; NumPy's global random state is never consumed.
    """

    confidence_level: float = 0.95
    method: Literal["auto", "bootstrap"] = "auto"
    rate_interval: Literal["wilson", "exact"] = "wilson"
    resampling: Literal["iid", "stratified", "cluster"] = "iid"
    cluster: Optional[str] = None
    random_state: Optional[int] = None
    missing: Literal["exclude", "include", "error"] = "exclude"
    min_valid_fraction: float = 0.95
    min_resamples: int = 100

    def __post_init__(self) -> None:
        if not np.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if self.rate_interval not in {"wilson", "exact"}:
            raise ValueError("rate_interval must be wilson or exact.")
        if self.method not in {"auto", "bootstrap"}:
            raise ValueError("method must be auto or bootstrap.")
        if self.resampling not in {"iid", "stratified", "cluster"}:
            raise ValueError("resampling must be iid, stratified, or cluster.")
        if (self.resampling == "cluster") != (self.cluster is not None):
            raise ValueError("Supply cluster only with cluster resampling.")
        if self.missing not in {"exclude", "include", "error"}:
            raise ValueError("missing must be exclude, include, or error.")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1].")
        if (
            isinstance(self.min_resamples, bool)
            or not isinstance(self.min_resamples, int)
            or self.min_resamples < 2
        ):
            raise ValueError("min_resamples must be an integer >= 2.")
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None.")


@dataclass
class LevelMetric:
    """
    Object to store the evaluation results for one metric of one level of a feature.
    (for example, AUC for one category of finding)

    Args:
        name (str): Name of the current feature level metric
        score (Union[float, int]): Score for the current feature level metric
        interval (tuple[float, float], optional): Optional lower and upper confidence
        bounds for the current feature level metric (defaults to None)
    """

    name: str
    label: str
    score: Union[float, int]
    interval: Optional[tuple[float, float]] = None
    status: str = "ok"
    interval_status: str = "not_requested"
    interval_method: Optional[str] = None
    denominator: Optional[int] = None
    valid_resamples: int = 0
    requested_resamples: int = 0
    nonfinite_resamples: int = 0
    direction: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class LevelEvaluation:
    """
    Object to store the evaluation results for one level of a feature
    (for example, all metrics for one category of finding).

    Args:
        name (str): Name of the current feature level
        metrics (dict[str, LevelMetric]): Metrics for the current feature level
        (defaults to an empty dict)
    """

    name: str
    metrics: dict[str, LevelMetric] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)

    def update(self, metric_name: str, metric_label: str, metric_score: float) -> None:
        """Add or update a metric for this level.

        Args:
            metric_name: Unique identifier for the metric.
            metric_label: Display label for the metric.
            metric_score: Computed metric value.
        """
        self.metrics[metric_name] = LevelMetric(
            name=metric_name, label=metric_label, score=metric_score
        )

    def update_intervals(
        self, metric_intervals: dict[str, tuple[float, float]]
    ) -> None:
        """Update confidence intervals for existing metrics.

        Args:
            metric_intervals: Dictionary mapping metric names to (lower, upper) bounds.
        """
        for metric_name, confidence_interval in metric_intervals.items():
            self.metrics[metric_name].interval = confidence_interval

    def to_dataframe(
        self, n_decimals: int = 3, add_index: bool = False, metric_labels: bool = False
    ) -> pd.DataFrame:
        """Convert level evaluation to a pandas DataFrame.

        Args:
            n_decimals: Number of decimal places for formatting scores.
            add_index: Unused parameter (kept for API consistency).
            metric_labels: If True, use metric labels as column names; else use names.

        Returns:
            Single-row DataFrame with metrics as columns.
        """
        metric_data: dict[str, str] = dict()
        for metric in self.metrics.values():
            # get the key name for the current metric (label if metric_labels is True)
            metric_key: str = metric.label if metric_labels else metric.name

            if metric.interval is not None:
                metric_data[metric_key] = (
                    f"{metric.score:.{n_decimals}f} ({metric.interval[0]:.{n_decimals}f}, {metric.interval[1]:.{n_decimals}f})"
                )
            elif isinstance(metric.score, float):
                metric_data[metric_key] = f"{metric.score:.{n_decimals}f}"
            else:
                # integer scores (default to comma delimited for now)
                metric_data[metric_key] = f"{metric.score:,}"

        return pd.DataFrame(metric_data, index=[self.name])

    def style_dataframe(
        self,
        n_decimals: int = 3,
        metric_labels: bool = False,
        include_count_metrics: bool = False,
        low_color: str = "#f8d7da",
        medium_color: str = "#fff3cd",
        high_color: str = "#d4edda",
        rank: bool = False,
    ) -> pd.io.formats.style.Styler:
        """Convert level evaluation to a styled pandas DataFrame for Jupyter display.

        Neutral by default. Set rank=True for descriptive within-feature ranks;
        colors do not express significance, equivalence, or practical importance.

        Args:
            n_decimals: Number of decimal places for formatting scores.
            metric_labels: If True, use metric labels as column names; else use names.
            include_count_metrics: If True, include count metrics in tier styling.
            low_color: Background color for low performance tier.
            medium_color: Background color for medium performance tier.
            high_color: Background color for high performance tier.

        Returns:
            A pandas Styler object with tier-based coloring applied.
        """
        # Build the display DataFrame (same as to_dataframe)
        display_df = self.to_dataframe(
            n_decimals=n_decimals, metric_labels=metric_labels
        )

        # Build parallel numeric DataFrame for styling decisions
        numeric_data = {}
        metric_names = []
        for metric in self.metrics.values():
            metric_key = metric.label if metric_labels else metric.name
            numeric_data[metric_key] = metric.score
            metric_names.append(metric_key)
        numeric_df = pd.DataFrame(numeric_data, index=[self.name])

        # Apply tier styling
        return _apply_tier_styling(
            display_df=display_df,
            numeric_df=numeric_df,
            metric_names=metric_names,
            include_count_metrics=include_count_metrics,
            low_color=low_color,
            medium_color=medium_color,
            high_color=high_color,
            rank=rank,
            directions={
                m.label if metric_labels else m.name: m.direction
                for m in self.metrics.values()
            },
        )


@dataclass
class FeatureEvaluation:
    """
    Object to store the evaluation results for one feature type
    (for example, metrics associated with different types of findings)

    Args:
        name (str): Name of the current feature
        name (str): Label for the current feature
        levels (dict[str, LevelEvaluation]): Levels of the current feature
        (defaults to an empty dict)
    """

    name: str
    label: str
    levels: dict[str, LevelEvaluation] = field(default_factory=dict)
    excluded_n: int = 0
    total_n: int = 0

    def update(
        self, metric_name: str, metric_label: str, data: dict[str, float]
    ) -> None:
        """Update metrics for all levels from a metric-level dictionary.

        Args:
            metric_name: Unique identifier for the metric.
            metric_label: Display label for the metric.
            data: Dictionary mapping level names to metric scores,
                e.g., {'levelA': 0.5, 'levelB': 0.5}.
        """
        # expects a dict for one metric type: {'levelA': 0.5, 'levelB': 0.5}
        # and maps them to child level metric dicts
        for level_name, level_metric in data.items():
            # try to get the level item and instantiate a new one if it doesn't exist yet
            level_eval: LevelEvaluation = self.levels.get(
                level_name, LevelEvaluation(name=level_name)
            )
            # update the metrics for that level eval object and save it back to the dict
            level_eval.update(
                metric_name=metric_name,
                metric_label=metric_label,
                metric_score=level_metric,
            )
            self.levels[level_name] = level_eval

    def update_intervals(
        self, level_name: str, metric_intervals: dict[str, tuple[float, float]]
    ) -> None:
        """Update confidence intervals for metrics at a specific level.

        Args:
            level_name: Name of the level to update.
            metric_intervals: Dictionary mapping metric names to (lower, upper) bounds.
        """
        self.levels[level_name].update_intervals(metric_intervals=metric_intervals)

    def to_dataframe(
        self, n_decimals: int = 3, add_index: bool = False, metric_labels: bool = False
    ) -> pd.DataFrame:
        """Convert feature evaluation to a pandas DataFrame.

        Args:
            n_decimals: Number of decimal places for formatting scores.
            add_index: If True, add feature label as a hierarchical index level.
            metric_labels: If True, use metric labels as column names; else use names.

        Returns:
            DataFrame with levels as rows and metrics as columns.
        """
        data: list[pd.DataFrame] = []
        for level_data in self.levels.values():
            data.append(
                level_data.to_dataframe(
                    n_decimals=n_decimals, metric_labels=metric_labels
                )
            )

        frame = pd.concat(data, axis=0) if data else pd.DataFrame()
        return pd.concat({self.label: frame}) if add_index else frame

    def style_dataframe(
        self,
        n_decimals: int = 3,
        metric_labels: bool = False,
        include_count_metrics: bool = False,
        low_color: str = "#f8d7da",
        medium_color: str = "#fff3cd",
        high_color: str = "#d4edda",
        rank: bool = False,
    ) -> pd.io.formats.style.Styler:
        """Convert feature evaluation to a styled pandas DataFrame for Jupyter display.

        Neutral by default. Set rank=True for descriptive within-feature ranks;
        colors do not express significance, equivalence, or practical importance.

        Args:
            n_decimals: Number of decimal places for formatting scores.
            metric_labels: If True, use metric labels as column names; else use names.
            include_count_metrics: If True, include count metrics in tier styling.
            low_color: Background color for low performance tier.
            medium_color: Background color for medium performance tier.
            high_color: Background color for high performance tier.

        Returns:
            A pandas Styler object with tier-based coloring applied.
        """
        # Build the display DataFrame (same as to_dataframe)
        display_df = self.to_dataframe(
            n_decimals=n_decimals, metric_labels=metric_labels
        )

        # Build parallel numeric DataFrame for styling decisions
        numeric_data_list = []
        metric_names = set()

        for level_eval in self.levels.values():
            level_numeric = {}
            for metric in level_eval.metrics.values():
                metric_key = metric.label if metric_labels else metric.name
                level_numeric[metric_key] = metric.score
                metric_names.add(metric_key)
            numeric_data_list.append(level_numeric)

        numeric_df = pd.DataFrame(numeric_data_list, index=display_df.index)

        # Apply tier styling
        return _apply_tier_styling(
            display_df=display_df,
            numeric_df=numeric_df,
            metric_names=list(metric_names),
            include_count_metrics=include_count_metrics,
            low_color=low_color,
            medium_color=medium_color,
            high_color=high_color,
            rank=rank,
            directions={
                m.label if metric_labels else m.name: m.direction
                for level in self.levels.values()
                for m in level.metrics.values()
            },
        )


@dataclass
class ScoreEvaluation:
    """Container for all evaluation results for a single score.

    Organizes evaluation results hierarchically by feature, then by level
    within each feature.

    Attributes:
        name: Name of the score being evaluated.
        label: Display label for the score.
        features: Dictionary mapping feature names to FeatureEvaluation objects.
    """

    name: str
    label: str
    features: dict[str, FeatureEvaluation] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_numeric_dataframe(self) -> pd.DataFrame:
        """Unrounded long-form results, support, diagnostics, and stable IDs.

        Evaluation provenance is copied into ``frame.attrs["metadata"]``.
        Intervals are pointwise, conditional on the supplied predictions.
        """
        from copy import deepcopy

        rows = []
        for feature in self.features.values():
            for level in feature.levels.values():
                for metric in level.metrics.values():
                    rows.append(
                        {
                            "score": self.name,
                            "feature": feature.name,
                            "level": level.name,
                            "metric": metric.name,
                            "estimate": metric.score,
                            "lower": metric.interval[0]
                            if metric.interval
                            else float("nan"),
                            "upper": metric.interval[1]
                            if metric.interval
                            else float("nan"),
                            "status": metric.status,
                            "interval_status": metric.interval_status,
                            "interval_method": metric.interval_method,
                            "denominator": metric.denominator,
                            "requested_resamples": metric.requested_resamples,
                            "valid_resamples": metric.valid_resamples,
                            "nonfinite_resamples": metric.nonfinite_resamples,
                            "direction": metric.direction,
                            "parameters": deepcopy(metric.parameters),
                            "excluded_n": feature.excluded_n,
                            "total_n": feature.total_n,
                            **level.support,
                        }
                    )
        frame = pd.DataFrame(rows)
        frame.attrs["metadata"] = deepcopy(self.metadata)
        return frame

    def to_dataframe(
        self, n_decimals: int = 3, add_index: bool = False, metric_labels: bool = False
    ) -> pd.DataFrame:
        """Convert score evaluation to a pandas DataFrame.

        Args:
            n_decimals: Number of decimal places for formatting scores.
            add_index: If True, add score label as a hierarchical index level.
            metric_labels: If True, use metric labels as column names; else use names.

        Returns:
            DataFrame with hierarchical index (feature, level) and metrics as columns.
        """
        data: list[pd.DataFrame] = []
        for feature_data in self.features.values():
            data.append(
                feature_data.to_dataframe(
                    n_decimals=n_decimals, add_index=True, metric_labels=metric_labels
                )
            )

        frame = pd.concat(data, axis=0) if data else pd.DataFrame()
        return pd.concat({self.label: frame}) if add_index else frame

    def style_dataframe(
        self,
        n_decimals: int = 3,
        metric_labels: bool = False,
        include_count_metrics: bool = False,
        low_color: str = "#f8d7da",
        medium_color: str = "#fff3cd",
        high_color: str = "#d4edda",
        rank: bool = False,
    ) -> pd.io.formats.style.Styler:
        """Convert score evaluation to a styled pandas DataFrame for Jupyter display.

        Neutral by default. Set rank=True for descriptive within-feature ranks;
        colors do not express significance, equivalence, or practical importance.

        Args:
            n_decimals: Number of decimal places for formatting scores.
            metric_labels: If True, use metric labels as column names; else use names.
            include_count_metrics: If True, include count metrics in tier styling.
            low_color: Background color for low performance tier.
            medium_color: Background color for medium performance tier.
            high_color: Background color for high performance tier.

        Returns:
            A pandas Styler object with tier-based coloring applied.
        """
        # Build the display DataFrame (same as to_dataframe)
        display_df = self.to_dataframe(
            n_decimals=n_decimals, metric_labels=metric_labels
        )

        # Build parallel numeric DataFrame for styling decisions
        numeric_data_list = []
        metric_names = set()

        for feature_eval in self.features.values():
            for level_eval in feature_eval.levels.values():
                level_numeric = {}
                for metric in level_eval.metrics.values():
                    metric_key = metric.label if metric_labels else metric.name
                    level_numeric[metric_key] = metric.score
                    metric_names.add(metric_key)
                numeric_data_list.append(level_numeric)

        numeric_df = pd.DataFrame(numeric_data_list, index=display_df.index)

        # Apply tier styling
        return _apply_tier_styling(
            display_df=display_df,
            numeric_df=numeric_df,
            metric_names=list(metric_names),
            include_count_metrics=include_count_metrics,
            low_color=low_color,
            medium_color=medium_color,
            high_color=high_color,
            rank=rank,
            directions={
                m.label if metric_labels else m.name: m.direction
                for feature in self.features.values()
                for level in feature.levels.values()
                for m in level.metrics.values()
            },
        )

    def plot_metric_intervals(
        self,
        metric: str,
        feature_names: Optional[list[str]] = None,
        include_overall: bool = True,
        rotate_plots: bool = False,
        include_sample_size: bool = True,
        include_class_balance: bool = True,
    ) -> dict:
        """Create interval plots for a metric across feature levels.

        For each selected feature, produces one matplotlib figure with an
        error-bar per level: the point marks the metric estimate (score) and
        the whiskers extend to the bootstrap CI lower and upper bounds.

        By default each feature subplot prepends an ``Overall`` comparator
        level (drawn from the ``"overall"`` synthetic feature when present)
        so subgroup performance can be read against the global baseline in
        the same figure.  The standalone ``"overall"`` feature is never
        rendered as its own subplot.

        Levels without plottable CI data (nonfinite score, missing interval,
        nonfinite bounds, or reversed bounds) are explicitly listed with their
        status rather than silently disappearing.

        Args:
            metric: Metric to plot.  Matched first by exact name, then by
                label.  Example: ``"sensitivity"`` or ``"Sensitivity"``.
            feature_names: Feature names to include.  ``None`` plots all
                features except the synthetic ``"overall"`` feature.
                Passing ``"overall"`` in this list has no effect (it is
                silently filtered out).
            include_overall: If ``True`` (default) and an ``"overall"``
                feature is present, prepend an ``Overall`` level to each
                feature subplot for visual comparison.
            rotate_plots: If ``False`` (default), error bars are horizontal
                (metric on x-axis, levels on y-axis).  If ``True``, error
                bars are vertical (levels on x-axis, metric on y-axis).
            include_sample_size: If ``True`` (default), annotate each level
                with its sample size as ``"N: {n} ({pct_of_overall}%)"``
                (``NA`` placeholders when counts are unavailable).  In
                horizontal mode the annotations are left-aligned at a
                consistent x-position to the left of the class labels.
            include_class_balance: If ``True`` (default), annotate each
                level with its positive-class count as
                ``"N Pos: {n_pos} ({pct_positive}%)"`` (``NA`` placeholders
                when counts are unavailable).

        Returns:
            Dictionary mapping feature name to
            ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` tuples,
            one entry per selected feature.

        Raises:
            ImportError: If matplotlib is not installed.
            ValueError: If ``self.features`` is empty, ``feature_names``
                contains unknown names, no plottable features remain after
                filtering ``"overall"``, or the metric is not found in any
                selected feature.
        """
        from model_auditor.plotting.intervals import plot_metric_intervals

        return plot_metric_intervals(
            self,
            metric,
            feature_names,
            include_overall,
            rotate_plots,
            include_sample_size,
            include_class_balance,
        )


@dataclass
class AuditorFeature:
    """Configuration for a stratification feature.

    Defines a column in the data that will be used to stratify metric
    evaluation into subgroups.

    Attributes:
        name: Column name in the DataFrame.
        label: Display label for the feature (defaults to name if None).
    """

    name: str
    label: Optional[str] = None


@dataclass
class ConditionalThreshold:
    """Conditional score threshold specification based on feature levels.

    Attributes:
        feature: Feature column name used to select the threshold per row.
        levels: Mapping of feature level values to numeric thresholds.
        default: Optional fallback threshold for levels missing from `levels`.
    """

    feature: str
    levels: dict[Any, float]
    default: Optional[float] = None


ThresholdSpec = float | ConditionalThreshold


@dataclass
class AuditorScore:
    """Configuration for a prediction score column.

    Defines a continuous score column that will be evaluated against
    the ground truth outcome.

    Attributes:
        name: Column name in the DataFrame containing prediction scores.
        label: Display label for the score (defaults to name if None).
        threshold: Optional scalar or conditional threshold for binarizing scores.
    """

    name: str
    label: Optional[str] = None
    threshold: Optional[ThresholdSpec] = None


@dataclass
class AuditorOutcome:
    """Configuration for the ground truth outcome column.

    Defines the outcome (label) column that predictions are compared against.

    Attributes:
        name: Column name in the DataFrame containing ground truth labels.
        mapping: Optional dictionary to convert outcome values to binary (0/1),
            e.g., {"positive": 1, "negative": 0}.
    """

    name: str
    mapping: Optional[dict[Any, int]] = None


@dataclass
class ErrorEvaluation:
    """Container for confusion-matrix group error analysis for a single score.

    Groups evaluation results by confusion-matrix category (TP, TN, FP, FN),
    each of which is itself a ScoreEvaluation holding per-feature odds-ratio
    metrics.

    Attributes:
        name: Name of the score that was evaluated.
        label: Display label for the score.
        threshold: Binarization threshold used during evaluation.
        groups: Dictionary mapping group keys ('tp', 'tn', 'fp', 'fn') to
            ScoreEvaluation objects containing per-feature metrics.
    """

    name: str
    label: str
    threshold: ThresholdSpec
    groups: dict[str, ScoreEvaluation] = field(default_factory=dict)
    global_total_n: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Sidecar support counts: {group_col: {feature_name: {level_name: {"n": int, "pct_overall": float, "pct_group": float}}}}
    # Used by to_dataframe() to compute class balance, overall N, and %-of-group metrics.
    support_data: dict[str, dict[str, dict[str, dict[str, float]]]] = field(
        default_factory=dict
    )

    def to_numeric_dataframe(self) -> pd.DataFrame:
        """Long-form enrichment estimates and diagnostics; not error-rate contrasts."""
        from copy import deepcopy

        frames = []
        for group, evaluation in self.groups.items():
            frame = evaluation.to_numeric_dataframe()
            frame["score"] = self.name
            frame["confusion_group"] = group
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        result.attrs["metadata"] = deepcopy(self.metadata)
        return result

    def to_dataframe(
        self, n_decimals: int = 3, metric_labels: bool = False
    ) -> pd.DataFrame:
        """Convert error evaluation to a wide cross-group analysis DataFrame.

        Returns a numeric DataFrame suitable for fairness and error auditing without
        additional reshaping:

        - Row index: MultiIndex(feature_label, level_name)
        - Column index: MultiIndex(section, metric)

        Sections and sub-columns:
          - ("Overall", "N")          : total rows at this level (TP+TN+FP+FN)
          - ("Overall", "% overall")   : Overall N / global_total_n
          - ("Overall", "N_pos")       : positives (TP+FN) at this level
          - ("Overall", "N_neg")       : negatives (TN+FP) at this level
          - ("Overall", "Pos %")       : N_pos / (N_pos + N_neg) per row
          - For each group in TP/TN/FP/FN:
            - (GROUP, "N")                     : rows in this group at this level
            - (GROUP, "% overall")             : group N / global_total_n
            - (GROUP, "% group")              : group N / retained rows in that group
                                                after the feature missingness policy
            - (GROUP, or_col_name)            : odds ratio (NaN for overall/Overall
                                               row — OR is undefined when all rows
                                               belong to the level)
            - (GROUP, or_ci_lower_name)       : CI lower bound for OR (see metadata for level/method)
            - (GROUP, or_ci_upper_name)       : CI upper bound for OR
                                               (NaN when n_bootstraps was None)

        Args:
            n_decimals: Ignored; kept for API compatibility. Output is always numeric.
            metric_labels: If True, use "Odds Ratio" and CI labels containing
                the configured confidence level; else use the machine-readable
                "odds_ratio" / "odds_ratio_ci_lower" / "odds_ratio_ci_upper".


        Returns:
            Numeric DataFrame with MultiIndex rows and columns, or an empty DataFrame
            when no groups have been evaluated.
        """
        if not self.groups:
            return pd.DataFrame()

        or_col_name = "Odds Ratio" if metric_labels else "odds_ratio"
        confidence = 100 * self.metadata.get("inference", {}).get(
            "confidence_level", 0.95
        )
        or_ci_lower_name = (
            f"OR {confidence:g}% CI Lower" if metric_labels else "odds_ratio_ci_lower"
        )
        or_ci_upper_name = (
            f"OR {confidence:g}% CI Upper" if metric_labels else "odds_ratio_ci_upper"
        )
        group_order = [g for g in ("tp", "tn", "fp", "fn") if g in self.groups]

        # Use the first group's feature ordering to determine all (feature, level) rows.
        # All groups are evaluated over the same features and levels.
        first_group = next(iter(self.groups.values()))
        feature_order: list[
            tuple[str, str, str]
        ] = []  # (feature_name, feature_label, level_name)
        for fname, feval in first_group.features.items():
            for lname in feval.levels:
                feature_order.append((fname, feval.label, lname))

        rows: list[dict] = []
        index_tuples: list[tuple[str, str]] = []

        for feature_name, feature_label, level_name in feature_order:
            # The overall/Overall row has an undefined OR (all rows belong to the
            # level; no comparator population exists).  Emit NaN to signal that
            # this cell is not analytically meaningful.
            is_overall_level = feature_name == "overall" and level_name == "Overall"

            # Pull raw support counts from the sidecar populated by evaluate_errors().
            def _n(group_col: str) -> int:
                return int(
                    self.support_data.get(group_col, {})
                    .get(feature_name, {})
                    .get(level_name, {})
                    .get("n", 0)
                )

            def _pct(group_col: str, key: str) -> float:
                return float(
                    self.support_data.get(group_col, {})
                    .get(feature_name, {})
                    .get(level_name, {})
                    .get(key, 0.0)
                )

            tp_n = _n("tp")
            tn_n = _n("tn")
            fp_n = _n("fp")
            fn_n = _n("fn")

            n_pos = tp_n + fn_n  # true class-positive count
            n_neg = tn_n + fp_n  # true class-negative count
            overall_n = tp_n + tn_n + fp_n + fn_n
            denom = self.global_total_n if self.global_total_n > 0 else None
            # cb_denom is the per-row class total; equals overall_n but expresses
            # intent: Pos % is a class fraction within this level's rows.
            cb_denom = n_pos + n_neg

            row: dict[tuple[str, str], Any] = {
                ("Overall", "N"): overall_n,
                ("Overall", "% overall"): overall_n / denom if denom else float("nan"),
                # Class-balance sub-columns sit in the Overall section so they stay
                # co-located with total sample size.  Neg % is omitted because it is
                # fully determined by Pos % (Neg % = 1 - Pos %).
                ("Overall", "N_pos"): n_pos,
                ("Overall", "N_neg"): n_neg,
                ("Overall", "Pos %"): n_pos / cb_denom
                if cb_denom > 0
                else float("nan"),
            }

            for group_col in group_order:
                group_label = group_col.upper()
                g_n = _n(group_col)
                g_pct_overall = _pct(group_col, "pct_overall")
                g_pct_group = _pct(group_col, "pct_group")

                row[(group_label, "N")] = g_n
                row[(group_label, "% overall")] = g_pct_overall
                row[(group_label, "% group")] = g_pct_group

                if is_overall_level:
                    row[(group_label, or_col_name)] = float("nan")
                    row[(group_label, or_ci_lower_name)] = float("nan")
                    row[(group_label, or_ci_upper_name)] = float("nan")
                else:
                    lm = (
                        self.groups[group_col]
                        .features[feature_name]
                        .levels[level_name]
                        .metrics.get("odds_ratio")
                    )
                    row[(group_label, or_col_name)] = (
                        lm.score if lm is not None else float("nan")
                    )
                    if lm is not None and lm.interval is not None:
                        row[(group_label, or_ci_lower_name)] = lm.interval[0]
                        row[(group_label, or_ci_upper_name)] = lm.interval[1]
                    else:
                        row[(group_label, or_ci_lower_name)] = float("nan")
                        row[(group_label, or_ci_upper_name)] = float("nan")

            rows.append(row)
            index_tuples.append((feature_label, level_name))

        index = pd.MultiIndex.from_tuples(index_tuples, names=["feature", "level"])
        df = pd.DataFrame(rows, index=index)
        df.columns = pd.MultiIndex.from_tuples(list(df.columns))  # type: ignore
        return df

    def style_dataframe(
        self,
        n_decimals: int = 3,
        metric_labels: bool = False,
        include_count_metrics: bool = False,
        low_color: str = "#f8d7da",
        medium_color: str = "#fff3cd",
        high_color: str = "#d4edda",
    ) -> pd.io.formats.style.Styler:
        """Convert error evaluation to a styled pandas DataFrame for Jupyter display.

        Uses neutral formatting for enrichment odds ratios in the
        wide cross-group table.  Several key behaviours differ from the raw numeric
        output of to_dataframe():

        - OR cells display the point estimate with inline CI when available:
            '1.952 (1.344, 2.753)'  — when CI exists
            '1.952'                 — when CI is absent (n_bootstraps=None)
            '\u2014'                     — when OR is NaN (Overall/Overall row)
        - CI bound columns are omitted from the styled output; they are folded
          into the OR cell text, making the table narrower and self-contained.
        - Enrichment has no universal better/worse direction; cells are neutral.

        Args:
            n_decimals: Decimal places used when formatting float cells.
            metric_labels: Passed through to to_dataframe().
            include_count_metrics: Unused; kept for API consistency with other
                style_dataframe() methods.
            low_color: Background colour for low odds-ratio tier.
            medium_color: Background colour for medium odds-ratio tier.
            high_color: Background colour for high odds-ratio tier.

        Returns:
            A neutrally formatted pandas Styler. Enrichment has no universal
            performance direction.
        """
        from model_auditor._styling import style_dataframe

        return style_dataframe(
            self,
            n_decimals,
            metric_labels,
            include_count_metrics,
            low_color,
            medium_color,
            high_color,
        )


@dataclass
class CalibrationEvaluation:
    """Binned observed frequencies and probability-metric summary.

    Bins are fixed equal-width intervals on [0, 1], including empty bins.
    Bin intervals are pointwise and concern observed event frequency only;
    they are not simultaneous bands or uncertainty in mean predicted scores.
    """

    bins: pd.DataFrame
    summary: ScoreEvaluation
    metadata: dict[str, Any] = field(default_factory=dict)
