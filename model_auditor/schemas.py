"""Data schemas for model auditor evaluation results.

This module defines the data structures used to store and organize
evaluation results, including features, scores, outcomes, and their
associated metrics at various levels of aggregation.
"""

from __future__ import annotations

from typing import Any, Optional, Union, Callable
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


# Helper functions for styling DataFrames


def _is_count_metric(metric_name: str) -> bool:
    """Check if a metric is a count metric (excluded from performance styling by default).

    Count metrics are identified by exact matching against known names,
    not prefix matching. This prevents misclassifying performance
    metrics like tpr/tnr/fpr/fnr as count metrics.

    Args:
        metric_name: Name of the metric to check.

    Returns:
        True if the metric is a count metric, False otherwise.
    """
    metric_name_upper = metric_name.upper()
    # Count metrics: exact matches only (case-insensitive)
    # Names with underscore prefix: n, n_tp, n_tn, n_fp, n_fn, n_pos, n_neg
    # Labels without prefix: N, TP, TN, FP, FN, Pos, Neg (with or without dots)
    count_metrics = {
        "N",
        "TP",
        "TN",
        "FP",
        "FN",
        "POS",
        "NEG",
        "POS.",
        "NEG.",  # labels (with and without dots)
        "N_TP",
        "N_TN",
        "N_FP",
        "N_FN",
        "N_POS",
        "N_NEG",  # with underscore
    }
    return metric_name_upper in count_metrics


def _is_lower_better_metric(metric_name: str) -> bool:
    """Check if a metric is lower-is-better (e.g., FPR, FNR).

    Args:
        metric_name: Name of the metric to check.

    Returns:
        True if the metric is lower-is-better, False otherwise.
    """
    metric_name_upper = metric_name.upper()
    lower_better_metrics = ["FPR", "FNR", "FALSE_POSITIVE_RATE", "FALSE_NEGATIVE_RATE"]
    return metric_name_upper in lower_better_metrics


def _get_metric_tier(
    value: float, values: pd.Series, lower_better: bool = False
) -> str:
    """Get the performance tier for a metric value relative to other values.

    Uses percentile-based tiering with thresholds at 0.33 and 0.66 for robustness
    across small samples and ties. Handles NaN values gracefully.

    Args:
        value: The value to classify.
        values: Series of all values for the metric (including the value).
        lower_better: If True, lower values are considered better performance.

    Returns:
        One of 'high', 'medium', 'low', or 'none' (for NaN).
    """
    if pd.isna(value):
        return "none"

    # Guard against empty series
    if values.count() == 0:
        return "none"

    # Calculate percentile rank (0-1) using strict less-than comparison
    # This handles ties by using consistent ranking across all values
    percentile = (values < value).sum() / values.count()

    # Define tier thresholds for 3-way split
    # Low: < 1/3, Medium: 1/3 to 2/3, High: >= 2/3
    if not lower_better:
        # Higher values are better: high tier has highest percentile
        if percentile >= 2 / 3:
            return "high"
        elif percentile >= 1 / 3:
            return "medium"
        else:
            return "low"
    else:
        # Lower values are better: invert the logic.
        # Use strict < boundaries so they mirror the higher_better >= boundaries.
        # higher_better: [0,1/3)→low, [1/3,2/3)→medium, [2/3,1]→high
        # lower_better:  [0,1/3)→high, [1/3,2/3)→medium, [2/3,1]→low
        if percentile < 1 / 3:
            return "high"
        elif percentile < 2 / 3:
            return "medium"
        else:
            return "low"


def _apply_tier_styling(
    display_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    metric_names: list[str],
    include_count_metrics: bool = False,
    low_color: str = "#f8d7da",
    medium_color: str = "#fff3cd",
    high_color: str = "#d4edda",
) -> pd.io.formats.style.Styler:
    """Apply tier-based coloring to a DataFrame.

    Args:
        display_df: DataFrame with formatted display values (strings).
        numeric_df: DataFrame with raw numeric values for tier classification.
        metric_names: List of metric names to apply styling to.
        include_count_metrics: If True, include count metrics in styling.
        low_color: Background color for low performance tier.
        medium_color: Background color for medium performance tier.
        high_color: Background color for high performance tier.

    Returns:
        A pandas Styler object with tier-based coloring applied.
    """
    # Initialize style matrix with all empty strings
    style_df = pd.DataFrame("", index=display_df.index, columns=display_df.columns)

    for metric_name in metric_names:
        if metric_name not in display_df.columns:
            continue

        # Skip count metrics unless explicitly included
        if not include_count_metrics and _is_count_metric(metric_name):
            continue

        # Get the numeric values for this metric
        numeric_values = numeric_df[metric_name]

        # Determine if this is a lower-is-better metric
        lower_better = _is_lower_better_metric(metric_name)

        # Apply tier coloring for each row
        for idx in display_df.index:
            value = numeric_values.loc[idx]
            tier = _get_metric_tier(value, numeric_values, lower_better=lower_better)

            if tier == "low":
                style_df.loc[idx, metric_name] = f"background-color: {low_color}"
            elif tier == "medium":
                style_df.loc[idx, metric_name] = f"background-color: {medium_color}"
            elif tier == "high":
                style_df.loc[idx, metric_name] = f"background-color: {high_color}"

    # Create and return the Styler
    return display_df.style.apply(lambda x: style_df, axis=None)



# ---------------------------------------------------------------------------
# Private helpers for interval plotting (used by ScoreEvaluation)
# ---------------------------------------------------------------------------


def _is_plottable_level(lm: LevelMetric) -> bool:
    """Return True iff a LevelMetric has a non-NaN score and finite CI bounds.

    Levels with NaN scores (e.g., unobserved categorical placeholders) or
    None/NaN intervals (e.g., count metrics or missing bootstrap runs) are
    excluded from interval plots.
    """
    if pd.isna(lm.score):
        return False
    if lm.interval is None:
        return False
    lo, hi = lm.interval
    return not (pd.isna(lo) or pd.isna(hi))


def _resolve_metric_key(
    metric: str,
    selected_features: list[str],
    features: dict[str, FeatureEvaluation],
) -> str:
    """Resolve a metric selector (name or label) to the internal metric name key.

    Performs exact name match first, then label match across all selected
    features.  Raises ValueError with actionable context if neither matches.

    Args:
        metric: User-supplied metric selector.
        selected_features: Feature names to search through.
        features: Feature evaluation dict from ScoreEvaluation.

    Returns:
        The internal metric name (key in LevelEvaluation.metrics).

    Raises:
        ValueError: If the metric is not found in any selected feature.
    """
    label_match: Optional[str] = None
    for fname in selected_features:
        feval = features[fname]
        for leval in feval.levels.values():
            # Exact name match takes priority over label match.
            if metric in leval.metrics:
                return metric
            # Record the first label match as a fallback.
            if label_match is None:
                for key, lm in leval.metrics.items():
                    if lm.label == metric:
                        label_match = key
    if label_match is not None:
        return label_match

    # Build a helpful error message from the first non-empty level.
    seen_names: list[str] = []
    seen_labels: list[str] = []
    for fname in selected_features:
        for leval in features[fname].levels.values():
            if leval.metrics:
                seen_names = sorted(leval.metrics.keys())
                seen_labels = sorted(lm.label for lm in leval.metrics.values())
                break
        if seen_names:
            break
    raise ValueError(
        f"Metric {metric!r} not found by name or label in the selected features. "
        f"Available names: {seen_names!r}, labels: {seen_labels!r}"
    )


def _get_metric_display_label(metric_key: str, feval: FeatureEvaluation) -> str:
    """Return the display label for a metric, falling back to its key.

    Looks up the label from the first level that contains the metric.
    """
    for leval in feval.levels.values():
        lm = leval.metrics.get(metric_key)
        if lm is not None:
            return lm.label
    return metric_key


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
    ) -> pd.io.formats.style.Styler:
        """Convert level evaluation to a styled pandas DataFrame for Jupyter display.

        Styles cells based on relative performance tiers within each metric column.

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

        if add_index:
            return pd.concat({self.label: pd.concat(data, axis=0)})
        else:
            return pd.concat(data, axis=0)

    def style_dataframe(
        self,
        n_decimals: int = 3,
        metric_labels: bool = False,
        include_count_metrics: bool = False,
        low_color: str = "#f8d7da",
        medium_color: str = "#fff3cd",
        high_color: str = "#d4edda",
    ) -> pd.io.formats.style.Styler:
        """Convert feature evaluation to a styled pandas DataFrame for Jupyter display.

        Styles cells based on relative performance tiers within each metric column.

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

        if add_index:
            return pd.concat({self.label: pd.concat(data, axis=0)})
        else:
            return pd.concat(data, axis=0)

    def style_dataframe(
        self,
        n_decimals: int = 3,
        metric_labels: bool = False,
        include_count_metrics: bool = False,
        low_color: str = "#f8d7da",
        medium_color: str = "#fff3cd",
        high_color: str = "#d4edda",
    ) -> pd.io.formats.style.Styler:
        """Convert score evaluation to a styled pandas DataFrame for Jupyter display.

        Styles cells based on relative performance tiers within each metric column.

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
        )

    def plot_metric_intervals(
        self,
        metric: str,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Create horizontal interval plots for a metric across feature levels.

        For each selected feature, produces one matplotlib figure with a
        horizontal error-bar per level: the point marks the metric mean (score)
        and the whiskers extend to the bootstrap CI lower and upper bounds.
        Levels without plottable CI data (NaN score, None interval, or NaN
        bounds) are silently excluded from that feature's plot.

        Args:
            metric: Metric to plot.  Matched first by exact name, then by
                label.  Example: ``"sensitivity"`` or ``"Sensitivity``.
            feature_names: Feature names to include.  ``None`` plots all
                features in the order they appear in ``self.features``.

        Returns:
            Dictionary mapping feature name to ``(matplotlib.figure.Figure,
            matplotlib.axes.Axes)`` tuples, one entry per selected feature.

        Raises:
            ImportError: If matplotlib is not installed.
            ValueError: If ``self.features`` is empty, ``feature_names``
                contains unknown names, the metric is not found in any
                selected feature, or a selected feature has no levels with
                plottable CI data.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for interval plots. "
                "Install it with: pip install matplotlib"
            ) from exc

        if not self.features:
            raise ValueError("ScoreEvaluation has no features to plot.")

        # Validate and resolve the feature list.
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

        # Resolve metric selector (name or label) to an internal metric key.
        metric_key = _resolve_metric_key(metric, selected_features, self.features)

        plots: dict[str, tuple] = {}
        for fname in selected_features:
            feval = self.features[fname]

            # Collect plottable tuples in level insertion order (preserves
            # categorical order where applicable).
            plottable: list[tuple[str, float, float, float]] = []
            for level_name, leval in feval.levels.items():
                lm = leval.metrics.get(metric_key)
                if lm is not None and _is_plottable_level(lm):
                    lower, upper = lm.interval  # type: ignore[misc]
                    plottable.append(
                        (level_name, float(lm.score), lower, upper)
                    )

            if not plottable:
                raise ValueError(
                    f"Feature {fname!r}: no levels have plottable CI data for "
                    f"metric {metric!r}. Ensure n_bootstraps was set during "
                    f"evaluation and the metric is CI-eligible."
                )

            level_names = [p[0] for p in plottable]
            scores = [p[1] for p in plottable]
            lowers = [p[2] for p in plottable]
            uppers = [p[3] for p in plottable]

            y = list(range(len(plottable)))
            # errorbar asymmetric xerr shape: [[left_deltas], [right_deltas]].
            xerr_low = [s - lo for s, lo in zip(scores, lowers)]
            xerr_high = [hi - s for s, hi in zip(scores, uppers)]

            fig, ax = plt.subplots()
            ax.errorbar(
                x=scores,
                y=y,
                xerr=[xerr_low, xerr_high],
                fmt="o",
                capsize=4,
            )
            ax.set_yticks(y)
            ax.set_yticklabels(level_names)
            # Invert y so the first level appears at the top, matching
            # the row order of to_dataframe().
            ax.invert_yaxis()

            metric_label = _get_metric_display_label(metric_key, feval)
            ax.set_xlabel(metric_label)
            ax.set_title(f"{feval.label}: {metric_label}")
            fig.tight_layout()
            plots[fname] = (fig, ax)

        return plots


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
class AuditorScore:
    """Configuration for a prediction score column.

    Defines a continuous score column that will be evaluated against
    the ground truth outcome.

    Attributes:
        name: Column name in the DataFrame containing prediction scores.
        label: Display label for the score (defaults to name if None).
        threshold: Optional threshold for binarizing continuous scores.
    """

    name: str
    label: Optional[str] = None
    threshold: Optional[float] = None


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
    threshold: float
    groups: dict[str, ScoreEvaluation] = field(default_factory=dict)
    global_total_n: int = 0
    # Sidecar support counts: {group_col: {feature_name: {level_name: {"n": int, "pct_overall": float, "pct_group": float}}}}
    # Used by to_dataframe() to compute class balance, overall N, and %-of-group metrics.
    support_data: dict = field(default_factory=dict)

    def to_dataframe(
        self, n_decimals: int = 3, metric_labels: bool = False
    ) -> pd.DataFrame:
        """Convert error evaluation to a wide cross-group analysis DataFrame.

        Returns a numeric DataFrame suitable for fairness and error auditing without
        additional reshaping:

        - Row index: MultiIndex(feature_label, level_name)
        - Column index: MultiIndex(section, metric)

        Sections and sub-columns:
          - ("Class Balance", "N_pos") : positives (TP+FN) at this level
          - ("Class Balance", "N_neg") : negatives (TN+FP) at this level
          - ("Class Balance", "Pos %") : N_pos / (N_pos + N_neg) per row
          - ("Class Balance", "Neg %") : N_neg / (N_pos + N_neg) per row
          - ("Overall", "N")          : total rows at this level (TP+TN+FP+FN)
          - ("Overall", "% overall")   : Overall N / global_total_n
          - For each group in TP/TN/FP/FN:
            - (GROUP, "N")                     : rows in this group at this level
            - (GROUP, "% overall")             : group N / global_total_n
            - (GROUP, "% group")              : group N / total rows in that group
            - (GROUP, or_col_name)            : odds ratio (NaN for overall/Overall
                                               row — OR is undefined when all rows
                                               belong to the level)
            - (GROUP, or_ci_lower_name)       : bootstrap 95% CI lower bound for OR
            - (GROUP, or_ci_upper_name)       : bootstrap 95% CI upper bound for OR
                                               (NaN when n_bootstraps was None)

        Args:
            n_decimals: Ignored; kept for API compatibility. Output is always numeric.
            metric_labels: If True, use "Odds Ratio" / "OR 95% CI Lower" /
                "OR 95% CI Upper" as column names; else use the machine-readable
                "odds_ratio" / "odds_ratio_ci_lower" / "odds_ratio_ci_upper".


        Returns:
            Numeric DataFrame with MultiIndex rows and columns, or an empty DataFrame
            when no groups have been evaluated.
        """
        if not self.groups:
            return pd.DataFrame()

        or_col_name = "Odds Ratio" if metric_labels else "odds_ratio"
        or_ci_lower_name = "OR 95% CI Lower" if metric_labels else "odds_ratio_ci_lower"
        or_ci_upper_name = "OR 95% CI Upper" if metric_labels else "odds_ratio_ci_upper"
        group_order = [g for g in ("tp", "tn", "fp", "fn") if g in self.groups]

        # Use the first group's feature ordering to determine all (feature, level) rows.
        # All groups are evaluated over the same features and levels.
        first_group = next(iter(self.groups.values()))
        feature_order: list[tuple[str, str, str]] = []  # (feature_name, feature_label, level_name)
        for fname, feval in first_group.features.items():
            for lname in feval.levels:
                feature_order.append((fname, feval.label, lname))

        rows: list[dict] = []
        index_tuples: list[tuple[str, str]] = []

        for feature_name, feature_label, level_name in feature_order:
            # The overall/Overall row has an undefined OR (all rows belong to the
            # level; no comparator population exists).  Emit NaN to signal that
            # this cell is not analytically meaningful.
            is_overall_level = (feature_name == "overall" and level_name == "Overall")

            # Pull raw support counts from the sidecar populated by evaluate_errors().
            def _n(group_col: str) -> int:
                return int(
                    self.support_data
                    .get(group_col, {})
                    .get(feature_name, {})
                    .get(level_name, {})
                    .get("n", 0)
                )

            def _pct(group_col: str, key: str) -> float:
                return float(
                    self.support_data
                    .get(group_col, {})
                    .get(feature_name, {})
                    .get(level_name, {})
                    .get(key, 0.0)
                )

            tp_n = _n("tp")
            tn_n = _n("tn")
            fp_n = _n("fp")
            fn_n = _n("fn")

            n_pos = tp_n + fn_n   # true class-positive count
            n_neg = tn_n + fp_n   # true class-negative count
            overall_n = tp_n + tn_n + fp_n + fn_n
            denom = self.global_total_n if self.global_total_n > 0 else None
            # cb_denom is the per-row class total; equals overall_n but expresses
            # intent: Pos%/Neg% are class fractions within this level's rows.
            cb_denom = n_pos + n_neg

            row: dict[tuple[str, str], Any] = {
                ("Overall", "N"): overall_n,
                ("Overall", "% overall"): overall_n / denom if denom else float("nan"),
                ("Class Balance", "N_pos"): n_pos,
                ("Class Balance", "N_neg"): n_neg,
                ("Class Balance", "Pos %"): n_pos / cb_denom if cb_denom > 0 else float("nan"),
                ("Class Balance", "Neg %"): n_neg / cb_denom if cb_denom > 0 else float("nan"),
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
                    row[(group_label, or_col_name)] = lm.score if lm is not None else float("nan")
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
        df.columns = pd.MultiIndex.from_tuples(list(df.columns)) # type: ignore
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

        Applies tier-based background colouring to the odds-ratio point-estimate
        columns of the wide cross-group table returned by to_dataframe().  Count
        columns (N, % overall, % group), class-balance columns, and CI bound
        columns are formatted but not coloured.

        Args:
            n_decimals: Decimal places used when formatting float cells.
            metric_labels: Passed through to to_dataframe().
            include_count_metrics: Unused; kept for API consistency with other
                style_dataframe() methods.
            low_color: Background colour for low odds-ratio tier.
            medium_color: Background colour for medium odds-ratio tier.
            high_color: Background colour for high odds-ratio tier.

        Returns:
            A pandas Styler with tier-based colouring applied to odds-ratio
            point-estimate columns.  Count, percentage, and CI cells are
            formatted but not coloured.
        """
        numeric_df = self.to_dataframe(metric_labels=metric_labels)
        if numeric_df.empty:
            return numeric_df.style

        or_col_name = "Odds Ratio" if metric_labels else "odds_ratio"
        group_order = [g for g in ("tp", "tn", "fp", "fn") if g in self.groups]

        # Format each column for display.
        def _fmt(val: Any, col: tuple[str, str]) -> str:
            if pd.isna(val):
                return "\u2014"
            section, metric = col
            # Count columns: integer display with thousands separator.
            if metric == "N" or metric in ("N_pos", "N_neg"):
                return f"{int(val):,}"
            # Percentage / ratio / CI columns: fixed decimal.
            return f"{val:.{n_decimals}f}"

        display_df = numeric_df.copy().astype(object)
        for col in numeric_df.columns:
            display_df[col] = [_fmt(v, col) for v in numeric_df[col]] # type: ignore

        # Build a style DataFrame for odds-ratio point-estimate columns only.
        # CI bound columns are left unstyled to avoid misleading visual encoding.
        style_df = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
        for group_col in group_order:
            group_label = group_col.upper()
            or_key = (group_label, or_col_name)
            if or_key not in numeric_df.columns:
                continue
            or_values = numeric_df[or_key]
            for idx in numeric_df.index:
                tier = _get_metric_tier(or_values.loc[idx], or_values)
                if tier == "low":
                    style_df.loc[idx, or_key] = f"background-color: {low_color}"
                elif tier == "medium":
                    style_df.loc[idx, or_key] = f"background-color: {medium_color}"
                elif tier == "high":
                    style_df.loc[idx, or_key] = f"background-color: {high_color}"

        return display_df.style.apply(lambda x: style_df, axis=None)