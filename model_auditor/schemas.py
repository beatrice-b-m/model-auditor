"""Data schemas for model auditor evaluation results.

This module defines the data structures used to store and organize
evaluation results, including features, scores, outcomes, and their
associated metrics at various levels of aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from model_auditor._styling import _apply_tier_styling

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas.io.formats.style import Styler


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable inference policies for fixed binary predictions.

    Args:
        confidence_level: Pointwise coverage probability in (0, 1); default 0.95.
        method: "auto" (default) uses analytic intervals for IID binomial rates
            and odds ratios, otherwise percentile bootstrap. "bootstrap" forces
            resampling. Neither policy refits models or corrects selection bias.
        rate_interval: "wilson" (default) approximate binomial interval, or "exact"
            for conservative Clopper-Pearson. Applies only to IID auto rates.
        resampling: "iid" (default) samples rows; "stratified" samples within truth
            classes, fixing observed class sizes; "cluster" samples whole clusters
            with replacement while retaining a row-weighted estimand.
        cluster: Column of nonmissing cluster IDs; required exactly when resampling
            is "cluster", otherwise None (default). Intervals need >=2 clusters.
        random_state: Nonnegative integer seed or None (default, nondeterministic).
            Uses local generators; never consumes NumPy's global random state.
        missing: Feature-level policy: "exclude" (default) drops missing levels per
            feature and records excluded_n; "include" labels them "(Missing)";
            "error" rejects missing levels. Truth/scores must always be nonmissing.
        min_valid_fraction: Required fraction of finite bootstrap estimates in
            (0, 1]; default 0.95. Intervals from fewer than all valid resamples
            are explicitly marked conditional_on_valid_resamples.
        min_resamples: Minimum finite resamples, integer >=2; default 100.
            Degenerate distributions also suppress intervals except recognized
            identical-model contrasts. Inspect LevelMetric.interval_status.

    Raises:
        ValueError: Invalid choices, ranges, seed, or cluster/resampling pairing.

    Notes:
        These arguments are also read-only dataclass fields. Passing
        n_bootstraps=None to evaluation disables analytic and bootstrap intervals.
        Configuration can be serialized with dataclasses.asdict()."""

    confidence_level: float = 0.95
    """Pointwise coverage probability in (0, 1); default 0.95."""
    method: Literal["auto", "bootstrap"] = "auto"
    """"auto" (default) uses analytic intervals for IID binomial rates
    and odds ratios, otherwise percentile bootstrap. "bootstrap" forces
    resampling. Neither policy refits models or corrects selection bias."""
    rate_interval: Literal["wilson", "exact"] = "wilson"
    """"wilson" (default) approximate binomial interval, or "exact"
    for conservative Clopper-Pearson. Applies only to IID auto rates."""
    resampling: Literal["iid", "stratified", "cluster"] = "iid"
    """"iid" (default) samples rows; "stratified" samples within truth
    classes, fixing observed class sizes; "cluster" samples whole clusters
    with replacement while retaining a row-weighted estimand."""
    cluster: Optional[str] = None
    """Column of nonmissing cluster IDs; required exactly when resampling
    is "cluster", otherwise None (default). Intervals need >=2 clusters."""
    random_state: Optional[int] = None
    """Nonnegative integer seed or None (default, nondeterministic).
    Uses local generators; never consumes NumPy's global random state."""
    missing: Literal["exclude", "include", "error"] = "exclude"
    """Feature-level policy: "exclude" (default) drops missing levels per
    feature and records excluded_n; "include" labels them "(Missing)";
    "error" rejects missing levels. Truth/scores must always be nonmissing."""
    min_valid_fraction: float = 0.95
    """Required fraction of finite bootstrap estimates in
    (0, 1]; default 0.95. Intervals from fewer than all valid resamples
    are explicitly marked conditional_on_valid_resamples."""
    min_resamples: int = 100
    """Minimum finite resamples, integer >=2; default 100.
    Degenerate distributions also suppress intervals except recognized
    identical-model contrasts. Inspect LevelMetric.interval_status."""

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
    """Mutable estimate and interval diagnostics for one metric in one subgroup.

    Args:
        name: Stable metric identifier used as the dictionary key.
        label: Human-readable display label.
        score: Original-sample estimate; NaN when undefined, infinity when unbounded.
        interval: Optional (lower, upper) pointwise bounds; default None. Absence
            does not imply certainty. A percentile interval need not contain score.
        status: Estimate status, normally "ok", "undefined" or "unbounded";
            default "ok". Direct construction does not infer status from score.
        interval_status: Default "not_requested". Evaluation may set "ok",
            "undefined_estimate", "insufficient_clusters", "insufficient_resamples",
            "too_many_invalid_resamples", "degenerate_distribution" or
            "conditional_on_valid_resamples". Check this when interval is None.
        interval_method: None (default), "wilson", "clopper_pearson",
            "conditional_exact", or an iid/stratified/cluster "_percentile" method.
        denominator: Binomial rate denominator when available; default None.
        valid_resamples: Finite bootstrap estimate count; default 0.
        requested_resamples: Actual attempted bootstrap count; default 0, including
            for analytic intervals that did not require resampling.
        nonfinite_resamples: NaN/infinite bootstrap estimate count; default 0.
        direction: "higher", "lower", "none" or None (default, unknown); descriptive
            styling metadata, not a significance or practical-importance claim.
        parameters: Metric configuration metadata; new empty dict by default.

    Notes:
        All arguments are public mutable fields. Dataclasses.asdict() preserves
        NaN/infinity; strict JSON consumers must explicitly encode these values."""

    name: str
    """Stable metric identifier used as the dictionary key."""
    label: str
    """Human-readable display label."""
    score: Union[float, int, np.integer]
    """Original-sample estimate; NaN when undefined, infinity when unbounded."""
    interval: Optional[tuple[float, float]] = None
    """Optional (lower, upper) pointwise bounds; default None. Absence
    does not imply certainty. A percentile interval need not contain score."""
    status: str = "ok"
    """Estimate status, normally "ok", "undefined" or "unbounded";
    default "ok". Direct construction does not infer status from score."""
    interval_status: str = "not_requested"
    """Default "not_requested". Evaluation may set "ok",
    "undefined_estimate", "insufficient_clusters", "insufficient_resamples",
    "too_many_invalid_resamples", "degenerate_distribution" or
    "conditional_on_valid_resamples". Check this when interval is None."""
    interval_method: Optional[str] = None
    """None (default), "wilson", "clopper_pearson",
    "conditional_exact", or an iid/stratified/cluster "_percentile" method."""
    denominator: Optional[int] = None
    """Binomial rate denominator when available; default None."""
    valid_resamples: int = 0
    """Finite bootstrap estimate count; default 0."""
    requested_resamples: int = 0
    """Actual attempted bootstrap count; default 0, including
    for analytic intervals that did not require resampling."""
    nonfinite_resamples: int = 0
    """NaN/infinite bootstrap estimate count; default 0."""
    direction: Optional[str] = None
    """"higher", "lower", "none" or None (default, unknown); descriptive
    styling metadata, not a significance or practical-importance claim."""
    parameters: dict[str, Any] = field(default_factory=dict)
    """Metric configuration metadata; new empty dict by default."""


@dataclass
class LevelEvaluation:
    """Mutable metric results and support counts for one string-labelled level.

    Args:
        name: String representation of the subgroup level.
        metrics: Mapping from metric ID to LevelMetric; new empty dict by default.
        support: New empty dict by default. Evaluation records n, n_pos, n_neg,
            and, when predictions exist, n_pred_pos and n_pred_neg. Cluster metric
            evaluation additionally records n_clusters.

    Notes:
        Arguments are public fields and supplied containers are stored by reference.
        Metrics and exported columns retain dictionary insertion order."""

    name: str
    """String representation of the subgroup level."""
    metrics: dict[str, LevelMetric] = field(default_factory=dict)
    """Mapping from metric ID to LevelMetric; new empty dict by default."""
    support: dict[str, int] = field(default_factory=dict)
    """New empty dict by default. Evaluation records n, n_pos, n_neg,
    and, when predictions exist, n_pred_pos and n_pred_neg. Cluster metric
    evaluation additionally records n_clusters."""

    def update(
        self,
        metric_name: str,
        metric_label: str,
        metric_score: float | int | np.integer,
    ) -> None:
        """Replace a metric for this level, resetting its interval and diagnostics.

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

        Raises:
            KeyError: A requested level or metric is absent.

        Notes:
            Mutates interval fields in place without changing diagnostic statuses.
            Updates are not transactional: earlier entries can change before an error."""
        for metric_name, confidence_interval in metric_intervals.items():
            self.metrics[metric_name].interval = confidence_interval

    def to_dataframe(
        self, n_decimals: int = 3, add_index: bool = False, metric_labels: bool = False
    ) -> pd.DataFrame:
        """Convert level evaluation to a pandas DataFrame.

        Args:
            n_decimals: Decimal places for display; default 3. Estimates are unchanged.
            add_index: Ignored; default False, retained for API compatibility.
            metric_labels: Use display labels if True; default False uses stable IDs.

        Returns:
            New single-row DataFrame of formatted strings, with metrics as columns
            and the level name as index; absent intervals are omitted from text.
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
    ) -> Styler:
        """Convert level evaluation to a styled pandas DataFrame for Jupyter display.

        Neutral by default. Set rank=True for descriptive within-feature ranks;
        colors do not express significance, equivalence, or practical importance.

        Args:
            n_decimals: Decimal places for display; default 3. Estimates are unchanged.
            metric_labels: Use display labels if True; default False uses stable IDs.
            include_count_metrics: False (default) excludes counts from ranks. True
                permits them, but a known higher/lower direction is still required.
            low_color: Low-tier background; default #f8d7da.
            medium_color: Middle-tier background; default #fff3cd.
            high_color: Background color for high performance tier; default #d4edda.
            rank: False (default) keeps neutral formatting. True applies relative
                ranks within features using known higher/lower directions.

        Returns:
            New pandas Styler with neutral formatting unless rank=True. Requires
            the styling extra (Jinja2); does not mutate result objects.

        Raises:
            ImportError: The optional styling dependency is unavailable.
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
    """Mutable evaluation results for a feature, in display order.

    Args:
        name: Stable feature column name.
        label: Display label used in formatted DataFrame indexes.
        levels: Mapping from string level labels to LevelEvaluation; new empty dict
            by default. Evaluation preserves categorical declaration order, including
            unobserved levels; noncategorical levels are sorted as strings.
        excluded_n: Rows excluded for missing values of this feature; default 0.
        total_n: Rows before this feature's exclusions; default 0.

    Notes:
        Arguments are public fields. Supplied containers are retained by reference.
        Missing levels can instead be included or rejected through InferenceConfig."""

    name: str
    """Stable feature column name."""
    label: str
    """Display label used in formatted DataFrame indexes."""
    levels: dict[str, LevelEvaluation] = field(default_factory=dict)
    """Mapping from string level labels to LevelEvaluation; new empty dict
    by default. Evaluation preserves categorical declaration order, including
    unobserved levels; noncategorical levels are sorted as strings."""
    excluded_n: int = 0
    """Rows excluded for missing values of this feature; default 0."""
    total_n: int = 0
    """Rows before this feature's exclusions; default 0."""

    def update(
        self,
        metric_name: str,
        metric_label: str,
        data: _Mapping[str, float | int | np.integer],
    ) -> None:
        """Replace one metric across levels, creating missing levels as needed.

        Resets intervals and diagnostics for that metric; other metrics remain.

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

        Raises:
            KeyError: A requested level or metric is absent.

        Notes:
            Mutates interval fields in place without changing diagnostic statuses.
            Updates are not transactional: earlier entries can change before an error."""
        self.levels[level_name].update_intervals(metric_intervals=metric_intervals)

    def to_dataframe(
        self, n_decimals: int = 3, add_index: bool = False, metric_labels: bool = False
    ) -> pd.DataFrame:
        """Convert feature evaluation to a pandas DataFrame.

        Args:
            n_decimals: Decimal places for display; default 3. Estimates are unchanged.
            add_index: If True, add feature label as a hierarchical index level.
            metric_labels: Use display labels if True; default False uses stable IDs.

        Returns:
            New DataFrame of formatted strings with levels in dictionary order
            as rows and metrics as columns. Does not mutate stored estimates.
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
    ) -> Styler:
        """Convert feature evaluation to a styled pandas DataFrame for Jupyter display.

        Neutral by default. Set rank=True for descriptive within-feature ranks;
        colors do not express significance, equivalence, or practical importance.

        Args:
            n_decimals: Decimal places for display; default 3. Estimates are unchanged.
            metric_labels: Use display labels if True; default False uses stable IDs.
            include_count_metrics: False (default) excludes counts from ranks. True
                permits them, but a known higher/lower direction is still required.
            low_color: Low-tier background; default #f8d7da.
            medium_color: Middle-tier background; default #fff3cd.
            high_color: Background color for high performance tier; default #d4edda.
            rank: False (default) keeps neutral formatting. True applies relative
                ranks within features using known higher/lower directions.

        Returns:
            New pandas Styler with neutral formatting unless rank=True. Requires
            the styling extra (Jinja2); does not mutate result objects.

        Raises:
            ImportError: The optional styling dependency is unavailable.
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
    """Mutable results for a score, comparison, or confusion-group analysis.

    Args:
        name: Stable score identifier (or confusion group key for error results).
        label: Display label for this evaluation.
        features: Mapping to FeatureEvaluation, new empty dict by default.
            Evaluations normally place "overall" first, then registered features;
            group comparisons contain only the requested feature.
        metadata: Provenance mapping, new empty dict by default. Evaluation records
            version, metric parameters, threshold, outcome mapping, cohort, inference
            and estimand. Supplied dictionaries are retained by reference.

    Notes:
        Arguments are public fields. Navigate results as
        ``result.features[feature].levels[level].metrics[metric]``. Numeric exports
        preserve NaN/infinity and copy metadata; formatted exports contain strings.
        DataFrame attrs are not preserved by every file format: serialize provenance
        separately when exporting CSV, and encode NaN/infinity for strict JSON."""

    name: str
    """Stable score identifier (or confusion group key for error results)."""
    label: str
    """Display label for this evaluation."""
    features: dict[str, FeatureEvaluation] = field(default_factory=dict)
    """Mapping to FeatureEvaluation, new empty dict by default.
    Evaluations normally place "overall" first, then registered features;
    group comparisons contain only the requested feature."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Provenance mapping, new empty dict by default. Evaluation records
    version, metric parameters, threshold, outcome mapping, cohort, inference
    and estimand. Supplied dictionaries are retained by reference."""

    def to_numeric_dataframe(self) -> pd.DataFrame:
        """Unrounded long-form results, support, diagnostics, and stable IDs.

        Evaluation provenance is copied into ``frame.attrs["metadata"]``.
        Intervals are pointwise, conditional on the supplied predictions.

        Returns:
            New DataFrame with one row per feature/level/metric in dictionary order.
            Columns include score, feature, level, metric, estimate, lower, upper,
            status, interval_status, interval_method, denominator, resample counts,
            direction, parameters, excluded_n, total_n and available support counts.
            Missing intervals are NaN. Empty results have no fixed column schema.
            Nested metric parameters and attrs["metadata"] are deep-copied; editing
            the frame does not update result objects. File formats may drop attrs."""
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
            n_decimals: Decimal places for display; default 3. Estimates are unchanged.
            add_index: If True, add score label as a hierarchical index level.
            metric_labels: Use display labels if True; default False uses stable IDs.

        Returns:
            New DataFrame of formatted strings, hierarchical (feature label,
            level) index and metric columns, in stored order; no result mutation.
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
    ) -> Styler:
        """Convert score evaluation to a styled pandas DataFrame for Jupyter display.

        Neutral by default. Set rank=True for descriptive within-feature ranks;
        colors do not express significance, equivalence, or practical importance.

        Args:
            n_decimals: Decimal places for display; default 3. Estimates are unchanged.
            metric_labels: Use display labels if True; default False uses stable IDs.
            include_count_metrics: False (default) excludes counts from ranks. True
                permits them, but a known higher/lower direction is still required.
            low_color: Low-tier background; default #f8d7da.
            medium_color: Middle-tier background; default #fff3cd.
            high_color: Background color for high performance tier; default #d4edda.
            rank: False (default) keeps neutral formatting. True applies relative
                ranks within features using known higher/lower directions.

        Returns:
            New pandas Styler with neutral formatting unless rank=True. Requires
            the styling extra (Jinja2); does not mutate result objects.

        Raises:
            ImportError: The optional styling dependency is unavailable.
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
    ) -> dict[str, tuple[Figure, Axes]]:
        """Create interval plots for a metric across feature levels.

        For each selected feature, produces one matplotlib figure with an
        error-bar per level: the point marks the metric estimate (score) and
        the whiskers extend to stored analytic or bootstrap CI bounds.

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
            one entry per selected feature in selected/stored order. Does not
            mutate results or show figures. Callers own the figures and should
            close them with matplotlib.pyplot.close(fig) after use.

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
    """Column name in the DataFrame."""
    label: Optional[str] = None
    """Display label for the feature (defaults to name if None)."""


@dataclass
class ConditionalThreshold:
    """Select a finite decision threshold from a row's feature value.

    Args:
        feature: Existing column used for lookup; need not be registered as a subgroup.
        levels: Mapping of original feature values (not display strings) to finite
            numeric thresholds. Uses exact pandas mapping semantics.
        default: Finite fallback for unmapped or missing values; default None.
            Missing/unmapped rows without a default raise ValueError during use.

    Notes:
        Arguments are mutable dataclass fields. Validation happens when evaluated.
        Scores >= their selected threshold are positive. Thresholds use the score's
        units and need not be in [0, 1]. Calls may override the whole configuration.
        Mapping keys can be nonstrings; dataclasses.asdict() is not necessarily
        JSON-safe without application-specific key encoding.

    Example:
        >>> policy = ConditionalThreshold("region", {"North": 0.4}, default=0.5)
        >>> policy.levels["North"]
        0.4"""

    feature: str
    """Existing column used for lookup; need not be registered as a subgroup."""
    levels: dict[Any, float]
    """Mapping of original feature values (not display strings) to finite
    numeric thresholds. Uses exact pandas mapping semantics."""
    default: Optional[float] = None
    """Finite fallback for unmapped or missing values; default None.
    Missing/unmapped rows without a default raise ValueError during use."""


ThresholdSpec = float | ConditionalThreshold
"""Finite scalar or row-conditional policy; positive predictions use score >= threshold."""


@dataclass
class AuditorScore:
    """Configuration for a prediction score column.

    Defines a continuous score column that will be evaluated against
    the ground truth outcome.

    Attributes:
        name: Column name in the DataFrame containing prediction scores.
        label: Display label for the score (defaults to name if None).
        threshold: Optional scalar or conditional threshold; default None.
            Scores >= threshold are positive. Only decision metrics require a
            threshold. Stored by reference and validated at evaluation time.
    """

    name: str
    """Column name in the DataFrame containing prediction scores."""
    label: Optional[str] = None
    """Display label for the score (defaults to name if None)."""
    threshold: Optional[ThresholdSpec] = None
    """Optional scalar or conditional threshold; default None.
    Scores >= threshold are positive. Only decision metrics require a
    threshold. Stored by reference and validated at evaluation time."""


@dataclass
class AuditorOutcome:
    """Configuration for the ground truth outcome column.

    Defines the outcome (label) column that predictions are compared against.

    Attributes:
        name: Column name in the DataFrame containing ground truth labels.
        mapping: Optional dictionary to convert outcome values to binary (0/1),
            e.g., {"positive": 1, "negative": 0}. Default None uses values
            directly. Unmapped values become missing; evaluation rejects them.
            The dataclass retains mapping by reference; Auditor registration
            deep-copies it.
    """

    name: str
    """Column name in the DataFrame containing ground truth labels."""
    mapping: Optional[dict[Any, int]] = None
    """Optional dictionary to convert outcome values to binary (0/1),
    e.g., {"positive": 1, "negative": 0}. Default None uses values
    directly. Unmapped values become missing; evaluation rejects them.
    The dataclass retains mapping by reference; Auditor registration
    deep-copies it."""


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
            ScoreEvaluation objects containing per-feature metrics. New empty
            dict by default, ordered tp, tn, fp, fn in evaluated results.
        global_total_n: Row count before per-feature exclusions; default 0.
        metadata: Provenance mapping; new empty dict by default.
        support_data: New empty dict by default. Nested group -> feature -> level
            -> support mapping, containing n, pct_overall and pct_group. Despite
            their names, pct fields are fractions in [0, 1], not percentages.
            pct_overall uses global_total_n; pct_group uses all rows in that
            confusion group after per-feature exclusions.

    Notes:
        Arguments are mutable fields and supplied containers are retained by
        reference. Use to_numeric_dataframe() for custom error metrics; legacy
        wide output expects odds_ratio. NaN and infinity require explicit
        encoding for strict JSON; DataFrame attrs need separate CSV storage.
    """

    name: str
    """Name of the score that was evaluated."""
    label: str
    """Display label for the score."""
    threshold: ThresholdSpec
    """Binarization threshold used during evaluation."""
    groups: dict[str, ScoreEvaluation] = field(default_factory=dict)
    """Dictionary mapping group keys ('tp', 'tn', 'fp', 'fn') to
    ScoreEvaluation objects containing per-feature metrics. New empty
    dict by default, ordered tp, tn, fp, fn in evaluated results."""
    global_total_n: int = 0
    """Row count before per-feature exclusions; default 0."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Provenance mapping; new empty dict by default."""
    # Sidecar support counts: {group_col: {feature_name: {level_name: {"n": int, "pct_overall": float, "pct_group": float}}}}
    # Used by to_dataframe() to compute class balance, overall N, and %-of-group metrics.
    support_data: dict[str, dict[str, dict[str, dict[str, float]]]] = field(
        default_factory=dict
    )
    """New empty dict by default. Nested group -> feature -> level
    -> support mapping, containing n, pct_overall and pct_group. Despite
    their names, pct fields are fractions in [0, 1], not percentages.
    pct_overall uses global_total_n; pct_group uses all rows in that
    confusion group after per-feature exclusions."""

    def to_numeric_dataframe(self) -> pd.DataFrame:
        """Long-form enrichment estimates and diagnostics; not error-rate contrasts.

        Returns:
            New DataFrame with one row per feature/level/metric in dictionary order.
            Columns include score, feature, level, metric, estimate, lower, upper,
            status, interval_status, interval_method, denominator, resample counts,
            direction, parameters, excluded_n, total_n and available support counts.
            Missing intervals are NaN. Empty results have no fixed column schema.
            Nested metric parameters and attrs["metadata"] are deep-copied; editing
            the frame does not update result objects. File formats may drop attrs.
            Adds confusion_group and uses the original score ID across groups."""
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
    ) -> Styler:
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
            low_color: Ignored compatibility argument; default #f8d7da.
            medium_color: Ignored compatibility argument; default #fff3cd.
            high_color: Ignored compatibility argument; default #d4edda.

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
    """Mutable reliability bins and probability-metric results.

    Args:
        bins: DataFrame with feature, level, zero-based bin, bin_lower, bin_upper,
            n, n_pos, mean_prediction, observed_frequency, lower, upper,
            interval_status and interval_method. Bounds and frequencies use [0, 1]
            probability units. Bins are left-closed/right-open except the final bin
            includes 1. Empty bins have n=0 and NaN estimates. Only observed subgroup
            levels have bin rows. attrs["metadata"] contains evaluation provenance.
        summary: ScoreEvaluation with Brier score, log loss, calibration intercept
            and slope, including declared unobserved categories.
        metadata: Provenance mapping; new empty dict by default.

    Notes:
        Arguments are public fields retained by reference. Bin intervals are
        pointwise for observed event frequency, not mean-prediction uncertainty
        or simultaneous bands. Preserve metadata separately when serializing to
        formats such as CSV that discard DataFrame attrs."""

    bins: pd.DataFrame
    """DataFrame with feature, level, zero-based bin, bin_lower, bin_upper,
    n, n_pos, mean_prediction, observed_frequency, lower, upper,
    interval_status and interval_method. Bounds and frequencies use [0, 1]
    probability units. Bins are left-closed/right-open except the final bin
    includes 1. Empty bins have n=0 and NaN estimates. Only observed subgroup
    levels have bin rows. attrs["metadata"] contains evaluation provenance."""
    summary: ScoreEvaluation
    """ScoreEvaluation with Brier score, log loss, calibration intercept
    and slope, including declared unobserved categories."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Provenance mapping; new empty dict by default."""
