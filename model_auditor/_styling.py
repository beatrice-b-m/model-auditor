"""Presentation helpers for formatted evaluation tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from model_auditor.schemas import ErrorEvaluation


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


def _tier_styles(
    values: pd.Series,
    lower_better: bool,
    low_color: str,
    medium_color: str,
    high_color: str,
) -> list[str]:
    """Rank a column once, retaining strict-less-than tie behavior and NaN blanks."""
    count = values.count()
    if not count:
        return [""] * len(values)
    if values.nunique() == 1:
        return [
            "" if pd.isna(v) else f"background-color: {medium_color}" for v in values
        ]
    percentiles = (values.rank(method="min") - 1) / count
    low, high = (high_color, low_color) if lower_better else (low_color, high_color)
    return [
        ""
        if pd.isna(rank)
        else f"background-color: {high if rank >= 2 / 3 else medium_color if rank >= 1 / 3 else low}"
        for rank in percentiles
    ]


def _apply_tier_styling(
    display_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    metric_names: list[str],
    include_count_metrics: bool = False,
    low_color: str = "#f8d7da",
    medium_color: str = "#fff3cd",
    high_color: str = "#d4edda",
    rank: bool = False,
    directions: dict[str, str | None] | None = None,
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
    if not rank:
        return display_df.style

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
        direction = (directions or {}).get(metric_name)
        if direction not in {"higher", "lower"}:
            continue
        lower_better = direction == "lower"

        if isinstance(numeric_values.index, pd.MultiIndex):
            # Rank within a feature, never against unrelated grouping variables.
            styles = numeric_values.groupby(level=0, sort=False).transform(
                lambda values: _tier_styles(
                    values, lower_better, low_color, medium_color, high_color
                )
            )
            style_df[metric_name] = styles
        else:
            style_df[metric_name] = _tier_styles(
                numeric_values, lower_better, low_color, medium_color, high_color
            )

    # Create and return the Styler
    return display_df.style.apply(lambda x: style_df, axis=None)


# ---------------------------------------------------------------------------
# Private helpers for interval plotting (used by ScoreEvaluation)
# ---------------------------------------------------------------------------


def style_dataframe(
    evaluation: ErrorEvaluation,
    n_decimals: int = 3,
    metric_labels: bool = False,
    include_count_metrics: bool = False,
    low_color: str = "#f8d7da",
    medium_color: str = "#fff3cd",
    high_color: str = "#d4edda",
) -> pd.io.formats.style.Styler:
    """Render style dataframe; see the public method for options."""
    numeric_df = evaluation.to_dataframe(metric_labels=metric_labels)
    if numeric_df.empty:
        return numeric_df.style

    or_col_name = "Odds Ratio" if metric_labels else "odds_ratio"
    confidence = 100 * evaluation.metadata.get("inference", {}).get(
        "confidence_level", 0.95
    )
    or_ci_lower_name = (
        f"OR {confidence:g}% CI Lower" if metric_labels else "odds_ratio_ci_lower"
    )
    or_ci_upper_name = (
        f"OR {confidence:g}% CI Upper" if metric_labels else "odds_ratio_ci_upper"
    )

    # CI bound columns are folded into the OR display string; drop them from
    # the visible output so the table stays narrow.
    ci_col_names = {or_ci_lower_name, or_ci_upper_name}
    display_cols = [c for c in numeric_df.columns if c[1] not in ci_col_names]

    # Build display DataFrame (string-formatted, no CI bound columns).
    display_df = pd.DataFrame(
        index=numeric_df.index,
        columns=pd.MultiIndex.from_tuples(display_cols),
        dtype=object,
    )
    for col in display_cols:
        section, metric = col
        if metric == or_col_name:
            # Fold CI bounds inline: 'or (lo, hi)' when available.
            ci_lower_col = (section, or_ci_lower_name)
            ci_upper_col = (section, or_ci_upper_name)
            formatted = []
            for idx in numeric_df.index:
                or_val = numeric_df.loc[idx, col]
                if pd.isna(or_val):
                    formatted.append("\u2014")
                else:
                    lo = (
                        numeric_df.loc[idx, ci_lower_col]
                        if ci_lower_col in numeric_df.columns
                        else float("nan")
                    )
                    hi = (
                        numeric_df.loc[idx, ci_upper_col]
                        if ci_upper_col in numeric_df.columns
                        else float("nan")
                    )
                    if not pd.isna(lo) and not pd.isna(hi):
                        formatted.append(
                            f"{or_val:.{n_decimals}f} ({lo:.{n_decimals}f}, {hi:.{n_decimals}f})"
                        )
                    else:
                        formatted.append(f"{or_val:.{n_decimals}f}")
            display_df[col] = formatted
        elif metric in ("N", "N_pos", "N_neg"):
            # Integer counts: thousands separator, em dash for NaN.
            display_df[col] = [
                "\u2014" if pd.isna(v) else f"{int(v):,}" for v in numeric_df[col]
            ]
        else:
            # Percentages and ratios: fixed decimal, em dash for NaN.
            display_df[col] = [
                "\u2014" if pd.isna(v) else f"{v:.{n_decimals}f}"
                for v in numeric_df[col]
            ]

    return display_df.style
