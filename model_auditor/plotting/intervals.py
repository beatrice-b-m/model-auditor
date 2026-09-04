"""Matplotlib interval plots for evaluation results (loaded on demand)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from model_auditor.schemas import (
        FeatureEvaluation,
        LevelEvaluation,
        LevelMetric,
        ScoreEvaluation,
    )


def _is_plottable_level(lm: LevelMetric) -> bool:
    """Return True iff a LevelMetric has a non-NaN score and finite CI bounds.

    Levels with NaN scores (e.g., unobserved categorical placeholders) or
    None/NaN intervals (e.g., count metrics or missing bootstrap runs) are
    excluded from interval plots.
    """
    if not np.isfinite(lm.score):
        return False
    if lm.interval is None:
        return False
    lo, hi = lm.interval
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo <= hi)


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


def _extract_level_counts(
    leval: "LevelEvaluation",
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract (n, n_pos, n_neg) from a LevelEvaluation's metrics.

    Tries direct metric lookup first (metric names ``"n"``, ``"n_pos"``,
    ``"n_neg"``), then derives missing values from confusion-matrix component
    counts (n_tp, n_tn, n_fp, n_fn).  Returns ``None`` for any count that
    cannot be determined from the available metrics.
    """

    def _count(key: str) -> Optional[int]:
        if key in leval.support:
            return leval.support[key]
        lm = leval.metrics.get(key)
        if lm is not None and not pd.isna(lm.score):
            return int(lm.score)
        return None

    n = _count("n")
    n_pos = _count("n_pos")
    n_neg = _count("n_neg")
    n_tp = _count("n_tp")
    n_tn = _count("n_tn")
    n_fp = _count("n_fp")
    n_fn = _count("n_fn")

    # Derive n_pos and n_neg from confusion-matrix components when direct
    # count metrics are absent.
    if n_pos is None and n_tp is not None and n_fn is not None:
        n_pos = n_tp + n_fn
    if n_neg is None and n_tn is not None and n_fp is not None:
        n_neg = n_tn + n_fp

    # Derive total n last so it can use the (potentially derived) n_pos/n_neg.
    if n is None:
        if n_pos is not None and n_neg is not None:
            n = n_pos + n_neg
        elif (
            n_tp is not None
            and n_tn is not None
            and n_fp is not None
            and n_fn is not None
        ):
            n = n_tp + n_tn + n_fp + n_fn

    return n, n_pos, n_neg


def _format_level_annotation(
    n_level: Optional[int],
    n_overall: Optional[int],
    n_pos_level: Optional[int],
    n_neg_level: Optional[int],
    include_sample_size: bool,
    include_class_balance: bool,
) -> str:
    """Build annotation text for a single level point.

    Sample size fragment:    ``"N: {n_level} ({pct_of_overall}%)"``
    Class balance fragment:  ``"N Pos: {n_pos_level} ({pct_positive}%)"``

    ``NA`` placeholders are emitted for any value that cannot be computed
    (missing counts or zero denominator).  Returns an empty string when both
    ``include_sample_size`` and ``include_class_balance`` are ``False``.
    """
    fragments: list[str] = []

    if include_sample_size:
        if n_level is not None:
            if n_overall is not None and n_overall > 0:
                pct = 100.0 * n_level / n_overall
                fragments.append(f"N: {n_level} ({pct:.1f}%)")
            else:
                fragments.append(f"N: {n_level} (NA)")
        else:
            fragments.append("N: NA (NA)")

    if include_class_balance:
        if n_pos_level is not None:
            denom = n_pos_level + n_neg_level if n_neg_level is not None else None
            if denom is not None and denom > 0:
                pct = 100.0 * n_pos_level / denom
                fragments.append(f"N Pos: {n_pos_level} ({pct:.1f}%)")
            else:
                fragments.append(f"N Pos: {n_pos_level} (NA)")
        else:
            fragments.append("N Pos: NA (NA)")

    return "\n".join(fragments)


# -- Figure sizing constants for interval plots --------------------------
# Height scales linearly with the number of plotted levels so every level
# gets consistent vertical space.  Width is fixed.
_INTERVAL_PLOT_WIDTH = 8.0
_INTERVAL_PLOT_HEIGHT_PER_LEVEL = 0.55
_INTERVAL_PLOT_MIN_HEIGHT = 2.5


def _interval_plot_figsize(n_levels: int) -> tuple[float, float]:
    """Compute figure dimensions for an interval plot.

    Args:
        n_levels: Number of levels that will be rendered (including an
            ``Overall`` comparator if present).

    Returns:
        ``(width, height)`` tuple suitable for ``plt.subplots(figsize=...)``.
    """
    height = max(
        _INTERVAL_PLOT_MIN_HEIGHT,
        n_levels * _INTERVAL_PLOT_HEIGHT_PER_LEVEL,
    )
    return (_INTERVAL_PLOT_WIDTH, height)


def plot_metric_intervals(
    evaluation: ScoreEvaluation,
    metric: str,
    feature_names: Optional[list[str]] = None,
    include_overall: bool = True,
    rotate_plots: bool = False,
    include_sample_size: bool = True,
    include_class_balance: bool = True,
) -> dict:
    """Render plot metric intervals; see the public method for options."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for interval plots. "
            "Install it with: pip install matplotlib"
        ) from exc

    if not evaluation.features:
        raise ValueError("ScoreEvaluation has no features to plot.")

    # Always exclude the synthetic "overall" feature from standalone
    # subplots — it is shown as a comparator level inside each feature
    # subplot when include_overall=True.
    if feature_names is None:
        selected_features = [f for f in evaluation.features.keys() if f != "overall"]
    else:
        unknown = [f for f in feature_names if f not in evaluation.features]
        if unknown:
            raise ValueError(
                f"Unknown feature(s): {unknown!r}. "
                f"Available features: {list(evaluation.features.keys())!r}"
            )
        selected_features = [f for f in feature_names if f != "overall"]

    if not selected_features:
        raise ValueError(
            "No plottable features remain after excluding 'overall'. "
            "Provide at least one non-'overall' feature in feature_names."
        )

    # Resolve metric selector (name or label) to an internal metric key.
    metric_key = _resolve_metric_key(metric, selected_features, evaluation.features)

    # Resolve overall level data once: used both for prepending the
    # Overall comparator level and as the n_overall denominator in
    # sample-size annotations.
    overall_leval: Optional[LevelEvaluation] = None
    if "overall" in evaluation.features:
        overall_leval = evaluation.features["overall"].levels.get("Overall")
    overall_lm: Optional[LevelMetric] = None
    if overall_leval is not None:
        overall_lm = overall_leval.metrics.get(metric_key)
    n_overall: Optional[int] = None
    if overall_leval is not None:
        n_overall, _, _ = _extract_level_counts(overall_leval)

    plots: dict[str, tuple] = {}
    for fname in selected_features:
        feval = evaluation.features[fname]

        # Collect plottable entries as parallel lists.
        # Level insertion order is preserved so the plot matches
        # the row order of to_dataframe().
        plot_names: list[str] = []
        plot_scores: list[float] = []
        plot_lowers: list[float] = []
        plot_uppers: list[float] = []
        plot_levals: list[LevelEvaluation] = []

        # Prepend Overall comparator when requested and CI data exists.
        if (
            include_overall
            and overall_leval is not None
            and overall_lm is not None
            and _is_plottable_level(overall_lm)
        ):
            lower, upper = overall_lm.interval  # type: ignore[misc]
            plot_names.append("Overall")
            plot_scores.append(float(overall_lm.score))
            plot_lowers.append(lower)
            plot_uppers.append(upper)
            plot_levals.append(overall_leval)

        for level_name, leval in feval.levels.items():
            lm = leval.metrics.get(metric_key)
            if lm is not None and _is_plottable_level(lm):
                lower, upper = lm.interval  # type: ignore[misc]
                plot_names.append(level_name)
                plot_scores.append(float(lm.score))
                plot_lowers.append(lower)
                plot_uppers.append(upper)
                plot_levals.append(leval)

        omitted = []
        candidates = list(feval.levels.items())
        if include_overall and overall_leval is not None:
            candidates.insert(0, ("Overall", overall_leval))
        for name, level in candidates:
            lm = level.metrics.get(metric_key)
            if lm is not None and not _is_plottable_level(lm):
                reason = (
                    lm.interval_status
                    if lm.interval is None
                    else "unbounded or invalid interval"
                )
                omitted.append(f"{name}: {reason}")
        if not plot_names:
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.set_title(f"{feval.label}: {metric_key}")
            ax.text(
                0,
                1,
                "No estimable intervals\n" + "\n".join(omitted),
                va="top",
                transform=ax.transAxes,
            )
            plots[fname] = (fig, ax)
            continue

        metric_label = _get_metric_display_label(metric_key, feval)
        fig, ax = plt.subplots(figsize=_interval_plot_figsize(len(plot_names)))

        # Percentile CIs need not contain the original-sample estimate. Draw
        # whiskers about an in-interval anchor, and draw the estimate separately.
        anchors = np.clip(plot_scores, plot_lowers, plot_uppers)
        errors = [(anchors - plot_lowers).tolist(), (plot_uppers - anchors).tolist()]
        if not rotate_plots:
            import matplotlib.transforms as mtransforms

            # Horizontal: metric value on x-axis, levels on y-axis.
            y = list(range(len(plot_names)))
            (points,) = ax.plot(plot_scores, y, "o")
            ax.errorbar(
                x=anchors,
                y=y,
                xerr=errors,
                fmt="none",
                ecolor=points.get_color(),
                capsize=4,
            )
            ax.set_yticks(y)
            ax.set_yticklabels(plot_names)
            # Invert y so the first level appears at the top, matching
            # the row order of to_dataframe().
            ax.invert_yaxis()
            ax.set_xlabel(metric_label)
            ax.set_title(f"{feval.label}: {metric_label}")

            if include_sample_size or include_class_balance:
                # Place all annotations at a fixed x in axes coordinates
                # (left of the y-axis class labels) with y in data
                # coordinates so each annotation aligns with its level.
                ann_transform = mtransforms.blended_transform_factory(
                    ax.transAxes, ax.transData
                )
                # Negative axes-x places annotations left of the plot
                # area, to the left of the y-tick class labels.
                ann_x = -0.02
                renderer = fig.canvas.get_renderer()
                label_width = (
                    max(
                        label.get_window_extent(renderer).width
                        for label in ax.get_yticklabels()
                    )
                    * 72
                    / fig.dpi
                )

                for i, ann_leval in enumerate(plot_levals):
                    n_lev, n_pos_lev, n_neg_lev = _extract_level_counts(ann_leval)
                    text = _format_level_annotation(
                        n_lev,
                        n_overall,
                        n_pos_lev,
                        n_neg_lev,
                        include_sample_size,
                        include_class_balance,
                    )
                    if text:
                        ax.annotate(
                            text,
                            xy=(ann_x, y[i]),
                            xycoords=ann_transform,
                            xytext=(-label_width - 8, 0),
                            textcoords="offset points",
                            ha="right",
                            va="center",
                            fontsize=7,
                        )
        else:
            # Rotated: levels on x-axis, metric value on y-axis.
            x = list(range(len(plot_names)))
            (points,) = ax.plot(x, plot_scores, "o")
            ax.errorbar(
                x=x,
                y=anchors,
                yerr=errors,
                fmt="none",
                ecolor=points.get_color(),
                capsize=4,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(plot_names, rotation=45, ha="right")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{feval.label}: {metric_label}")

            if include_sample_size or include_class_balance:
                for i, (score, ann_leval) in enumerate(zip(plot_scores, plot_levals)):
                    n_lev, n_pos_lev, n_neg_lev = _extract_level_counts(ann_leval)
                    text = _format_level_annotation(
                        n_lev,
                        n_overall,
                        n_pos_lev,
                        n_neg_lev,
                        include_sample_size,
                        include_class_balance,
                    )
                    if text:
                        ax.annotate(
                            text,
                            xy=(x[i], score),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7,
                        )

        if omitted:
            fig.text(
                0.01, 0.01, "Not drawn: " + "; ".join(omitted), fontsize=8, wrap=True
            )
        fig.tight_layout(rect=(0, 0.08 if omitted else 0, 1, 1))
        plots[fname] = (fig, ax)

    return plots
