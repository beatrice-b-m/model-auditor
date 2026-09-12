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


# Human-readable explanations for omitted levels, keyed by interval_status.
_INTERVAL_STATUS_MESSAGES = {
    "not_requested": "interval not requested",
    "undefined_estimate": "estimate undefined",
    "insufficient_resamples": "too few valid resamples",
    "too_many_invalid_resamples": "too many invalid resamples",
    "degenerate_distribution": "degenerate resampling distribution",
    "insufficient_clusters": "too few clusters for resampling",
    "conditional_on_valid_resamples": "conditional on valid resamples",
}


def _describe_omission(lm: LevelMetric) -> str:
    """Return a readable reason why a level is not drawn."""
    if lm.interval is None:
        return _INTERVAL_STATUS_MESSAGES.get(lm.interval_status, lm.interval_status)
    return "nonfinite or reversed interval bounds"


def _boxes_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    pad: float = 0.0,
) -> bool:
    """Return True when two ``(x0, y0, x1, y1)`` boxes overlap (inflated by pad)."""
    return not (
        first[2] + pad <= second[0]
        or second[2] + pad <= first[0]
        or first[3] + pad <= second[1]
        or second[3] + pad <= first[1]
    )


def _stack_annotations(
    ax,
    fig,
    x_positions: list[float],
    y_positions: list[float],
    texts: list[str],
    fontsize: float = 7.0,
    base_offset_pts: float = 5.0,
    row_gap_pts: float = 20.0,
    pad_pts: float = 4.0,
) -> None:
    """Annotate rotated points, stacking overlapping labels into extra rows.

    Labels are anchored above their point and assigned to the lowest row whose
    rendered box does not collide with an already-placed label.  Placement uses
    real display geometry (including each point's own vertical position), so
    labels stay disjoint even when point estimates differ.  The y-axis grows
    until the tallest label row fits inside the axes.
    """
    if not texts:
        return
    # Force a full draw so transforms reflect final layout.
    fig.canvas.draw()

    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    text_to_path = TextToPath()
    properties = FontProperties(size=fontsize)
    pixels_per_point = fig.dpi / 72.0
    widths: list[float] = []
    heights: list[float] = []
    for text in texts:
        lines = text.splitlines() or [""]
        widths.append(
            max(
                text_to_path.get_text_width_height_descent(line, properties, False)[0]
                for line in lines
            )
            * pixels_per_point
        )
        heights.append(len(lines) * fontsize * 1.25 * pixels_per_point)

    pad = pad_pts * pixels_per_point
    assignments: list[int] = []
    y0, y1 = ax.get_ylim()
    for _ in range(8):
        centers = [
            ax.transData.transform((x, y)) for x, y in zip(x_positions, y_positions)
        ]
        placed: list[tuple[float, float, float, float]] = []
        assignments = []
        highest = 0.0
        for (center_x, center_y), width, height in zip(centers, widths, heights):
            row = 0
            while True:
                offset = (base_offset_pts + row * row_gap_pts) * pixels_per_point
                box = (
                    center_x - width / 2,
                    center_y + offset,
                    center_x + width / 2,
                    center_y + offset + height,
                )
                if all(not _boxes_overlap(box, other, pad) for other in placed):
                    break
                row += 1
            placed.append(box)
            assignments.append(row)
            highest = max(highest, box[3])
        overflow = highest - ax.bbox.y1
        if overflow <= 0.5 or ax.bbox.height <= 0:
            break
        units_per_pixel = (y1 - y0) / ax.bbox.height
        y1 = y1 + overflow * units_per_pixel * 1.02
        ax.set_ylim(y0, y1)

    for index, text in enumerate(texts):
        if not text:
            continue
        offset = base_offset_pts + assignments[index] * row_gap_pts
        ax.annotate(
            text,
            xy=(x_positions[index], y_positions[index]),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _layout_interval_figure(fig, ax, rotated: bool, omitted: bool) -> None:
    """Reserve physical space for rotated labels before placing annotations."""
    rect = (0, 0.08 if omitted else 0, 1, 1)
    if rotated:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_height = (
            max(
                (
                    label.get_window_extent(renderer).height
                    for label in ax.get_xticklabels()
                ),
                default=0,
            )
            / fig.dpi
        )
        fig.set_figheight(max(fig.get_figheight(), label_height + 3.5))
    fig.tight_layout(rect=rect)
    if rotated:
        # A numeric-axis expansion cannot create space when tick labels consume
        # the figure. Keep at least three inches for the data and annotations.
        for _ in range(3):
            shortfall = 3.0 - ax.get_window_extent().height / fig.dpi
            if shortfall <= 0:
                break
            fig.set_figheight(fig.get_figheight() + shortfall + 0.2)
            fig.tight_layout(rect=rect)


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
                omitted.append(f"{name}: {_describe_omission(lm)}")
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

        if omitted:
            fig.text(
                0.01, 0.01, "Not drawn: " + "; ".join(omitted), fontsize=8, wrap=True
            )
        _layout_interval_figure(fig, ax, rotate_plots, bool(omitted))

        if rotate_plots and (include_sample_size or include_class_balance):
            # Stack after layout so collision detection uses final geometry.
            texts: list[str] = []
            for ann_leval in plot_levals:
                n_lev, n_pos_lev, n_neg_lev = _extract_level_counts(ann_leval)
                texts.append(
                    _format_level_annotation(
                        n_lev,
                        n_overall,
                        n_pos_lev,
                        n_neg_lev,
                        include_sample_size,
                        include_class_balance,
                    )
                )
            # Start above both the estimate and its interval, including
            # percentile intervals that do not contain the point estimate.
            _stack_annotations(ax, fig, x, np.maximum(plot_scores, plot_uppers), texts)

        plots[fname] = (fig, ax)

    return plots
