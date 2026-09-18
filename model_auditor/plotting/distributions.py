"""Matplotlib score distributions from configured auditor data."""

from __future__ import annotations

from collections.abc import Sequence as _Sequence
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray as _NDArray

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from model_auditor.core import Auditor
    from model_auditor.schemas import AuditorScore


def plot_score_distributions(
    auditor: Auditor,
    score_name: str,
    feature_names: Optional[list[str]] = None,
    bins: int | str | _Sequence[float] | _NDArray[np.floating] = 30,
    density: bool = True,
    split_classes: bool = False,
) -> dict[str, tuple[Figure, _NDArray[np.object_]]]:
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
        auditor: Configured Auditor; its data is read without mutation.
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
            null values are filtered out."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for score distribution plots. "
            "Install it with: pip install matplotlib"
        ) from exc

    if auditor.data is None:
        raise ValueError("Please add data with .add_data() first")

    if not auditor.features:
        raise ValueError(
            "No features configured. Add features with .add_feature() first"
        )

    if score_name not in auditor.scores:
        available = ", ".join(auditor.scores.keys()) or "(none)"
        raise ValueError(
            f"Score '{score_name}' not found. Available scores: {available}"
        )

    if split_classes:
        if "_truth" not in auditor.data:
            raise ValueError(
                "Please define an outcome with .add_outcome() before splitting classes."
            )
        if not auditor.data["_truth"].isin([0, 1]).all():
            raise ValueError(
                "Class splitting requires binary outcomes (0 or 1) with no missing values."
            )

    score: AuditorScore = auditor.scores[score_name]
    score_label = score.label if score.label is not None else score.name

    # Resolve feature selection.
    if feature_names is None:
        selected_features = list(auditor.features.keys())
    else:
        unknown = [f for f in feature_names if f not in auditor.features]
        if unknown:
            raise ValueError(
                f"Unknown feature(s): {unknown!r}. "
                f"Available features: {list(auditor.features.keys())!r}"
            )
        selected_features = list(feature_names)

    plots: dict[str, tuple] = {}

    for fname in selected_features:
        from model_auditor._evaluation import _prepare_feature_data

        _prepare_feature_data(auditor.data, fname)
        feature = auditor.features[fname]
        feature_label = feature.label if feature.label is not None else feature.name
        feature_col = feature.name

        # Drop rows where the feature value or score value is null.
        feature_slice = auditor.data.dropna(subset=[feature_col, score.name]).copy()

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
            levels = feature_slice[feature_col].astype(str).drop_duplicates().tolist()

        if not levels:
            raise ValueError(
                f"Feature '{fname}' has no plottable levels after filtering "
                "null values from the data."
            )

        # Shared bin edges — computed once from the full feature slice so
        # every level subplot uses an identical grid.
        bin_edges = np.histogram_bin_edges(
            feature_slice[score.name].values,
            bins=bins,  # type: ignore
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
            level_rows = feature_slice.loc[str_col == level_name]
            if split_classes:
                for truth, label, color in (
                    (0, "Negative", "C0"),
                    (1, "Positive", "C1"),
                ):
                    values = level_rows.loc[level_rows["_truth"] == truth, score.name]
                    if not values.empty:
                        ax.hist(
                            values,
                            bins=bin_edges,
                            density=density,
                            zorder=2,
                            alpha=0.5,
                            label=label,
                            color=color,
                        )
                ax.legend()
            else:
                ax.hist(
                    level_rows[score.name], bins=bin_edges, density=density, zorder=2
                )

            positive_n = (
                int((level_rows["_truth"] == 1).sum())
                if "_truth" in level_rows
                else None
            )
            counts = f"N={len(level_rows)}" + (
                f", positive={positive_n}" if positive_n is not None else ""
            )
            ax.text(
                0.99,
                0.95,
                counts,
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
            )
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
        normalization = (
            "Density normalized within each class"
            if density and split_classes
            else "Density normalized within each level"
            if density
            else "Counts; panel vertical scales differ"
        )
        excluded = len(auditor.data) - len(feature_slice)
        fig.text(0.01, 0.01, f"{normalization}; excluded rows: {excluded}", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))

        plots[fname] = (fig, axes)

    return plots
