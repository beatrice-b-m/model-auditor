"""Matplotlib score distributions from configured auditor data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from model_auditor.core import Auditor
    from model_auditor.schemas import AuditorScore


def plot_score_distributions(
    auditor: Auditor,
    score_name: str,
    feature_names: Optional[list[str]] = None,
    bins: Union[int, str] = 30,
    density: bool = True,
    split_classes: bool = False,
) -> dict:
    """Render plot score distributions; see the public method for options."""
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
