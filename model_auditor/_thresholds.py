"""Threshold validation, conditional resolution, and score binarization."""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from model_auditor.schemas import AuditorScore, ConditionalThreshold, ThresholdSpec


def coerce_threshold_value(value: Any, context: str) -> float:
    """Convert a threshold candidate to a finite floating-point value."""
    try:
        threshold_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must be a finite numeric value, got {value!r}."
        ) from exc

    if not np.isfinite(threshold_value):
        raise ValueError(f"{context} must be a finite numeric value, got {value!r}.")
    return threshold_value


def resolve_threshold(
    score: AuditorScore, threshold: Optional[ThresholdSpec]
) -> ThresholdSpec:
    """Resolve and validate the effective threshold specification for a score."""
    resolved_threshold: Optional[ThresholdSpec] = (
        threshold if threshold is not None else score.threshold
    )
    if resolved_threshold is None:
        raise ValueError(
            f"Threshold for score '{score.name}' must be defined via "
            "add_score(threshold=...) or passed to the evaluation method."
        )

    if isinstance(resolved_threshold, ConditionalThreshold):
        if resolved_threshold.feature == "":
            raise ValueError(
                "Conditional threshold feature name must be a non-empty string."
            )
        validated_levels = {
            level: coerce_threshold_value(
                value=level_threshold,
                context=f"Threshold for level {level!r} in feature '{resolved_threshold.feature}'",
            )
            for level, level_threshold in resolved_threshold.levels.items()
        }
        validated_default = (
            None
            if resolved_threshold.default is None
            else coerce_threshold_value(
                value=resolved_threshold.default,
                context=f"Default threshold for feature '{resolved_threshold.feature}'",
            )
        )
        return ConditionalThreshold(
            feature=resolved_threshold.feature,
            levels=validated_levels,
            default=validated_default,
        )

    return coerce_threshold_value(
        value=resolved_threshold,
        context=f"Threshold for score '{score.name}'",
    )


def build_threshold_series(data: pd.DataFrame, threshold: ThresholdSpec) -> pd.Series:
    """Build a per-row threshold series from a scalar or conditional spec."""
    if isinstance(threshold, ConditionalThreshold):
        feature_name = threshold.feature
        if feature_name not in data.columns:
            raise ValueError(
                f"Conditional threshold feature '{feature_name}' not found in evaluation data."
            )

        feature_series = data[feature_name]
        threshold_series = feature_series.map(threshold.levels).astype(float)
        if threshold.default is not None:
            threshold_series = threshold_series.fillna(threshold.default)

        unresolved_level_mask = feature_series.notna() & threshold_series.isna()
        if unresolved_level_mask.any():
            unresolved_levels = (
                feature_series.loc[unresolved_level_mask]
                .drop_duplicates()
                .astype(str)
                .tolist()
            )
            raise ValueError(
                f"Conditional threshold for feature '{feature_name}' is missing "
                f"mappings for levels: {unresolved_levels}. Add level thresholds "
                "or set a default threshold."
            )

        unresolved_null_mask = feature_series.isna() & threshold_series.isna()
        if unresolved_null_mask.any():
            raise ValueError(
                f"Conditional threshold feature '{feature_name}' contains null "
                "values. Provide a default threshold to handle null rows."
            )

        return threshold_series.astype(float)

    scalar_threshold = coerce_threshold_value(value=threshold, context="Threshold")
    return pd.Series(scalar_threshold, index=data.index, dtype=float)


def binarize(score_data: pd.Series, threshold: Union[float, pd.Series]) -> pd.Series:
    """Convert continuous scores to binary predictions using per-row thresholds."""
    if isinstance(threshold, pd.Series):
        threshold_series = threshold.reindex(score_data.index)
    else:
        scalar_threshold = coerce_threshold_value(value=threshold, context="Threshold")
        threshold_series = pd.Series(
            scalar_threshold, index=score_data.index, dtype=float
        )

    if threshold_series.isna().any():
        raise ValueError(
            "Threshold resolution produced null values; ensure all rows resolve "
            "to a finite threshold."
        )
    return (score_data >= threshold_series).astype(int)
