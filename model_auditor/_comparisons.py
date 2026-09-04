"""Shared-resample contrasts for fixed binary model and subgroup comparisons."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from model_auditor._evaluation import bootstrap_summary, resample
from model_auditor.schemas import InferenceConfig, LevelMetric


def contrast_estimate(left: float, right: float, contrast: str) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return (
        left - right
        if contrast == "difference"
        else left / right
        if right != 0
        else float("nan")
    )


def evaluate_contrast(
    data: pd.DataFrame,
    statistic: Callable,
    name: str,
    label: str,
    n_bootstraps: int | None,
    config: InferenceConfig,
    *,
    identical: bool = False,
) -> LevelMetric:
    """Pointwise contrast interval; shared rows retain overlap and model pairing."""
    estimate = statistic(data)
    result = LevelMetric(name, label, estimate)
    if not np.isfinite(estimate):
        result.status = "undefined"
        result.interval_status = "undefined_estimate"
        return result
    if n_bootstraps is None:
        return result
    if config.resampling == "cluster" and data[config.cluster].nunique() < 2:
        result.interval_status = "insufficient_clusters"
        return result
    rng = np.random.default_rng(config.random_state)
    values = np.asarray(
        [statistic(resample(data, config, rng)) for _ in range(n_bootstraps)]
    )
    bootstrap_summary(result, values, config, allow_degenerate=identical)
    return result
