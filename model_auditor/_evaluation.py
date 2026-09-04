"""Subgroup aggregation and bootstrap calculations, independent of Auditor state."""

from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from model_auditor.error_metrics import AuditorErrorMetric
from model_auditor.metrics import AuditorMetric
from model_auditor.schemas import AuditorFeature, FeatureEvaluation, LevelEvaluation


def _prepare_feature_data(
    data: pd.DataFrame, feature: str
) -> tuple[pd.DataFrame, list[str] | None]:
    """Drop null levels and retain declared categorical order and placeholders."""
    categories = None
    if isinstance(data[feature].dtype, pd.CategoricalDtype):
        categories = [str(value) for value in data[feature].cat.categories]
    result = data.dropna(subset=[feature]).copy()
    if categories is None:
        result[feature] = result[feature].astype(str)
    return result, categories


def _level_counts(data: pd.DataFrame, feature: str) -> dict[str, int]:
    return {
        str(key): int(count)
        for key, count in data.groupby(feature, observed=True).size().items()
    }


def validate_n_bootstraps(n_bootstraps: int | None) -> None:
    """Require a positive integer or the explicit no-bootstrap sentinel."""
    if n_bootstraps is not None and (
        isinstance(n_bootstraps, (bool, np.bool_))
        or not isinstance(n_bootstraps, (int, np.integer))
        or n_bootstraps <= 0
    ):
        raise ValueError("n_bootstraps must be a positive integer or None.")


def evaluate_feature(
    metrics: list[AuditorMetric],
    data: pd.DataFrame,
    feature: AuditorFeature,
    n_bootstraps: Optional[int],
) -> FeatureEvaluation:
    """Evaluate all metrics for a single feature across its levels.

    When the feature column carries a categorical dtype, levels are ordered
    according to the declared category order.  Categories present in the
    declaration but absent in the data appear as placeholder rows whose
    metric scores are NaN.

    Args:
        data: DataFrame containing the evaluation data with metric input columns.
        feature: The feature to stratify evaluation by.
        n_bootstraps: Number of bootstrap samples for CI calculation, or None.

    Returns:
        FeatureEvaluation containing metrics for each level of the feature.
    """
    feature_data, declared_categories = _prepare_feature_data(data, feature.name)
    feature_eval = FeatureEvaluation(
        name=feature.name,
        label=feature.label if feature.label is not None else feature.name,
    )

    # Iterate each group once, retaining its feature column for custom metrics.
    # Unlike GroupBy.apply, this is stable across pandas versions and handles
    # an all-null feature without inventing metric rows from an empty DataFrame.
    for level_name, level_data in feature_data.groupby(feature.name, observed=True):
        level_eval = LevelEvaluation(name=str(level_name))
        for metric in metrics:
            level_eval.update(metric.name, metric.label, metric.data_call(level_data))
        if n_bootstraps is not None:
            level_eval.update_intervals(
                evaluate_confidence_interval(metrics, level_data, n_bootstraps)
            )
        feature_eval.levels[str(level_name)] = level_eval

    if declared_categories is not None:
        ordered_levels = {}
        for category in declared_categories:
            level = feature_eval.levels.get(category)
            if level is None:
                level = LevelEvaluation(name=category)
                for metric in metrics:
                    level.update(metric.name, metric.label, float("nan"))
            ordered_levels[category] = level
        feature_eval.levels = ordered_levels

    return feature_eval


def evaluate_error_feature(
    data: pd.DataFrame,
    group_col: str,
    feature: AuditorFeature,
    metric: AuditorErrorMetric,
    n_bootstraps: Optional[int],
    global_total_n: int,
) -> tuple[FeatureEvaluation, dict[str, dict[str, float]]]:
    """Compute odds ratios for one feature within one confusion-matrix group.

    Calculates the canonical 2x2 odds ratio for each feature level versus all
    other levels combined, then optionally runs bootstrap resampling to derive
    confidence intervals and replace the point estimate with the bootstrap mean.

    Categorical dtype is honoured: declared-but-unobserved categories appear
    as NaN placeholder rows (same behaviour as _evaluate_feature).

    Args:
        data: Full data slice including confusion indicator columns.
        group_col: Column name of the confusion indicator ('tp', 'tn', etc.).
        feature: The feature whose levels are being analysed.
        metric: Error metric to compute (e.g. OddsRatio).
        n_bootstraps: Bootstrap iterations, or None to skip.
        global_total_n: Total rows in the full data slice; used as the
            denominator for pct_overall in the returned support counts.

    Returns:
        Two-tuple (feature_eval, support) where support maps each level name to
        {"n": int, "pct_overall": float, "pct_group": float}.
    """
    feature_col = feature.name

    full_data, declared_categories = _prepare_feature_data(data, feature_col)
    full_total = len(full_data)
    full_counts = _level_counts(full_data, feature_col)
    group_data = full_data[full_data[group_col] == 1]
    group_total = len(group_data)
    group_counts = _level_counts(group_data, feature_col)
    all_levels = (
        declared_categories if declared_categories is not None else list(full_counts)
    )

    feature_eval = FeatureEvaluation(
        name=feature.name,
        label=feature.label if feature.label is not None else feature.name,
    )

    # Point estimates (raw ratio, or bootstrap mean if bootstraps requested).
    level_scores: dict[str, float] = {}
    for level_name in all_levels:
        full_count = full_counts.get(level_name, 0)
        group_count = group_counts.get(level_name, 0)
        level_scores[level_name] = metric.compute(
            group_count=group_count,
            group_total=group_total,
            full_count=full_count,
            full_total=full_total,
        )

    for level_name in all_levels:
        feature_eval.update(
            metric_name=metric.name,
            metric_label=metric.label,
            data={level_name: level_scores[level_name]},
        )

    # Bootstrap: replaces point estimates with bootstrap mean and adds CI.
    # Levels with an undefined baseline (NaN score) are excluded — they cannot
    # yield a meaningful CI and the NaN placeholder should be preserved.
    if n_bootstraps is not None and metric.ci_eligible:
        valid_levels = [
            level for level in all_levels if not np.isnan(level_scores[level])
        ]

        if valid_levels:
            bootstrap_results: dict[str, NDArray[np.float64]] = {
                level: np.empty(n_bootstraps, dtype=np.float64)
                for level in valid_levels
            }
            n = len(data)  # resample from the full slice (all confusion groups)

            for i in range(n_bootstraps):
                boot = data.sample(n, replace=True)
                boot_full, _ = _prepare_feature_data(boot, feature_col)
                boot_full_total = len(boot_full)
                boot_group = boot_full[boot_full[group_col] == 1]
                boot_group_total = len(boot_group)
                full_bootstrap_counts = _level_counts(boot_full, feature_col)
                group_bootstrap_counts = _level_counts(boot_group, feature_col)

                for level_name in valid_levels:
                    bootstrap_results[level_name][i] = metric.compute(
                        group_count=group_bootstrap_counts.get(level_name, 0),
                        group_total=boot_group_total,
                        full_count=full_bootstrap_counts.get(level_name, 0),
                        full_total=boot_full_total,
                    )

            for level_name in valid_levels:
                bs = bootstrap_results[level_name]
                if np.isnan(bs).all():
                    point_estimate = lower = upper = float("nan")
                else:
                    # Infinite sparse-table ORs are valid. Their percentile
                    # interpolation is repaired below; suppress only the known
                    # invalid arithmetic from that interpolation.
                    point_estimate = float(np.nanmean(bs))
                    with np.errstate(invalid="ignore"):
                        lower, upper = np.nanpercentile(bs, [2.5, 97.5])
                # np.nanpercentile returns NaN when interpolating between
                # infinities (inf - inf = NaN). Recover the bound from the
                # observed sign of infinite bootstrap samples.
                if np.isnan(lower):
                    if np.any(np.isneginf(bs)):
                        lower = float("-inf")
                    elif np.any(np.isposinf(bs)):
                        lower = float("inf")
                if np.isnan(upper):
                    if np.any(np.isposinf(bs)):
                        upper = float("inf")
                    elif np.any(np.isneginf(bs)):
                        upper = float("-inf")
                lm = feature_eval.levels[level_name].metrics[metric.name]
                lm.score = point_estimate
                lm.interval = (float(lower), float(upper))

    # Build sidecar support counts per level for the wide-format DataFrame export.
    # These are derived from the same full_data / group_data counts already computed above.
    # For categorical placeholders (levels in declared_categories but absent from full_counts),
    # all counts are zero.
    support: dict[str, dict[str, float]] = {}
    for level_name in all_levels:
        g_n = group_counts.get(level_name, 0)
        g_pct_overall = g_n / global_total_n if global_total_n > 0 else float("nan")
        g_pct_group = g_n / group_total if group_total > 0 else 0.0
        support[level_name] = {
            "n": g_n,
            "pct_overall": g_pct_overall,
            "pct_group": g_pct_group,
        }

    return feature_eval, support


def evaluate_confidence_interval(
    metrics: list[AuditorMetric], data: pd.DataFrame, n_bootstraps: int
) -> dict[str, tuple[float, float]]:
    """Calculate bootstrap confidence intervals for all CI-eligible metrics.

    Uses bootstrap resampling to estimate 95% confidence intervals for
    metrics that have ci_eligible=True.

    Args:
        data: DataFrame containing the data for a single feature level.
        n_bootstraps: Number of bootstrap samples to draw.

    Returns:
        Dictionary mapping metric names to (lower, upper) confidence bounds.
    """
    validate_n_bootstraps(n_bootstraps)
    eligible_metrics = [metric for metric in metrics if metric.ci_eligible]
    if not eligible_metrics:
        return {}

    bootstrap_results = {
        metric.name: np.empty(n_bootstraps, dtype=np.float64)
        for metric in eligible_metrics
    }
    for i in range(n_bootstraps):
        boot_data = data.sample(len(data), replace=True)
        for metric in eligible_metrics:
            bootstrap_results[metric.name][i] = metric.data_call(boot_data)

    intervals = {}
    for name, samples in bootstrap_results.items():
        if np.isnan(samples).all():
            intervals[name] = (float("nan"), float("nan"))
        else:
            lower, upper = np.nanpercentile(samples, [2.5, 97.5])
            intervals[name] = (float(lower), float(upper))
    return intervals
