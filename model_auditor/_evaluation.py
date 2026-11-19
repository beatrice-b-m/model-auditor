"""Binary subgroup estimates and inference, independent of Auditor state."""

from collections.abc import Callable
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from scipy.stats.contingency import odds_ratio

from model_auditor.error_metrics import AuditorErrorMetric, OddsRatio
from model_auditor.metrics import AuditorMetric
from model_auditor.schemas import (
    AuditorFeature,
    FeatureEvaluation,
    InferenceConfig,
    LevelEvaluation,
    LevelMetric,
)


def _prepare_feature_data(
    data: pd.DataFrame, feature: str, missing: str = "exclude"
) -> tuple[pd.DataFrame, list[str] | None]:
    """Preserve declared order and reject ambiguous display identities."""
    values = data[feature]
    categorical = isinstance(values.dtype, pd.CategoricalDtype)
    identities = list(
        values.cat.categories if categorical else values.dropna().unique()
    )
    labels = [str(value) for value in identities]
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"Feature {feature!r} has distinct values with the same string label; rename the levels before evaluation."
        )
    if values.isna().any() and missing == "error":
        raise ValueError(f"Feature {feature!r} contains missing values.")
    categories = labels if categorical else None
    if missing == "include" and values.isna().any():
        if "(Missing)" in labels:
            raise ValueError(
                f"Feature {feature!r} already contains the reserved missing label '(Missing)'."
            )
        result = data.copy()
        result[feature] = (
            values.astype(object).where(values.notna(), "(Missing)").astype(str)
        )
        if categories is not None:
            categories = [*categories, "(Missing)"]
        return result, categories
    result = data.dropna(subset=[feature]).copy()
    if not categorical:
        result[feature] = result[feature].astype(str)
    return result, categories


def _level_counts(data: pd.DataFrame, feature: str) -> dict[str, int]:
    return {
        str(key): int(count)
        for key, count in data.groupby(feature, observed=True).size().items()
    }


def validate_n_bootstraps(n_bootstraps: int | None) -> None:
    if n_bootstraps is not None and (
        isinstance(n_bootstraps, (bool, np.bool_))
        or not isinstance(n_bootstraps, (int, np.integer))
        or n_bootstraps <= 0
    ):
        raise ValueError("n_bootstraps must be a positive integer or None.")


def support_counts(data: pd.DataFrame) -> dict[str, int]:
    result = {
        "n": len(data),
        "n_pos": int((data["_truth"] == 1).sum()),
        "n_neg": int((data["_truth"] == 0).sum()),
    }
    if "_binary_pred" in data and data["_binary_pred"].notna().all():
        result.update(
            n_pred_pos=int((data["_binary_pred"] == 1).sum()),
            n_pred_neg=int((data["_binary_pred"] == 0).sum()),
        )
    return result


def metric_scale(metric: AuditorMetric) -> float:
    """Return a metric's optional constant reporting scale, validated.

    A missing ``scale`` attribute means 1.0. The scale must be a finite,
    positive real number (booleans are rejected).
    """
    scale = getattr(metric, "scale", 1.0)
    if (
        isinstance(scale, (bool, np.bool_))
        or not isinstance(scale, (int, float, np.integer, np.floating))
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            f"Metric {metric.name!r} scale must be a finite positive number."
        )
    return float(scale)


def metric_value(metric: AuditorMetric, data: pd.DataFrame) -> float:
    """Evaluate a metric on prepared rows and apply its reporting scale.

    The raw ``data_call`` result is returned unchanged (including NumPy integer
    counts) when the scale is 1.0, so unscaled outputs keep their types.
    """
    value = metric.data_call(data)
    scale = metric_scale(metric)
    return value if scale == 1.0 else value * scale


def metric_parameters(metric: AuditorMetric) -> dict:
    """Copy a metric's parameters, adding ``scale`` only when it is not 1.0."""
    parameters = deepcopy(getattr(metric, "parameters", {}))
    scale = metric_scale(metric)
    if scale != 1.0:
        parameters["scale"] = scale
    return parameters


def binomial_counts(
    metric: AuditorMetric, data: pd.DataFrame
) -> tuple[int, int] | None:
    columns = getattr(metric, "binomial_columns", None)
    if columns is not None:
        successes = int(data[columns[0]].sum())
        return successes, successes + int(data[columns[1]].sum())
    indicator = getattr(metric, "binomial_indicator", None)
    if indicator is not None:
        return int(data[indicator].sum()), len(data)
    return None


def wilson_interval(
    successes: int, total: int, confidence_level: float
) -> tuple[float, float]:
    """Wilson score interval for independent Bernoulli observations."""
    if total == 0:
        return float("nan"), float("nan")
    z = norm.ppf((1 + confidence_level) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return float(max(0, center - radius)), float(min(1, center + radius))


def make_sampler(
    data: pd.DataFrame, config: InferenceConfig
) -> Callable[[np.random.Generator], pd.DataFrame]:
    """Build a resampler for ``data`` that groups rows once, not on every draw.

    Each call of the returned function draws shared row indices, whole clusters,
    or truth-stratified observations from ``rng`` and returns a new DataFrame.
    Group order and generator calls are fixed, so a seeded generator gives the
    same draws as regrouping on every call.

    Raises:
        ValueError: Cluster resampling with a missing cluster column or
            missing cluster IDs.
    """
    if config.resampling == "cluster":
        if config.cluster not in data or data[config.cluster].isna().any():
            raise ValueError("Cluster IDs must exist and have no missing values.")
        clusters = list(
            data.groupby(config.cluster, sort=False, observed=True).indices.values()
        )

        def draw(rng: np.random.Generator) -> pd.DataFrame:
            indices = np.concatenate(
                [clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]
            )
            return data.iloc[indices].copy()

    elif config.resampling == "stratified":
        strata = list(data.groupby("_truth", sort=False).indices.values())

        def draw(rng: np.random.Generator) -> pd.DataFrame:
            indices = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in strata]
            )
            return data.iloc[indices].copy()

    else:
        n = len(data)

        def draw(rng: np.random.Generator) -> pd.DataFrame:
            return data.iloc[rng.integers(n, size=n)].copy()

    return draw


def bootstrap_summary(
    result: LevelMetric,
    samples: np.ndarray,
    config: InferenceConfig,
    *,
    allow_degenerate: bool = False,
) -> None:
    """Retain estimates and explicitly diagnose unusable bootstrap distributions."""
    result.interval_method = f"{config.resampling}_percentile"
    result.requested_resamples = len(samples)
    finite = samples[np.isfinite(samples)]
    result.valid_resamples = len(finite)
    result.nonfinite_resamples = len(samples) - len(finite)
    if len(finite) < config.min_resamples:
        result.interval_status = "insufficient_resamples"
    elif len(finite) / len(samples) < config.min_valid_fraction:
        result.interval_status = "too_many_invalid_resamples"
    elif np.ptp(finite) == 0 and not allow_degenerate:
        result.interval_status = "degenerate_distribution"
    else:
        alpha = (1 - config.confidence_level) / 2
        lower, upper = np.quantile(finite, [alpha, 1 - alpha])
        result.interval = float(lower), float(upper)
        result.interval_status = (
            "ok" if len(finite) == len(samples) else "conditional_on_valid_resamples"
        )


def evaluate_level(
    metrics: list[AuditorMetric],
    data: pd.DataFrame,
    name: str,
    n_bootstraps: int | None,
    config: InferenceConfig,
) -> LevelEvaluation:
    level = LevelEvaluation(name=name, support=support_counts(data))
    if config.cluster is not None:
        level.support["n_clusters"] = data[config.cluster].nunique()
    bootstrap_metrics = []
    for metric in metrics:
        scale = metric_scale(metric)
        score = metric_value(metric, data) if len(data) else float("nan")
        level.update(metric.name, metric.label, score)
        result = level.metrics[metric.name]
        result.direction = getattr(metric, "direction", None)
        result.parameters = metric_parameters(metric)
        counts = binomial_counts(metric, data)
        result.denominator = counts[1] if counts is not None else None
        if not np.isfinite(score):
            result.status = "undefined" if np.isnan(score) else "unbounded"
            result.interval_status = "undefined_estimate"
            continue
        if n_bootstraps is None or not metric.ci_eligible:
            continue
        if (
            counts is not None
            and config.method == "auto"
            and config.resampling == "iid"
        ):
            if config.rate_interval == "exact":
                ci = binomtest(*counts).proportion_ci(
                    confidence_level=config.confidence_level, method="exact"
                )
                lower, upper = float(ci.low), float(ci.high)
                result.interval_method = "clopper_pearson"
            else:
                lower, upper = wilson_interval(*counts, config.confidence_level)
                result.interval_method = "wilson"
            result.interval = (
                (lower, upper) if scale == 1.0 else (lower * scale, upper * scale)
            )
            result.interval_status = "ok"
        else:
            bootstrap_metrics.append(metric)
    if bootstrap_metrics:
        if config.resampling == "cluster" and data[config.cluster].nunique() < 2:
            for metric in bootstrap_metrics:
                level.metrics[metric.name].interval_status = "insufficient_clusters"
            return level
        rng = np.random.default_rng(config.random_state)
        sample = make_sampler(data, config)
        samples = {metric.name: [] for metric in bootstrap_metrics}
        for _ in range(n_bootstraps):
            boot = sample(rng)
            for metric in bootstrap_metrics:
                samples[metric.name].append(metric_value(metric, boot))
        for metric in bootstrap_metrics:
            bootstrap_summary(
                level.metrics[metric.name], np.asarray(samples[metric.name]), config
            )
    return level


def evaluate_feature(
    metrics: list[AuditorMetric],
    data: pd.DataFrame,
    feature: AuditorFeature,
    n_bootstraps: Optional[int],
    inference: InferenceConfig | None = None,
) -> FeatureEvaluation:
    config = inference or InferenceConfig()
    feature_data, categories = _prepare_feature_data(data, feature.name, config.missing)
    result = FeatureEvaluation(
        name=feature.name,
        label=feature.label or feature.name,
        excluded_n=len(data) - len(feature_data),
        total_n=len(data),
    )
    for name, group in feature_data.groupby(feature.name, observed=True):
        result.levels[str(name)] = evaluate_level(
            metrics, group, str(name), n_bootstraps, config
        )
    if categories is not None:
        result.levels = {
            name: result.levels[name]
            if name in result.levels
            else evaluate_level(metrics, feature_data.iloc[:0], name, None, config)
            for name in categories
        }
    return result


ERROR_GROUPS = ("tp", "tn", "fp", "fn")


def evaluate_error_features(
    data: pd.DataFrame,
    feature: AuditorFeature,
    metric: AuditorErrorMetric,
    n_bootstraps: Optional[int],
    global_total_n: int,
    inference: InferenceConfig | None = None,
    groups: tuple[str, ...] = ERROR_GROUPS,
) -> dict[str, tuple[FeatureEvaluation, dict[str, dict[str, float]]]]:
    """Confusion-membership enrichment versus rest, not conditional error rates.

    Evaluates every confusion group in ``groups`` for one feature. Point
    estimates always use the original table. IID auto inference uses a
    conditional exact interval for the population OR with the sample OR as the
    reported point estimate. Other designs use diagnosed percentile resampling,
    and every group is evaluated on the same ``n_bootstraps`` resamples.

    Returns:
        ``{group: (FeatureEvaluation, support)}`` in ``groups`` order.
    """
    config = inference or InferenceConfig()
    full, categories = _prepare_feature_data(data, feature.name, config.missing)
    full_counts = _level_counts(full, feature.name)
    levels = categories if categories is not None else list(full_counts)
    level_support = {
        name: support_counts(full.loc[full[feature.name].astype(str) == name])
        for name in levels
    }
    out = {}
    pending: dict[str, list[str]] = {}
    for group_col in groups:
        group = full[full[group_col] == 1]
        group_counts = _level_counts(group, feature.name)
        result = FeatureEvaluation(
            feature.name,
            feature.label or feature.name,
            excluded_n=len(data) - len(full),
            total_n=len(data),
        )
        support = {}
        for name in levels:
            count, group_count = full_counts.get(name, 0), group_counts.get(name, 0)
            estimate = metric.compute(group_count, len(group), count, len(full))
            result.update(metric.name, metric.label, {name: estimate})
            level = result.levels[name]
            level.support = deepcopy(level_support[name])
            lm = level.metrics[metric.name]
            lm.status = (
                "undefined"
                if np.isnan(estimate)
                else "unbounded"
                if np.isinf(estimate)
                else "ok"
            )
            lm.direction = "none"
            support[name] = {
                "n": group_count,
                "pct_overall": group_count / global_total_n
                if global_total_n
                else float("nan"),
                "pct_group": group_count / len(group) if len(group) else float("nan"),
            }
            if np.isnan(estimate):
                lm.interval_status = "undefined_estimate"
            elif (
                n_bootstraps is not None
                and metric.ci_eligible
                and config.method == "auto"
                and config.resampling == "iid"
                and type(metric) is OddsRatio
            ):
                a, b = group_count, count - group_count
                c = len(group) - group_count
                d = len(full) - count - c
                ci = odds_ratio(
                    [[a, b], [c, d]], kind="conditional"
                ).confidence_interval(confidence_level=config.confidence_level)
                lm.interval = float(ci.low), float(ci.high)
                lm.interval_method = "conditional_exact"
                lm.interval_status = "ok"
        group_pending = [
            name
            for name in levels
            if n_bootstraps is not None
            and metric.ci_eligible
            and result.levels[name].metrics[metric.name].interval_status
            == "not_requested"
        ]
        if group_pending:
            pending[group_col] = group_pending
        out[group_col] = (result, support)
    if not pending:
        return out
    if config.resampling == "cluster" and full[config.cluster].nunique() < 2:
        for group_col, names in pending.items():
            for name in names:
                out[group_col][0].levels[name].metrics[
                    metric.name
                ].interval_status = "insufficient_clusters"
        return out
    samples = {
        group_col: {name: [] for name in names} for group_col, names in pending.items()
    }
    rng = np.random.default_rng(config.random_state)
    sample = make_sampler(full, config)
    for _ in range(n_bootstraps):
        boot = sample(rng)
        counts = _level_counts(boot, feature.name)
        for group_col, names in pending.items():
            boot_group = boot[boot[group_col] == 1]
            group_counts = _level_counts(boot_group, feature.name)
            for name in names:
                samples[group_col][name].append(
                    metric.compute(
                        group_counts.get(name, 0),
                        len(boot_group),
                        counts.get(name, 0),
                        len(boot),
                    )
                )
    for group_col, names in pending.items():
        for name in names:
            bootstrap_summary(
                out[group_col][0].levels[name].metrics[metric.name],
                np.asarray(samples[group_col][name]),
                config,
            )
    return out


def evaluate_confidence_interval(
    metrics: list[AuditorMetric], data: pd.DataFrame, n_bootstraps: int
) -> dict[str, tuple[float, float]]:
    """Compatibility helper using the default metric-specific inference policy."""
    validate_n_bootstraps(n_bootstraps)
    level = evaluate_level(metrics, data, "", n_bootstraps, InferenceConfig())
    return {
        name: metric.interval
        if metric.interval is not None
        else (float("nan"), float("nan"))
        for name, metric in level.metrics.items()
        if next(m for m in metrics if m.name == name).ci_eligible
    }
