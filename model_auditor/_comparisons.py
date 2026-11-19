"""Shared-resample contrasts for fixed binary model and subgroup comparisons."""

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from model_auditor._evaluation import bootstrap_summary, make_sampler
from model_auditor.metrics import AuditorMetric
from model_auditor.schemas import HypothesisTestResult, InferenceConfig, LevelMetric

TESTS = ("noninferiority", "superiority", "equivalence")
_USABLE = {"ok", "conditional_on_valid_resamples"}


@dataclass(frozen=True)
class HypothesisSpec:
    """Validated per-metric hypothesis test on the reported contrast scale."""

    test: str
    alpha: float
    direction: str
    boundary: Optional[float] = None
    boundaries: Optional[tuple[float, float]] = None


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


def _real(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and bool(np.isfinite(value))
    )


def resolve_tests(
    metrics: Sequence[AuditorMetric],
    contrast: str,
    n_bootstraps: Optional[int],
    *,
    test: Optional[str],
    test_metrics: Optional[Sequence[str]],
    margin,
    boundary,
    alpha: Optional[float],
    direction: Optional[Mapping[str, str]],
) -> dict[str, HypothesisSpec]:
    """Validate hypothesis-test arguments and derive each tested metric's null."""
    if test is None:
        supplied = {
            "test_metrics": test_metrics,
            "margin": margin,
            "boundary": boundary,
            "alpha": alpha,
            "direction": direction,
        }
        given = [key for key, value in supplied.items() if value is not None]
        if given:
            raise ValueError(f"{', '.join(given)} require test to be set.")
        return {}
    if test not in TESTS:
        raise ValueError(f"test must be None or one of {', '.join(TESTS)}.")
    if n_bootstraps is None:
        raise ValueError("Hypothesis tests need resamples; set n_bootstraps.")
    if alpha is None or not _real(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha is required for a test and must be in (0, 0.5).")
    by_name = {metric.name: metric for metric in metrics}
    if test_metrics is None:
        names = list(by_name)
    else:
        if isinstance(test_metrics, str):
            raise ValueError("test_metrics must be a list of metric names.")
        names = list(test_metrics)
        if not names or len(set(names)) != len(names):
            raise ValueError("test_metrics must name each metric at most once.")
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise ValueError(f"test_metrics names unknown metrics: {unknown}.")

    def per_metric(value, name: str, argument: str):
        if isinstance(value, Mapping):
            unknown = set(value) - set(names)
            if unknown:
                raise ValueError(
                    f"{argument} names metrics that are not tested: {sorted(unknown)}."
                )
            return value.get(name)
        return value

    if direction is not None:
        if not isinstance(direction, Mapping):
            raise ValueError("direction must map metric names to higher or lower.")
        per_metric(direction, "", "direction")
        bad = {k: v for k, v in direction.items() if v not in {"higher", "lower"}}
        if bad:
            raise ValueError(f"direction overrides must be higher or lower: {bad}.")
    if test == "superiority" and (margin is not None or boundary is not None):
        raise ValueError(
            "Superiority tests against no difference; margin and boundary do not apply."
        )
    null = 0.0 if contrast == "difference" else 1.0
    specs = {}
    for name in names:
        metric = by_name[name]
        if not metric.ci_eligible:
            raise ValueError(
                f"Metric {name!r} is not interval-eligible and cannot be tested."
            )
        side = (direction or {}).get(name) or getattr(metric, "direction", None)
        if test != "equivalence" and side not in {"higher", "lower"}:
            raise ValueError(
                f"Metric {name!r} has direction {side!r}; a {test} test needs a "
                "higher or lower direction. Pass direction={...} or exclude it "
                "with test_metrics."
            )
        side = side if side in {"higher", "lower"} else "none"
        metric_margin = per_metric(margin, name, "margin")
        metric_boundary = per_metric(boundary, name, "boundary")
        if test == "superiority":
            specs[name] = HypothesisSpec(test, float(alpha), side, boundary=null)
            continue
        if (metric_margin is None) == (metric_boundary is None):
            raise ValueError(
                f"Metric {name!r}: supply exactly one of margin or boundary for a "
                f"{test} test."
            )
        if metric_margin is not None and (
            not _real(metric_margin) or metric_margin <= 0
        ):
            raise ValueError(f"Metric {name!r}: margin must be a positive number.")
        if test == "noninferiority":
            if metric_boundary is None:
                value = (
                    null - metric_margin if side == "higher" else null + metric_margin
                )
            elif not _real(metric_boundary):
                raise ValueError(f"Metric {name!r}: boundary must be a finite number.")
            else:
                value = float(metric_boundary)
            if (side == "higher" and not value < null) or (
                side == "lower" and not value > null
            ):
                raise ValueError(
                    f"Metric {name!r}: the non-inferiority boundary must lie on the "
                    f"unfavourable side of {null:g}."
                )
            if contrast == "ratio" and value <= 0:
                raise ValueError(
                    f"Metric {name!r}: a ratio boundary must be positive; use a "
                    "margin below 1."
                )
            specs[name] = HypothesisSpec(
                test, float(alpha), side, boundary=float(value)
            )
        else:
            if metric_boundary is None:
                low, high = null - metric_margin, null + metric_margin
            elif (
                isinstance(metric_boundary, (str, bytes))
                or not isinstance(metric_boundary, Sequence)
                or len(metric_boundary) != 2
                or not all(_real(v) for v in metric_boundary)
            ):
                raise ValueError(
                    f"Metric {name!r}: an equivalence boundary is a (lower, upper) "
                    "pair of finite numbers."
                )
            else:
                low, high = (float(v) for v in metric_boundary)
            if not low < null < high:
                raise ValueError(
                    f"Metric {name!r}: equivalence boundaries must satisfy "
                    f"lower < {null:g} < upper."
                )
            if contrast == "ratio" and low <= 0:
                raise ValueError(
                    f"Metric {name!r}: a ratio boundary must be positive; use a "
                    "margin below 1."
                )
            specs[name] = HypothesisSpec(
                test, float(alpha), side, boundaries=(float(low), float(high))
            )
    return specs


def hypothesis_metadata(
    specs: Mapping[str, HypothesisSpec], margin, boundary
) -> Optional[dict]:
    """Provenance block for a hypothesis test, or None when no test ran."""
    if not specs:
        return None
    first = next(iter(specs.values()))
    return {
        "test": first.test,
        "alpha": first.alpha,
        "margin": dict(margin) if isinstance(margin, Mapping) else margin,
        "boundary_override": dict(boundary)
        if isinstance(boundary, Mapping)
        else boundary,
        "metrics": {
            name: {
                "direction": spec.direction,
                "null_boundary": spec.boundary
                if spec.boundaries is None
                else list(spec.boundaries),
            }
            for name, spec in specs.items()
        },
        "decision_rule": "one_sided_percentile_bound",
        "p_value": "approximate_resampling_(k+1)/(B+1)",
        "scope": "pointwise_no_multiplicity_adjustment",
    }


def _one_sided(
    finite: np.ndarray, alpha: float, boundary: float, favourable: str
) -> tuple[float, float, bool]:
    """Bound, p-value and rejection for H1: contrast above/below boundary."""
    if favourable == "higher":
        bound = float(np.quantile(finite, alpha))
        k = int(np.sum(finite <= boundary))
        reject = bound > boundary
    else:
        bound = float(np.quantile(finite, 1 - alpha))
        k = int(np.sum(finite >= boundary))
        reject = bound < boundary
    return bound, (k + 1) / (len(finite) + 1), reject


def run_hypothesis(
    result: LevelMetric, spec: HypothesisSpec, samples: Optional[np.ndarray]
) -> HypothesisTestResult:
    """Test a contrast on the same finite draws used for its interval."""
    outcome = HypothesisTestResult(
        test=spec.test,
        alpha=spec.alpha,
        direction=spec.direction,
        null_boundary=spec.boundary,
        equivalence_boundaries=spec.boundaries,
    )
    if samples is None or result.interval_status not in _USABLE:
        outcome.decision = "undefined"
        outcome.decision_status = result.interval_status
        return outcome
    finite = samples[np.isfinite(samples)]
    if spec.boundaries is None:
        assert spec.boundary is not None
        bound, p_value, reject = _one_sided(
            finite, spec.alpha, spec.boundary, spec.direction
        )
        outcome.one_sided_bound = bound
    else:
        low, high = spec.boundaries
        lower, p_low, reject_low = _one_sided(finite, spec.alpha, low, "higher")
        upper, p_high, reject_high = _one_sided(finite, spec.alpha, high, "lower")
        outcome.equivalence_bounds = (lower, upper)
        p_value, reject = max(p_low, p_high), reject_low and reject_high
    outcome.p_value = float(p_value)
    outcome.decision = "reject_null" if reject else "fail_to_reject"
    outcome.decision_status = result.interval_status
    return outcome


@dataclass(frozen=True)
class ContrastSpec:
    """One contrast evaluated on a shared prepared frame.

    ``statistic`` receives the output of the call's ``prepare`` step, for the
    original rows and for every resample. ``resample`` is False for metrics
    that are not interval-eligible or when intervals are disabled.
    """

    name: str
    label: str
    statistic: Callable[[Any], float]
    resample: bool
    test: Optional[HypothesisSpec] = None


def evaluate_contrasts(
    data: pd.DataFrame,
    prepare: Callable[[pd.DataFrame], Any],
    specs: Mapping[Hashable, ContrastSpec],
    n_bootstraps: int | None,
    config: InferenceConfig,
    *,
    identical: bool = False,
) -> dict[Hashable, LevelMetric]:
    """Pointwise contrast intervals from one shared set of resamples.

    Shared rows retain overlap and model pairing. Each of the ``n_bootstraps``
    resamples is drawn and prepared once, then every pending contrast is
    evaluated on it, so all contrasts see the same draws. With a test spec, the
    one-sided bound and p-value reuse the interval's finite resampled
    contrasts, so they agree with the interval at matched levels.

    Statistics must not mutate the prepared frame, because it is shared.
    """
    prepared = prepare(data)
    results: dict[Hashable, LevelMetric] = {}
    pending = []
    for key, spec in specs.items():
        estimate = spec.statistic(prepared)
        result = LevelMetric(spec.name, spec.label, estimate)
        results[key] = result
        if not np.isfinite(estimate):
            result.status = "undefined"
            result.interval_status = "undefined_estimate"
        elif n_bootstraps is not None and spec.resample:
            pending.append(key)
    samples: dict[Hashable, np.ndarray] = {}
    if (
        pending
        and config.resampling == "cluster"
        and data[config.cluster].nunique() < 2
    ):
        for key in pending:
            results[key].interval_status = "insufficient_clusters"
    elif pending:
        assert n_bootstraps is not None
        rng = np.random.default_rng(config.random_state)
        sample = make_sampler(data, config)
        draws: dict[Hashable, list] = {key: [] for key in pending}
        for _ in range(n_bootstraps):
            boot = prepare(sample(rng))
            for key in pending:
                draws[key].append(specs[key].statistic(boot))
        for key in pending:
            samples[key] = np.asarray(draws[key])
            bootstrap_summary(
                results[key], samples[key], config, allow_degenerate=identical
            )
    for key, spec in specs.items():
        if spec.test is not None:
            results[key].hypothesis = run_hypothesis(
                results[key], spec.test, samples.get(key)
            )
    return results
