"""Constant reporting scale for metrics (issue #5, part 2)."""

import numpy as np
import pandas as pd
import pytest

from model_auditor import Auditor, InferenceConfig
from model_auditor.metrics import AUROC, Sensitivity


class ScaledSensitivity(Sensitivity):
    name = "sensitivity_per_1000"
    label = "Sensitivity per 1,000"
    scale = 1000


class ScaledAUROC(AUROC):
    name = "auroc_per_1000"
    label = "AUROC x1000"
    scale = 1000


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "y": y,
            "s": np.clip(0.3 * y + rng.uniform(0, 0.7, n), 0, 1),
            "r": np.clip(0.2 * y + rng.uniform(0, 0.8, n), 0, 1),
            "g": rng.choice(["a", "b", "c"], n),
            "cluster": rng.integers(0, 40, n),
        }
    )


def make_auditor(df):
    a = Auditor(
        data=df,
        metrics=[Sensitivity(), ScaledSensitivity(), AUROC(), ScaledAUROC()],
    )
    a.add_outcome("y")
    a.add_score("s", threshold=0.5)
    a.add_score("r", threshold=0.5)
    a.add_feature("g")
    return a


def pairs(result):
    for feature in result.features.values():
        for level in feature.levels.values():
            yield level.metrics


def assert_scaled(raw, scaled, factor=1000):
    assert scaled.score == pytest.approx(raw.score * factor, rel=1e-12)
    if raw.interval is None:
        assert scaled.interval is None
    else:
        assert scaled.interval == pytest.approx(
            (raw.interval[0] * factor, raw.interval[1] * factor), rel=1e-12
        )
    assert scaled.denominator == raw.denominator
    assert scaled.interval_status == raw.interval_status
    assert scaled.interval_method == raw.interval_method
    assert scaled.parameters == {**raw.parameters, "scale": 1000.0}


@pytest.mark.parametrize(
    "inference",
    [
        InferenceConfig(random_state=3),
        InferenceConfig(random_state=3, rate_interval="exact"),
        InferenceConfig(random_state=3, resampling="stratified"),
        InferenceConfig(random_state=3, resampling="cluster", cluster="cluster"),
    ],
)
def test_scaled_estimates_and_bounds_are_exact_multiples(inference):
    result = make_auditor(make_data()).evaluate_metrics(
        "s", n_bootstraps=200, inference=inference
    )
    for metrics in pairs(result):
        assert_scaled(metrics["sensitivity"], metrics["sensitivity_per_1000"])
        assert_scaled(metrics["auroc"], metrics["auroc_per_1000"])
    level = result.features["overall"].levels["Overall"]
    assert level.metrics["sensitivity"].interval is not None


def test_scale_is_recorded_only_when_not_one():
    result = make_auditor(make_data()).evaluate_metrics("s", n_bootstraps=None)
    entries = {m["name"]: m["parameters"] for m in result.metadata["metrics"]}
    assert entries["sensitivity"] == {}
    assert entries["sensitivity_per_1000"] == {"scale": 1000.0}
    assert "scale" not in entries["auroc"]
    assert entries["auroc_per_1000"]["scale"] == 1000.0


def test_support_is_unchanged_and_numeric_export_is_scaled():
    result = make_auditor(make_data()).evaluate_metrics("s", n_bootstraps=None)
    frame = result.to_numeric_dataframe()
    raw = frame[frame["metric"] == "sensitivity"].reset_index(drop=True)
    scaled = frame[frame["metric"] == "sensitivity_per_1000"].reset_index(drop=True)
    assert np.allclose(scaled["estimate"], raw["estimate"] * 1000)
    assert (scaled["n"] == raw["n"]).all()


@pytest.mark.parametrize("method", ["compare_scores", "compare_groups"])
def test_difference_contrast_is_scaled_and_ratio_is_not(method):
    a = make_auditor(make_data())
    config = InferenceConfig(random_state=5)

    def run(contrast):
        if method == "compare_scores":
            return a.compare_scores(
                "s", "r", contrast=contrast, n_bootstraps=200, inference=config
            )
        return a.compare_groups(
            "s", "g", "a", contrast=contrast, n_bootstraps=200, inference=config
        )

    for metrics in pairs(run("difference")):
        assert_scaled(metrics["sensitivity"], metrics["sensitivity_per_1000"])
        assert_scaled(metrics["auroc"], metrics["auroc_per_1000"])
    for metrics in pairs(run("ratio")):
        raw, scaled = metrics["sensitivity"], metrics["sensitivity_per_1000"]
        assert scaled.score == pytest.approx(raw.score, rel=1e-12)
        assert scaled.interval == pytest.approx(raw.interval, rel=1e-12)
        assert "scale" not in scaled.parameters


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "10"])
def test_invalid_scale_is_rejected(scale):
    class Bad(Sensitivity):
        name = "bad"

    Bad.scale = scale
    with pytest.raises(ValueError, match="scale must be a finite positive number"):
        Auditor(metrics=[Bad()])


def test_counts_keep_their_type_without_scale():
    from model_auditor.metrics import nTP

    a = Auditor(data=make_data(), metrics=[nTP()])
    a.add_outcome("y")
    a.add_score("s", threshold=0.5)
    score = (
        a.evaluate_metrics("s", n_bootstraps=None)
        .features["overall"]
        .levels["Overall"]
        .metrics[nTP.name]
        .score
    )
    assert isinstance(score, (int, np.integer))
