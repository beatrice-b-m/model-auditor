"""Non-inferiority, superiority and equivalence tests on contrasts (issue #4)."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from model_auditor import Auditor, InferenceConfig
from model_auditor.metrics import FPR, Prevalence, Sensitivity, nTP

CONFIG = InferenceConfig(random_state=11)


def paired_data(n=800, miss_rate=0.0, fp_rate=0.0, seed=0):
    """Binary reference policy and a new policy that misses some positives."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    reference = np.where(y == 1, rng.uniform(size=n) < 0.8, rng.uniform(size=n) < 0.2)
    new = reference.copy()
    new[(y == 1) & reference & (rng.uniform(size=n) < miss_rate)] = False
    new[(y == 0) & ~reference & (rng.uniform(size=n) < fp_rate)] = True
    return pd.DataFrame(
        {
            "y": y,
            "new": new.astype(float),
            "ref": reference.astype(float),
            "g": rng.choice(["a", "b"], n),
            "cluster": rng.integers(0, 80, n),
        }
    )


def make_auditor(df, metrics=None):
    a = Auditor(data=df, metrics=metrics or [Sensitivity(), FPR()])
    a.add_outcome("y")
    a.add_score("new", threshold=0.5)
    a.add_score("ref", threshold=0.5)
    a.add_feature("g")
    return a


def overall(result, metric):
    return result.features["overall"].levels["Overall"].metrics[metric]


def compare(a, **kwargs):
    kwargs.setdefault("n_bootstraps", 400)
    kwargs.setdefault("inference", CONFIG)
    return a.compare_scores("new", "ref", **kwargs)


def test_noninferiority_rejects_when_clearly_above_margin():
    a = make_auditor(paired_data(miss_rate=0.01))
    result = compare(
        a, test="noninferiority", test_metrics=["sensitivity"], margin=0.05, alpha=0.025
    )
    hyp = overall(result, "sensitivity").hypothesis
    assert hyp.null_boundary == -0.05
    assert hyp.direction == "higher"
    assert hyp.one_sided_bound > -0.05
    assert hyp.decision == "reject_null"
    assert hyp.decision_status == "ok"
    assert hyp.p_value < 0.025


def test_noninferiority_fails_when_clearly_below_margin():
    a = make_auditor(paired_data(miss_rate=0.3))
    result = compare(
        a, test="noninferiority", test_metrics=["sensitivity"], margin=0.05, alpha=0.025
    )
    hyp = overall(result, "sensitivity").hypothesis
    assert hyp.one_sided_bound < -0.05
    assert hyp.decision == "fail_to_reject"
    assert hyp.p_value > 0.5


def test_lower_is_better_metric_uses_upper_bound_and_positive_margin():
    good = make_auditor(paired_data(fp_rate=0.01))
    bad = make_auditor(paired_data(fp_rate=0.4))
    kwargs = dict(test="noninferiority", test_metrics=["fpr"], margin=0.05, alpha=0.025)
    good_hyp = overall(compare(good, **kwargs), "fpr").hypothesis
    bad_hyp = overall(compare(bad, **kwargs), "fpr").hypothesis
    for hyp in (good_hyp, bad_hyp):
        assert hyp.direction == "lower"
        assert hyp.null_boundary == 0.05
    assert good_hyp.one_sided_bound < 0.05
    assert good_hyp.decision == "reject_null"
    assert bad_hyp.one_sided_bound > 0.05
    assert bad_hyp.decision == "fail_to_reject"


@pytest.mark.parametrize("method", ["compare_scores", "compare_groups"])
@pytest.mark.parametrize(
    "inference",
    [
        InferenceConfig(random_state=2, confidence_level=0.9),
        InferenceConfig(random_state=2, resampling="stratified"),
        InferenceConfig(random_state=2, resampling="cluster", cluster="cluster"),
    ],
)
def test_bound_matches_interval_end_at_matched_alpha(method, inference):
    a = make_auditor(paired_data(miss_rate=0.05, fp_rate=0.05))
    alpha = (1 - inference.confidence_level) / 2
    kwargs = dict(
        n_bootstraps=300,
        inference=inference,
        test="noninferiority",
        margin=0.05,
        alpha=alpha,
    )
    if method == "compare_scores":
        result = a.compare_scores("new", "ref", **kwargs)
    else:
        result = a.compare_groups("new", "g", "a", **kwargs)
    checked = 0
    for feature in result.features.values():
        for level in feature.levels.values():
            for name, end in [("sensitivity", 0), ("fpr", 1)]:
                metric = level.metrics[name]
                assert metric.hypothesis.decision_status == metric.interval_status
                assert metric.hypothesis.one_sided_bound == metric.interval[end]
                checked += 1
    assert checked >= 2


def test_equivalence_bounds_match_interval_and_require_both_sides():
    a = make_auditor(paired_data(miss_rate=0.01))
    result = compare(
        a,
        test="equivalence",
        test_metrics=["sensitivity"],
        margin=0.05,
        alpha=0.025,
    )
    metric = overall(result, "sensitivity")
    hyp = metric.hypothesis
    assert hyp.null_boundary is None and hyp.one_sided_bound is None
    assert hyp.equivalence_boundaries == (-0.05, 0.05)
    assert hyp.equivalence_bounds == metric.interval
    assert hyp.decision == "reject_null"
    shifted = compare(
        a,
        test="equivalence",
        test_metrics=["sensitivity"],
        boundary=(-0.001, 0.05),
        alpha=0.025,
    )
    assert overall(shifted, "sensitivity").hypothesis.decision == "fail_to_reject"


def test_equivalence_allows_metrics_without_direction():
    a = make_auditor(paired_data(fp_rate=0.01), [Prevalence(), Sensitivity()])
    result = compare(
        a, test="equivalence", test_metrics=["prevalence"], margin=0.05, alpha=0.05
    )
    hyp = overall(result, "prevalence").hypothesis
    assert hyp.direction == "none"
    assert hyp.decision in {"reject_null", "fail_to_reject", "undefined"}


def test_identical_columns_under_superiority_are_defined_and_not_rejected():
    df = paired_data()
    df["ref"] = df["new"]
    result = compare(
        make_auditor(df), test="superiority", test_metrics=["sensitivity"], alpha=0.025
    )
    hyp = overall(result, "sensitivity").hypothesis
    assert hyp.one_sided_bound == 0
    assert hyp.null_boundary == 0
    assert hyp.decision == "fail_to_reject"
    assert hyp.p_value == 1


def test_superiority_after_noninferiority_on_the_same_contrast():
    df = paired_data(fp_rate=0.0)
    df["new"] = np.where(df["y"] == 1, 1.0, df["ref"])  # catches every positive
    a = make_auditor(df)
    shared = dict(test_metrics=["sensitivity"], alpha=0.025)
    ni = overall(
        compare(a, test="noninferiority", margin=0.02, **shared), "sensitivity"
    )
    sup = overall(compare(a, test="superiority", **shared), "sensitivity")
    assert ni.hypothesis.decision == "reject_null"
    assert sup.hypothesis.decision == "reject_null"
    assert sup.hypothesis.null_boundary == 0
    assert sup.interval == ni.interval


def test_ratio_contrast_boundaries():
    a = make_auditor(paired_data(miss_rate=0.01))
    ni = compare(
        a,
        contrast="ratio",
        test="noninferiority",
        test_metrics=["sensitivity"],
        margin=0.1,
        alpha=0.025,
    )
    hyp = overall(ni, "sensitivity").hypothesis
    assert hyp.null_boundary == pytest.approx(0.9)
    assert hyp.decision == "reject_null"
    override = compare(
        a,
        contrast="ratio",
        test="noninferiority",
        test_metrics=["sensitivity"],
        boundary=0.95,
        alpha=0.025,
    )
    assert overall(override, "sensitivity").hypothesis.null_boundary == 0.95
    sup = compare(
        a,
        contrast="ratio",
        test="superiority",
        test_metrics=["fpr"],
        alpha=0.025,
    )
    assert overall(sup, "fpr").hypothesis.null_boundary == 1.0


def test_too_few_clusters_gives_undefined_decision():
    df = paired_data()
    df.loc[df["g"] == "b", "cluster"] = 999
    result = compare(
        make_auditor(df),
        test="noninferiority",
        test_metrics=["sensitivity"],
        margin=0.05,
        alpha=0.025,
        inference=InferenceConfig(
            random_state=0, resampling="cluster", cluster="cluster"
        ),
    )
    metric = result.features["g"].levels["b"].metrics["sensitivity"]
    assert metric.interval_status == "insufficient_clusters"
    assert metric.hypothesis.decision == "undefined"
    assert metric.hypothesis.decision_status == "insufficient_clusters"
    assert metric.hypothesis.p_value is None
    assert "undefined (insufficient_clusters)" in result.to_dataframe().to_string()


def test_undefined_estimate_and_unobserved_category_are_undefined():
    df = paired_data()
    df["g"] = pd.Categorical(df["g"], categories=["a", "b", "c"])
    a = make_auditor(df)
    kwargs = dict(
        test="noninferiority", test_metrics=["sensitivity"], margin=0.05, alpha=0.025
    )
    scores = compare(a, **kwargs)
    groups = a.compare_groups(
        "new", "g", "a", n_bootstraps=200, inference=CONFIG, **kwargs
    )
    for result in (scores, groups):
        hyp = result.features["g"].levels["c"].metrics["sensitivity"].hypothesis
        assert hyp.decision == "undefined"
        assert hyp.decision_status == "undefined_estimate"


def test_compare_groups_tests_each_level_against_reference():
    df = paired_data()
    rng = np.random.default_rng(4)
    worse = (df["g"] == "b") & (df["y"] == 1) & (rng.uniform(size=len(df)) < 0.5)
    df.loc[worse, "new"] = 0.0
    a = make_auditor(df)
    result = a.compare_groups(
        "new",
        "g",
        "a",
        n_bootstraps=300,
        inference=CONFIG,
        test="noninferiority",
        test_metrics=["sensitivity"],
        margin=0.05,
        alpha=0.025,
    )
    hyp = result.features["g"].levels["b"].metrics["sensitivity"].hypothesis
    assert hyp.decision == "fail_to_reject"
    assert result.metadata["hypothesis_test"]["metrics"]["sensitivity"] == {
        "direction": "higher",
        "null_boundary": -0.05,
    }


def test_margin_is_on_the_scaled_contrast():
    class PerThousand(Sensitivity):
        name = "sensitivity_per_1000"
        scale = 1000

    a = make_auditor(paired_data(miss_rate=0.05), [Sensitivity(), PerThousand()])
    result = compare(
        a,
        test="noninferiority",
        margin={"sensitivity": 0.05, "sensitivity_per_1000": 50.0},
        alpha=0.025,
    )
    raw = overall(result, "sensitivity").hypothesis
    scaled = overall(result, "sensitivity_per_1000").hypothesis
    assert scaled.null_boundary == -50.0
    assert scaled.one_sided_bound == pytest.approx(raw.one_sided_bound * 1000)
    assert scaled.p_value == raw.p_value
    assert scaled.decision == raw.decision


def test_direction_override_and_missing_direction_error():
    a = make_auditor(paired_data(), [Prevalence(), Sensitivity()])
    with pytest.raises(ValueError, match="'prevalence' has direction 'none'"):
        compare(a, test="superiority", alpha=0.025)
    result = compare(
        a, test="superiority", alpha=0.025, direction={"prevalence": "higher"}
    )
    assert overall(result, "prevalence").hypothesis.direction == "higher"
    assert overall(result, "sensitivity").hypothesis.direction == "higher"


def test_exports_and_metadata_only_change_when_a_test_runs():
    a = make_auditor(paired_data(miss_rate=0.02))
    plain = compare(a)
    tested = compare(
        a, test="noninferiority", test_metrics=["sensitivity"], margin=0.05, alpha=0.025
    )
    assert "hypothesis_test" not in plain.metadata
    assert "test" not in plain.to_numeric_dataframe().columns
    assert list(plain.to_dataframe().columns) == ["sensitivity", "fpr"]
    assert all(
        asdict(m)["hypothesis"] is None
        for f in plain.features.values()
        for level in f.levels.values()
        for m in level.metrics.values()
    )
    # Tests reuse the interval resamples, so estimates and intervals are unchanged.
    for feature in plain.features:
        for level in plain.features[feature].levels:
            for name in ["sensitivity", "fpr"]:
                left = plain.features[feature].levels[level].metrics[name]
                right = tested.features[feature].levels[level].metrics[name]
                assert (left.score, left.interval) == (right.score, right.interval)
    display = tested.to_dataframe()
    assert list(display.columns) == ["sensitivity", "sensitivity test", "fpr"]
    assert display.iloc[0]["sensitivity test"].startswith("reject_null (p=")
    numeric = tested.to_numeric_dataframe()
    for column in [
        "test",
        "test_alpha",
        "null_boundary",
        "one_sided_bound",
        "p_value",
        "decision",
        "decision_status",
    ]:
        assert column in numeric.columns
    assert "equivalence_lower_bound" not in numeric.columns
    row = numeric[numeric["metric"] == "sensitivity"].iloc[0]
    assert row["decision"] == "reject_null" and row["null_boundary"] == -0.05
    assert numeric[numeric["metric"] == "fpr"]["test"].isna().all()
    meta = tested.metadata["hypothesis_test"]
    assert meta["test"] == "noninferiority" and meta["alpha"] == 0.025
    assert meta["margin"] == 0.05 and meta["boundary_override"] is None
    assert meta["scope"] == "pointwise_no_multiplicity_adjustment"
    equivalence = compare(
        a, test="equivalence", test_metrics=["sensitivity"], margin=0.05, alpha=0.025
    ).to_numeric_dataframe()
    assert "equivalence_upper_bound" in equivalence.columns
    assert "one_sided_bound" not in equivalence.columns


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(test="bogus", alpha=0.025), "test must be None or one of"),
        (dict(test="superiority"), "alpha is required"),
        (dict(test="superiority", alpha=0.5), "alpha is required"),
        (dict(test="superiority", alpha=0.025, n_bootstraps=None), "set n_bootstraps"),
        (dict(test="superiority", alpha=0.025, margin=0.1), "do not apply"),
        (dict(test="noninferiority", alpha=0.025), "exactly one of margin or boundary"),
        (
            dict(test="noninferiority", alpha=0.025, margin=0.1, boundary=-0.1),
            "exactly one of margin or boundary",
        ),
        (dict(test="noninferiority", alpha=0.025, margin=-0.1), "positive number"),
        (
            dict(
                test="noninferiority", alpha=0.025, test_metrics=["fpr"], boundary=-0.1
            ),
            "unfavourable side",
        ),
        (
            dict(test="equivalence", alpha=0.025, boundary=(0.01, 0.1)),
            "lower < 0 < upper",
        ),
        (
            dict(test="superiority", alpha=0.025, test_metrics=["missing"]),
            "unknown metrics",
        ),
        (dict(test="superiority", alpha=0.025, test_metrics="fpr"), "list of metric"),
        (
            dict(
                test="noninferiority",
                alpha=0.025,
                test_metrics=["fpr"],
                margin={"sensitivity": 0.1},
            ),
            "not tested",
        ),
        (
            dict(test="superiority", alpha=0.025, direction={"fpr": "none"}),
            "higher or lower",
        ),
        (dict(margin=0.1), "require test to be set"),
        (
            dict(
                contrast="ratio",
                test="noninferiority",
                alpha=0.025,
                test_metrics=["sensitivity"],
                margin=1.5,
            ),
            "ratio boundary must be positive",
        ),
    ],
)
def test_invalid_test_arguments(kwargs, message):
    a = make_auditor(paired_data())
    with pytest.raises(ValueError, match=message):
        compare(a, **kwargs)


def test_non_interval_metric_cannot_be_tested():
    a = make_auditor(paired_data(), [nTP(), Sensitivity()])
    with pytest.raises(ValueError, match="not interval-eligible"):
        compare(a, test="superiority", alpha=0.025, direction={"n_tp": "higher"})
    result = compare(a, test="superiority", alpha=0.025, test_metrics=["sensitivity"])
    assert overall(result, "n_tp").hypothesis is None
