"""Independent regressions for binary estimands, uncertainty and provenance."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom
from sklearn.metrics import brier_score_loss, log_loss

from model_auditor import Auditor, InferenceConfig
from model_auditor._evaluation import wilson_interval
from model_auditor.metrics import (
    AUROC,
    FNR,
    FPR,
    AveragePrecision,
    BrierScore,
    CalibrationIntercept,
    CalibrationSlope,
    ExpectedCost,
    FBetaScore,
    LogLoss,
    NegativePredictiveValue,
    Sensitivity,
    Specificity,
)


def make_auditor(df, metrics=None, threshold=0.5):
    a = Auditor(data=df, metrics=metrics or [Sensitivity(), Specificity()])
    a.add_outcome("y")
    a.add_score("s", threshold=threshold)
    if "g" in df:
        a.add_feature("g")
    return a


def overall(result, metric):
    return result.features["overall"].levels["Overall"].metrics[metric]


def test_no_positive_cases_are_undefined_with_denominators():
    a = make_auditor(
        pd.DataFrame({"y": [0] * 10, "s": [0.1] * 10}), [Sensitivity(), FNR()]
    )
    result = a.evaluate_metrics("s")
    for name in ["sensitivity", "fnr"]:
        metric = overall(result, name)
        assert np.isnan(metric.score)
        assert metric.denominator == 0
        assert metric.status == "undefined"
        assert metric.interval is None
    assert result.features["overall"].levels["Overall"].support["n_neg"] == 10


def test_boundary_sensitivity_has_nonzero_uncertainty():
    a = make_auditor(pd.DataFrame({"y": [1] * 10, "s": [0.9] * 10}), [Sensitivity()])
    metric = overall(a.evaluate_metrics("s"), "sensitivity")
    assert metric.score == 1
    assert metric.interval == pytest.approx((0.7224672001371107, 1))
    assert metric.interval_method == "wilson"


def test_exact_enumerated_wilson_coverage_for_review_scenario():
    # Enumerate the binomial experiment; do not compare our own bootstrap loop.
    bounds = [wilson_interval(k, 10, 0.95) for k in range(11)]
    coverage = sum(
        binom.pmf(k, 10, 0.95) for k, (lo, hi) in enumerate(bounds) if lo <= 0.95 <= hi
    )
    assert coverage > 0.90


def test_sparse_or_point_estimates_remain_reciprocal():
    a = make_auditor(
        pd.DataFrame(
            {"g": ["A", "A", "B", "B"], "y": [0] * 4, "s": [0.9, 0.1, 0.9, 0.1]}
        )
    )
    raw = a.evaluate_errors("s", n_bootstraps=None)
    inferred = a.evaluate_errors("s", inference=InferenceConfig(random_state=42))
    for name in ["A", "B"]:
        m = inferred.groups["fp"].features["g"].levels[name].metrics["odds_ratio"]
        assert (
            m.score
            == raw.groups["fp"].features["g"].levels[name].metrics["odds_ratio"].score
            == 1
        )
        assert m.interval_method == "conditional_exact"
        assert m.interval[0] < 1 < m.interval[1]


def test_tiny_auc_does_not_report_a_certain_interval():
    a = make_auditor(pd.DataFrame({"y": [1, 0], "s": [0.9, 0.1]}), [AUROC()], None)
    m = overall(
        a.evaluate_metrics("s", inference=InferenceConfig(random_state=42)), "auroc"
    )
    assert m.score == 1 and m.interval is None
    assert m.nonfinite_resamples > 0
    assert m.interval_status == "too_many_invalid_resamples"


@pytest.mark.parametrize("categorical", [False, True])
def test_ambiguous_subgroup_labels_rejected(categorical):
    df = pd.DataFrame({"g": [1, 1, "1", "1"], "y": [1] * 4, "s": [0.9, 0.9, 0.1, 0.1]})
    if categorical:
        df["g"] = pd.Categorical(df["g"], categories=[1, "1"])
    a = make_auditor(df)
    for method in [a.evaluate_metrics, a.evaluate_errors]:
        with pytest.raises(ValueError, match="same string label"):
            method("s", n_bootstraps=None)


def test_missingness_is_recorded_and_can_be_included_or_rejected():
    a = make_auditor(pd.DataFrame({"g": ["A", None], "y": [1, 0], "s": [0.9, 0.1]}))
    excluded = a.evaluate_metrics("s", n_bootstraps=None).features["g"]
    assert excluded.excluded_n == 1 and excluded.total_n == 2
    included = a.evaluate_metrics(
        "s", n_bootstraps=None, inference=InferenceConfig(missing="include")
    )
    assert included.features["g"].levels["(Missing)"].support["n"] == 1
    with pytest.raises(ValueError, match="missing"):
        a.evaluate_metrics("s", inference=InferenceConfig(missing="error"))


def test_cluster_bootstrap_does_not_treat_copies_as_new_subjects():
    df = pd.DataFrame({"id": range(20), "y": [1] * 20, "s": [0.9, 0.1] * 10})
    config = InferenceConfig(resampling="cluster", cluster="id", random_state=12)
    first = overall(
        make_auditor(df, [Sensitivity()]).evaluate_metrics(
            "s", n_bootstraps=200, inference=config
        ),
        "sensitivity",
    )
    copied = df.loc[df.index.repeat(5)].reset_index(drop=True)
    second = overall(
        make_auditor(copied, [Sensitivity()]).evaluate_metrics(
            "s", n_bootstraps=200, inference=config
        ),
        "sensitivity",
    )
    assert first.interval is not None
    assert first.interval == second.interval
    assert first.interval_method == "cluster_percentile"


def test_seed_is_local_and_provenance_and_numeric_export_survive():
    df = pd.DataFrame({"y": [1, 0] * 20, "s": [0.7, 0.3, 0.2, 0.8] * 10})
    a = make_auditor(df, [AUROC()], None)
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    config = InferenceConfig(random_state=5)
    result = a.evaluate_metrics(
        "s", n_bootstraps=100, inference=config, cohort="external"
    )
    assert np.random.random() == expected
    repeat = a.evaluate_metrics("s", n_bootstraps=100, inference=config)
    assert overall(result, "auroc").interval == overall(repeat, "auroc").interval
    frame = result.to_numeric_dataframe()
    assert pd.api.types.is_numeric_dtype(frame.estimate)
    assert frame.attrs["metadata"]["cohort"] == "external"
    assert frame.attrs["metadata"]["threshold"] is None
    frame.attrs["metadata"]["inference"]["random_state"] = 10
    assert result.metadata["inference"]["random_state"] == 5


def test_identical_paired_models_have_exact_zero_contrast():
    df = pd.DataFrame({"y": [1, 0] * 30, "s": [0.8, 0.7, 0.3, 0.2] * 15})
    df["other"] = df.s
    a = make_auditor(df, [AUROC()], None)
    a.add_score("other")
    result = a.compare_scores(
        "s", "other", n_bootstraps=100, inference=InferenceConfig(random_state=3)
    )
    metric = overall(result, "auroc")
    assert metric.score == 0 and metric.interval == (0, 0)
    assert result.metadata["estimand"] == "paired_model_contrast"


def test_group_rate_comparison_is_not_membership_enrichment():
    rows = []
    for g, tp, fn, fp, tn in [("A", 72, 18, 1, 9), ("B", 8, 2, 9, 81)]:
        for y, s, n in [(1, 0.9, tp), (1, 0.1, fn), (0, 0.9, fp), (0, 0.1, tn)]:
            rows.extend([{"g": g, "y": y, "s": s}] * n)
    a = make_auditor(pd.DataFrame(rows), [FPR(), FNR()])
    result = a.compare_groups("s", "g", "A", n_bootstraps=None)
    assert result.features["g"].levels["B"].metrics["fpr"].score == 0
    assert result.features["g"].levels["B"].metrics["fnr"].score == 0


def test_probability_metrics_match_independent_implementations():
    df = pd.DataFrame({"_truth": [1, 0, 1, 0], "_pred": [0.8, 0.4, 0.3, 0.1]})
    assert BrierScore().data_call(df) == pytest.approx(
        brier_score_loss(df._truth, df._pred)
    )
    assert LogLoss().data_call(df) == pytest.approx(log_loss(df._truth, df._pred))
    with pytest.raises(ValueError, match="Probability"):
        BrierScore().data_call(df.assign(_pred=2))
    assert AveragePrecision().data_call(
        df.assign(_truth=0)
    ) != AveragePrecision().data_call(df.assign(_truth=0))


def test_calibration_bins_include_empty_and_endpoint_bins():
    a = make_auditor(pd.DataFrame({"y": [0, 1, 1, 0], "s": [0.0, 1.0, 0.75, 0.25]}))
    result = a.evaluate_calibration("s", bins=5, n_bootstraps=None)
    assert result.bins.n.sum() == 4
    assert len(result.bins) == 5
    assert np.isnan(result.bins.loc[result.bins.n == 0, "observed_frequency"]).all()
    assert overall(result.summary, "brier_score").score == pytest.approx(0.03125)
    assert overall(result.summary, "calibration_slope").status == "undefined"


def test_calibration_fits_known_binomial_frequencies():
    # Two probability groups with exact observed frequencies equal to scores.
    df = pd.DataFrame(
        {
            "_pred": [0.25] * 100 + [0.75] * 100,
            "_truth": [1] * 25 + [0] * 75 + [1] * 75 + [0] * 25,
        }
    )
    assert CalibrationIntercept().data_call(df) == pytest.approx(0, abs=1e-6)
    assert CalibrationSlope().data_call(df) == pytest.approx(1, abs=1e-6)


def test_decision_curve_and_cost_have_explicit_estimands():
    a = make_auditor(
        pd.DataFrame({"y": [1, 1, 0, 0], "s": [0.9, 0.2, 0.7, 0.1]}),
        [ExpectedCost(2, 3)],
    )
    assert (
        overall(a.evaluate_metrics("s", n_bootstraps=None), "expected_cost").score
        == 1.25
    )
    curve = a.decision_curve("s", [0.5])
    assert curve.net_benefit.iloc[0] == 0 and curve.act_all.iloc[0] == 0
    assert curve.selection_rate.iloc[0] == 0.5


def test_new_rate_and_parameter_validation():
    assert (
        NegativePredictiveValue().data_call(pd.DataFrame({"tn": [3], "fn": [1]}))
        == 0.75
    )
    assert FBetaScore(1.01).name != FBetaScore(1.04).name
    for value in [-1, 0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            FBetaScore(value)


def test_neutral_default_and_unknown_custom_direction():
    a = make_auditor(pd.DataFrame({"y": [1, 0], "s": [0.9, 0.1]}))
    result = a.evaluate_metrics("s", n_bootstraps=None)
    assert "background-color" not in result.style_dataframe().to_html()


def test_intersections_preserve_missingness_and_component_boundaries():
    df = pd.DataFrame(
        {
            "y": [1, 0, 1],
            "s": [0.8, 0.1, 0.7],
            "a": ["a & b", "a", None],
            "b": ["c", "b & c", "c"],
        }
    )
    a = make_auditor(df)
    a.add_intersection("joint", ["a", "b"])
    result = a.evaluate_metrics("s", n_bootstraps=None)
    assert len(result.features["joint"].levels) == 2
    assert result.features["joint"].excluded_n == 1
    assert "joint" not in df


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 0},
        {"confidence_level": np.nan},
        {"resampling": "cluster"},
        {"cluster": "id"},
        {"method": "unknown"},
        {"missing": "drop"},
        {"random_state": -1},
        {"min_valid_fraction": 2},
    ],
)
def test_inference_configuration_rejects_invalid_policies(kwargs):
    with pytest.raises(ValueError):
        InferenceConfig(**kwargs)


def test_replacing_data_requires_outcome_registration_without_deleting_raw_column():
    a = make_auditor(pd.DataFrame({"y": [0, 1], "s": [0.1, 0.9]}))
    replacement = pd.DataFrame({"_truth": [1, 0], "s": [0.9, 0.1]})
    a.add_data(replacement)
    with pytest.raises(ValueError, match="outcome"):
        a.evaluate_metrics("s", n_bootstraps=None)
    a.add_outcome("_truth")
    assert overall(a.evaluate_metrics("s", n_bootstraps=None), "sensitivity").score == 1
    pd.testing.assert_frame_equal(
        replacement, pd.DataFrame({"_truth": [1, 0], "s": [0.9, 0.1]})
    )


def test_cluster_requires_nonmissing_ids_and_multiple_clusters():
    df = pd.DataFrame({"id": [1, 1], "y": [1, 1], "s": [0.9, 0.1]})
    config = InferenceConfig(resampling="cluster", cluster="id", random_state=1)
    m = overall(
        make_auditor(df, [Sensitivity()]).evaluate_metrics("s", inference=config),
        "sensitivity",
    )
    assert m.interval_status == "insufficient_clusters"
    with pytest.raises(ValueError, match="Cluster"):
        make_auditor(df.assign(id=np.nan)).evaluate_metrics("s", inference=config)


def test_explicit_stratification_and_degeneracy_diagnostics():
    a = make_auditor(pd.DataFrame({"y": [1, 0], "s": [0.9, 0.1]}), [AUROC()], None)
    m = overall(
        a.evaluate_metrics(
            "s",
            n_bootstraps=100,
            inference=InferenceConfig(resampling="stratified", random_state=1),
        ),
        "auroc",
    )
    assert m.valid_resamples == 100 and m.nonfinite_resamples == 0
    assert m.interval is None and m.interval_status == "degenerate_distribution"


def test_error_confidence_labels_follow_configuration():
    df = pd.DataFrame(
        {"g": ["A", "A", "B", "B"], "y": [0] * 4, "s": [0.9, 0.1, 0.9, 0.1]}
    )
    result = make_auditor(df).evaluate_errors(
        "s", inference=InferenceConfig(confidence_level=0.9)
    )
    assert ("FP", "OR 90% CI Lower") in result.to_dataframe(metric_labels=True)
    styled = result.style_dataframe(metric_labels=True).data
    assert ("FP", "OR 90% CI Lower") not in styled
    assert "(" in styled.loc[("g", "A"), ("FP", "Odds Ratio")]


def test_bootstrap_option_retains_sparse_point_estimate_and_exposes_failures():
    df = pd.DataFrame(
        {"g": ["A", "A", "B", "B"], "y": [0] * 4, "s": [0.9, 0.1, 0.9, 0.1]}
    )
    result = make_auditor(df).evaluate_errors(
        "s", inference=InferenceConfig(method="bootstrap", random_state=42)
    )
    m = result.groups["fp"].features["g"].levels["A"].metrics["odds_ratio"]
    assert m.score == 1 and m.nonfinite_resamples > 0
    assert m.interval is None and m.interval_status == "too_many_invalid_resamples"


def test_custom_error_metric_is_available_in_generic_export():
    class GroupFraction:
        name = "group_fraction"
        label = "Group fraction"
        ci_eligible = False

        def compute(self, group_count, group_total, full_count, full_total):
            return group_count / group_total if group_total else float("nan")

    df = pd.DataFrame(
        {"g": ["A", "A", "B", "B"], "y": [0] * 4, "s": [0.9, 0.1, 0.9, 0.1]}
    )
    result = make_auditor(df).evaluate_errors("s", error_metric=GroupFraction())
    frame = result.to_numeric_dataframe()
    row = frame.loc[
        (frame.feature == "g") & (frame.level == "A") & (frame.confusion_group == "fp")
    ].iloc[0]
    assert row.score == "s" and row.metric == "group_fraction" and row.estimate == 0.5


def test_perfect_rank_ties_receive_neutral_color_when_ranking_is_requested():
    from model_auditor.schemas import FeatureEvaluation, LevelEvaluation, LevelMetric

    result = FeatureEvaluation("g", "g")
    for name in ["A", "B", "C"]:
        result.levels[name] = LevelEvaluation(
            name,
            metrics={
                "sensitivity": LevelMetric(
                    "sensitivity", "Sensitivity", 1.0, direction="higher"
                )
            },
        )
    html = result.style_dataframe(rank=True).to_html()
    assert "#fff3cd" in html and "#f8d7da" not in html and "#d4edda" not in html


def test_exact_binomial_option_handles_all_successes():
    a = make_auditor(pd.DataFrame({"y": [1] * 10, "s": [0.9] * 10}), [Sensitivity()])
    result = a.evaluate_metrics("s", inference=InferenceConfig(rate_interval="exact"))
    m = overall(result, "sensitivity")
    assert m.interval == pytest.approx((0.6915028921812393, 1))
    assert m.interval_method == "clopper_pearson"
