"""Tests for Auditor.evaluate_errors() and related result types.

Coverage:
- evaluate_errors() returns ErrorEvaluation with all four confusion groups.
- RepresentationRatio formula correctness on deterministic synthetic data.
- NaN baseline for levels absent from the full dataset (full_count == 0).
- Zero-in-group: a level that appears in the full dataset but has no rows in a
  specific confusion group yields ratio == 0.0 (not NaN).
- Bootstrap mean replaces the raw ratio as point estimate when n_bootstraps > 0.
- Bootstrap CI bounds are present and satisfy lower <= upper.
- No CI for levels with NaN baseline, even when bootstraps are requested.
- Categorical ordering is preserved across all groups.
- to_dataframe() produces the expected MultiIndex structure.
- Validation errors: missing data, missing outcome, bad score name, no threshold.
"""

import math

import numpy as np
import pandas as pd
import pytest

from model_auditor import Auditor
from model_auditor.schemas import ErrorEvaluation, ScoreEvaluation


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

# Layout (threshold = 0.5):
#   Female: 4 TP, 1 FP, 1 FN, 2 TN  → 8 rows
#   Male:   3 TP, 1 FP, 1 FN, 3 TN  → 8 rows
#   Other:  1 TP, 0 FP, 1 FN, 2 TN  → 4 rows
#   Total:  8 TP, 2 FP, 3 FN, 7 TN  → 20 rows
#
# Representation ratios for Female in TP group:
#   P(Female | full) = 8/20 = 0.4
#   P(Female | TP)   = 4/8  = 0.5
#   ratio            = 0.5 / 0.4 = 1.25
#
# Representation ratio for Other in FP group:
#   P(Other | full) = 4/20  = 0.2
#   P(Other | FP)   = 0/2   = 0.0
#   ratio            = 0.0 / 0.2 = 0.0  (zero-in-group case)


def _make_df(include_unknown: bool = False) -> pd.DataFrame:
    """Return a deterministic binary-classification DataFrame.

    Groups Female, Male, Other are always present.  If include_unknown=False,
    'Unknown' is declared as a category but never observed, giving a
    declared-but-unobserved level to test NaN baseline behaviour.
    """
    rows = []

    # Female
    rows.extend([{"gender": "Female", "score": 0.8, "label": 1}] * 4)  # TP
    rows.append({"gender": "Female", "score": 0.7, "label": 0})          # FP
    rows.append({"gender": "Female", "score": 0.3, "label": 1})          # FN
    rows.extend([{"gender": "Female", "score": 0.2, "label": 0}] * 2)   # TN

    # Male
    rows.extend([{"gender": "Male", "score": 0.9, "label": 1}] * 3)     # TP
    rows.append({"gender": "Male", "score": 0.6, "label": 0})            # FP
    rows.append({"gender": "Male", "score": 0.4, "label": 1})            # FN
    rows.extend([{"gender": "Male", "score": 0.1, "label": 0}] * 3)     # TN

    # Other
    rows.append({"gender": "Other", "score": 0.8, "label": 1})           # TP
    rows.append({"gender": "Other", "score": 0.3, "label": 1})           # FN
    rows.extend([{"gender": "Other", "score": 0.1, "label": 0}] * 2)    # TN

    if include_unknown:
        rows.extend([{"gender": "Unknown", "score": 0.8, "label": 1}] * 2)
        rows.extend([{"gender": "Unknown", "score": 0.2, "label": 0}] * 2)

    df = pd.DataFrame(rows)
    categories = ["Female", "Male", "Other", "Unknown"]
    df["gender"] = pd.Categorical(df["gender"], categories=categories, ordered=True)
    return df


def _make_auditor(df: pd.DataFrame) -> Auditor:
    a = Auditor()
    a.add_data(df)
    a.add_feature(name="gender")
    a.add_score(name="score", threshold=0.5)
    a.add_outcome(name="label")
    return a


# ---------------------------------------------------------------------------
# Helpers for exact-ratio assertions
# ---------------------------------------------------------------------------


def _ratio(group_count: int, group_total: int, full_count: int, full_total: int) -> float:
    if full_total == 0 or full_count == 0:
        return float("nan")
    baseline = full_count / full_total
    group_rate = group_count / group_total if group_total > 0 else 0.0
    return group_rate / baseline


# ---------------------------------------------------------------------------
# TestEvaluateErrorsStructure
# ---------------------------------------------------------------------------


class TestEvaluateErrorsStructure:
    """evaluate_errors() returns the expected container structure."""

    def test_returns_error_evaluation(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        assert isinstance(result, ErrorEvaluation)

    def test_name_and_label(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        assert result.name == "score"
        assert result.label == "score"

    def test_threshold_stored(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        assert result.threshold == 0.5

    def test_all_four_groups_present(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        assert set(result.groups.keys()) == {"tp", "tn", "fp", "fn"}

    def test_each_group_is_score_evaluation(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        for group_eval in result.groups.values():
            assert isinstance(group_eval, ScoreEvaluation)

    def test_gender_feature_present_in_every_group(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        for group_eval in result.groups.values():
            assert "gender" in group_eval.features

    def test_overall_feature_present_in_every_group(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        for group_eval in result.groups.values():
            assert "overall" in group_eval.features

    def test_representation_ratio_metric_present_in_levels(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        gender = result.groups["tp"].features["gender"]
        for level_eval in gender.levels.values():
            assert "representation_ratio" in level_eval.metrics


# ---------------------------------------------------------------------------
# TestRepresentationRatioValues
# ---------------------------------------------------------------------------


class TestRepresentationRatioValues:
    """Ratio formula correctness on deterministic data."""

    def setup_method(self):
        df = _make_df()
        self.result = _make_auditor(df).evaluate_errors(score_name="score", n_bootstraps=None)

    def _ratio_for(self, group: str, level: str) -> float:
        return self.result.groups[group].features["gender"].levels[level].metrics["representation_ratio"].score

    def test_female_in_tp(self):
        # P(Female|full)=8/20=0.4, P(Female|TP)=4/8=0.5 → 1.25
        expected = _ratio(4, 8, 8, 20)
        assert abs(self._ratio_for("tp", "Female") - expected) < 1e-10

    def test_male_in_tp(self):
        # P(Male|full)=8/20=0.4, P(Male|TP)=3/8=0.375 → 0.9375
        expected = _ratio(3, 8, 8, 20)
        assert abs(self._ratio_for("tp", "Male") - expected) < 1e-10

    def test_other_in_tp(self):
        # P(Other|full)=4/20=0.2, P(Other|TP)=1/8=0.125 → 0.625
        expected = _ratio(1, 8, 4, 20)
        assert abs(self._ratio_for("tp", "Other") - expected) < 1e-10

    def test_female_in_fn(self):
        # FN total = 3; Female FN = 1
        expected = _ratio(1, 3, 8, 20)
        assert abs(self._ratio_for("fn", "Female") - expected) < 1e-10

    def test_overall_ratio_is_one_for_nonempty_group(self):
        # P(Overall|group) / P(Overall|full) = 1.0 by definition when group non-empty.
        for group in ("tp", "tn", "fp", "fn"):
            overall_ratio = (
                self.result.groups[group]
                .features["overall"]
                .levels["Overall"]
                .metrics["representation_ratio"]
                .score
            )
            assert abs(overall_ratio - 1.0) < 1e-10, (
                f"Overall ratio for group '{group}' should be 1.0, got {overall_ratio}"
            )


# ---------------------------------------------------------------------------
# TestZeroInGroup
# ---------------------------------------------------------------------------


class TestZeroInGroup:
    """Level present in full dataset but absent from a confusion group → ratio 0.0."""

    def test_other_in_fp_is_zero(self):
        # Other has no FP rows (score threshold 0.5; all Other scores are 0.3/0.1).
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        fp_other = result.groups["fp"].features["gender"].levels["Other"]
        ratio = fp_other.metrics["representation_ratio"].score
        assert ratio == 0.0, f"Expected 0.0 for Other in FP, got {ratio}"

    def test_zero_ratio_is_not_nan(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        ratio = (
            result.groups["fp"].features["gender"].levels["Other"]
            .metrics["representation_ratio"].score
        )
        assert not math.isnan(ratio), "Zero-in-group ratio must not be NaN"


# ---------------------------------------------------------------------------
# TestNaNBaseline
# ---------------------------------------------------------------------------


class TestNaNBaseline:
    """Level absent from the full dataset (full_count == 0) → ratio NaN, no CI."""

    def test_unobserved_category_nan_ratio(self):
        # 'Unknown' is a declared category with zero rows in the full dataset.
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        unknown = result.groups["tp"].features["gender"].levels["Unknown"]
        ratio = unknown.metrics["representation_ratio"].score
        assert math.isnan(ratio), f"Expected NaN for unobserved 'Unknown', got {ratio}"

    def test_unobserved_category_no_ci(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=500)
        for group in ("tp", "tn", "fp", "fn"):
            unknown = result.groups[group].features["gender"].levels["Unknown"]
            lm = unknown.metrics["representation_ratio"]
            assert lm.interval is None, (
                f"NaN-baseline level 'Unknown' in group '{group}' must have no CI, "
                f"got {lm.interval}"
            )

    def test_unobserved_category_nan_score_with_bootstraps(self):
        """Bootstrap must preserve NaN score for levels with undefined baseline."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=50)
        unknown = result.groups["tp"].features["gender"].levels["Unknown"]
        assert math.isnan(unknown.metrics["representation_ratio"].score)


# ---------------------------------------------------------------------------
# TestBootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    """Bootstrap path: point estimate becomes bootstrap mean; CI bounds populated."""

    def test_observed_levels_have_ci(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=100)
        for level in ("Female", "Male", "Other"):
            lm = result.groups["tp"].features["gender"].levels[level].metrics["representation_ratio"]
            assert lm.interval is not None, (
                f"Observed level '{level}' must have a CI after bootstrapping"
            )

    def test_ci_lower_le_upper(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=100)
        for group in ("tp", "tn", "fp", "fn"):
            for level in ("Female", "Male", "Other"):
                lm = (
                    result.groups[group]
                    .features["gender"]
                    .levels[level]
                    .metrics["representation_ratio"]
                )
                if lm.interval is not None:
                    lower, upper = lm.interval
                    assert lower <= upper, (
                        f"CI for {level} in {group}: lower={lower} > upper={upper}"
                    )

    def test_bootstrap_mean_is_float(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=50)
        lm = result.groups["tp"].features["gender"].levels["Female"].metrics["representation_ratio"]
        assert isinstance(lm.score, float)

    def test_no_bootstraps_no_ci(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        for group in ("tp", "tn", "fp", "fn"):
            for level in ("Female", "Male", "Other"):
                lm = (
                    result.groups[group]
                    .features["gender"]
                    .levels[level]
                    .metrics["representation_ratio"]
                )
                assert lm.interval is None, (
                    f"No bootstraps requested; {level} in {group} should have no CI"
                )


# ---------------------------------------------------------------------------
# TestCategoricalOrdering
# ---------------------------------------------------------------------------


DECLARED_ORDER = ["Female", "Male", "Other", "Unknown"]


class TestCategoricalOrdering:
    """Declared categorical order is preserved in every group's feature evaluation."""

    def test_gender_levels_follow_declared_order_in_all_groups(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        for group_name, group_eval in result.groups.items():
            keys = list(group_eval.features["gender"].levels.keys())
            assert keys == DECLARED_ORDER, (
                f"Group '{group_name}': expected {DECLARED_ORDER}, got {keys}"
            )

    def test_unobserved_unknown_present_in_all_groups(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        for group_name, group_eval in result.groups.items():
            assert "Unknown" in group_eval.features["gender"].levels, (
                f"'Unknown' placeholder missing from group '{group_name}'"
            )


# ---------------------------------------------------------------------------
# TestToDataframe
# ---------------------------------------------------------------------------


class TestToDataframe:
    """ErrorEvaluation.to_dataframe() structure and content."""

    def test_returns_dataframe(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_multiindex_has_three_levels(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        df = result.to_dataframe()
        assert df.index.nlevels == 3, (
            f"Expected 3-level MultiIndex (group, feature, level), got {df.index.nlevels}"
        )

    def test_all_four_groups_in_top_index_level(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        df = result.to_dataframe()
        top_level_groups = set(df.index.get_level_values(0).unique())
        assert top_level_groups == {"TP", "TN", "FP", "FN"}

    def test_representation_ratio_column_present(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        df = result.to_dataframe()
        assert "representation_ratio" in df.columns

    def test_metric_labels_flag(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        df = result.to_dataframe(metric_labels=True)
        assert "Representation Ratio" in df.columns

    def test_n_decimals_applied(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        df2 = result.to_dataframe(n_decimals=2)
        df5 = result.to_dataframe(n_decimals=5)
        # Different precision → different string representations for non-NaN values
        assert df2["representation_ratio"].iloc[0] != df5["representation_ratio"].iloc[0]

    def test_empty_groups_dict_returns_empty_dataframe(self):
        from model_auditor.schemas import ErrorEvaluation
        empty = ErrorEvaluation(name="x", label="x", threshold=0.5)
        df = empty.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    """evaluate_errors() raises descriptive errors for invalid state."""

    def test_no_data_raises(self):
        a = Auditor()
        a.add_score(name="score", threshold=0.5)
        with pytest.raises(ValueError, match="add_data"):
            a.evaluate_errors(score_name="score")

    def test_no_outcome_raises(self):
        a = Auditor()
        a.add_data(pd.DataFrame({"score": [0.5], "label": [1]}))
        a.add_score(name="score", threshold=0.5)
        with pytest.raises(ValueError, match="add_outcome"):
            a.evaluate_errors(score_name="score")

    def test_unknown_score_name_raises(self):
        a = _make_auditor(_make_df())
        with pytest.raises(ValueError, match="not found"):
            a.evaluate_errors(score_name="nonexistent")

    def test_no_threshold_raises(self):
        a = Auditor()
        df = _make_df()
        a.add_data(df)
        a.add_feature(name="gender")
        a.add_score(name="score")  # no threshold
        a.add_outcome(name="label")
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            a.evaluate_errors(score_name="score")

    def test_threshold_passed_at_call_time(self):
        """Threshold can be overridden per-call even when one is set on the score."""
        a = _make_auditor(_make_df())
        result = a.evaluate_errors(score_name="score", threshold=0.6, n_bootstraps=None)
        assert result.threshold == 0.6
