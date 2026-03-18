"""Tests for Auditor.evaluate_errors() and related result types.

Coverage:
- evaluate_errors() returns ErrorEvaluation with all four confusion groups.
- OddsRatio formula correctness on deterministic synthetic data.
- NaN baseline for levels absent from the full dataset (full_count == 0).
- NaN when no comparator population (overall/Overall: full_total == full_count).
- Zero-in-group: a level present in the full dataset but absent from a specific
  confusion group yields OR == 0.0 (not NaN).
- Bootstrap mean replaces the raw OR as point estimate when n_bootstraps > 0.
- Bootstrap CI bounds are present and satisfy lower <= upper.
- No CI for levels with NaN baseline, even when bootstraps are requested.
- Categorical ordering is preserved across all groups.
- to_dataframe() produces the expected wide cross-group format.
- to_dataframe() includes OR CI columns populated from LevelMetric.interval.
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
# Canonical OR for Female in TP group:
#   a=4 (Female∩TP), b=4 (Female∩not-TP), c=4 (not-Female∩TP), d=8 (not-Female∩not-TP)
#   OR = (4*8)/(4*4) = 2.0
#
# OR for Other in FP group (zero-in-group):
#   a=0, b=4, c=2, d=14 → OR = 0/8 = 0.0


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
# Helpers for exact OR assertions
# ---------------------------------------------------------------------------


def _or(group_count: int, group_total: int, full_count: int, full_total: int) -> float:
    """Canonical 2x2 odds ratio matching OddsRatio.compute()."""
    if full_count == 0:
        return float("nan")
    if full_total - full_count == 0:
        return float("nan")
    a = group_count
    b = full_count - group_count
    c = group_total - group_count
    d = (full_total - full_count) - c
    numerator = a * d
    denominator = b * c
    if denominator == 0:
        if numerator == 0:
            return float("nan")
        return float("inf")
    return numerator / denominator


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

    def test_odds_ratio_metric_present_in_levels(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        gender = result.groups["tp"].features["gender"]
        for level_eval in gender.levels.values():
            assert "odds_ratio" in level_eval.metrics


# ---------------------------------------------------------------------------
# TestOddsRatioValues
# ---------------------------------------------------------------------------


class TestOddsRatioValues:
    """OR formula correctness on deterministic data."""

    def setup_method(self):
        df = _make_df()
        self.result = _make_auditor(df).evaluate_errors(score_name="score", n_bootstraps=None)

    def _or_for(self, group: str, level: str) -> float:
        return self.result.groups[group].features["gender"].levels[level].metrics["odds_ratio"].score

    def test_female_in_tp(self):
        # a=4, b=4, c=4, d=8 → OR=32/16=2.0
        expected = _or(4, 8, 8, 20)
        assert abs(self._or_for("tp", "Female") - expected) < 1e-10

    def test_male_in_tp(self):
        # a=3, b=5, c=5, d=7 → OR=21/25=0.84
        expected = _or(3, 8, 8, 20)
        assert abs(self._or_for("tp", "Male") - expected) < 1e-10

    def test_other_in_tp(self):
        # a=1, b=3, c=7, d=9 → OR=9/21=3/7
        expected = _or(1, 8, 4, 20)
        assert abs(self._or_for("tp", "Other") - expected) < 1e-10

    def test_female_in_fn(self):
        # FN total=3; Female FN=1; a=1, b=7, c=2, d=10 → OR=10/14=5/7
        expected = _or(1, 3, 8, 20)
        assert abs(self._or_for("fn", "Female") - expected) < 1e-10

    def test_overall_or_is_nan(self):
        # The overall level covers all rows: full_total - full_count == 0 → NaN.
        for group in ("tp", "tn", "fp", "fn"):
            overall_or = (
                self.result.groups[group]
                .features["overall"]
                .levels["Overall"]
                .metrics["odds_ratio"]
                .score
            )
            assert math.isnan(overall_or), (
                f"Overall OR for group '{group}' should be NaN (no comparator), got {overall_or}"
            )


# ---------------------------------------------------------------------------
# TestZeroInGroup
# ---------------------------------------------------------------------------


class TestZeroInGroup:
    """Level present in full dataset but absent from a confusion group → OR == 0.0."""

    def test_other_in_fp_is_zero(self):
        # Other has no FP rows (all Other scores are 0.3/0.1 < threshold=0.5).
        # a=0, b=4, c=2, d=14 → OR=0/8=0.0
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        fp_other = result.groups["fp"].features["gender"].levels["Other"]
        ratio = fp_other.metrics["odds_ratio"].score
        assert ratio == 0.0, f"Expected 0.0 for Other in FP, got {ratio}"

    def test_zero_or_is_not_nan(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        ratio = (
            result.groups["fp"].features["gender"].levels["Other"]
            .metrics["odds_ratio"].score
        )
        assert not math.isnan(ratio), "Zero-in-group OR must not be NaN"


# ---------------------------------------------------------------------------
# TestNaNBaseline
# ---------------------------------------------------------------------------


class TestNaNBaseline:
    """Level absent from the full dataset (full_count == 0) → OR NaN, no CI."""

    def test_unobserved_category_nan_or(self):
        # 'Unknown' is a declared category with zero rows in the full dataset.
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        unknown = result.groups["tp"].features["gender"].levels["Unknown"]
        ratio = unknown.metrics["odds_ratio"].score
        assert math.isnan(ratio), f"Expected NaN for unobserved 'Unknown', got {ratio}"

    def test_unobserved_category_no_ci(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=500)
        for group in ("tp", "tn", "fp", "fn"):
            unknown = result.groups[group].features["gender"].levels["Unknown"]
            lm = unknown.metrics["odds_ratio"]
            assert lm.interval is None, (
                f"NaN-baseline level 'Unknown' in group '{group}' must have no CI, "
                f"got {lm.interval}"
            )

    def test_unobserved_category_nan_score_with_bootstraps(self):
        """Bootstrap must preserve NaN score for levels with undefined baseline."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=50)
        unknown = result.groups["tp"].features["gender"].levels["Unknown"]
        assert math.isnan(unknown.metrics["odds_ratio"].score)


# ---------------------------------------------------------------------------
# TestBootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    """Bootstrap path: point estimate becomes bootstrap mean; CI bounds populated."""

    def test_observed_levels_have_ci(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=100)
        for level in ("Female", "Male", "Other"):
            lm = result.groups["tp"].features["gender"].levels[level].metrics["odds_ratio"]
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
                    .metrics["odds_ratio"]
                )
                if lm.interval is not None:
                    lower, upper = lm.interval
                    assert lower <= upper, (
                        f"CI for {level} in {group}: lower={lower} > upper={upper}"
                    )

    def test_bootstrap_mean_is_float(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=50)
        lm = result.groups["tp"].features["gender"].levels["Female"].metrics["odds_ratio"]
        assert isinstance(lm.score, float)

    def test_no_bootstraps_no_ci(self):
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        for group in ("tp", "tn", "fp", "fn"):
            for level in ("Female", "Male", "Other"):
                lm = (
                    result.groups[group]
                    .features["gender"]
                    .levels[level]
                    .metrics["odds_ratio"]
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
# TestToDataframe — wide cross-group format contract
# ---------------------------------------------------------------------------


class TestToDataframe:
    """ErrorEvaluation.to_dataframe() wide cross-group format contract.

    Dataset used (threshold=0.5, global_total_n=20):
      Female: 4 TP, 1 FP, 1 FN, 2 TN  → 8 rows
      Male:   3 TP, 1 FP, 1 FN, 3 TN  → 8 rows
      Other:  1 TP, 0 FP, 1 FN, 2 TN  → 4 rows
      Unknown: 0 rows (declared but unobserved)
    """

    def setup_method(self):
        self.result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        self.df = self.result.to_dataframe()

    # -- basic structure -------------------------------------------------------

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_row_index_has_two_levels(self):
        assert self.df.index.nlevels == 2, (
            f"Expected 2-level MultiIndex (feature, level), got {self.df.index.nlevels}"
        )

    def test_row_index_names(self):
        assert list(self.df.index.names) == ["feature", "level"]

    def test_columns_are_multiindex(self):
        assert isinstance(self.df.columns, pd.MultiIndex)

    # -- row membership --------------------------------------------------------

    def test_overall_row_present(self):
        assert ("Overall", "Overall") in self.df.index

    def test_gender_observed_levels_present(self):
        for level in ("Female", "Male", "Other"):
            assert ("gender", level) in self.df.index, (
                f"Row ('gender', {level!r}) missing from index"
            )

    def test_gender_unobserved_level_present(self):
        """Declared-but-unobserved 'Unknown' category must appear as a row."""
        assert ("gender", "Unknown") in self.df.index

    # -- column section presence -----------------------------------------------

    def test_class_balance_section_absent(self):
        """Class Balance is no longer a top-level section; its columns moved to Overall."""
        top_cols = set(self.df.columns.get_level_values(0))
        assert "Class Balance" not in top_cols

    def test_overall_section_present(self):
        top_cols = set(self.df.columns.get_level_values(0))
        assert "Overall" in top_cols

    def test_all_group_sections_present(self):
        top_cols = set(self.df.columns.get_level_values(0))
        assert {"TP", "TN", "FP", "FN"}.issubset(top_cols)

    # -- column sub-column completeness ----------------------------------------

    def test_overall_section_sub_columns(self):
        """Overall section includes N, % overall, and the class-balance sub-columns."""
        overall_cols = set(self.df["Overall"].columns)
        assert overall_cols == {"N", "% overall", "N_pos", "N_neg", "Pos %"}

    def test_group_section_sub_columns_default_names(self):
        for group in ("TP", "TN", "FP", "FN"):
            group_cols = set(self.df[group].columns)
            expected = {"N", "% overall", "% group", "odds_ratio",
                        "odds_ratio_ci_lower", "odds_ratio_ci_upper"}
            assert expected.issubset(group_cols), (
                f"Group {group} missing columns; got {group_cols}"
            )

    def test_metric_labels_flag_renames_or_columns(self):
        df_labels = self.result.to_dataframe(metric_labels=True)
        for group in ("TP", "TN", "FP", "FN"):
            assert "Odds Ratio" in df_labels[group].columns
            assert "OR 95% CI Lower" in df_labels[group].columns
            assert "OR 95% CI Upper" in df_labels[group].columns
            assert "odds_ratio" not in df_labels[group].columns
            assert "odds_ratio_ci_lower" not in df_labels[group].columns
            assert "odds_ratio_ci_upper" not in df_labels[group].columns

    # -- numeric types ---------------------------------------------------------

    def test_all_columns_are_numeric(self):
        """to_dataframe() must return numeric data, never preformatted strings."""
        for col in self.df.columns:
            non_nan = self.df[col].dropna()
            if len(non_nan) == 0:
                continue
            assert non_nan.dtype.kind in ("f", "i", "u"), (
                f"Column {col} has non-numeric dtype {self.df[col].dtype}"
            )

    # -- overall/Overall row: OR and CI columns must be NaN -------------------

    def test_overall_row_group_or_is_nan(self):
        """(Overall, Overall) row emits NaN for all group odds ratios.

        The OR is undefined when all rows belong to the level (no comparator
        population exists).
        """
        overall_row = self.df.loc[("Overall", "Overall")]
        for group in ("TP", "TN", "FP", "FN"):
            or_val = overall_row[(group, "odds_ratio")]
            assert pd.isna(or_val), (
                f"Overall/Overall OR for {group} must be NaN, got {or_val}"
            )

    def test_overall_row_ci_columns_are_nan(self):
        """(Overall, Overall) CI columns are also NaN."""
        overall_row = self.df.loc[("Overall", "Overall")]
        for group in ("TP", "TN", "FP", "FN"):
            assert pd.isna(overall_row[(group, "odds_ratio_ci_lower")])
            assert pd.isna(overall_row[(group, "odds_ratio_ci_upper")])

    # -- no-bootstrap run: CI columns are NaN --------------------------------

    def test_ci_columns_are_nan_without_bootstraps(self):
        """When n_bootstraps=None, all CI bound columns must be NaN."""
        for group in ("TP", "TN", "FP", "FN"):
            assert self.df[(group, "odds_ratio_ci_lower")].isna().all(), (
                f"CI lower column for {group} should be all-NaN without bootstraps"
            )
            assert self.df[(group, "odds_ratio_ci_upper")].isna().all(), (
                f"CI upper column for {group} should be all-NaN without bootstraps"
            )

    def test_overall_n_pos_female(self):
        """Female N_pos = TP_n(4) + FN_n(1) = 5 (now in Overall section)."""
        row = self.df.loc[("gender", "Female")]
        assert row[("Overall", "N_pos")] == 5

    def test_overall_n_neg_female(self):
        """Female N_neg = TN_n(2) + FP_n(1) = 3 (now in Overall section)."""
        row = self.df.loc[("gender", "Female")]
        assert row[("Overall", "N_neg")] == 3

    def test_overall_pos_pct_female(self):
        """Female Pos % = 5 / 8 (fraction of Female rows that are positive)."""
        row = self.df.loc[("gender", "Female")]
        assert abs(row[("Overall", "Pos %")] - 5 / 8) < 1e-10

    def test_neg_pct_absent(self):
        """Neg % is redundant (= 1 - Pos %) and must not appear anywhere."""
        leaf_cols = [col[1] for col in self.df.columns]
        assert "Neg %" not in leaf_cols


    def test_first_two_columns_are_overall(self):
        """First two leaf columns are ('Overall', 'N') then ('Overall', '% overall')."""
        cols = list(self.df.columns)
        assert cols[0] == ("Overall", "N"), f"First column should be ('Overall', 'N'), got {cols[0]}"
        assert cols[1] == ("Overall", "% overall"), (
            f"Second column should be ('Overall', '% overall'), got {cols[1]}"
        )

    # -- pct_overall uses global N denominator (not per-feature N) -------------

    def test_pct_overall_female_tp(self):
        """Female % overall for TP = 4/20 (global N=20)."""
        row = self.df.loc[("gender", "Female")]
        assert abs(row[("TP", "% overall")] - 4 / 20) < 1e-10

    # -- pct_group uses confusion-group total as denominator -------------------

    def test_pct_group_female_tp(self):
        """Female % group for TP = 4/8 (total TP rows = 8)."""
        row = self.df.loc[("gender", "Female")]
        assert abs(row[("TP", "% group")] - 4 / 8) < 1e-10

    # -- support counts for unobserved category --------------------------------

    def test_unobserved_level_n_is_zero(self):
        """Unobserved 'Unknown' level must have N=0 for all groups."""
        row = self.df.loc[("gender", "Unknown")]
        for group in ("TP", "TN", "FP", "FN"):
            assert row[(group, "N")] == 0, (
                f"Unobserved 'Unknown' should have N=0 for {group}, got {row[(group, 'N')]}"
            )

    # -- empty container -------------------------------------------------------

    def test_empty_groups_dict_returns_empty_dataframe(self):
        empty = ErrorEvaluation(name="x", label="x", threshold=0.5)
        df = empty.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# TestToDataframeWithBootstrap — CI columns populated after bootstrapping
# ---------------------------------------------------------------------------


class TestToDataframeWithBootstrap:
    """CI columns in to_dataframe() are populated when n_bootstraps is set."""

    def setup_method(self):
        self.result = _make_auditor(_make_df()).evaluate_errors(
            score_name="score", n_bootstraps=100
        )
        self.df = self.result.to_dataframe()

    def test_ci_columns_present_in_schema(self):
        for group in ("TP", "TN", "FP", "FN"):
            assert (group, "odds_ratio_ci_lower") in self.df.columns
            assert (group, "odds_ratio_ci_upper") in self.df.columns

    def test_observed_level_ci_columns_are_numeric(self):
        """Observed levels must have non-NaN CI values after bootstrapping.

        CI bounds may be +inf (OR is unbounded when b*c==0 in a bootstrap sample)
        but must never be NaN for observed levels.
        """
        for group in ("TP", "TN", "FP", "FN"):
            for level in ("Female", "Male", "Other"):
                row = self.df.loc[("gender", level)]
                lower = row[(group, "odds_ratio_ci_lower")]
                upper = row[(group, "odds_ratio_ci_upper")]
                # CI bounds must not be NaN; +inf is allowed (sparse-table bootstrap).
                assert not pd.isna(lower), (
                    f"CI lower for {level}/{group} should not be NaN after bootstrapping"
                )
                assert not pd.isna(upper), (
                    f"CI upper for {level}/{group} should not be NaN after bootstrapping"
                )

    def test_ci_lower_le_upper_in_dataframe(self):
        """CI lower bound must not exceed upper bound in the DataFrame."""
        for group in ("TP", "TN", "FP", "FN"):
            for level in ("Female", "Male", "Other"):
                row = self.df.loc[("gender", level)]
                lower = row[(group, "odds_ratio_ci_lower")]
                upper = row[(group, "odds_ratio_ci_upper")]
                if pd.isna(lower) or pd.isna(upper):
                    continue
                assert lower <= upper, (
                    f"{level}/{group}: CI lower={lower} > upper={upper}"
                )

    def test_ci_columns_numeric_dtype(self):
        """CI columns must have floating-point dtype."""
        for group in ("TP", "TN", "FP", "FN"):
            for suffix in ("odds_ratio_ci_lower", "odds_ratio_ci_upper"):
                col = self.df[(group, suffix)]
                assert col.dtype.kind == "f", (
                    f"({group}, {suffix}) should be float dtype, got {col.dtype}"
                )

    def test_unobserved_level_ci_still_nan(self):
        """Declared-but-unobserved level 'Unknown' must retain NaN CI even after bootstrapping."""
        for group in ("TP", "TN", "FP", "FN"):
            row = self.df.loc[("gender", "Unknown")]
            assert pd.isna(row[(group, "odds_ratio_ci_lower")])
            assert pd.isna(row[(group, "odds_ratio_ci_upper")])

    def test_overall_row_ci_still_nan(self):
        """(Overall, Overall) OR is undefined; its CI columns must be NaN."""
        overall_row = self.df.loc[("Overall", "Overall")]
        for group in ("TP", "TN", "FP", "FN"):
            assert pd.isna(overall_row[(group, "odds_ratio_ci_lower")])
            assert pd.isna(overall_row[(group, "odds_ratio_ci_upper")])


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



# ---------------------------------------------------------------------------
# TestStyleDataframe — style_dataframe() rendering and tier behaviour
# ---------------------------------------------------------------------------


def _render_html(styler: "pd.io.formats.style.Styler") -> str:
    """Render Styler to HTML, compatible with pandas < 1.4 and >= 1.4."""
    try:
        return styler.to_html()
    except AttributeError:
        return styler.render()  # type: ignore[attr-defined]


def _cell_styles(html: str) -> "dict[str, str]":
    """Parse rendered Styler HTML to {cell_text: css_declaration}.

    Maps the display text of each <td> to the CSS declaration for that cell
    (empty string when unstyled).  When multiple cells share the same display
    text the last processed occurrence wins; callers must ensure the values of
    interest are unique across the table.
    """
    import re
    id_to_css: dict = {}
    for rule_m in re.finditer(
        r"((?:#T_\w+_row\d+_col\d+\s*,?\s*)+)\{([^}]+)\}", html
    ):
        css = rule_m.group(2).strip()
        for sel_m in re.finditer(r"T_\w+_row\d+_col\d+", rule_m.group(1)):
            id_to_css[sel_m.group(0)] = css
    result: dict = {}
    for m in re.finditer(
        r'id="(T_\w+_row\d+_col\d+)"[^>]*>(.*?)</td>', html, re.DOTALL
    ):
        cell_id = m.group(1)
        value = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        css = id_to_css.get(cell_id, "")
        result[value] = css
    return result


class TestStyleDataframe:
    """ErrorEvaluation.style_dataframe() rendering contract.

    Uses the canonical 20-row synthetic dataset (Female / Male / Other / Unknown,
    threshold=0.5).  OR values are deterministic; see OR derivations in the
    TestOddsRatioValues comments at the top of this file.
    """

    # -- OR column formatting ---------------------------------------------------

    def test_or_shows_inline_ci_with_bootstraps(self):
        """With bootstraps, OR cells show 'value (lo, hi)' inline."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=100)
        styler = result.style_dataframe(n_decimals=3)
        display_df = styler.data
        for group in ("TP", "TN", "FP", "FN"):
            for level in ("Female", "Male", "Other"):
                val = display_df.at[("gender", level), (group, "odds_ratio")]
                assert "(" in str(val) and "," in str(val), (
                    f"Expected inline CI for {level}/{group}, got {val!r}"
                )

    def test_or_shows_value_only_without_bootstraps(self):
        """Without bootstraps, OR cells show just the numeric value (no parens)."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        styler = result.style_dataframe(n_decimals=3)
        display_df = styler.data
        for group in ("TP", "TN", "FP", "FN"):
            for level in ("Female", "Male", "Other"):
                val = display_df.at[("gender", level), (group, "odds_ratio")]
                assert "(" not in str(val), (
                    f"Expected no CI parens for {level}/{group} without bootstraps, got {val!r}"
                )
                assert val != "\u2014", (
                    f"{level}/{group} has a defined OR; must not show em dash"
                )

    def test_nan_or_shows_em_dash(self):
        """NaN OR values (Overall/Overall row) must render as em dash."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        styler = result.style_dataframe()
        display_df = styler.data
        for group in ("TP", "TN", "FP", "FN"):
            val = display_df.at[("Overall", "Overall"), (group, "odds_ratio")]
            assert val == "\u2014", (
                f"Expected em dash for Overall OR in {group}, got {val!r}"
            )

    def test_n_decimals_respected_in_or_cell(self):
        """OR cell text uses the requested decimal precision."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        styler = result.style_dataframe(n_decimals=2)
        display_df = styler.data
        val = display_df.at[("gender", "Female"), ("TP", "odds_ratio")]
        assert val == "2.00", f"Expected '2.00' for Female/TP OR at n_decimals=2, got {val!r}"

    # -- CI columns absent from styled output -----------------------------------

    def test_ci_columns_absent_from_styled_output(self):
        """CI bound columns must not appear in the Styler display DataFrame."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=100)
        styler = result.style_dataframe()
        leaf_cols = [col[1] for col in styler.data.columns]
        assert "odds_ratio_ci_lower" not in leaf_cols, "CI lower must be hidden in styled output"
        assert "odds_ratio_ci_upper" not in leaf_cols, "CI upper must be hidden in styled output"

    def test_ci_columns_absent_with_metric_labels(self):
        """CI columns absent even when metric_labels=True swaps column names."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=100)
        styler = result.style_dataframe(metric_labels=True)
        leaf_cols = [col[1] for col in styler.data.columns]
        assert "OR 95% CI Lower" not in leaf_cols
        assert "OR 95% CI Upper" not in leaf_cols

    # -- OR column present in styled output ------------------------------------

    def test_or_column_present_in_styled_output(self):
        """OR column is present in the Styler for every group section."""
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        styler = result.style_dataframe()
        for group in ("TP", "TN", "FP", "FN"):
            assert (group, "odds_ratio") in styler.data.columns, (
                f"OR column missing from styled output for group {group}"
            )

    # -- FP/FN tier inversion ---------------------------------------------------

    def test_tp_highest_or_gets_green(self):
        """For TP (higher-is-better): the level with the highest OR gets the green tier.

        Female TP OR = 2.0 (highest in TP column) → should be green (#d4edda).
        This value is unique in the table so the _cell_styles lookup is unambiguous.
        """
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        html = _render_html(result.style_dataframe(n_decimals=3))
        styles = _cell_styles(html)
        assert "#d4edda" in styles.get("2.000", ""), (
            "Female TP OR (highest in column) must receive green background (high tier), "
            f"got CSS: {styles.get('2.000', '<absent>')}"
        )

    def test_fp_highest_or_does_not_get_green(self):
        """For FP (lower-is-better): the level with the highest OR must NOT be green.

        Female and Male FP OR = 11/7 ≈ 1.571 (highest in FP column); with
        inversion they should receive medium (yellow) tier, not high (green).
        '1.571' is unique in the display table (no other column produces this value).
        """
        result = _make_auditor(_make_df()).evaluate_errors(score_name="score", n_bootstraps=None)
        html = _render_html(result.style_dataframe(n_decimals=3))
        styles = _cell_styles(html)
        css = styles.get("1.571", "")
        assert "#d4edda" not in css, (
            "Female/Male FP OR (highest in FP column) must NOT be green with inversion, "
            f"got CSS: {css}"
        )
        # Three distinct values [0.0, 1.571, 1.571], percentile for 1.571 = 1/3.
        # lower_better=True: percentile 1/3 is not < 1/3 but < 2/3 → medium.
        assert "#fff3cd" in css, (
            "Female/Male FP OR should be medium (yellow) tier with inversion, "
            f"got CSS: {css}"
        )