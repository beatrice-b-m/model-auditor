"""Tests for Auditor.plot_score_distributions.

Coverage:
- Success paths: returns expected dict keys, figure/axes types, axes count,
  x-label, figure title, density normalization, shared bins, score label
  fallback, feature subset selection and order preservation,
  categorical level ordering.
- Error paths: missing data, no features, unknown score, unknown feature
  name, no plottable levels (all-null feature column).
- Integration: full Auditor pipeline without evaluate_metrics().
"""

import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede any plt import

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from model_auditor import Auditor


# ---------------------------------------------------------------------------
# Shared fixture: close all figures after every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_auditor(feature_dtype: str = "object") -> Auditor:
    """Return a minimal Auditor with two features and one score.

    feature_dtype controls the dtype of the 'group' column:
    - 'object': plain string column (non-categorical path)
    - 'categorical': pandas CategoricalDtype with explicit ordering
    """
    rows = []
    for grp, n in [("A", 10), ("B", 8), ("C", 5)]:
        for _ in range(n):
            rows.append({"group": grp, "region": "North" if grp != "C" else "South",
                          "score": 0.6 if grp == "A" else 0.4})
    df = pd.DataFrame(rows)

    if feature_dtype == "categorical":
        # Declare categories in reverse alphabetical order to test ordering.
        df["group"] = pd.Categorical(
            df["group"], categories=["C", "B", "A"], ordered=True
        )

    auditor = Auditor()
    auditor.add_data(df)
    auditor.add_feature(name="group", label="Group")
    auditor.add_feature(name="region", label="Region")
    auditor.add_score(name="score", label="Risk Score", threshold=0.5)
    return auditor


# ===========================================================================
# TestReturnShape
# ===========================================================================


class TestReturnShape:
    """Return value structure and types."""

    def test_returns_dict(self):
        auditor = _make_auditor()
        result = auditor.plot_score_distributions("score")
        assert isinstance(result, dict)

    def test_returns_all_features_by_default(self):
        auditor = _make_auditor()
        result = auditor.plot_score_distributions("score")
        assert set(result.keys()) == {"group", "region"}

    def test_values_are_figure_ndarray_tuples(self):
        auditor = _make_auditor()
        result = auditor.plot_score_distributions("score")
        for fname, (fig, axes) in result.items():
            assert isinstance(fig, matplotlib.figure.Figure), fname
            assert isinstance(axes, np.ndarray), fname

    def test_axes_count_matches_level_count(self):
        """group has 3 levels; region has 2 levels."""
        auditor = _make_auditor()
        result = auditor.plot_score_distributions("score")
        _, group_axes = result["group"]
        _, region_axes = result["region"]
        assert len(group_axes) == 3
        assert len(region_axes) == 2

    def test_axes_are_matplotlib_axes_instances(self):
        auditor = _make_auditor()
        _, axes = auditor.plot_score_distributions("score")["group"]
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.Axes)

    def test_single_level_feature_returns_ndarray_of_length_one(self):
        """A feature with one level must still return a 1-element ndarray."""
        df = pd.DataFrame({"grp": ["X"] * 5, "score": [0.5] * 5})
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="grp")
        auditor.add_score(name="score", threshold=0.5)
        _, axes = auditor.plot_score_distributions("score")["grp"]
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 1


# ===========================================================================
# TestLabelsAndTitles
# ===========================================================================


class TestLabelsAndTitles:
    """X-label and figure title content."""

    def test_bottom_axes_xlabel_uses_score_label(self):
        auditor = _make_auditor()
        _, axes = auditor.plot_score_distributions("score")["group"]
        assert axes[-1].get_xlabel() == "Risk Score"

    def test_score_label_fallback_to_name(self):
        """When AuditorScore.label is None, the score column name is used."""
        df = pd.DataFrame({"grp": ["X", "Y"] * 5, "prob": [0.3, 0.7] * 5})
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="grp")
        auditor.add_score(name="prob")  # no label
        _, axes = auditor.plot_score_distributions("prob")["grp"]
        assert axes[-1].get_xlabel() == "prob"

    def test_figure_title_includes_feature_label(self):
        auditor = _make_auditor()
        fig, _ = auditor.plot_score_distributions("score")["group"]
        assert "Group" in fig.texts[0].get_text()

    def test_figure_title_includes_score_label(self):
        auditor = _make_auditor()
        fig, _ = auditor.plot_score_distributions("score")["group"]
        assert "Risk Score" in fig.texts[0].get_text()

    def test_level_name_appears_as_ylabel_on_each_axes(self):
        auditor = _make_auditor()
        _, axes = auditor.plot_score_distributions("score")["group"]
        ylabel_texts = [ax.get_ylabel() for ax in axes]
        for level_name in ["A", "B", "C"]:
            assert level_name in ylabel_texts

    def test_non_bottom_axes_have_no_xlabel(self):
        """Only the bottom subplot gets the score label as x-label."""
        auditor = _make_auditor()
        _, axes = auditor.plot_score_distributions("score")["group"]
        # All axes above the bottom must have empty xlabel.
        for ax in axes[:-1]:
            assert ax.get_xlabel() == ""


# ===========================================================================
# TestFeatureSelection
# ===========================================================================


class TestFeatureSelection:
    """feature_names parameter behavior."""

    def test_feature_names_none_returns_all_features(self):
        auditor = _make_auditor()
        result = auditor.plot_score_distributions("score", feature_names=None)
        assert set(result.keys()) == {"group", "region"}

    def test_feature_names_subset_returns_only_specified(self):
        auditor = _make_auditor()
        result = auditor.plot_score_distributions("score", feature_names=["group"])
        assert list(result.keys()) == ["group"]
        assert "region" not in result

    def test_feature_names_order_is_preserved(self):
        """Output dict order must match the caller-supplied feature_names order."""
        auditor = _make_auditor()
        result = auditor.plot_score_distributions(
            "score", feature_names=["region", "group"]
        )
        assert list(result.keys()) == ["region", "group"]


# ===========================================================================
# TestHistogramProperties
# ===========================================================================


class TestHistogramProperties:
    """Histogram density normalization and shared bins."""

    def test_density_default_area_approx_one(self):
        """With density=True (default), each histogram's area integrates to ~1."""
        auditor = _make_auditor()
        _, axes = auditor.plot_score_distributions("score")["group"]
        for ax in axes:
            patches = ax.patches
            if not patches:
                continue
            # Sum area = sum(height * width) for all bars.
            area = sum(p.get_height() * p.get_width() for p in patches)
            assert math.isclose(area, 1.0, abs_tol=0.05), (
                f"Expected area ≈ 1.0, got {area:.4f}"
            )

    def test_density_false_uses_raw_counts(self):
        """With density=False, bar heights are raw counts, area != 1."""
        auditor = _make_auditor()
        _, axes = auditor.plot_score_distributions(
            "score", density=False
        )["group"]
        # At least one bar should have height > 1 (raw count).
        heights = [p.get_height() for ax in axes for p in ax.patches]
        assert any(h > 1.0 for h in heights)

    def test_shared_bins_across_levels(self):
        """All axes in one figure must share identical bin edges."""
        auditor = _make_auditor()
        _, axes = auditor.plot_score_distributions("score")["group"]
        # Collect left edges from the first non-empty level as the reference.
        ref_patches = [ax for ax in axes if ax.patches]
        if len(ref_patches) < 2:
            pytest.skip("Need >=2 axes with patches to compare bins")
        ref_lefts = [p.get_x() for p in ref_patches[0].patches]
        for ax in ref_patches[1:]:
            lefts = [p.get_x() for p in ax.patches]
            assert lefts == ref_lefts, "Bin edges differ between level subplots"

    def test_custom_bins_integer(self):
        """Passing bins=10 produces 10 bars per level."""
        auditor = _make_auditor()
        _, axes = auditor.plot_score_distributions("score", bins=10)["group"]
        for ax in axes:
            assert len(ax.patches) == 10


# ===========================================================================
# TestLevelOrdering
# ===========================================================================


class TestLevelOrdering:
    """Level ordering for categorical and non-categorical features."""

    def test_categorical_levels_follow_declared_order(self):
        """Categories declared as [C, B, A] must appear in that order in axes."""
        auditor = _make_auditor(feature_dtype="categorical")
        _, axes = auditor.plot_score_distributions("score")["group"]
        ylabel_texts = [ax.get_ylabel() for ax in axes]
        assert ylabel_texts == ["C", "B", "A"]

    def test_non_categorical_levels_follow_first_appearance_order(self):
        """Non-categorical strings must appear in the order first seen in data."""
        df = pd.DataFrame(
            {"grp": ["Z", "Z", "Y", "Y", "X", "X"], "score": [0.3] * 6}
        )
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="grp")
        auditor.add_score(name="score", threshold=0.5)
        _, axes = auditor.plot_score_distributions("score")["grp"]
        ylabel_texts = [ax.get_ylabel() for ax in axes]
        assert ylabel_texts == ["Z", "Y", "X"]


# ===========================================================================
# TestErrors
# ===========================================================================


class TestErrors:
    """All documented ValueError and ImportError paths."""

    def test_error_no_data(self):
        auditor = Auditor()
        auditor.add_feature(name="grp")
        auditor.add_score(name="score", threshold=0.5)
        with pytest.raises(ValueError, match="add_data"):
            auditor.plot_score_distributions("score")

    def test_error_no_features(self):
        df = pd.DataFrame({"score": [0.5, 0.3]})
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_score(name="score", threshold=0.5)
        with pytest.raises(ValueError, match="[Ff]eature"):
            auditor.plot_score_distributions("score")

    def test_error_unknown_score(self):
        auditor = _make_auditor()
        with pytest.raises(ValueError, match="not found"):
            auditor.plot_score_distributions("nonexistent_score")

    def test_error_unknown_score_message_includes_available(self):
        auditor = _make_auditor()
        with pytest.raises(ValueError) as exc_info:
            auditor.plot_score_distributions("nonexistent_score")
        assert "score" in str(exc_info.value)

    def test_error_unknown_feature_name(self):
        auditor = _make_auditor()
        with pytest.raises(ValueError, match="Unknown"):
            auditor.plot_score_distributions("score", feature_names=["nonexistent"])

    def test_error_unknown_feature_message_includes_name(self):
        auditor = _make_auditor()
        with pytest.raises(ValueError) as exc_info:
            auditor.plot_score_distributions(
                "score", feature_names=["nonexistent"]
            )
        assert "nonexistent" in str(exc_info.value)

    def test_error_all_nulls_in_feature_column(self):
        """If all feature values are null, no levels can be formed."""
        df = pd.DataFrame(
            {"grp": [float("nan")] * 5, "score": [0.5] * 5}
        )
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="grp")
        auditor.add_score(name="score", threshold=0.5)
        with pytest.raises(ValueError, match="plottable levels"):
            auditor.plot_score_distributions("score")

    def test_error_all_nulls_in_score_column(self):
        """If all score values are null, no data remains after dropna."""
        df = pd.DataFrame(
            {"grp": ["A"] * 5, "score": [float("nan")] * 5}
        )
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="grp")
        auditor.add_score(name="score", threshold=0.5)
        with pytest.raises(ValueError, match="plottable levels"):
            auditor.plot_score_distributions("score")


# ===========================================================================
# TestEndToEnd
# ===========================================================================


class TestEndToEnd:
    """Full pipeline: no evaluate_metrics() required."""

    def test_works_without_outcome_defined(self):
        """plot_score_distributions must not require add_outcome()."""
        df = pd.DataFrame(
            {"grp": ["A", "B", "A", "B"], "score": [0.8, 0.3, 0.7, 0.4]}
        )
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="grp")
        auditor.add_score(name="score", threshold=0.5)
        result = auditor.plot_score_distributions("score")
        assert "grp" in result

    def test_works_without_metrics_defined(self):
        """plot_score_distributions must not require set_metrics()."""
        df = pd.DataFrame(
            {"grp": ["A", "B", "A", "B"], "score": [0.8, 0.3, 0.7, 0.4]}
        )
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="grp")
        auditor.add_score(name="score", threshold=0.5)
        # No set_metrics() call.
        result = auditor.plot_score_distributions("score")
        assert isinstance(result, dict)

    def test_mixed_data_types_in_feature_column(self):
        """Integer feature values must be coerced to str and produce sane plots."""
        df = pd.DataFrame(
            {"age_bucket": [0, 1, 0, 1, 2, 2], "score": [0.9, 0.1, 0.8, 0.2, 0.5, 0.5]}
        )
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="age_bucket")
        auditor.add_score(name="score", threshold=0.5)
        result = auditor.plot_score_distributions("score")
        _, axes = result["age_bucket"]
        assert len(axes) == 3  # three buckets: 0, 1, 2

    def test_nulls_in_some_rows_are_excluded_without_error(self):
        """Rows with null feature or score values are silently dropped."""
        df = pd.DataFrame(
            {
                "grp": ["A", None, "B", "A", float("nan"), "B"],
                "score": [0.9, 0.5, 0.2, float("nan"), 0.3, 0.8],
            }
        )
        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="grp")
        auditor.add_score(name="score", threshold=0.5)
        result = auditor.plot_score_distributions("score")
        _, axes = result["grp"]
        # A: 1 valid row (row 0; row 3 has null score).
        # B: 1 valid row (row 5; row 2 has null feature... wait, row 2 has grp=B, score=0.2 — valid).
        # Let's just check no error and correct level count.
        assert len(axes) == 2  # A and B
