"""Tests for feature-level ordering in DataFrame exports.

Coverage:
- Categorical feature columns produce rows in declared category order.
- Unobserved categories appear as placeholder rows with NaN metric scores.
- Non-categorical features retain existing (insertion-order) behaviour.
- Both to_dataframe() and style_dataframe() honour the category order.
- ScoreEvaluation-level exports inherit the same feature-level order.
- Bootstrap CI computation is unaffected (observed groups only; placeholders
  keep None intervals).
"""

import math

import numpy as np
import pandas as pd
import pytest

from model_auditor import Auditor
from model_auditor.metrics import Sensitivity, Specificity, nData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORIES = ["C", "B", "A", "D"]  # custom non-alphabetical order; 'D' unobserved by default


def _make_df(include_d: bool = False) -> pd.DataFrame:
    """Return a synthetic binary-classification DataFrame.

    Groups A, B, C are always present.  Group D is included only when
    include_d=True.  The 'group' column is categorical with declared order
    ['C', 'B', 'A', 'D'].
    """
    rows = []
    for grp, n_pos, n_neg in [("A", 3, 2), ("B", 2, 3), ("C", 4, 1)]:
        rows += [{"group": grp, "score": 0.8, "label": 1}] * n_pos
        rows += [{"group": grp, "score": 0.2, "label": 0}] * n_neg
    if include_d:
        rows += [{"group": "D", "score": 0.8, "label": 1}] * 2
        rows += [{"group": "D", "score": 0.2, "label": 0}] * 2

    df = pd.DataFrame(rows)
    df["group"] = pd.Categorical(df["group"], categories=CATEGORIES, ordered=True)
    return df


def _make_auditor(df: pd.DataFrame, metrics=None) -> Auditor:
    if metrics is None:
        metrics = [Sensitivity(), nData()]
    auditor = Auditor()
    auditor.add_data(df)
    auditor.add_feature(name="group")
    auditor.add_score(name="score", threshold=0.5)
    auditor.add_outcome(name="label")
    auditor.set_metrics(metrics)
    return auditor


def _evaluate(include_d: bool = False, n_bootstraps=None, metrics=None):
    df = _make_df(include_d=include_d)
    return _make_auditor(df, metrics=metrics).evaluate_metrics(score_name="score", n_bootstraps=n_bootstraps)


# ---------------------------------------------------------------------------
# TestCategoricalLevelOrder — ordering and placeholders (primary feature)
# ---------------------------------------------------------------------------


class TestCategoricalLevelOrder:
    """Categorical feature columns follow declared category order in all exports."""

    # -- levels dict order ---------------------------------------------------

    def test_levels_dict_order_matches_declared_categories(self):
        """FeatureEvaluation.levels preserves declared categorical order."""
        results = _evaluate()
        keys = list(results.features["group"].levels.keys())
        assert keys == CATEGORIES, f"Expected {CATEGORIES}, got {keys}"

    # -- to_dataframe order --------------------------------------------------

    def test_feature_to_dataframe_row_order(self):
        """FeatureEvaluation.to_dataframe() index follows declared category order."""
        results = _evaluate()
        df = results.features["group"].to_dataframe()
        assert list(df.index) == CATEGORIES, (
            f"Expected {CATEGORIES}, got {list(df.index)}"
        )

    def test_score_to_dataframe_group_row_order(self):
        """ScoreEvaluation.to_dataframe() group rows follow declared category order."""
        results = _evaluate()
        df = results.to_dataframe()
        # MultiIndex: level 0 = feature label, level 1 = level name
        group_df = df.loc["group"]
        assert list(group_df.index) == CATEGORIES, (
            f"Expected {CATEGORIES}, got {list(group_df.index)}"
        )

    # -- style_dataframe order -----------------------------------------------

    def test_feature_style_dataframe_row_order(self):
        """FeatureEvaluation.style_dataframe() Styler.data index follows category order."""
        results = _evaluate()
        styler = results.features["group"].style_dataframe()
        assert list(styler.data.index) == CATEGORIES, (
            f"Expected {CATEGORIES}, got {list(styler.data.index)}"
        )

    def test_score_style_dataframe_group_row_order(self):
        """ScoreEvaluation.style_dataframe() group rows follow declared category order."""
        results = _evaluate()
        styler = results.style_dataframe()
        group_rows = styler.data.loc["group"]
        assert list(group_rows.index) == CATEGORIES, (
            f"Expected {CATEGORIES}, got {list(group_rows.index)}"
        )


# ---------------------------------------------------------------------------
# TestUnobservedCategoryPlaceholders
# ---------------------------------------------------------------------------


class TestUnobservedCategoryPlaceholders:
    """Categories declared but absent in data appear as NaN rows."""

    def test_unobserved_category_row_exists(self):
        """Row for 'D' (unobserved) is present in to_dataframe() output."""
        results = _evaluate()
        df = results.features["group"].to_dataframe()
        assert "D" in df.index, "Missing row for unobserved category 'D'"

    def test_unobserved_category_raw_metrics_are_nan(self):
        """LevelEvaluation for unobserved 'D' has NaN scores for every metric."""
        results = _evaluate()
        level_d = results.features["group"].levels["D"]
        assert len(level_d.metrics) > 0, (
            "Placeholder level 'D' must have metric entries (not an empty dict)"
        )
        for name, lm in level_d.metrics.items():
            assert math.isnan(lm.score), (
                f"Metric '{name}' for unobserved 'D' must be NaN, got {lm.score}"
            )

    def test_unobserved_category_no_confidence_interval(self):
        """Placeholder level for 'D' must not carry a confidence interval."""
        results = _evaluate()
        level_d = results.features["group"].levels["D"]
        for name, lm in level_d.metrics.items():
            assert lm.interval is None, (
                f"Metric '{name}' for unobserved 'D' should have no CI, got {lm.interval}"
            )

    def test_unobserved_category_row_in_score_dataframe(self):
        """Row for 'D' appears in ScoreEvaluation.to_dataframe() under 'group'."""
        results = _evaluate()
        df = results.to_dataframe()
        group_df = df.loc["group"]
        assert "D" in group_df.index

    def test_observed_categories_have_expected_metric_values(self):
        """Observed categories have exact known metric values."""
        results = _evaluate()
        feature = results.features["group"]

        expected_n = {"A": 5, "B": 5, "C": 5}
        for cat in ("A", "B", "C"):
            level = feature.levels[cat]
            assert level.metrics["sensitivity"].score == pytest.approx(1.0)
            assert level.metrics["n"].score == expected_n[cat]

    def test_placeholder_metric_set_matches_observed_levels(self):
        """Placeholder 'D' has the same set of metric names as an observed level."""
        results = _evaluate()
        feature = results.features["group"]
        observed_metric_names = set(feature.levels["A"].metrics.keys())
        placeholder_metric_names = set(feature.levels["D"].metrics.keys())
        assert placeholder_metric_names == observed_metric_names, (
            f"Placeholder metric names {placeholder_metric_names} != observed {observed_metric_names}"
        )


# ---------------------------------------------------------------------------
# TestAllCategoriesObserved
# ---------------------------------------------------------------------------


class TestAllCategoriesObserved:
    """When every declared category has rows, ordering still follows declaration."""

    def test_all_observed_level_order(self):
        results = _evaluate(include_d=True)
        keys = list(results.features["group"].levels.keys())
        assert keys == CATEGORIES

    def test_all_observed_levels_have_expected_values(self):
        """When D is observed, every level has exact expected metric values."""
        results = _evaluate(include_d=True)
        feature = results.features["group"]

        expected_n = {"A": 5, "B": 5, "C": 5, "D": 4}
        for cat in CATEGORIES:
            level = feature.levels[cat]
            assert level.metrics["sensitivity"].score == pytest.approx(1.0)
            assert level.metrics["n"].score == expected_n[cat]


# ---------------------------------------------------------------------------
# TestBootstrapCIWithCategorical
# ---------------------------------------------------------------------------


class TestBootstrapCIWithCategorical:
    """CI computation runs only on observed groups; placeholders stay CI-free."""

    def test_observed_groups_get_ci_for_eligible_metrics(self):
        """Observed levels receive confidence intervals for ci_eligible metrics."""
        results = _evaluate(n_bootstraps=50)
        feature = results.features["group"]
        for cat in ("A", "B", "C"):
            sensitivity_metric = feature.levels[cat].metrics.get("sensitivity")
            assert sensitivity_metric is not None
            assert sensitivity_metric.interval is not None, (
                f"Observed '{cat}' should have a CI for Sensitivity"
            )

    def test_unobserved_placeholder_has_no_ci(self):
        """The NaN placeholder for 'D' carries no CI even when bootstraps are run."""
        results = _evaluate(n_bootstraps=50)
        level_d = results.features["group"].levels["D"]
        for name, lm in level_d.metrics.items():
            assert lm.interval is None, (
                f"Placeholder '{name}' for unobserved 'D' must not have a CI"
            )

    def test_count_metric_has_no_ci(self):
        """nData (ci_eligible=False) has no interval for any observed level."""
        results = _evaluate(n_bootstraps=50)
        feature = results.features["group"]
        for cat in ("A", "B", "C"):
            n_metric = feature.levels[cat].metrics.get("n")
            assert n_metric is not None
            assert n_metric.interval is None, (
                f"nData should never carry a CI, got {n_metric.interval}"
            )


# ---------------------------------------------------------------------------
# TestNonCategoricalPreservation
# ---------------------------------------------------------------------------


class TestNonCategoricalPreservation:
    """Non-categorical feature columns retain existing string-based behaviour."""

    def _results_string_feature(self):
        df = _make_df()
        # Strip categorical dtype → plain object/string column
        df["group"] = df["group"].astype(str)
        return _make_auditor(df).evaluate_metrics(score_name="score", n_bootstraps=None)

    def test_non_categorical_evaluates_without_error(self):
        """String-typed feature evaluates successfully."""
        results = self._results_string_feature()
        assert "group" in results.features

    def test_non_categorical_no_unobserved_placeholders(self):
        """No phantom row for 'D' when the feature is not categorical."""
        results = self._results_string_feature()
        assert "D" not in results.features["group"].levels, (
            "Non-categorical feature must not generate phantom 'D' row"
        )

    def test_non_categorical_observed_levels_only(self):
        """Only A, B, C appear as levels for the non-categorical feature."""
        results = self._results_string_feature()
        assert set(results.features["group"].levels.keys()) == {"A", "B", "C"}

    def test_non_categorical_metrics_match_expected_values(self):
        """Non-categorical feature keeps the same deterministic metric values."""
        results = self._results_string_feature()
        expected_n = {"A": 5, "B": 5, "C": 5}
        for cat in ("A", "B", "C"):
            level = results.features["group"].levels[cat]
            assert level.metrics["sensitivity"].score == pytest.approx(1.0)
            assert level.metrics["n"].score == expected_n[cat]


# ---------------------------------------------------------------------------
# TestMultipleMetrics
# ---------------------------------------------------------------------------


class TestMultipleMetrics:
    """Ordering and placeholders work across a richer metric set."""

    def test_multiple_metrics_preserve_category_order(self):
        """All metrics populate in category order; placeholders cover every metric."""
        results = _evaluate(metrics=[Sensitivity(), Specificity(), nData()])
        keys = list(results.features["group"].levels.keys())
        assert keys == CATEGORIES

        # Placeholder 'D' has one entry per declared metric
        metric_names = {"sensitivity", "specificity", "n"}
        placeholder_names = set(results.features["group"].levels["D"].metrics.keys())
        assert placeholder_names == metric_names
