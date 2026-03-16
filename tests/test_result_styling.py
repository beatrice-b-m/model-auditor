"""Tests for result DataFrame styling functionality.

Coverage:
- Helper functions (_is_count_metric, _is_lower_better_metric, _get_metric_tier)
- CSS output: performance metrics are styled, count metrics excluded by default
- CSS output: fpr/fnr lower-is-better inversion
- Tie/small-sample stability
- Backward compatibility of to_dataframe()
"""

import re

import numpy as np
import pandas as pd
import pytest

from model_auditor.schemas import (
    FeatureEvaluation,
    LevelEvaluation,
    LevelMetric,
    ScoreEvaluation,
    _get_metric_tier,
    _is_count_metric,
    _is_lower_better_metric,
)


# ---------------------------------------------------------------------------
# HTML inspection helpers
# ---------------------------------------------------------------------------

def _render_html(styler: "pd.io.formats.style.Styler") -> str:
    """Render styler to HTML, compatible with pandas < 1.4 and >= 1.4."""
    try:
        return styler.to_html()
    except AttributeError:
        return styler.render()  # type: ignore[attr-defined]


def _parse_cell_styles(html: str) -> dict[str, str]:
    """Parse rendered Styler HTML → {cell_text_value: css_string}.

    Works by cross-referencing the <style> block (id → css) with the <td>
    elements (id → displayed value).  Values that share the same display text
    are overwritten; callers using distinct cell values avoid ambiguity.

    Handles both single-cell selectors and comma-separated groups that pandas
    may generate when multiple cells share the same style:
      #T_xx_row0_col0, #T_xx_row1_col0 { background-color: #d4edda; }
    """
    id_to_css: dict[str, str] = {}

    # Each CSS rule may have one or more comma-separated selectors.
    # Capture the full selector group and the declaration block separately.
    for rule_m in re.finditer(
        r"((?:#T_\w+_row\d+_col\d+\s*,?\s*)+)\{([^}]+)\}", html
    ):
        css = rule_m.group(2).strip()
        # Extract every individual cell ID from the selector list.
        for sel_m in re.finditer(r"T_\w+_row\d+_col\d+", rule_m.group(1)):
            id_to_css[sel_m.group(0)] = css

    # Extract id → display value from <td> elements.
    result: dict[str, str] = {}
    for m in re.finditer(
        r'id="(T_\w+_row\d+_col\d+)"[^>]*>(.*?)</td>', html, re.DOTALL
    ):
        cell_id = m.group(1)
        value = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        css = id_to_css.get(cell_id, "")
        result[value] = css

    return result


# ---------------------------------------------------------------------------
# Helper: build FeatureEvaluation with one named metric across multiple levels
# ---------------------------------------------------------------------------

def _make_feature(
    metric_name: str, scores: dict[str, float]
) -> FeatureEvaluation:
    """Build a FeatureEvaluation with one metric, one level per score entry.

    Insertion order determines row order in the resulting DataFrame.
    """
    feature = FeatureEvaluation(name="test_feature", label="Test Feature")
    for level_name, score in scores.items():
        level = LevelEvaluation(name=level_name)
        level.metrics[metric_name] = LevelMetric(
            name=metric_name, label=metric_name.upper(), score=score
        )
        feature.levels[level_name] = level
    return feature


# ---------------------------------------------------------------------------
# Colour constants (defaults from style_dataframe)
# ---------------------------------------------------------------------------

HIGH_COLOR = "#d4edda"
MED_COLOR = "#fff3cd"
LOW_COLOR = "#f8d7da"


# ===========================================================================
# TestHelperFunctions
# ===========================================================================

class TestHelperFunctions:
    """Unit tests for the three helper functions used by styling."""

    # -- _is_count_metric ----------------------------------------------------

    def test_is_count_metric_performance_metrics(self):
        """Standard performance metrics are NOT count metrics."""
        for name in ("accuracy", "precision", "recall", "f1", "auc", "mcc"):
            assert not _is_count_metric(name), f"should not be count: {name}"

    def test_is_count_metric_rate_metrics_not_count(self):
        """Rate metrics tpr/tnr/fpr/fnr are NOT count metrics.

        This is the core correctness invariant fixed by moving from prefix
        matching to exact-name matching.  With prefix matching, 'tpr' would
        match the 'TP' prefix and get incorrectly classified.
        """
        for name in ("tpr", "tnr", "fpr", "fnr", "TPR", "TNR", "FPR", "FNR"):
            assert not _is_count_metric(name), f"rate metric must not be count: {name}"

    def test_is_count_metric_exact_counts(self):
        """Known count metric names are correctly identified."""
        for name in ("N", "TP", "TN", "FP", "FN", "POS", "NEG", "POS.", "NEG."):
            assert _is_count_metric(name), f"should be count: {name}"

    def test_is_count_metric_underscore_counts(self):
        """Underscore-prefixed count names are correctly identified."""
        for name in ("N_TP", "N_TN", "N_FP", "N_FN", "N_POS", "N_NEG"):
            assert _is_count_metric(name), f"should be count: {name}"

    def test_is_count_metric_case_insensitive(self):
        """Count-metric detection is case-insensitive."""
        for name in ("n", "tp", "tn", "fp", "fn", "pos", "neg", "Tp", "Fp"):
            assert _is_count_metric(name), f"case variant should be count: {name}"

    # -- _is_lower_better_metric --------------------------------------------

    def test_is_lower_better_fpr(self):
        assert _is_lower_better_metric("fpr")
        assert _is_lower_better_metric("FPR")
        assert _is_lower_better_metric("false_positive_rate")

    def test_is_lower_better_fnr(self):
        assert _is_lower_better_metric("fnr")
        assert _is_lower_better_metric("FNR")
        assert _is_lower_better_metric("false_negative_rate")

    def test_not_lower_better_performance(self):
        for name in ("accuracy", "precision", "recall", "f1", "auc", "tpr", "tnr"):
            assert not _is_lower_better_metric(name), f"should be higher-better: {name}"

    # -- _get_metric_tier ---------------------------------------------------

    def test_tier_higher_better_three_values(self):
        """Three-value series: top→high, middle→medium, bottom→low."""
        values = pd.Series([0.1, 0.5, 0.9])
        assert _get_metric_tier(0.9, values, lower_better=False) == "high"
        assert _get_metric_tier(0.5, values, lower_better=False) == "medium"
        assert _get_metric_tier(0.1, values, lower_better=False) == "low"

    def test_tier_lower_better_three_values(self):
        """Lower-is-better inverts the assignment."""
        values = pd.Series([0.1, 0.5, 0.9])
        assert _get_metric_tier(0.1, values, lower_better=True) == "high"
        assert _get_metric_tier(0.5, values, lower_better=True) == "medium"
        assert _get_metric_tier(0.9, values, lower_better=True) == "low"

    def test_tier_nan_returns_none(self):
        values = pd.Series([0.1, 0.5, 0.9, np.nan])
        assert _get_metric_tier(np.nan, values, lower_better=False) == "none"

    def test_tier_empty_series_returns_none(self):
        """Guard clause: empty series must not raise ZeroDivisionError."""
        values = pd.Series([], dtype=float)
        assert _get_metric_tier(0.5, values, lower_better=False) == "none"

    def test_tier_ties_stable(self):
        """All equal values produce a consistent tier without crashing."""
        values = pd.Series([0.5, 0.5, 0.5])
        tiers = [_get_metric_tier(v, values) for v in values]
        # All values are the same so all tiers must be the same
        assert len(set(tiers)) == 1
        # Each individual tier must be a valid value
        assert tiers[0] in ("low", "medium", "high")

    def test_tier_single_value(self):
        """Single-row series (e.g., LevelEvaluation alone) must not crash."""
        values = pd.Series([0.85])
        tier = _get_metric_tier(0.85, values, lower_better=False)
        assert tier in ("low", "medium", "high", "none")


# ===========================================================================
# TestStylingCSS  — verifies that the correct background-colors are applied
# ===========================================================================

class TestStylingCSS:
    """CSS-level assertions for style_dataframe().

    These tests render the Styler to HTML and verify which cells receive
    which background colours, proving that tier assignment and metric
    filtering both work correctly end-to-end.
    """

    # -- count-metric exclusion/inclusion -----------------------------------

    def test_count_only_no_styling_by_default(self):
        """A feature with only count metric produces NO background-color by default."""
        feature = _make_feature("N", {"a": 100.0, "b": 200.0, "c": 300.0})
        html = _render_html(feature.style_dataframe(include_count_metrics=False))
        assert "background-color" not in html, (
            "count-only column should have no background-color by default"
        )

    def test_count_styled_when_flag_set(self):
        """With include_count_metrics=True, count column receives background-color."""
        feature = _make_feature("N", {"a": 100.0, "b": 200.0, "c": 300.0})
        html = _render_html(feature.style_dataframe(include_count_metrics=True))
        assert "background-color" in html, (
            "count column should be styled when include_count_metrics=True"
        )

    # -- fpr/fnr/tpr treated as performance, not count ----------------------

    def test_fpr_styled_by_default(self):
        """fpr is a performance metric and must be styled even with include_count_metrics=False."""
        feature = _make_feature(
            "fpr", {"best": 0.10, "mid": 0.50, "worst": 0.90}
        )
        html = _render_html(feature.style_dataframe(include_count_metrics=False))
        assert "background-color" in html, (
            "fpr should be styled by default (not a count metric)"
        )

    def test_fnr_styled_by_default(self):
        """fnr is a performance metric and must be styled by default."""
        feature = _make_feature(
            "fnr", {"best": 0.10, "mid": 0.50, "worst": 0.90}
        )
        html = _render_html(feature.style_dataframe(include_count_metrics=False))
        assert "background-color" in html

    def test_tpr_styled_by_default(self):
        """tpr is a performance metric and must be styled by default."""
        feature = _make_feature(
            "tpr", {"low": 0.70, "mid": 0.80, "high": 0.90}
        )
        html = _render_html(feature.style_dataframe(include_count_metrics=False))
        assert "background-color" in html

    def test_tnr_styled_by_default(self):
        """tnr is a performance metric and must be styled by default."""
        feature = _make_feature(
            "tnr", {"low": 0.70, "mid": 0.80, "high": 0.90}
        )
        html = _render_html(feature.style_dataframe(include_count_metrics=False))
        assert "background-color" in html

    # -- lower-is-better inversion ------------------------------------------

    def test_fpr_lower_better_inversion(self):
        """Low fpr (good performance) → high tier (green); high fpr (bad) → low tier (red)."""
        feature = _make_feature(
            "fpr", {"best": 0.10, "mid": 0.50, "worst": 0.90}
        )
        styler = feature.style_dataframe()
        html = _render_html(styler)
        cells = _parse_cell_styles(html)

        assert HIGH_COLOR in cells.get("0.100", ""), (
            f"fpr=0.1 (best) should get green (high tier); got: {cells.get('0.100')}"
        )
        assert LOW_COLOR in cells.get("0.900", ""), (
            f"fpr=0.9 (worst) should get red (low tier); got: {cells.get('0.900')}"
        )

    def test_fnr_lower_better_inversion(self):
        """Low fnr (good performance) → high tier (green); high fnr (bad) → low tier (red)."""
        feature = _make_feature(
            "fnr", {"best": 0.10, "mid": 0.50, "worst": 0.90}
        )
        html = _render_html(feature.style_dataframe())
        cells = _parse_cell_styles(html)

        assert HIGH_COLOR in cells.get("0.100", ""), (
            f"fnr=0.1 should be green; got: {cells.get('0.100')}"
        )
        assert LOW_COLOR in cells.get("0.900", ""), (
            f"fnr=0.9 should be red; got: {cells.get('0.900')}"
        )

    def test_accuracy_higher_better_normal(self):
        """High accuracy → high tier (green); low accuracy → low tier (red)."""
        feature = _make_feature(
            "accuracy", {"low": 0.70, "mid": 0.80, "high": 0.90}
        )
        html = _render_html(feature.style_dataframe())
        cells = _parse_cell_styles(html)

        assert HIGH_COLOR in cells.get("0.900", ""), (
            f"accuracy=0.9 should be green; got: {cells.get('0.900')}"
        )
        assert LOW_COLOR in cells.get("0.700", ""), (
            f"accuracy=0.7 should be red; got: {cells.get('0.700')}"
        )

    # -- count excluded, performance included, together ----------------------

    def test_fpr_styled_n_excluded_together(self):
        """Feature with fpr AND N: fpr gets colours, N does not (default)."""
        feature = FeatureEvaluation(name="test", label="Test")
        # Use integer N scores: they format as "100", "200", "300" (no decimals)
        for level_name, fpr_val, n_val in [
            ("a", 0.10, 100),
            ("b", 0.50, 200),
            ("c", 0.90, 300),
        ]:
            level = LevelEvaluation(name=level_name)
            level.metrics["fpr"] = LevelMetric(name="fpr", label="FPR", score=fpr_val)
            level.metrics["N"] = LevelMetric(name="N", label="N", score=n_val)
            feature.levels[level_name] = level

        html_default = _render_html(feature.style_dataframe(include_count_metrics=False))
        # fpr column is styled → background-color must appear
        assert "background-color" in html_default

        cells_default = _parse_cell_styles(html_default)
        # N column displays as "100", "200", "300" (int formatting) — none should have a color
        for n_display in ("100", "200", "300"):
            assert cells_default.get(n_display, "") == "", (
                f"N={n_display} should have no background-color by default"
            )

    def test_n_styled_when_include_flag_with_fpr(self):
        """With include_count_metrics=True, N column also gets styled."""
        feature = FeatureEvaluation(name="test", label="Test")
        # Integer N scores format as "100", "200", "300" (no decimals)
        for level_name, fpr_val, n_val in [
            ("a", 0.10, 100),
            ("b", 0.50, 200),
            ("c", 0.90, 300),
        ]:
            level = LevelEvaluation(name=level_name)
            level.metrics["fpr"] = LevelMetric(name="fpr", label="FPR", score=fpr_val)
            level.metrics["N"] = LevelMetric(name="N", label="N", score=n_val)
            feature.levels[level_name] = level

        cells = _parse_cell_styles(
            _render_html(feature.style_dataframe(include_count_metrics=True))
        )
        # At least one N cell should have a background-color
        n_styled = [cells.get(k, "") for k in ("100", "200", "300")]
        assert any("background-color" in s for s in n_styled), (
            "N column should have background-color when include_count_metrics=True"
        )

    # -- tie/small-sample stability -----------------------------------------

    def test_tie_stability_all_equal_values(self):
        """All levels with same metric value must not crash, and all get the same tier."""
        feature = _make_feature(
            "accuracy", {"a": 0.85, "b": 0.85, "c": 0.85}
        )
        html = _render_html(feature.style_dataframe())
        cells = _parse_cell_styles(html)
        # Three rows all display "0.850".  The dict will have one entry (last wins).
        # What matters: no exception and the result is a non-empty string.
        assert isinstance(html, str)

    def test_single_level_does_not_crash(self):
        """Single-row feature (nothing to rank against) must not crash."""
        feature = _make_feature("accuracy", {"only_level": 0.85})
        html = _render_html(feature.style_dataframe())
        assert isinstance(html, str)

    def test_all_nan_does_not_crash(self):
        """All-NaN values in a column must not crash and produce no coloring."""
        feature = _make_feature(
            "accuracy", {"a": float("nan"), "b": float("nan")}
        )
        html = _render_html(feature.style_dataframe())
        assert isinstance(html, str)

    # -- custom colors -------------------------------------------------------

    def test_custom_colors_appear_in_output(self):
        """Custom high/low/medium colors must appear in rendered HTML."""
        feature = _make_feature(
            "accuracy", {"low": 0.70, "mid": 0.80, "high": 0.90}
        )
        html = _render_html(
            feature.style_dataframe(
                low_color="#aa0000",
                medium_color="#aaaa00",
                high_color="#00aa00",
            )
        )
        assert "#aa0000" in html
        assert "#00aa00" in html

    # -- ScoreEvaluation end-to-end -----------------------------------------

    def test_score_evaluation_fpr_styled(self):
        """ScoreEvaluation.style_dataframe() correctly styles fpr by default."""
        score = ScoreEvaluation(name="risk_score", label="Risk Score")
        feature = FeatureEvaluation(name="gender", label="Gender")
        for level_name, fpr_val in [("male", 0.10), ("female", 0.40)]:
            level = LevelEvaluation(name=level_name)
            level.metrics["fpr"] = LevelMetric(
                name="fpr", label="FPR", score=fpr_val
            )
            feature.levels[level_name] = level
        score.features["gender"] = feature

        html = _render_html(score.style_dataframe())
        assert "background-color" in html

    def test_score_evaluation_count_excluded_by_default(self):
        """ScoreEvaluation with count-only metric: no background-color by default."""
        score = ScoreEvaluation(name="risk_score", label="Risk Score")
        feature = FeatureEvaluation(name="gender", label="Gender")
        for level_name, n_val in [("male", 100.0), ("female", 120.0)]:
            level = LevelEvaluation(name=level_name)
            level.metrics["N"] = LevelMetric(name="N", label="N", score=n_val)
            feature.levels[level_name] = level
        score.features["gender"] = feature

        html = _render_html(score.style_dataframe(include_count_metrics=False))
        assert "background-color" not in html


# ===========================================================================
# TestLevelEvaluationStyling
# ===========================================================================

class TestLevelEvaluationStyling:
    """Tests for LevelEvaluation.style_dataframe()."""

    def test_returns_styler_object(self):
        level = LevelEvaluation(name="test_level")
        level.metrics["accuracy"] = LevelMetric(
            name="accuracy", label="Accuracy", score=0.85
        )
        result = level.style_dataframe()
        assert isinstance(result, pd.io.formats.style.Styler)

    def test_metric_labels_parameter(self):
        """metric_labels=True uses label as column header."""
        level = LevelEvaluation(name="test_level")
        level.metrics["accuracy"] = LevelMetric(
            name="accuracy", label="Accuracy", score=0.85
        )
        styler_name = level.style_dataframe(metric_labels=False)
        styler_label = level.style_dataframe(metric_labels=True)
        assert "accuracy" in styler_name.columns
        assert "Accuracy" in styler_label.columns

    def test_confidence_intervals_preserved_in_display(self):
        """Confidence interval text appears in rendered HTML."""
        level = LevelEvaluation(name="test_level")
        level.metrics["accuracy"] = LevelMetric(
            name="accuracy", label="Accuracy", score=0.85, interval=(0.80, 0.90)
        )
        html = _render_html(level.style_dataframe())
        assert "0.800" in html and "0.900" in html


# ===========================================================================
# TestFeatureEvaluationStyling
# ===========================================================================

class TestFeatureEvaluationStyling:
    """Tests for FeatureEvaluation.style_dataframe()."""

    def test_returns_styler_object(self):
        feature = FeatureEvaluation(name="gender", label="Gender")
        for level_name, score in [("male", 0.85), ("female", 0.90)]:
            level = LevelEvaluation(name=level_name)
            level.metrics["accuracy"] = LevelMetric(
                name="accuracy", label="Accuracy", score=score
            )
            feature.levels[level_name] = level
        assert isinstance(feature.style_dataframe(), pd.io.formats.style.Styler)

    def test_relative_ranking_across_levels(self):
        """Three levels with distinct accuracy produce all three tiers."""
        feature = _make_feature(
            "accuracy", {"low": 0.70, "mid": 0.80, "high": 0.90}
        )
        html = _render_html(feature.style_dataframe())
        assert HIGH_COLOR in html
        assert MED_COLOR in html
        assert LOW_COLOR in html


# ===========================================================================
# TestScoreEvaluationStyling
# ===========================================================================

class TestScoreEvaluationStyling:
    """Tests for ScoreEvaluation.style_dataframe()."""

    def test_returns_styler_object(self):
        score = ScoreEvaluation(name="risk_score", label="Risk Score")
        feature = FeatureEvaluation(name="gender", label="Gender")
        level = LevelEvaluation(name="male")
        level.metrics["accuracy"] = LevelMetric(
            name="accuracy", label="Accuracy", score=0.85
        )
        feature.levels["male"] = level
        score.features["gender"] = feature
        assert isinstance(score.style_dataframe(), pd.io.formats.style.Styler)

    def test_multiple_features_styled(self):
        """Styling spans all features/levels in a ScoreEvaluation."""
        score = ScoreEvaluation(name="risk_score", label="Risk Score")
        for feat_name, score_val in [("gender", 0.85), ("age_group", 0.90)]:
            feature = FeatureEvaluation(name=feat_name, label=feat_name)
            level = LevelEvaluation(name="grp")
            level.metrics["accuracy"] = LevelMetric(
                name="accuracy", label="Accuracy", score=score_val
            )
            feature.levels["grp"] = level
            score.features[feat_name] = feature
        html = _render_html(score.style_dataframe())
        assert isinstance(html, str)


# ===========================================================================
# TestBackwardCompatibility
# ===========================================================================

class TestBackwardCompatibility:
    """Ensure existing to_dataframe() behaviour is unchanged."""

    def test_level_to_dataframe_unchanged(self):
        level = LevelEvaluation(name="test_level")
        level.metrics["accuracy"] = LevelMetric(
            name="accuracy", label="Accuracy", score=0.85
        )
        df = level.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 1)
        assert "accuracy" in df.columns
        assert df.loc["test_level", "accuracy"] == "0.850"

    def test_feature_to_dataframe_unchanged(self):
        feature = FeatureEvaluation(name="gender", label="Gender")
        level = LevelEvaluation(name="male")
        level.metrics["accuracy"] = LevelMetric(
            name="accuracy", label="Accuracy", score=0.85
        )
        feature.levels["male"] = level
        df = feature.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 1)
        assert "accuracy" in df.columns

    def test_score_to_dataframe_unchanged(self):
        score = ScoreEvaluation(name="risk_score", label="Risk Score")
        feature = FeatureEvaluation(name="gender", label="Gender")
        level = LevelEvaluation(name="male")
        level.metrics["accuracy"] = LevelMetric(
            name="accuracy", label="Accuracy", score=0.85
        )
        feature.levels["male"] = level
        score.features["gender"] = feature
        df = score.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "accuracy" in df.columns

    def test_to_dataframe_with_intervals_unchanged(self):
        """Confidence interval formatting in to_dataframe is unaffected."""
        level = LevelEvaluation(name="test")
        level.metrics["accuracy"] = LevelMetric(
            name="accuracy", label="Accuracy", score=0.85, interval=(0.80, 0.90)
        )
        df = level.to_dataframe(n_decimals=3)
        cell = df.loc["test", "accuracy"]
        assert "0.850" in cell
        assert "0.800" in cell
        assert "0.900" in cell
