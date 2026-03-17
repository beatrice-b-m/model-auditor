"""Tests for ScoreEvaluation.plot_metric_intervals.

Coverage:
- Private helper functions (_is_plottable_level, _resolve_metric_key,
  _get_metric_display_label).
- Success paths: name match, label match, feature_names=None, subset,
  level exclusion (NaN score / None interval), output dict order.
- Failure paths: empty features, unknown feature, unknown metric,
  all-excluded levels.
- End-to-end integration via a full Auditor + bootstrap evaluation.
"""

import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede any plt import

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pytest

from model_auditor.schemas import (
    FeatureEvaluation,
    LevelEvaluation,
    LevelMetric,
    ScoreEvaluation,
    _get_metric_display_label,
    _is_plottable_level,
    _resolve_metric_key,
)


# ---------------------------------------------------------------------------
# Shared fixture: close all figures after every test to avoid memory leaks
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_level(
    name: str,
    score: float,
    interval: "tuple[float, float] | None" = None,
) -> LevelEvaluation:
    """Build a LevelEvaluation with one metric (sensitivity)."""
    leval = LevelEvaluation(name=name)
    leval.metrics["sensitivity"] = LevelMetric(
        name="sensitivity",
        label="Sensitivity",
        score=score,
        interval=interval,
    )
    return leval


def _make_feature(
    name: str,
    label: str,
    levels: "dict[str, LevelEvaluation]",
) -> FeatureEvaluation:
    """Build a FeatureEvaluation from a {level_name: LevelEvaluation} dict."""
    feval = FeatureEvaluation(name=name, label=label)
    feval.levels.update(levels)
    return feval


def _make_score_eval(features: "dict[str, FeatureEvaluation]") -> ScoreEvaluation:
    """Build a ScoreEvaluation from a {feature_name: FeatureEvaluation} dict."""
    seval = ScoreEvaluation(name="score", label="Score")
    seval.features.update(features)
    return seval


def _two_feature_score_eval() -> ScoreEvaluation:
    """Score evaluation with two features, all levels having finite CI bounds."""
    return _make_score_eval(
        {
            "sex": _make_feature(
                "sex",
                "Sex",
                {
                    "M": _make_level("M", 0.85, (0.80, 0.90)),
                    "F": _make_level("F", 0.78, (0.72, 0.84)),
                },
            ),
            "age": _make_feature(
                "age",
                "Age Group",
                {
                    "Young": _make_level("Young", 0.82, (0.76, 0.88)),
                    "Old": _make_level("Old", 0.75, (0.68, 0.82)),
                },
            ),
        }
    )


# ===========================================================================
# TestIsPlottableLevel
# ===========================================================================


class TestIsPlottableLevel:
    """Unit tests for _is_plottable_level."""

    def test_plottable_with_valid_score_and_interval(self):
        lm = LevelMetric(name="s", label="S", score=0.8, interval=(0.7, 0.9))
        assert _is_plottable_level(lm)

    def test_not_plottable_nan_score(self):
        lm = LevelMetric(
            name="s", label="S", score=float("nan"), interval=(0.7, 0.9)
        )
        assert not _is_plottable_level(lm)

    def test_not_plottable_none_interval(self):
        lm = LevelMetric(name="s", label="S", score=0.8, interval=None)
        assert not _is_plottable_level(lm)

    def test_not_plottable_nan_lower_bound(self):
        lm = LevelMetric(
            name="s", label="S", score=0.8, interval=(float("nan"), 0.9)
        )
        assert not _is_plottable_level(lm)

    def test_not_plottable_nan_upper_bound(self):
        lm = LevelMetric(
            name="s", label="S", score=0.8, interval=(0.7, float("nan"))
        )
        assert not _is_plottable_level(lm)

    def test_plottable_integer_score(self):
        # Integer scores are valid; the check must not crash on them.
        lm = LevelMetric(name="n", label="N", score=100, interval=(90, 110))
        assert _is_plottable_level(lm)

    def test_plottable_score_zero(self):
        """Score of 0.0 is a valid float, not NaN."""
        lm = LevelMetric(name="s", label="S", score=0.0, interval=(0.0, 0.05))
        assert _is_plottable_level(lm)


# ===========================================================================
# TestResolveMetricKey
# ===========================================================================


class TestResolveMetricKey:
    """Unit tests for _resolve_metric_key."""

    def _features(self) -> "dict[str, FeatureEvaluation]":
        feval = _make_feature(
            "sex", "Sex", {"M": _make_level("M", 0.85, (0.80, 0.90))}
        )
        return {"sex": feval}

    def test_exact_name_match(self):
        features = self._features()
        key = _resolve_metric_key("sensitivity", ["sex"], features)
        assert key == "sensitivity"

    def test_label_match(self):
        features = self._features()
        key = _resolve_metric_key("Sensitivity", ["sex"], features)
        assert key == "sensitivity"

    def test_name_takes_priority_over_label(self):
        """When name and label are both valid selectors, name wins."""
        # Build a feature where a metric's label happens to equal the name of
        # a *different* metric.  The name-match must win.
        feval = FeatureEvaluation(name="f", label="F")
        leval = LevelEvaluation(name="lvl")
        leval.metrics["auroc"] = LevelMetric(
            name="auroc", label="AUPRC", score=0.9, interval=(0.85, 0.95)
        )
        leval.metrics["auprc"] = LevelMetric(
            name="auprc", label="auroc", score=0.8, interval=(0.75, 0.85)
        )
        feval.levels["lvl"] = leval
        key = _resolve_metric_key("auroc", ["f"], {"f": feval})
        assert key == "auroc"

    def test_unknown_metric_raises_value_error(self):
        features = self._features()
        with pytest.raises(ValueError, match="not found"):
            _resolve_metric_key("nonexistent", ["sex"], features)

    def test_error_message_lists_available_names_and_labels(self):
        features = self._features()
        with pytest.raises(ValueError) as exc_info:
            _resolve_metric_key("nonexistent", ["sex"], features)
        msg = str(exc_info.value)
        # Both the internal name and the display label must appear in the message.
        assert "sensitivity" in msg
        assert "Sensitivity" in msg


# ===========================================================================
# TestGetMetricDisplayLabel
# ===========================================================================


class TestGetMetricDisplayLabel:
    """Unit tests for _get_metric_display_label."""

    def test_returns_label_from_first_matching_level(self):
        feval = _make_feature(
            "sex", "Sex", {"M": _make_level("M", 0.85, (0.80, 0.90))}
        )
        assert _get_metric_display_label("sensitivity", feval) == "Sensitivity"

    def test_falls_back_to_key_for_empty_feature(self):
        feval = FeatureEvaluation(name="f", label="F")
        assert _get_metric_display_label("missing_key", feval) == "missing_key"

    def test_falls_back_to_key_when_metric_absent_from_all_levels(self):
        feval = _make_feature(
            "sex", "Sex", {"M": _make_level("M", 0.85, (0.80, 0.90))}
        )
        assert _get_metric_display_label("auroc", feval) == "auroc"


# ===========================================================================
# TestPlotMetricIntervalsSuccess
# ===========================================================================


class TestPlotMetricIntervalsSuccess:
    """Success paths for ScoreEvaluation.plot_metric_intervals."""

    def test_returns_dict_keyed_by_feature_name(self):
        results = _two_feature_score_eval()
        plots = results.plot_metric_intervals("sensitivity")
        assert set(plots.keys()) == {"sex", "age"}

    def test_each_value_is_figure_axes_tuple(self):
        results = _two_feature_score_eval()
        plots = results.plot_metric_intervals("sensitivity")
        for fname, (fig, ax) in plots.items():
            assert isinstance(fig, matplotlib.figure.Figure), fname
            assert isinstance(ax, matplotlib.axes.Axes), fname

    def test_metric_by_name(self):
        results = _two_feature_score_eval()
        plots = results.plot_metric_intervals("sensitivity")
        assert len(plots) == 2

    def test_metric_by_label(self):
        results = _two_feature_score_eval()
        plots = results.plot_metric_intervals("Sensitivity")
        assert len(plots) == 2

    def test_metric_by_name_and_label_produce_same_keys(self):
        results = _two_feature_score_eval()
        by_name = results.plot_metric_intervals("sensitivity")
        by_label = results.plot_metric_intervals("Sensitivity")
        assert set(by_name.keys()) == set(by_label.keys())

    def test_feature_names_none_returns_all_features(self):
        results = _two_feature_score_eval()
        plots = results.plot_metric_intervals("sensitivity", feature_names=None)
        assert set(plots.keys()) == {"sex", "age"}

    def test_feature_names_subset_returns_only_specified(self):
        results = _two_feature_score_eval()
        plots = results.plot_metric_intervals("sensitivity", feature_names=["sex"])
        assert list(plots.keys()) == ["sex"]
        assert "age" not in plots

    def test_feature_names_order_preserved_in_output(self):
        """Output dict preserves the order of feature_names, not features dict order."""
        results = _two_feature_score_eval()
        plots = results.plot_metric_intervals(
            "sensitivity", feature_names=["age", "sex"]
        )
        assert list(plots.keys()) == ["age", "sex"]

    def test_ytick_labels_match_level_names(self):
        results = _two_feature_score_eval()
        fig, ax = results.plot_metric_intervals("sensitivity")["sex"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["M", "F"]

    def test_xlabel_is_metric_display_label(self):
        results = _two_feature_score_eval()
        fig, ax = results.plot_metric_intervals("sensitivity")["sex"]
        assert ax.get_xlabel() == "Sensitivity"

    def test_title_includes_feature_label(self):
        results = _two_feature_score_eval()
        fig, ax = results.plot_metric_intervals("sensitivity")["sex"]
        assert "Sex" in ax.get_title()

    def test_title_includes_metric_label(self):
        results = _two_feature_score_eval()
        fig, ax = results.plot_metric_intervals("sensitivity")["sex"]
        assert "Sensitivity" in ax.get_title()

    def test_levels_without_ci_silently_excluded(self):
        """Levels with None interval are excluded; the rest are plotted."""
        feval = _make_feature(
            "sex",
            "Sex",
            {
                "M": _make_level("M", 0.85, (0.80, 0.90)),  # plottable
                "F": _make_level("F", 0.78, None),           # excluded
            },
        )
        results = _make_score_eval({"sex": feval})
        plots = results.plot_metric_intervals("sensitivity")
        labels = [t.get_text() for t in plots["sex"][1].get_yticklabels()]
        assert labels == ["M"]
        assert "F" not in labels

    def test_nan_score_levels_silently_excluded(self):
        """Levels with NaN scores (e.g., categorical placeholders) are excluded."""
        feval = _make_feature(
            "sex",
            "Sex",
            {
                "M": _make_level("M", 0.85, (0.80, 0.90)),
                "Unknown": _make_level("Unknown", float("nan"), None),
            },
        )
        results = _make_score_eval({"sex": feval})
        plots = results.plot_metric_intervals("sensitivity")
        labels = [t.get_text() for t in plots["sex"][1].get_yticklabels()]
        assert labels == ["M"]
        assert "Unknown" not in labels

    def test_level_insertion_order_preserved(self):
        """Y-axis order matches level insertion order in FeatureEvaluation.levels."""
        feval = _make_feature(
            "grp",
            "Group",
            {
                "C": _make_level("C", 0.9, (0.85, 0.95)),
                "B": _make_level("B", 0.8, (0.75, 0.85)),
                "A": _make_level("A", 0.7, (0.65, 0.75)),
            },
        )
        results = _make_score_eval({"grp": feval})
        fig, ax = results.plot_metric_intervals("sensitivity")["grp"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # Insertion order C → B → A must be preserved (y-axis is inverted,
        # so first-inserted is at the top which is displayed first).
        assert labels == ["C", "B", "A"]


# ===========================================================================
# TestPlotMetricIntervalsErrors
# ===========================================================================


class TestPlotMetricIntervalsErrors:
    """Error paths for ScoreEvaluation.plot_metric_intervals."""

    def test_empty_features_raises_value_error(self):
        results = ScoreEvaluation(name="s", label="S")
        with pytest.raises(ValueError, match="no features"):
            results.plot_metric_intervals("sensitivity")

    def test_unknown_feature_in_feature_names_raises(self):
        results = _two_feature_score_eval()
        with pytest.raises(ValueError, match="Unknown feature"):
            results.plot_metric_intervals(
                "sensitivity", feature_names=["nonexistent"]
            )

    def test_unknown_feature_error_message_lists_available(self):
        results = _two_feature_score_eval()
        with pytest.raises(ValueError) as exc_info:
            results.plot_metric_intervals(
                "sensitivity", feature_names=["nonexistent"]
            )
        msg = str(exc_info.value)
        # At least one valid feature name must appear in the error message.
        assert "sex" in msg or "age" in msg

    def test_unknown_metric_raises_value_error(self):
        results = _two_feature_score_eval()
        with pytest.raises(ValueError, match="not found"):
            results.plot_metric_intervals("completely_made_up_metric")

    def test_unknown_metric_error_message_lists_available(self):
        results = _two_feature_score_eval()
        with pytest.raises(ValueError) as exc_info:
            results.plot_metric_intervals("completely_made_up_metric")
        msg = str(exc_info.value)
        assert "sensitivity" in msg

    def test_all_levels_no_ci_raises_value_error(self):
        """When every level in a feature lacks CI data, raise rather than plot empty."""
        feval = _make_feature(
            "sex",
            "Sex",
            {
                "M": _make_level("M", 0.85, None),
                "F": _make_level("F", 0.78, None),
            },
        )
        results = _make_score_eval({"sex": feval})
        with pytest.raises(ValueError, match="no levels have plottable CI data"):
            results.plot_metric_intervals("sensitivity")

    def test_all_levels_nan_score_raises_value_error(self):
        """All-NaN scores (all-placeholder feature) must raise, not produce an empty plot."""
        feval = _make_feature(
            "grp",
            "Group",
            {
                "A": _make_level("A", float("nan"), (0.80, 0.90)),
                "B": _make_level("B", float("nan"), (0.72, 0.84)),
            },
        )
        results = _make_score_eval({"grp": feval})
        with pytest.raises(ValueError, match="no levels have plottable CI data"):
            results.plot_metric_intervals("sensitivity")

    def test_no_plottable_ci_error_message_names_feature(self):
        """ValueError for no CI data must mention the offending feature name."""
        feval = _make_feature(
            "diagnosis",
            "Diagnosis",
            {"pos": _make_level("pos", 0.9, None)},
        )
        results = _make_score_eval({"diagnosis": feval})
        with pytest.raises(ValueError) as exc_info:
            results.plot_metric_intervals("sensitivity")
        assert "diagnosis" in str(exc_info.value)

    def test_first_valid_feature_success_second_no_ci_raises(self):
        """Error from a later feature must still raise after earlier features succeed."""
        results = _make_score_eval(
            {
                "sex": _make_feature(
                    "sex",
                    "Sex",
                    {"M": _make_level("M", 0.85, (0.80, 0.90))},  # OK
                ),
                "age": _make_feature(
                    "age",
                    "Age",
                    {"Y": _make_level("Y", 0.78, None)},  # no CI
                ),
            }
        )
        with pytest.raises(ValueError, match="no levels have plottable CI data"):
            results.plot_metric_intervals("sensitivity")


# ===========================================================================
# TestEndToEnd — integration smoke test via full Auditor pipeline
# ===========================================================================


class TestEndToEndPlotting:
    """Full pipeline: Auditor.evaluate_metrics → plot_metric_intervals.

    Verifies that CI data produced by bootstrap resampling is correctly
    surfaced in the interval plots.
    """

    @pytest.fixture(scope="class")
    def results(self):
        import pandas as pd

        from model_auditor import Auditor
        from model_auditor.metrics import Sensitivity, Specificity

        rows = []
        for grp, n_pos, n_neg in [("A", 8, 5), ("B", 6, 7), ("C", 4, 3)]:
            rows += [{"group": grp, "score": 0.8, "label": 1}] * n_pos
            rows += [{"group": grp, "score": 0.2, "label": 0}] * n_neg
        df = pd.DataFrame(rows)

        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="group")
        auditor.add_score(name="score", threshold=0.5)
        auditor.add_outcome(name="label")
        auditor.set_metrics([Sensitivity(), Specificity()])
        return auditor.evaluate_metrics(score_name="score", n_bootstraps=50)

    def test_returns_figure_per_feature(self, results):
        plots = results.plot_metric_intervals("sensitivity")
        assert "group" in plots
        fig, ax = plots["group"]
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_name_and_label_return_same_features(self, results):
        by_name = results.plot_metric_intervals("sensitivity")
        by_label = results.plot_metric_intervals("Sensitivity")
        assert set(by_name.keys()) == set(by_label.keys())

    def test_yticks_match_observed_levels(self, results):
        fig, ax = results.plot_metric_intervals("sensitivity")["group"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # All observed groups must appear in the plot.
        assert "A" in labels
        assert "B" in labels
        assert "C" in labels

    def test_specificity_also_plottable_by_label(self, results):
        plots = results.plot_metric_intervals("Specificity")
        assert "group" in plots

    def test_feature_names_subset_works_end_to_end(self, results):
        plots = results.plot_metric_intervals(
            "sensitivity", feature_names=["group"]
        )
        assert list(plots.keys()) == ["group"]



# ---------------------------------------------------------------------------
# Additional imports for new tests
# ---------------------------------------------------------------------------

from matplotlib.text import Annotation

from model_auditor.schemas import _extract_level_counts, _format_level_annotation


# ---------------------------------------------------------------------------
# Shared helper for annotation extraction
# ---------------------------------------------------------------------------


def _annotation_texts(ax) -> list[str]:
    """Return text content of all Annotation objects on the axes."""
    return [child.get_text() for child in ax.get_children() if isinstance(child, Annotation)]


# ---------------------------------------------------------------------------
# Helper: LevelEvaluation with count metrics
# ---------------------------------------------------------------------------


def _make_level_with_counts(
    name: str,
    score: float,
    interval: "tuple[float, float]",
    n: int,
    n_pos: int,
    n_neg: int,
) -> LevelEvaluation:
    """Build a LevelEvaluation with sensitivity and count metrics."""
    leval = LevelEvaluation(name=name)
    leval.metrics["sensitivity"] = LevelMetric(
        name="sensitivity", label="Sensitivity", score=score, interval=interval
    )
    leval.metrics["n"] = LevelMetric(name="n", label="N", score=n)
    leval.metrics["n_pos"] = LevelMetric(name="n_pos", label="Pos.", score=n_pos)
    leval.metrics["n_neg"] = LevelMetric(name="n_neg", label="Neg.", score=n_neg)
    return leval


# ---------------------------------------------------------------------------
# Helper: ScoreEvaluation with 'overall' and 'sex' features
# ---------------------------------------------------------------------------


def _score_eval_with_overall() -> ScoreEvaluation:
    """Score evaluation with synthetic 'overall' feature and 'sex' feature."""
    return _make_score_eval(
        {
            "overall": _make_feature(
                "overall",
                "Overall",
                {"Overall": _make_level("Overall", 0.82, (0.78, 0.86))},
            ),
            "sex": _make_feature(
                "sex",
                "Sex",
                {
                    "M": _make_level("M", 0.85, (0.80, 0.90)),
                    "F": _make_level("F", 0.78, (0.72, 0.84)),
                },
            ),
        }
    )


# ===========================================================================
# TestExtractLevelCounts
# ===========================================================================


class TestExtractLevelCounts:
    """Unit tests for _extract_level_counts."""

    def _leval(self, **kwargs) -> LevelEvaluation:
        leval = LevelEvaluation(name="test")
        for name, value in kwargs.items():
            leval.metrics[name] = LevelMetric(name=name, label=name.upper(), score=value)
        return leval

    def test_empty_level_returns_none_triple(self):
        assert _extract_level_counts(LevelEvaluation(name="t")) == (None, None, None)

    def test_direct_n_only(self):
        n, n_pos, n_neg = _extract_level_counts(self._leval(n=50))
        assert n == 50
        assert n_pos is None
        assert n_neg is None

    def test_all_direct_metrics(self):
        assert _extract_level_counts(self._leval(n=20, n_pos=8, n_neg=12)) == (20, 8, 12)

    def test_derives_n_pos_from_n_tp_n_fn(self):
        _, n_pos, _ = _extract_level_counts(self._leval(n_tp=3, n_fn=5))
        assert n_pos == 8

    def test_derives_n_neg_from_n_tn_n_fp(self):
        _, _, n_neg = _extract_level_counts(self._leval(n_tn=7, n_fp=2))
        assert n_neg == 9

    def test_derives_n_from_derived_n_pos_and_n_neg(self):
        n, n_pos, n_neg = _extract_level_counts(
            self._leval(n_tp=3, n_fn=5, n_tn=7, n_fp=2)
        )
        assert n_pos == 8
        assert n_neg == 9
        assert n == 17

    def test_derives_n_from_all_four_components(self):
        n, _, _ = _extract_level_counts(self._leval(n_tp=2, n_tn=3, n_fp=4, n_fn=5))
        assert n == 14

    def test_nan_count_metric_treated_as_missing(self):
        leval = LevelEvaluation(name="t")
        leval.metrics["n"] = LevelMetric(name="n", label="N", score=float("nan"))
        n, _, _ = _extract_level_counts(leval)
        assert n is None

    def test_direct_n_pos_wins_over_component_derivation(self):
        """Explicit n_pos metric overrides the n_tp+n_fn derivation path."""
        _, n_pos, _ = _extract_level_counts(self._leval(n_pos=10, n_tp=3, n_fn=5))
        assert n_pos == 10


# ===========================================================================
# TestFormatLevelAnnotation
# ===========================================================================


class TestFormatLevelAnnotation:
    """Unit tests for _format_level_annotation."""

    def test_both_disabled_empty_string(self):
        assert _format_level_annotation(10, 100, 4, 6, False, False) == ""

    def test_sample_size_only(self):
        assert _format_level_annotation(10, 100, 4, 6, True, False) == "10 (10.0%)"

    def test_class_balance_only(self):
        assert _format_level_annotation(10, 100, 4, 6, False, True) == "4 (40.0%)"

    def test_both_enabled_newline_joined(self):
        assert _format_level_annotation(10, 100, 4, 6, True, True) == "10 (10.0%)\n4 (40.0%)"

    def test_sample_size_none_n_level_gives_na_na(self):
        assert _format_level_annotation(None, 100, 4, 6, True, False) == "NA (NA)"

    def test_sample_size_none_n_overall_gives_n_na(self):
        assert _format_level_annotation(10, None, 4, 6, True, False) == "10 (NA)"

    def test_sample_size_zero_n_overall_gives_n_na(self):
        assert _format_level_annotation(10, 0, 4, 6, True, False) == "10 (NA)"

    def test_class_balance_none_n_pos_gives_na_na(self):
        assert _format_level_annotation(10, 100, None, 6, False, True) == "NA (NA)"

    def test_class_balance_none_n_neg_gives_n_na(self):
        assert _format_level_annotation(10, 100, 4, None, False, True) == "4 (NA)"

    def test_class_balance_zero_denom_gives_n_na(self):
        assert _format_level_annotation(10, 100, 0, 0, False, True) == "0 (NA)"


# ===========================================================================
# TestPlotMetricIntervalsOverall
# ===========================================================================


class TestPlotMetricIntervalsOverall:
    """Tests for overall feature exclusion and Optional[Overall] comparator level."""

    def test_overall_feature_not_in_plots_when_feature_names_none(self):
        results = _score_eval_with_overall()
        plots = results.plot_metric_intervals("sensitivity")
        assert "overall" not in plots
        assert "sex" in plots

    def test_overall_excluded_when_explicit_in_feature_names(self):
        results = _score_eval_with_overall()
        plots = results.plot_metric_intervals(
            "sensitivity", feature_names=["overall", "sex"]
        )
        assert "overall" not in plots
        assert "sex" in plots

    def test_feature_names_only_overall_raises(self):
        results = _score_eval_with_overall()
        with pytest.raises(ValueError, match="No plottable features"):
            results.plot_metric_intervals("sensitivity", feature_names=["overall"])

    def test_include_overall_true_prepends_overall_to_yticks(self):
        results = _score_eval_with_overall()
        fig, ax = results.plot_metric_intervals(
            "sensitivity", include_overall=True
        )["sex"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # Overall must be the first (top) level.
        assert labels[0] == "Overall"
        assert "M" in labels
        assert "F" in labels

    def test_include_overall_false_no_overall_in_yticks(self):
        results = _score_eval_with_overall()
        fig, ax = results.plot_metric_intervals(
            "sensitivity", include_overall=False
        )["sex"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "Overall" not in labels
        assert "M" in labels
        assert "F" in labels

    def test_include_overall_true_no_overall_feature_no_crash(self):
        """include_overall=True is a no-op when the 'overall' feature is absent."""
        results = _two_feature_score_eval()  # no 'overall' feature
        fig, ax = results.plot_metric_intervals(
            "sensitivity", include_overall=True
        )["sex"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "Overall" not in labels

    def test_include_overall_skipped_when_overall_has_no_ci(self):
        """Overall level is silently omitted when its CI is not plottable."""
        results = _make_score_eval(
            {
                "overall": _make_feature(
                    "overall",
                    "Overall",
                    {"Overall": _make_level("Overall", 0.82, None)},  # no CI
                ),
                "sex": _make_feature(
                    "sex",
                    "Sex",
                    {"M": _make_level("M", 0.85, (0.80, 0.90))},
                ),
            }
        )
        fig, ax = results.plot_metric_intervals(
            "sensitivity", include_overall=True
        )["sex"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "Overall" not in labels
        assert "M" in labels

    def test_overall_level_count_reflects_three_feature_levels(self):
        """When include_overall=True, the plot has N+1 y-ticks (Overall + feature levels)."""
        results = _score_eval_with_overall()
        fig, ax = results.plot_metric_intervals(
            "sensitivity", include_overall=True
        )["sex"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # 1 Overall + 2 feature levels (M, F)
        assert len(labels) == 3


# ===========================================================================
# TestPlotMetricIntervalsRotated
# ===========================================================================


class TestPlotMetricIntervalsRotated:
    """Tests for rotate_plots parameter."""

    def test_rotate_false_levels_on_yaxis(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity", rotate_plots=False
        )["sex"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "M" in labels
        assert "F" in labels

    def test_rotate_false_metric_on_xaxis(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity", rotate_plots=False
        )["sex"]
        assert ax.get_xlabel() == "Sensitivity"

    def test_rotate_true_levels_on_xaxis(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity", rotate_plots=True
        )["sex"]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "M" in labels
        assert "F" in labels

    def test_rotate_true_metric_on_yaxis(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity", rotate_plots=True
        )["sex"]
        assert ax.get_ylabel() == "Sensitivity"

    def test_rotate_true_xlabel_is_not_metric_label(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity", rotate_plots=True
        )["sex"]
        assert ax.get_xlabel() != "Sensitivity"

    def test_rotate_true_title_unchanged(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity", rotate_plots=True
        )["sex"]
        assert "Sex" in ax.get_title()
        assert "Sensitivity" in ax.get_title()

    def test_rotate_true_returns_figure_axes_tuple(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity", rotate_plots=True
        )["sex"]
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)


# ===========================================================================
# TestPlotMetricIntervalsAnnotations
# ===========================================================================


class TestPlotMetricIntervalsAnnotations:
    """Tests for annotation text on interval plots."""

    def test_default_annotations_present(self):
        """Annotations appear by default (both toggles True), showing NA placeholders."""
        fig, ax = _two_feature_score_eval().plot_metric_intervals("sensitivity")["sex"]
        texts = _annotation_texts(ax)
        # sex has two levels (M, F) → two annotations
        assert len(texts) == 2

    def test_default_annotations_contain_na_when_no_count_metrics(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals("sensitivity")["sex"]
        texts = _annotation_texts(ax)
        for t in texts:
            assert "NA (NA)" in t

    def test_both_disabled_no_annotations(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity",
            include_sample_size=False,
            include_class_balance=False,
        )["sex"]
        assert len(_annotation_texts(ax)) == 0

    def test_sample_size_only_single_line_per_annotation(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity",
            include_sample_size=True,
            include_class_balance=False,
        )["sex"]
        for t in _annotation_texts(ax):
            assert "\n" not in t

    def test_class_balance_only_single_line_per_annotation(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity",
            include_sample_size=False,
            include_class_balance=True,
        )["sex"]
        for t in _annotation_texts(ax):
            assert "\n" not in t

    def test_both_enabled_two_lines_per_annotation(self):
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity",
            include_sample_size=True,
            include_class_balance=True,
        )["sex"]
        for t in _annotation_texts(ax):
            assert "\n" in t

    def test_sample_size_annotation_formatted_correctly_with_count_metrics(self):
        """n=10, n_overall=20 → '10 (50.0%)' appears as an annotation."""
        overall_leval = _make_level_with_counts(
            "Overall", 0.82, (0.78, 0.86), n=20, n_pos=8, n_neg=12
        )
        m_leval = _make_level_with_counts("M", 0.85, (0.80, 0.90), n=10, n_pos=4, n_neg=6)
        f_leval = _make_level_with_counts("F", 0.78, (0.72, 0.84), n=10, n_pos=4, n_neg=6)
        results = _make_score_eval(
            {
                "overall": _make_feature(
                    "overall", "Overall", {"Overall": overall_leval}
                ),
                "sex": _make_feature(
                    "sex", "Sex", {"M": m_leval, "F": f_leval}
                ),
            }
        )
        fig, ax = results.plot_metric_intervals(
            "sensitivity",
            include_overall=False,
            include_sample_size=True,
            include_class_balance=False,
        )["sex"]
        texts = _annotation_texts(ax)
        # Each level: n=10, n_overall=20 → 50.0%
        assert "10 (50.0%)" in texts

    def test_class_balance_annotation_formatted_correctly_with_count_metrics(self):
        """n_pos=4, n_neg=6 → '4 (40.0%)'; n_pos=2, n_neg=8 → '2 (20.0%)'."""
        m_leval = _make_level_with_counts("M", 0.85, (0.80, 0.90), n=10, n_pos=4, n_neg=6)
        f_leval = _make_level_with_counts("F", 0.78, (0.72, 0.84), n=10, n_pos=2, n_neg=8)
        results = _make_score_eval(
            {"sex": _make_feature("sex", "Sex", {"M": m_leval, "F": f_leval})}
        )
        fig, ax = results.plot_metric_intervals(
            "sensitivity",
            include_sample_size=False,
            include_class_balance=True,
        )["sex"]
        texts = _annotation_texts(ax)
        assert any("4 (40.0%)" in t for t in texts)
        assert any("2 (20.0%)" in t for t in texts)

    def test_annotation_sample_size_na_when_n_overall_zero(self):
        """n_overall=0 is a zero denominator; percentage must show NA."""
        overall_leval = _make_level_with_counts(
            "Overall", 0.82, (0.78, 0.86), n=0, n_pos=0, n_neg=0
        )
        m_leval = _make_level_with_counts("M", 0.85, (0.80, 0.90), n=10, n_pos=4, n_neg=6)
        results = _make_score_eval(
            {
                "overall": _make_feature(
                    "overall", "Overall", {"Overall": overall_leval}
                ),
                "sex": _make_feature("sex", "Sex", {"M": m_leval}),
            }
        )
        fig, ax = results.plot_metric_intervals(
            "sensitivity",
            include_overall=False,
            include_sample_size=True,
            include_class_balance=False,
        )["sex"]
        texts = _annotation_texts(ax)
        # n=10 is available, but n_overall=0 → NA for %
        assert any("10 (NA)" in t for t in texts)

    def test_annotation_class_balance_na_when_zero_denom(self):
        """n_pos=0, n_neg=0 → denominator zero → '0 (NA)'."""
        m_leval = _make_level_with_counts("M", 0.85, (0.80, 0.90), n=10, n_pos=0, n_neg=0)
        results = _make_score_eval(
            {"sex": _make_feature("sex", "Sex", {"M": m_leval})}
        )
        fig, ax = results.plot_metric_intervals(
            "sensitivity",
            include_sample_size=False,
            include_class_balance=True,
        )["sex"]
        texts = _annotation_texts(ax)
        assert any("0 (NA)" in t for t in texts)

    def test_annotations_also_appear_in_rotated_mode(self):
        """Annotations are rendered regardless of orientation."""
        fig, ax = _two_feature_score_eval().plot_metric_intervals(
            "sensitivity",
            rotate_plots=True,
            include_sample_size=True,
            include_class_balance=False,
        )["sex"]
        # sex has two levels (M, F)
        assert len(_annotation_texts(ax)) == 2