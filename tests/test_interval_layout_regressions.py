"""Regression coverage for interval plot layout fixes.

Guards two concrete defects: rotated annotations that overlap when many
levels are plotted, and omitted-level reasons that exposed internal status
codes instead of readable text.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede any plt import

import matplotlib.pyplot as plt
import pytest

from model_auditor.plotting.intervals import _describe_omission
from model_auditor.schemas import (
    FeatureEvaluation,
    LevelEvaluation,
    LevelMetric,
    ScoreEvaluation,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _level(
    name: str,
    score: float,
    interval: "tuple[float, float] | None",
    interval_status: str = "ok",
    n: int = 100,
) -> LevelEvaluation:
    level = LevelEvaluation(name=name, support={"n": n, "n_pos": 40, "n_neg": n - 40})
    level.metrics["specificity"] = LevelMetric(
        name="specificity",
        label="Specificity",
        score=score,
        interval=interval,
        interval_status=interval_status,
    )
    return level


def _score_evaluation(levels: "dict[str, LevelEvaluation]") -> ScoreEvaluation:
    feature = FeatureEvaluation(name="clinic", label="Clinic")
    feature.levels.update(levels)
    evaluation = ScoreEvaluation(name="score", label="Score")
    evaluation.features["clinic"] = feature
    return evaluation


def _boxes_overlap(a, b, tolerance: float = 0.5) -> bool:
    return not (
        a.x1 - tolerance <= b.x0
        or b.x1 - tolerance <= a.x0
        or a.y1 - tolerance <= b.y0
        or b.y1 - tolerance <= a.y0
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("undefined_estimate", "estimate undefined"),
        ("too_many_invalid_resamples", "too many invalid resamples"),
        ("insufficient_resamples", "too few valid resamples"),
        ("degenerate_distribution", "degenerate resampling distribution"),
        ("insufficient_clusters", "too few clusters for resampling"),
        ("not_requested", "interval not requested"),
    ],
)
def test_describe_omission_maps_internal_statuses(status, expected):
    metric = LevelMetric(
        name="specificity",
        label="Specificity",
        score=float("nan"),
        interval=None,
        interval_status=status,
    )
    assert _describe_omission(metric) == expected


def test_describe_omission_reports_invalid_bounds():
    metric = LevelMetric(
        name="specificity",
        label="Specificity",
        score=0.5,
        interval=(0.9, 0.1),
    )
    assert _describe_omission(metric) == "nonfinite or reversed interval bounds"


def test_rotated_annotations_do_not_overlap():
    levels = {
        f"Clinic {index:02d}": _level(
            f"Clinic {index:02d}",
            score=0.90 - 0.01 * index,
            interval=(0.80 - 0.01 * index, 0.95 - 0.01 * index),
            n=100 - 5 * index,
        )
        for index in range(1, 11)
    }
    plots = _score_evaluation(levels).plot_metric_intervals(
        "specificity", feature_names=["clinic"], rotate_plots=True
    )
    figure, axes = plots["clinic"]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert len(axes.texts) == len(levels)
    boxes = [text.get_window_extent(renderer) for text in axes.texts]
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            assert not _boxes_overlap(boxes[first], boxes[second]), (
                f"annotations {first} and {second} overlap: "
                f"{boxes[first]} vs {boxes[second]}"
            )
    figure_box = figure.bbox
    for box in boxes:
        assert box.x0 >= figure_box.x0 - 0.5
        assert box.x1 <= figure_box.x1 + 0.5
        assert box.y0 >= figure_box.y0 - 0.5
        assert box.y1 <= figure_box.y1 + 0.5


def test_omitted_levels_use_readable_reasons():
    levels = {
        "Observed": _level("Observed", 0.8, (0.7, 0.9)),
        "Sparse": _level(
            "Sparse", 0.9, None, interval_status="too_many_invalid_resamples"
        ),
        "Unobserved": _level(
            "Unobserved", float("nan"), None, interval_status="undefined_estimate"
        ),
    }
    plots = _score_evaluation(levels).plot_metric_intervals(
        "specificity", feature_names=["clinic"], include_overall=False
    )
    figure, _ = plots["clinic"]
    caption = "\n".join(text.get_text() for text in figure.texts)
    assert "Not drawn" in caption
    assert "Sparse: too many invalid resamples" in caption
    assert "Unobserved: estimate undefined" in caption
    assert "too_many_invalid_resamples" not in caption


def test_long_labels_retain_usable_axes_and_title_clearance():
    names = [
        "Northeast Metropolitan Service Area",
        "Pacific Northwest Coastal Region",
        "Upper Midwest Rural District",
        "Southern Gulf Catchment",
    ]
    levels = {name: _level(name, 0.8, (0.7, 0.9)) for name in names}
    figure, axes = _score_evaluation(levels).plot_metric_intervals(
        "specificity", rotate_plots=True
    )["clinic"]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert axes.bbox.height / figure.dpi >= 2.9
    assert axes.get_ylim()[1] < 1.2
    title_box = axes.title.get_window_extent(renderer)
    for annotation in axes.texts:
        box = annotation.get_window_extent(renderer)
        assert not _boxes_overlap(box, title_box)
        assert box.y1 <= axes.bbox.y1 + 0.5
        assert box.y0 > axes.transData.transform((0, 0.9))[1]


def test_undefined_metric_does_not_claim_observed_subgroup_is_empty():
    import pandas as pd

    from model_auditor import Auditor
    from model_auditor.metrics import Sensitivity

    auditor = Auditor(
        pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "truth": [0, 0, 1, 0],
                "score": [0.1, 0.2, 0.9, 0.1],
            }
        )
    )
    auditor.add_feature("group")
    auditor.add_score("score", threshold=0.5)
    auditor.add_outcome("truth")
    auditor.set_metrics([Sensitivity()])
    result = auditor.evaluate_metrics("score")
    assert result.features["group"].levels["A"].support["n"] == 2
    figure, _ = result.plot_metric_intervals("sensitivity")["group"]
    caption = "\n".join(text.get_text() for text in figure.texts)
    assert "A: estimate undefined" in caption
    assert "no data" not in caption
