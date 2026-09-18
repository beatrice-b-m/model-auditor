"""Static consumer checks; checked by Pyright, not executed as an evaluation."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.io.formats.style import Styler
from typing_extensions import assert_type

from model_auditor import Auditor, InferenceConfig
from model_auditor.metric_inputs import TruePositives
from model_auditor.metrics import ExpectedCost, Sensitivity, nTP
from model_auditor.plotting import HierarchyPlotter
from model_auditor.plotting.distributions import plot_score_distributions
from model_auditor.plotting.intervals import plot_metric_intervals
from model_auditor.plotting.schemas import PlotterData
from model_auditor.schemas import (
    CalibrationEvaluation,
    ConditionalThreshold,
    ErrorEvaluation,
    FeatureEvaluation,
    LevelEvaluation,
    LevelMetric,
    ScoreEvaluation,
)

if TYPE_CHECKING:
    auditor = Auditor(pd.DataFrame({"score": [0.9, 0.1], "y": [1, 0]}))
    auditor.add_score("score", threshold=ConditionalThreshold("region", {}, 0.5))
    auditor.add_outcome("y")
    auditor.set_metrics([Sensitivity(), ExpectedCost(false_negative_cost=2)])
    inference = InferenceConfig(missing="include", random_state=42)
    result = auditor.evaluate_metrics("score", inference=inference)
    assert_type(result, ScoreEvaluation)
    assert_type(nTP().data_call(pd.DataFrame()), int | np.integer)
    assert_type(result.features["region"], FeatureEvaluation)
    level = result.features["region"].levels["North"]
    assert_type(level, LevelEvaluation)
    assert_type(level.metrics["sensitivity"], LevelMetric)
    assert_type(level.metrics["sensitivity"].interval, tuple[float, float] | None)
    assert_type(result.style_dataframe(rank=True), Styler)
    assert_type(auditor.evaluate_errors("score"), ErrorEvaluation)
    assert_type(auditor.evaluate_calibration("score", bins=5), CalibrationEvaluation)
    assert_type(auditor.compare_groups("score", "region", "North"), ScoreEvaluation)
    assert_type(auditor.compare_scores("score", "reference"), ScoreEvaluation)
    assert_type(result.plot_metric_intervals("sensitivity")["region"][0], Figure)
    assert_type(result.plot_metric_intervals("sensitivity")["region"][1], Axes)
    assert_type(plot_metric_intervals(result, "sensitivity")["region"][1], Axes)
    histograms = auditor.plot_score_distributions("score", bins=[0.0, 0.5, 1.0])
    assert_type(histograms["region"][0], Figure)
    assert_type(plot_score_distributions(auditor, "score")["region"][0], Figure)
    assert_type(TruePositives().data_transform(pd.DataFrame()), pd.DataFrame)
    plotter = HierarchyPlotter()
    plotter.set_aggregator(lambda group: float(group["score"].mean()))
    assert_type(plotter.compile("All"), PlotterData)
    assert_type(plotter.compile("All").colors, list[float | None])
