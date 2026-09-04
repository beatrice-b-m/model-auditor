"""Regression checks for shared preparation and evaluation edge cases."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from model_auditor import Auditor
from model_auditor.metrics import Sensitivity, nData
from model_auditor.schemas import (
    ConditionalThreshold,
    FeatureEvaluation,
    ScoreEvaluation,
)


@pytest.fixture
def auditor():
    result = Auditor(
        pd.DataFrame(
            {
                "group": ["B", "A", "B", "A"],
                "score": [0.8, 0.7, 0.2, 0.1],
                "outcome": [1, 0, 1, 0],
            }
        )
    )
    result.add_feature("group")
    result.add_score("score", threshold=0.5)
    result.add_outcome("outcome")
    result.set_metrics([nData(), Sensitivity()])
    return result


@pytest.mark.parametrize("method", ["evaluate_metrics", "evaluate_errors"])
@pytest.mark.parametrize("count", [0, -1, 1.5, True, np.bool_(True), float("nan")])
def test_bootstrap_count_validation(auditor, method, count):
    with pytest.raises(ValueError, match="positive integer or None"):
        getattr(auditor, method)("score", n_bootstraps=count)


@pytest.mark.parametrize("method", ["evaluate_metrics", "evaluate_errors"])
@pytest.mark.parametrize("bad_value", [None, 2, -1, "positive"])
def test_invalid_truth_is_rejected(auditor, method, bad_value):
    auditor.data["_truth"] = auditor.data["_truth"].astype(object)
    auditor.data.loc[0, "_truth"] = bad_value
    with pytest.raises(ValueError, match="Outcome values must be binary"):
        getattr(auditor, method)("score", n_bootstraps=None)


@pytest.mark.parametrize("method", ["evaluate_metrics", "evaluate_errors"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores_are_rejected(auditor, method, bad_value):
    auditor.data.loc[0, "score"] = bad_value
    with pytest.raises(ValueError, match="finite numeric values"):
        getattr(auditor, method)("score", n_bootstraps=None)


@pytest.mark.parametrize("method", ["evaluate_metrics", "evaluate_errors"])
def test_missing_registered_column_has_clear_error(auditor, method):
    auditor.add_feature("missing")
    with pytest.raises(ValueError, match="columns not found.*missing"):
        getattr(auditor, method)("score", n_bootstraps=None)


@pytest.mark.parametrize("method", ["evaluate_metrics", "evaluate_errors"])
def test_empty_dataset_is_rejected(auditor, method):
    auditor.data = auditor.data.iloc[:0]
    with pytest.raises(ValueError, match="at least one row"):
        getattr(auditor, method)("score", n_bootstraps=None)


@pytest.mark.parametrize("method", ["evaluate_metrics", "evaluate_errors"])
def test_duplicate_columns_are_rejected(auditor, method):
    auditor.data = pd.concat([auditor.data, auditor.data[["score"]]], axis=1)
    with pytest.raises(ValueError, match="unique column names"):
        getattr(auditor, method)("score", n_bootstraps=None)


@pytest.mark.parametrize(
    "name", ["overall", "_truth", "_pred", "_binary_pred", "tp", "tn", "fp", "fn"]
)
def test_feature_cannot_overwrite_generated_columns(auditor, name):
    with pytest.raises(ValueError, match="reserved"):
        auditor.add_feature(name)


def test_duplicate_metric_names_are_rejected():
    with pytest.raises(ValueError, match="Metric names must be unique"):
        Auditor(metrics=[Sensitivity(), Sensitivity()])


def test_set_metrics_owns_its_list(auditor):
    metrics = [nData()]
    auditor.set_metrics(metrics)
    metrics.clear()
    assert (
        auditor.evaluate_metrics("score", n_bootstraps=None)
        .features["overall"]
        .levels["Overall"]
        .metrics["n"]
        .score
        == 4
    )


@pytest.mark.parametrize("method", ["evaluate_metrics", "evaluate_errors"])
def test_categorical_conditional_default_handles_missing_level_and_null(
    auditor, method
):
    auditor.data["group"] = pd.Categorical(["B", "A", "B", None], categories=["B", "A"])
    threshold = ConditionalThreshold("group", {"A": 0.9}, default=0.5)
    result = getattr(auditor, method)("score", threshold=threshold, n_bootstraps=None)
    if method == "evaluate_metrics":
        assert (
            result.features["overall"].levels["Overall"].metrics["sensitivity"].score
            == 0.5
        )
    else:
        assert result.to_dataframe().loc[("Overall", "Overall"), ("TP", "N")] == 1
        assert result.to_dataframe().loc[("Overall", "Overall"), ("FP", "N")] == 0


def test_count_only_evaluation_does_not_resample(auditor, monkeypatch):
    auditor.set_metrics([nData()])

    def unexpected_sample(*args, **kwargs):
        pytest.fail("Count-only evaluation must not perform bootstrap resampling")

    monkeypatch.setattr(pd.DataFrame, "sample", unexpected_sample)
    result = auditor.evaluate_metrics("score")
    assert result.features["overall"].levels["Overall"].metrics["n"].score == 4


def test_custom_metric_can_read_binary_predictions_and_group_column(auditor):
    class PositiveFraction:
        name = "positive_fraction"
        label = "Positive fraction"
        inputs = ["_binary_pred"]
        ci_eligible = False

        def data_call(self, data):
            assert "group" in data.columns
            return float(data["_binary_pred"].mean())

    auditor.set_metrics([PositiveFraction()])
    result = auditor.evaluate_metrics("score", n_bootstraps=None)
    assert (
        result.features["group"].levels["A"].metrics["positive_fraction"].score == 0.5
    )


def test_unknown_input_has_clear_error(auditor):
    class BadMetric:
        name = "bad"
        label = "Bad"
        inputs = ["unknown_input"]
        ci_eligible = False

    auditor.set_metrics([BadMetric()])
    with pytest.raises(ValueError, match="Unknown metric input 'unknown_input'"):
        auditor.evaluate_metrics("score", n_bootstraps=None)


def test_all_null_feature_exports_empty_rows(auditor):
    auditor.data["group"] = None
    result = auditor.evaluate_metrics("score", n_bootstraps=None)
    assert result.features["group"].levels == {}
    assert result.features["group"].to_dataframe().empty
    assert list(result.to_dataframe().index) == [("Overall", "Overall")]


def test_empty_result_containers_export_empty_frames():
    assert FeatureEvaluation("group", "Group").to_dataframe().empty
    assert ScoreEvaluation("score", "Score").to_dataframe().empty


@pytest.mark.parametrize("method", ["evaluate_metrics", "evaluate_errors"])
def test_evaluation_does_not_mutate_stored_data(auditor, method):
    before = auditor.data.copy(deep=True)
    getattr(auditor, method)("score", n_bootstraps=5)
    pd.testing.assert_frame_equal(auditor.data, before)


def test_core_import_does_not_load_optional_renderers():
    code = """
import sys
from model_auditor import Auditor
assert not any(name.startswith(('matplotlib', 'plotly', 'jinja2')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
