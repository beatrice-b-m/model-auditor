"""Known-value integration tests for Auditor.evaluate_metrics()."""

import math

import pandas as pd
import pytest

from model_auditor import Auditor
from model_auditor.metrics import (
    F1Score,
    FNR,
    FPR,
    MatthewsCorrelationCoefficient,
    Precision,
    Sensitivity,
    Specificity,
    nData,
    nFN,
    nFP,
    nNegative,
    nPositive,
    nTN,
    nTP,
)


@pytest.fixture
def eval_df() -> pd.DataFrame:
    # threshold=0.5 confusion totals:
    # overall TP=3 FN=2 FP=2 TN=3
    # group A TP=2 FN=1 FP=1 TN=1
    # group B TP=1 FN=1 FP=1 TN=2
    return pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "score": [0.9, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.3, 0.2, 0.8],
            "label": [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        }
    )


@pytest.fixture
def auditor(eval_df: pd.DataFrame) -> Auditor:
    a = Auditor()
    a.add_data(eval_df)
    a.add_feature(name="group")
    a.add_score(name="score", threshold=0.5)
    a.add_outcome(name="label")
    a.set_metrics(
        [
            Sensitivity(),
            Specificity(),
            Precision(),
            F1Score(),
            FPR(),
            FNR(),
            MatthewsCorrelationCoefficient(),
            nData(),
            nTP(),
            nTN(),
            nFP(),
            nFN(),
            nPositive(),
            nNegative(),
        ]
    )
    return a


def _metric(result, feature: str, level: str, name: str) -> float:
    return result.features[feature].levels[level].metrics[name].score


def test_evaluate_metrics_known_values_overall_and_feature_levels(auditor: Auditor):
    result = auditor.evaluate_metrics(score_name="score", n_bootstraps=None)

    expected = {
        ("overall", "Overall"): {
            "sensitivity": 3 / 5,
            "specificity": 3 / 5,
            "precision": 3 / 5,
            "f1": 3 / 5,
            "fpr": 2 / 5,
            "fnr": 2 / 5,
            "mcc": 0.2,
            "n": 10,
            "n_tp": 3,
            "n_tn": 3,
            "n_fp": 2,
            "n_fn": 2,
            "n_pos": 5,
            "n_neg": 5,
        },
        ("group", "A"): {
            "sensitivity": 2 / 3,
            "specificity": 1 / 2,
            "precision": 2 / 3,
            "f1": 2 / 3,
            "fpr": 1 / 2,
            "fnr": 1 / 3,
            "mcc": 1 / 6,
            "n": 5,
            "n_tp": 2,
            "n_tn": 1,
            "n_fp": 1,
            "n_fn": 1,
            "n_pos": 3,
            "n_neg": 2,
        },
        ("group", "B"): {
            "sensitivity": 1 / 2,
            "specificity": 2 / 3,
            "precision": 1 / 2,
            "f1": 1 / 2,
            "fpr": 1 / 3,
            "fnr": 1 / 2,
            "mcc": 1 / 6,
            "n": 5,
            "n_tp": 1,
            "n_tn": 2,
            "n_fp": 1,
            "n_fn": 1,
            "n_pos": 2,
            "n_neg": 3,
        },
    }

    for (feature, level), metric_values in expected.items():
        for metric_name, expected_value in metric_values.items():
            got = _metric(result, feature, level, metric_name)
            assert got == pytest.approx(expected_value), (
                f"Mismatch for {feature}/{level}/{metric_name}: expected {expected_value}, got {got}"
            )


def test_threshold_override_changes_confusion_counts_and_rates(auditor: Auditor):
    # threshold=0.7 => overall TP=2 FN=3 FP=2 TN=3
    result = auditor.evaluate_metrics(
        score_name="score",
        threshold=0.7,
        n_bootstraps=None,
    )

    assert _metric(result, "overall", "Overall", "n_tp") == 2
    assert _metric(result, "overall", "Overall", "n_fn") == 3
    assert _metric(result, "overall", "Overall", "sensitivity") == pytest.approx(2 / 5)
    assert _metric(result, "overall", "Overall", "precision") == pytest.approx(1 / 2)


def test_count_metrics_never_populate_confidence_intervals(auditor: Auditor):
    result = auditor.evaluate_metrics(score_name="score", n_bootstraps=50)
    count_metric_names = {"n", "n_tp", "n_tn", "n_fp", "n_fn", "n_pos", "n_neg"}

    for feature_eval in result.features.values():
        for level_eval in feature_eval.levels.values():
            for name in count_metric_names:
                interval = level_eval.metrics[name].interval
                assert interval is None, f"{name} unexpectedly had CI {interval}"


def test_mcc_exact_value_not_epsilon_shifted(auditor: Auditor):
    result = auditor.evaluate_metrics(score_name="score", n_bootstraps=None)
    got = _metric(result, "overall", "Overall", "mcc")
    assert got == pytest.approx(0.2)
    assert not math.isclose(got, 0.2 + 1e-8)
