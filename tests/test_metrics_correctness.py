"""Deterministic correctness tests for built-in metrics."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from model_auditor.metrics import (
    AUPRC,
    AUROC,
    F1Score,
    FBetaScore,
    FNR,
    FPR,
    MatthewsCorrelationCoefficient,
    Precision,
    Recall,
    Sensitivity,
    Specificity,
    TNR,
    TPR,
    nData,
    nFN,
    nFP,
    nNegative,
    nPositive,
    nTN,
    nTP,
)


TRUTH = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
PRED_BINARY = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)

# Confusion totals for the fixture above:
# TP=3, FN=2, FP=1, TN=4
TP = 3
FN = 2
FP = 1
TN = 4
N = 10


@pytest.fixture
def confusion_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_truth": TRUTH,
            "_pred": np.array([0.95, 0.9, 0.8, 0.35, 0.2, 0.85, 0.3, 0.25, 0.15, 0.05]),
            "tp": ((TRUTH == 1.0) & (PRED_BINARY == 1.0)).astype(int),
            "fn": ((TRUTH == 1.0) & (PRED_BINARY == 0.0)).astype(int),
            "fp": ((TRUTH == 0.0) & (PRED_BINARY == 1.0)).astype(int),
            "tn": ((TRUTH == 0.0) & (PRED_BINARY == 0.0)).astype(int),
        }
    )


def test_confusion_rate_and_count_metrics_match_known_values(confusion_df: pd.DataFrame):
    expected = {
        "sensitivity": TP / (TP + FN),
        "specificity": TN / (TN + FP),
        "precision": TP / (TP + FP),
        "recall": TP / (TP + FN),
        "f1": 2 * TP / ((2 * TP) + FP + FN),
        "f0_5": (1 + 0.5**2) * ((TP / (TP + FP)) * (TP / (TP + FN))) / ((0.5**2 * (TP / (TP + FP))) + (TP / (TP + FN))),
        "fpr": FP / (FP + TN),
        "fnr": FN / (FN + TP),
        "tpr": TP / (TP + FN),
        "tnr": TN / (TN + FP),
        "mcc": ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)),
        "n": N,
        "n_tp": TP,
        "n_tn": TN,
        "n_fp": FP,
        "n_fn": FN,
        "n_pos": TP + FN,
        "n_neg": TN + FP,
    }

    metrics = {
        "sensitivity": Sensitivity(),
        "specificity": Specificity(),
        "precision": Precision(),
        "recall": Recall(),
        "f1": F1Score(),
        "f0_5": FBetaScore(beta=0.5),
        "fpr": FPR(),
        "fnr": FNR(),
        "tpr": TPR(),
        "tnr": TNR(),
        "mcc": MatthewsCorrelationCoefficient(),
        "n": nData(),
        "n_tp": nTP(),
        "n_tn": nTN(),
        "n_fp": nFP(),
        "n_fn": nFN(),
        "n_pos": nPositive(),
        "n_neg": nNegative(),
    }

    for name, metric in metrics.items():
        score = metric.data_call(confusion_df)
        assert score == pytest.approx(expected[name]), f"{name} mismatch"


def test_auroc_and_auprc_match_sklearn_oracle(confusion_df: pd.DataFrame):
    truth = confusion_df["_truth"]
    pred = confusion_df["_pred"]

    assert AUROC().data_call(confusion_df) == pytest.approx(roc_auc_score(truth, pred))
    assert AUPRC().data_call(confusion_df) == pytest.approx(
        average_precision_score(truth, pred)
    )


def test_auroc_returns_nan_for_single_class_truth():
    df = pd.DataFrame({"_truth": [1, 1, 1], "_pred": [0.1, 0.7, 0.9]})
    score = AUROC().data_call(df)
    assert math.isnan(score)


@pytest.mark.parametrize(
    ("metric", "data"),
    [
        (Sensitivity(), pd.DataFrame({"tp": [0, 0], "fn": [0, 0]})),
        (Specificity(), pd.DataFrame({"tn": [0, 0], "fp": [0, 0]})),
        (Precision(), pd.DataFrame({"tp": [0, 0], "fp": [0, 0]})),
        (Recall(), pd.DataFrame({"tp": [0, 0], "fn": [0, 0]})),
        (F1Score(), pd.DataFrame({"tp": [0, 0], "fp": [0, 0], "fn": [0, 0]})),
        (FBetaScore(beta=2.0), pd.DataFrame({"tp": [0, 0], "fp": [0, 0], "fn": [0, 0]})),
        (FPR(), pd.DataFrame({"fp": [0, 0], "tn": [0, 0]})),
        (FNR(), pd.DataFrame({"fn": [0, 0], "tp": [0, 0]})),
        (
            MatthewsCorrelationCoefficient(),
            pd.DataFrame({"tp": [1, 1], "tn": [0, 0], "fp": [0, 0], "fn": [0, 0]}),
        ),
    ],
)
def test_zero_denominator_cases_return_zero(metric, data: pd.DataFrame):
    assert metric.data_call(data) == 0.0
