"""Deterministic and integration checks for bootstrap confidence intervals."""

import math

import numpy as np
import pandas as pd
import pytest

from model_auditor import Auditor
from model_auditor.metrics import Sensitivity, nData


@pytest.fixture
def bootstrap_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_truth": [1, 1, 1, 0, 0, 0, 1, 0],
            "_binary_pred": [1, 0, 1, 1, 0, 0, 0, 0],
            "tp": [1, 0, 1, 0, 0, 0, 0, 0],
            "fn": [0, 1, 0, 0, 0, 0, 1, 0],
            "fp": [0, 0, 0, 1, 0, 0, 0, 0],
            "tn": [0, 0, 0, 0, 1, 1, 0, 1],
        }
    )


def _bootstrap_oracle(data: pd.DataFrame, metric: Sensitivity, n_bootstraps: int) -> tuple[float, float]:
    n = len(data)
    scores = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        boot = data.sample(n, replace=True)
        scores[i] = metric.data_call(boot)
    lower, upper = np.nanpercentile(scores, [2.5, 97.5])
    return float(lower), float(upper)


def test_private_ci_computation_matches_seeded_oracle_exactly(bootstrap_df: pd.DataFrame):
    auditor = Auditor(metrics=[Sensitivity(), nData()])
    n_bootstraps = 64

    np.random.seed(12345)
    expected = _bootstrap_oracle(bootstrap_df, Sensitivity(), n_bootstraps)

    np.random.seed(12345)
    actual = auditor._evaluate_confidence_interval(bootstrap_df, n_bootstraps=n_bootstraps)

    assert set(actual.keys()) == {"sensitivity"}
    assert actual["sensitivity"] == pytest.approx(expected)


def test_evaluate_metrics_ci_rules_for_eligible_and_ineligible_metrics():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "B"],
            "score": [0.9, 0.2, 0.8, 0.7, 0.1, 0.6, 0.3],
            "label": [1, 1, 0, 1, 0, 0, 1],
        }
    )

    auditor = Auditor()
    auditor.add_data(df)
    auditor.add_feature(name="group")
    auditor.add_score(name="score", threshold=0.5)
    auditor.add_outcome(name="label")
    auditor.set_metrics([Sensitivity(), nData()])

    result = auditor.evaluate_metrics(score_name="score", n_bootstraps=80)

    for feature_eval in result.features.values():
        for level_eval in feature_eval.levels.values():
            sens_interval = level_eval.metrics["sensitivity"].interval
            n_interval = level_eval.metrics["n"].interval
            if math.isnan(level_eval.metrics["sensitivity"].score):
                assert sens_interval is None
            else:
                assert sens_interval is not None
                lower, upper = sens_interval
                assert lower <= upper
            assert n_interval is None


def test_evaluate_errors_bootstrap_ci_rules():
    df = pd.DataFrame(
        {
            "gender": ["Female", "Female", "Male", "Male", "Other", "Other"],
            "score": [0.9, 0.2, 0.8, 0.4, 0.7, 0.1],
            "label": [1, 1, 0, 1, 0, 0],
        }
    )
    df["gender"] = pd.Categorical(
        df["gender"], categories=["Female", "Male", "Other", "Unknown"], ordered=True
    )

    auditor = Auditor()
    auditor.add_data(df)
    auditor.add_feature(name="gender")
    auditor.add_score(name="score", threshold=0.5)
    auditor.add_outcome(name="label")

    result = auditor.evaluate_errors(score_name="score", n_bootstraps=80)

    for group in ("tp", "tn", "fp", "fn"):
        for level in ("Female", "Male", "Other"):
            lm = result.groups[group].features["gender"].levels[level].metrics["odds_ratio"]
            if not math.isnan(lm.score):
                assert lm.interval is not None
                lower, upper = lm.interval
                assert lower <= upper

        unknown_lm = result.groups[group].features["gender"].levels["Unknown"].metrics[
            "odds_ratio"
        ]
        assert math.isnan(unknown_lm.score)
        assert unknown_lm.interval is None
