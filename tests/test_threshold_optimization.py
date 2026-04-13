"""Tests for score-threshold optimization methods."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve

from model_auditor import Auditor
from model_auditor.metrics import Sensitivity, Specificity


BASE_DF = pd.DataFrame(
    {
        "risk_score": [0.95, 0.85, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1],
        "label": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    }
)

INFEASIBLE_SPECIFICITY_DF = pd.DataFrame(
    {
        "risk_score": [0.9, 0.8, 0.7, 0.6],
        "label": [0, 1, 1, 0],
    }
)


def _make_auditor(df: pd.DataFrame) -> Auditor:
    auditor = Auditor()
    auditor.add_data(df)
    auditor.add_score(name="risk_score")
    auditor.add_outcome(name="label")
    auditor.set_metrics([Sensitivity(), Specificity()])
    return auditor


def _overall_metric(result, metric_name: str) -> float:
    return result.features["overall"].levels["Overall"].metrics[metric_name].score


class TestOptimizeScoreThresholdForTarget:
    def test_sensitivity_selects_highest_feasible_threshold(self):
        target = 0.75
        auditor = _make_auditor(BASE_DF)

        with pytest.warns(UserWarning, match="sensitivity >="):
            threshold = auditor.optimize_score_threshold_for_target(
                score_name="risk_score",
                target=target,
                metric="sensitivity",
            )

        fpr, tpr, thresholds = roc_curve(
            BASE_DF["label"].astype(float),
            BASE_DF["risk_score"].astype(float),
            drop_intermediate=False,
        )
        valid = np.flatnonzero((tpr >= target) & np.isfinite(thresholds))
        expected = float(thresholds[valid[0]])

        assert np.isfinite(threshold)
        assert threshold == pytest.approx(expected)

    def test_specificity_selects_lowest_feasible_threshold(self):
        target = 0.60
        auditor = _make_auditor(BASE_DF)

        with pytest.warns(UserWarning, match="specificity >="):
            threshold = auditor.optimize_score_threshold_for_target(
                score_name="risk_score",
                target=target,
                metric="specificity",
            )

        fpr, _, thresholds = roc_curve(
            BASE_DF["label"].astype(float),
            BASE_DF["risk_score"].astype(float),
            drop_intermediate=False,
        )
        specificity = 1.0 - fpr
        valid = np.flatnonzero((specificity >= target) & np.isfinite(thresholds))
        expected = float(thresholds[valid[-1]])

        assert np.isfinite(threshold)
        assert threshold == pytest.approx(expected)

    def test_invalid_metric_raises_value_error(self):
        auditor = _make_auditor(BASE_DF)

        with pytest.raises(ValueError, match="metric must be either"):
            auditor.optimize_score_threshold_for_target(
                score_name="risk_score",
                target=0.70,
                metric="precision",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("bad_target", [-0.1, 1.1])
    def test_out_of_range_target_raises_value_error(self, bad_target: float):
        auditor = _make_auditor(BASE_DF)

        with pytest.raises(ValueError, match="target must be between 0.0 and 1.0"):
            auditor.optimize_score_threshold_for_target(
                score_name="risk_score",
                target=bad_target,
                metric="sensitivity",
            )

    def test_infeasible_finite_threshold_raises_with_achievable_range(self):
        auditor = _make_auditor(INFEASIBLE_SPECIFICITY_DF)

        with pytest.raises(
            ValueError,
            match=(
                r"No finite threshold for score 'risk_score' can satisfy "
                r"specificity >= 0\.750\. Achievable specificity range across "
                r"finite thresholds is \[0\.000, 0\.500\]\."
            ),
        ):
            auditor.optimize_score_threshold_for_target(
                score_name="risk_score",
                target=0.75,
                metric="specificity",
            )

    def test_selected_threshold_meets_target_in_evaluation(self):
        auditor = _make_auditor(BASE_DF)

        with pytest.warns(UserWarning, match="sensitivity >="):
            threshold = auditor.optimize_score_threshold_for_target(
                score_name="risk_score",
                target=0.75,
                metric="sensitivity",
            )

        result = auditor.evaluate_metrics(
            score_name="risk_score",
            threshold=threshold,
            n_bootstraps=None,
        )

        assert _overall_metric(result, "sensitivity") >= 0.75


class TestOptimizeScoreThresholdYoudenRegression:
    def test_existing_youden_optimizer_still_returns_finite_float(self):
        auditor = _make_auditor(BASE_DF)

        with pytest.warns(UserWarning, match="Optimal threshold"):
            threshold = auditor.optimize_score_threshold(score_name="risk_score")

        assert isinstance(threshold, float)
        assert np.isfinite(threshold)
