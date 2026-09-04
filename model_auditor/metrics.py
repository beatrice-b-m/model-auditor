"""Metrics module for model evaluation.

This module defines the AuditorMetric protocol and provides implementations
of common classification metrics including sensitivity, specificity, precision,
recall, F1, AUROC, AUPRC, MCC, and count-based metrics.

All metrics follow a protocol-based design allowing for easy extension with
custom metrics.

Example:
    Using built-in metrics::

        from model_auditor.metrics import Sensitivity, Specificity, AUROC

        metrics = [Sensitivity(), Specificity(), AUROC()]
        auditor.set_metrics(metrics)

    Creating a custom metric::

        class CustomMetric(AuditorMetric):
            name = "custom"
            label = "Custom Metric"
            inputs = ["tp", "tn"]
            ci_eligible = True

            def data_call(self, data: pd.DataFrame) -> float:
                return (data["tp"].sum() + data["tn"].sum()) / len(data)
"""

from typing import Protocol, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable ratio, returning NaN when the denominator is zero."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


class AuditorMetric(Protocol):
    """Protocol defining the interface for auditor metrics.

    All metrics must implement this protocol to be used with the Auditor.

    Attributes:
        name: Unique identifier for the metric (used as dictionary key).
        label: Human-readable display name for the metric.
        inputs: List of column names required to compute this metric.
        ci_eligible: Whether this metric supports confidence interval calculation.
    """

    name: str
    label: str
    inputs: list[str]
    ci_eligible: bool

    def data_call(self, data: pd.DataFrame) -> Union[float, int]:
        """Calculate the metric from a DataFrame.

        Args:
            data: DataFrame containing the required input columns.

        Returns:
            The computed metric value.
        """
        raise NotImplementedError


class Sensitivity(AuditorMetric):
    """Sensitivity (True Positive Rate) metric.

    Calculates TP / (TP + FN), the proportion of actual positives
    correctly identified.
    """

    binomial_columns = ("tp", "fn")
    direction = "higher"
    name: str = "sensitivity"
    label: str = "Sensitivity"
    inputs: list[str] = ["tp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate sensitivity from the data.

        Args:
            data: DataFrame with 'tp' and 'fn' columns.

        Returns:
            Sensitivity value between 0 and 1.
        """
        n_tp: int = data["tp"].sum()
        n_fn: int = data["fn"].sum()
        return _safe_ratio(n_tp, n_tp + n_fn)


class Specificity(AuditorMetric):
    """Specificity (True Negative Rate) metric.

    Calculates TN / (TN + FP), the proportion of actual negatives
    correctly identified.
    """

    binomial_columns = ("tn", "fp")
    direction = "higher"
    name: str = "specificity"
    label: str = "Specificity"
    inputs: list[str] = ["tn", "fp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate specificity from the data.

        Args:
            data: DataFrame with 'tn' and 'fp' columns.

        Returns:
            Specificity value between 0 and 1.
        """
        n_tn: int = data["tn"].sum()
        n_fp: int = data["fp"].sum()
        return _safe_ratio(n_tn, n_tn + n_fp)


class Precision(AuditorMetric):
    """Precision (Positive Predictive Value) metric.

    Calculates TP / (TP + FP), the proportion of positive predictions
    that are correct.
    """

    binomial_columns = ("tp", "fp")
    direction = "higher"
    name: str = "precision"
    label: str = "Precision"
    inputs: list[str] = ["tp", "fp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate precision from the data.

        Args:
            data: DataFrame with 'tp' and 'fp' columns.

        Returns:
            Precision value between 0 and 1.
        """
        n_tp: int = data["tp"].sum()
        n_fp: int = data["fp"].sum()
        return _safe_ratio(n_tp, n_tp + n_fp)


class Recall(Sensitivity):
    """Recall metric (alias for Sensitivity).

    Calculates TP / (TP + FN), identical to Sensitivity.
    """

    direction = "higher"
    name: str = "recall"
    label: str = "Recall"


class F1Score(AuditorMetric):
    """F1 Score metric.

    Calculates the harmonic mean of precision and recall:
    2 * (precision * recall) / (precision + recall).
    """

    direction = "higher"
    name: str = "f1"
    label: str = "F1 Score"
    inputs: list[str] = ["tp", "fp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate F1 score from the data.

        Args:
            data: DataFrame with 'tp', 'fp', and 'fn' columns.

        Returns:
            F1 score value between 0 and 1.
        """
        tp = float(data["tp"].sum())
        fp = float(data["fp"].sum())
        fn = float(data["fn"].sum())
        return _safe_ratio(2 * tp, 2 * tp + fp + fn)


class AUROC(AuditorMetric):
    """Area Under the Receiver Operating Characteristic curve metric.

    Uses sklearn's roc_auc_score to compute AUROC from continuous
    predictions and binary ground truth.
    """

    direction = "higher"
    name: str = "auroc"
    label: str = "AUROC"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate AUROC from the data.

        Args:
            data: DataFrame with '_truth' and '_pred' columns.

        Returns:
            AUROC value between 0 and 1, or NaN if calculation fails.
        """
        if data["_truth"].nunique() < 2:
            return float("nan")
        try:
            return float(roc_auc_score(data["_truth"], data["_pred"]))
        except ValueError:
            return float("nan")


class AUPRC(AuditorMetric):
    """Compatibility name for non-interpolated average precision (not trapezoidal area).

    Uses sklearn's average_precision_score to compute AUPRC from
    continuous predictions and binary ground truth.
    """

    direction = "higher"
    parameters = {"integration": "average_precision"}
    name: str = "auprc"
    label: str = "AUPRC"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate AUPRC from the data.

        Args:
            data: DataFrame with '_truth' and '_pred' columns.

        Returns:
            AUPRC value between 0 and 1, or NaN if calculation fails.
        """
        if data["_truth"].sum() == 0:
            return float("nan")
        try:
            return float(average_precision_score(data["_truth"], data["_pred"]))
        except ValueError:
            return float("nan")


class MatthewsCorrelationCoefficient(AuditorMetric):
    """Matthews Correlation Coefficient (MCC) metric.

    A balanced measure that accounts for all four confusion matrix values.
    Returns values between -1 (total disagreement) and +1 (perfect prediction),
    with 0 indicating zero measured correlation, not proof of randomness.
    """

    direction = "higher"
    name: str = "mcc"
    label: str = "Matthews Correlation Coefficient"
    inputs: list[str] = ["tp", "tn", "fp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate MCC from the data.

        Args:
            data: DataFrame with 'tp', 'tn', 'fp', and 'fn' columns.

        Returns:
            MCC value between -1 and 1.
        """
        tp = float(data["tp"].sum())
        tn = float(data["tn"].sum())
        fp = float(data["fp"].sum())
        fn = float(data["fn"].sum())

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if denominator == 0:
            return float("nan")
        return float(numerator / denominator)


class FBetaScore(AuditorMetric):
    """F-beta Score metric with configurable beta parameter.

    Generalizes F1 score by allowing different weightings of precision
    and recall. Beta < 1 weights precision higher, beta > 1 weights
    recall higher.
    """

    direction = "higher"
    name: str = "fbeta"
    label: str = "F-beta Score"
    inputs: list[str] = ["tp", "fp", "fn"]
    ci_eligible: bool = True

    def __init__(self, beta: float = 1.0) -> None:
        """Initialize FBetaScore with a specific beta value.

        Args:
            beta: Weight of recall vs precision. Default is 1.0 (F1 score).
        """
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive.")
        self.beta = float(beta)
        self.parameters = {"beta": self.beta}
        self.name = f"f{self.beta}".replace(".", "_")
        self.label = f"F{self.beta} Score"

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate F-beta score from the data.

        Args:
            data: DataFrame with columns needed by Precision and Recall.

        Returns:
            F-beta score value between 0 and 1.
        """
        tp = float(data["tp"].sum())
        fp = float(data["fp"].sum())
        fn = float(data["fn"].sum())
        beta_sq = self.beta**2
        return _safe_ratio((1 + beta_sq) * tp, (1 + beta_sq) * tp + beta_sq * fn + fp)


class TPR(Sensitivity):
    """True Positive Rate metric (alias for Sensitivity)."""

    direction = "higher"
    name: str = "tpr"
    label: str = "TPR"


class TNR(Specificity):
    """True Negative Rate metric (alias for Specificity)."""

    direction = "higher"
    name: str = "tnr"
    label: str = "TNR"


class FPR(AuditorMetric):
    """False Positive Rate metric.

    Calculates FP / (FP + TN), the proportion of actual negatives
    incorrectly identified as positive.
    """

    binomial_columns = ("fp", "tn")
    direction = "lower"
    name: str = "fpr"
    label: str = "FPR"
    inputs: list[str] = ["fp", "tn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate FPR from the data.

        Args:
            data: DataFrame with 'fp' and 'tn' columns.

        Returns:
            FPR value between 0 and 1.
        """
        n_fp: int = data["fp"].sum()
        n_tn: int = data["tn"].sum()
        return _safe_ratio(n_fp, n_fp + n_tn)


class FNR(AuditorMetric):
    """False Negative Rate metric.

    Calculates FN / (FN + TP), the proportion of actual positives
    incorrectly identified as negative.
    """

    binomial_columns = ("fn", "tp")
    direction = "lower"
    name: str = "fnr"
    label: str = "FNR"
    inputs: list[str] = ["fn", "tp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate FNR from the data.

        Args:
            data: DataFrame with 'fn' and 'tp' columns.

        Returns:
            FNR value between 0 and 1.
        """
        n_fn: int = data["fn"].sum()
        n_tp: int = data["tp"].sum()
        return _safe_ratio(n_fn, n_fn + n_tp)


class nData(AuditorMetric):
    """Sample size metric.

    Returns the number of rows in the data subset.
    """

    direction = "none"
    name: str = "n"
    label: str = "N"
    inputs: list[str] = []
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the number of samples in the data.

        Args:
            data: DataFrame to count rows from.

        Returns:
            Number of rows in the DataFrame.
        """
        return len(data)


class nTP(AuditorMetric):
    """True Positive count metric.

    Returns the total number of true positives in the data subset.
    """

    direction = "none"
    name: str = "n_tp"
    label: str = "TP"
    inputs: list[str] = ["tp"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of true positives.

        Args:
            data: DataFrame with 'tp' column.

        Returns:
            Sum of true positives.
        """
        return data["tp"].sum()


class nTN(AuditorMetric):
    """True Negative count metric.

    Returns the total number of true negatives in the data subset.
    """

    direction = "none"
    name: str = "n_tn"
    label: str = "TN"
    inputs: list[str] = ["tn"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of true negatives.

        Args:
            data: DataFrame with 'tn' column.

        Returns:
            Sum of true negatives.
        """
        return data["tn"].sum()


class nFP(AuditorMetric):
    """False Positive count metric.

    Returns the total number of false positives in the data subset.
    """

    direction = "none"
    name: str = "n_fp"
    label: str = "FP"
    inputs: list[str] = ["fp"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of false positives.

        Args:
            data: DataFrame with 'fp' column.

        Returns:
            Sum of false positives.
        """
        return data["fp"].sum()


class nFN(AuditorMetric):
    """False Negative count metric.

    Returns the total number of false negatives in the data subset.
    """

    direction = "none"
    name: str = "n_fn"
    label: str = "FN"
    inputs: list[str] = ["fn"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of false negatives.

        Args:
            data: DataFrame with 'fn' column.

        Returns:
            Sum of false negatives.
        """
        return data["fn"].sum()


class nPositive(AuditorMetric):
    """Positive class count metric.

    Returns the number of actual positive cases in the ground truth.
    """

    direction = "none"
    name: str = "n_pos"
    label: str = "Pos."
    inputs: list[str] = ["_truth"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of positive ground truth labels.

        Args:
            data: DataFrame with '_truth' column.

        Returns:
            Number of rows where _truth equals 1.
        """
        return (data["_truth"] == 1).astype(int).sum()


class nNegative(AuditorMetric):
    """Negative class count metric.

    Returns the number of actual negative cases in the ground truth.
    """

    direction = "none"
    name: str = "n_neg"
    label: str = "Neg."
    inputs: list[str] = ["_truth"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of negative ground truth labels.

        Args:
            data: DataFrame with '_truth' column.

        Returns:
            Number of rows where _truth equals 0.
        """
        return (data["_truth"] == 0).astype(int).sum()


class AveragePrecision(AUPRC):
    """Non-interpolated average precision; AUPRC remains a compatibility alias.

    Prevalence-dependent. Undefined without positives. This is not trapezoidal
    PR area and is not a probability calibration metric.
    """

    name = "average_precision"
    label = "Average precision"


class NegativePredictiveValue(AuditorMetric):
    """TN / (TN + FN); undefined without negative predictions."""

    name = "npv"
    label = "NPV"
    inputs = ["tn", "fn"]
    ci_eligible = True
    direction = "higher"
    binomial_columns = ("tn", "fn")

    def data_call(self, data: pd.DataFrame) -> float:
        return _safe_ratio(data["tn"].sum(), data["tn"].sum() + data["fn"].sum())


class BalancedAccuracy(AuditorMetric):
    """Mean sensitivity and specificity; requires both truth classes."""

    name = "balanced_accuracy"
    label = "Balanced accuracy"
    inputs = ["tp", "tn", "fp", "fn"]
    ci_eligible = True
    direction = "higher"

    def data_call(self, data: pd.DataFrame) -> float:
        return (Sensitivity().data_call(data) + Specificity().data_call(data)) / 2


class Prevalence(AuditorMetric):
    """Observed positive-class proportion; descriptive, not a performance rank."""

    name = "prevalence"
    label = "Prevalence"
    inputs = ["_truth"]
    ci_eligible = True
    direction = "none"
    binomial_indicator = "_truth"

    def data_call(self, data: pd.DataFrame) -> float:
        return float(data["_truth"].mean())


class SelectionRate(Prevalence):
    """Predicted-positive proportion at the chosen policy threshold."""

    name = "selection_rate"
    label = "Selection rate"
    inputs = ["_binary_pred"]
    binomial_indicator = "_binary_pred"

    def data_call(self, data: pd.DataFrame) -> float:
        return float(data["_binary_pred"].mean())


def validate_probabilities(data: pd.DataFrame) -> np.ndarray:
    """Probability metrics require finite values in [0, 1], unlike ranking scores."""
    probabilities = data["_pred"].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probability metrics require finite scores in [0, 1].")
    return probabilities


class BrierScore(AuditorMetric):
    """Mean squared probability error; measures calibration and discrimination."""

    name = "brier_score"
    label = "Brier score"
    inputs = ["_truth", "_pred"]
    ci_eligible = True
    direction = "lower"

    def data_call(self, data: pd.DataFrame) -> float:
        probabilities = validate_probabilities(data)
        return float(np.mean((probabilities - data["_truth"].to_numpy()) ** 2))


class LogLoss(BrierScore):
    """Binary log loss, clipping probabilities to float64 epsilon at endpoints."""

    name = "log_loss"
    label = "Log loss"
    parameters = {"clip_epsilon": float(np.finfo(float).eps)}

    def data_call(self, data: pd.DataFrame) -> float:
        p = np.clip(
            validate_probabilities(data), np.finfo(float).eps, 1 - np.finfo(float).eps
        )
        y = data["_truth"].to_numpy()
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _calibration_fit(data: pd.DataFrame, fit_slope: bool) -> float:
    """Logistic calibration with explicit rejection of unidentified/separated fits."""
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    p = validate_probabilities(data)
    y = data["_truth"].to_numpy(dtype=float)
    if len(np.unique(y)) != 2 or np.any((p == 0) | (p == 1)):
        return float("nan")
    x = logit(p)
    if fit_slope:
        if (
            np.ptp(x) == 0
            or x[y == 0].max() <= x[y == 1].min()
            or x[y == 1].max() <= x[y == 0].min()
        ):
            return float("nan")
        design = np.column_stack([np.ones(len(x)), x])
        offset = np.zeros(len(x))
        initial = np.array([0.0, 1.0])
    else:
        design = np.ones((len(x), 1))
        offset = x
        initial = np.array([0.0])

    def objective(beta):
        eta = design @ beta + offset
        return np.mean(np.logaddexp(0, eta) - y * eta), design.T @ (
            expit(eta) - y
        ) / len(y)

    fit = minimize(objective, initial, jac=True, method="BFGS", options={"gtol": 1e-8})
    if not fit.success or not np.isfinite(fit.x).all():
        return float("nan")
    return float(fit.x[-1])


class CalibrationIntercept(BrierScore):
    """Calibration-in-the-large: logistic intercept with logit(score) as offset.

    Ideal value is zero. Requires interior probabilities and both truth classes.
    Not a higher/lower-is-better metric. Invalid fits return NaN.
    """

    name = "calibration_intercept"
    label = "Calibration intercept"
    direction = "none"

    def data_call(self, data: pd.DataFrame) -> float:
        return _calibration_fit(data, fit_slope=False)


class CalibrationSlope(CalibrationIntercept):
    """Slope from logistic regression on logit(score), with a fitted intercept.

    Ideal value is one. Constant scores, separation and boundary probabilities
    are not estimable; bootstrap failures are exposed in interval diagnostics.
    """

    name = "calibration_slope"
    label = "Calibration slope"

    def data_call(self, data: pd.DataFrame) -> float:
        return _calibration_fit(data, fit_slope=True)


class ExpectedCost(AuditorMetric):
    """Mean FP/FN cost per evaluated row at the selected decision policy."""

    name = "expected_cost"
    label = "Expected cost"
    inputs = ["fp", "fn"]
    ci_eligible = True
    direction = "lower"

    def __init__(self, false_positive_cost: float = 1, false_negative_cost: float = 1):
        if not all(
            np.isfinite(v) and v >= 0
            for v in (false_positive_cost, false_negative_cost)
        ):
            raise ValueError("Costs must be finite and nonnegative.")
        self.parameters = {
            "false_positive_cost": float(false_positive_cost),
            "false_negative_cost": float(false_negative_cost),
        }

    def data_call(self, data: pd.DataFrame) -> float:
        return _safe_ratio(
            self.parameters["false_positive_cost"] * data["fp"].sum()
            + self.parameters["false_negative_cost"] * data["fn"].sum(),
            len(data),
        )
