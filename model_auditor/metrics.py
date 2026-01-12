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

from typing import Protocol, Union, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
    name: str = "sensitivity"
    label: str = "Sensitivity"
    inputs: list[str] = ["tp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        """Calculate sensitivity from the data.

        Args:
            data: DataFrame with 'tp' and 'fn' columns.
            eps: Small constant to avoid division by zero.

        Returns:
            Sensitivity value between 0 and 1.
        """
        n_tp: int = data["tp"].sum()
        n_fn: int = data["fn"].sum()
        return n_tp / (n_tp + n_fn + eps)


class Specificity(AuditorMetric):
    """Specificity (True Negative Rate) metric.

    Calculates TN / (TN + FP), the proportion of actual negatives
    correctly identified.
    """
    name: str = "specificity"
    label: str = "Specificity"
    inputs: list[str] = ["tn", "fp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        """Calculate specificity from the data.

        Args:
            data: DataFrame with 'tn' and 'fp' columns.
            eps: Small constant to avoid division by zero.

        Returns:
            Specificity value between 0 and 1.
        """
        n_tn: int = data["tn"].sum()
        n_fp: int = data["fp"].sum()
        return n_tn / (n_tn + n_fp + eps)


class Precision(AuditorMetric):
    """Precision (Positive Predictive Value) metric.

    Calculates TP / (TP + FP), the proportion of positive predictions
    that are correct.
    """
    name: str = "precision"
    label: str = "Precision"
    inputs: list[str] = ["tp", "fp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        """Calculate precision from the data.

        Args:
            data: DataFrame with 'tp' and 'fp' columns.
            eps: Small constant to avoid division by zero.

        Returns:
            Precision value between 0 and 1.
        """
        n_tp: int = data["tp"].sum()
        n_fp: int = data["fp"].sum()
        return n_tp / (n_tp + n_fp + eps)


class Recall(AuditorMetric):
    """Recall metric (alias for Sensitivity).

    Calculates TP / (TP + FN), identical to Sensitivity.
    """
    name: str = "recall"
    label: str = "Recall"
    inputs: list[str] = ["tp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        """Calculate recall from the data.

        Args:
            data: DataFrame with 'tp' and 'fn' columns.
            eps: Small constant to avoid division by zero.

        Returns:
            Recall value between 0 and 1.
        """
        n_tp: int = data["tp"].sum()
        n_fn: int = data["fn"].sum()
        return n_tp / (n_tp + n_fn + eps)


class F1Score(AuditorMetric):
    """F1 Score metric.

    Calculates the harmonic mean of precision and recall:
    2 * (precision * recall) / (precision + recall).
    """
    name: str = "f1"
    label: str = "F1 Score"
    inputs: list[str] = ["tp", "fp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        """Calculate F1 score from the data.

        Args:
            data: DataFrame with 'tp', 'fp', and 'fn' columns.
            eps: Small constant to avoid division by zero.

        Returns:
            F1 score value between 0 and 1.
        """
        # Recalculate to avoid dependency on ordering of metrics
        precision = Precision().data_call(data)
        recall = Recall().data_call(data)
        return 2 * (precision * recall) / (precision + recall + eps)


class AUROC(AuditorMetric):
    """Area Under the Receiver Operating Characteristic curve metric.

    Uses sklearn's roc_auc_score to compute AUROC from continuous
    predictions and binary ground truth.
    """
    name: str = "auroc"
    label: str = "AUROC"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate AUROC from the data.

        Args:
            data: DataFrame with '_truth' and '_pred' columns.

        Returns:
            AUROC value between 0 and 1, or 0.0 if calculation fails.
        """
        try:
            return float(roc_auc_score(data["_truth"], data["_pred"]))
        except ValueError:
            return 0.0


class AUPRC(AuditorMetric):
    """Area Under the Precision-Recall Curve metric.

    Uses sklearn's average_precision_score to compute AUPRC from
    continuous predictions and binary ground truth.
    """
    name: str = "auprc"
    label: str = "AUPRC"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate AUPRC from the data.

        Args:
            data: DataFrame with '_truth' and '_pred' columns.

        Returns:
            AUPRC value between 0 and 1, or 0.0 if calculation fails.
        """
        try:
            return float(average_precision_score(data["_truth"], data["_pred"]))
        except ValueError:
            return 0.0


class MatthewsCorrelationCoefficient(AuditorMetric):
    """Matthews Correlation Coefficient (MCC) metric.

    A balanced measure that accounts for all four confusion matrix values.
    Returns values between -1 (total disagreement) and +1 (perfect prediction),
    with 0 indicating random prediction.
    """
    name: str = "mcc"
    label: str = "Matthews Correlation Coefficient"
    inputs: list[str] = ["tp", "tn", "fp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        """Calculate MCC from the data.

        Args:
            data: DataFrame with 'tp', 'tn', 'fp', and 'fn' columns.
            eps: Small constant to avoid division by zero.

        Returns:
            MCC value between -1 and 1.
        """
        tp = data["tp"].sum()
        tn = data["tn"].sum()
        fp = data["fp"].sum()
        fn = data["fn"].sum()

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if denominator == 0:
            return 0.0
        return numerator / (denominator + eps)


class FBetaScore(AuditorMetric):
    """F-beta Score metric with configurable beta parameter.

    Generalizes F1 score by allowing different weightings of precision
    and recall. Beta < 1 weights precision higher, beta > 1 weights
    recall higher.
    """
    name: str = "fbeta"
    label: str = "F-beta Score"
    inputs: list[str] = ["precision", "recall"]
    ci_eligible: bool = True

    def __init__(self, beta: float = 1.0):
        """Initialize FBetaScore with a specific beta value.

        Args:
            beta: Weight of recall vs precision. Default is 1.0 (F1 score).
        """
        self.beta = beta
        self.name = f"f{beta:.1f}".replace(".", "_")  # e.g., "f0_5" or "f2_0"
        self.label = f"F{beta:.1f} Score"

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate F-beta score from the data.

        Args:
            data: DataFrame with columns needed by Precision and Recall.

        Returns:
            F-beta score value between 0 and 1.
        """
        precision = Precision().data_call(data)
        recall = Recall().data_call(data)
        beta_sq = self.beta**2

        if precision + recall == 0:
            return 0.0

        return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)


class TPR(Sensitivity):
    """True Positive Rate metric (alias for Sensitivity)."""

    name: str = "tpr"
    label: str = "TPR"


class TNR(Specificity):
    """True Negative Rate metric (alias for Specificity)."""
    name: str = "tnr"
    label: str = "TNR"


class FPR(AuditorMetric):
    """False Positive Rate metric.

    Calculates FP / (FP + TN), the proportion of actual negatives
    incorrectly identified as positive.
    """

    name: str = "fpr"
    label: str = "FPR"
    inputs: list[str] = ["fp", "tn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        """Calculate FPR from the data.

        Args:
            data: DataFrame with 'fp' and 'tn' columns.
            eps: Small constant to avoid division by zero.

        Returns:
            FPR value between 0 and 1.
        """
        n_fp: int = data["fp"].sum()
        n_tn: int = data["tn"].sum()
        return n_fp / (n_fp + n_tn + eps)


class FNR(AuditorMetric):
    """False Negative Rate metric.

    Calculates FN / (FN + TP), the proportion of actual positives
    incorrectly identified as negative.
    """
    name: str = "fnr"
    label: str = "FNR"
    inputs: list[str] = ["fn", "tp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        """Calculate FNR from the data.

        Args:
            data: DataFrame with 'fn' and 'tp' columns.
            eps: Small constant to avoid division by zero.

        Returns:
            FNR value between 0 and 1.
        """
        n_fn: int = data["fn"].sum()
        n_tp: int = data["tp"].sum()
        return n_fn / (n_fn + n_tp + eps)


class nData(AuditorMetric):
    """Sample size metric.

    Returns the number of rows in the data subset.
    """
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
    name: str = "n_tp"
    label: str = "TP"
    inputs: list[str] = ['tp']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of true positives.

        Args:
            data: DataFrame with 'tp' column.

        Returns:
            Sum of true positives.
        """
        return data['tp'].sum()


class nTN(AuditorMetric):
    """True Negative count metric.

    Returns the total number of true negatives in the data subset.
    """
    name: str = "n_tn"
    label: str = "TN"
    inputs: list[str] = ['tn']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of true negatives.

        Args:
            data: DataFrame with 'tn' column.

        Returns:
            Sum of true negatives.
        """
        return data['tn'].sum()


class nFP(AuditorMetric):
    """False Positive count metric.

    Returns the total number of false positives in the data subset.
    """
    name: str = "n_fp"
    label: str = "FP"
    inputs: list[str] = ['fp']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of false positives.

        Args:
            data: DataFrame with 'fp' column.

        Returns:
            Sum of false positives.
        """
        return data['fp'].sum()


class nFN(AuditorMetric):
    """False Negative count metric.

    Returns the total number of false negatives in the data subset.
    """
    name: str = "n_fn"
    label: str = "FN"
    inputs: list[str] = ['fn']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of false negatives.

        Args:
            data: DataFrame with 'fn' column.

        Returns:
            Sum of false negatives.
        """
        return data['fn'].sum()


class nPositive(AuditorMetric):
    """Positive class count metric.

    Returns the number of actual positive cases in the ground truth.
    """
    name: str = "n_pos"
    label: str = "Pos."
    inputs: list[str] = ['_truth']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of positive ground truth labels.

        Args:
            data: DataFrame with '_truth' column.

        Returns:
            Number of rows where _truth equals 1.
        """
        return (data['_truth'] == 1).astype(int).sum()


class nNegative(AuditorMetric):
    """Negative class count metric.

    Returns the number of actual negative cases in the ground truth.
    """
    name: str = "n_neg"
    label: str = "Neg."
    inputs: list[str] = ['_truth']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the count of negative ground truth labels.

        Args:
            data: DataFrame with '_truth' column.

        Returns:
            Number of rows where _truth equals 0.
        """
        return (data['_truth'] == 0).astype(int).sum()