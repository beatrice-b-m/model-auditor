from typing import Protocol, Union, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class AuditorMetric(Protocol):
    name: str
    label: str
    inputs: list[str]

    def data_call(self, data: pd.DataFrame) -> Union[float, int]:
        """
        method called on a dataframe to calculate a metric
        """
        raise NotImplementedError


class Sensitivity(AuditorMetric):
    name: str = "sensitivity"
    label: str = "Sensitivity"
    inputs: list[str] = ["tp", "fn"]

    def data_call(self, data: pd.DataFrame) -> float:
        n_tp: int = data["tp"].sum()
        n_fn: int = data["fn"].sum()
        return n_tp / (n_tp + n_fn)


class Specificity(AuditorMetric):
    name: str = "specificity"
    label: str = "Specificity"
    inputs: list[str] = ["tn", "fp"]

    def data_call(self, data: pd.DataFrame) -> float:
        n_tn: int = data["tn"].sum()
        n_fp: int = data["fp"].sum()
        return n_tn / (n_tn + n_fp)


class Precision(AuditorMetric):
    name: str = "precision"
    label: str = "Precision"
    inputs: list[str] = ["tp", "fp"]

    def data_call(self, data: pd.DataFrame) -> float:
        n_tp: int = data["tp"].sum()
        n_fp: int = data["fp"].sum()
        if n_tp + n_fp == 0:
            return 0.0
        return n_tp / (n_tp + n_fp)


class Recall(AuditorMetric):
    name: str = "recall"
    label: str = "Recall"
    inputs: list[str] = ["tp", "fn"]

    def data_call(self, data: pd.DataFrame) -> float:
        n_tp: int = data["tp"].sum()
        n_fn: int = data["fn"].sum()
        if n_tp + n_fn == 0:
            return 0.0
        return n_tp / (n_tp + n_fn)


class F1Score(AuditorMetric):
    name: str = "f1"
    label: str = "F1 Score"
    inputs: list[str] = ["precision", "recall"]

    def data_call(self, data: pd.DataFrame) -> float:
        # Recalculate to avoid dependency on ordering of metrics
        precision = Precision().data_call(data)
        recall = Recall().data_call(data)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class AUROC(AuditorMetric):
    name: str = "auroc"
    label: str = "AUROC"
    inputs: list[str] = ["_truth", "_pred"]

    def data_call(self, data: pd.DataFrame) -> float:
        try:
            return float(roc_auc_score(data["_truth"], data["_pred"]))
        except ValueError:
            return 0.0


class AUPRC(AuditorMetric):
    name: str = "auprc"
    label: str = "AUPRC"
    inputs: list[str] = ["_truth", "_pred"]

    def data_call(self, data: pd.DataFrame) -> float:
        try:
            return float(average_precision_score(data["_truth"], data["_pred"]))
        except ValueError:
            return 0.0


class MatthewsCorrelationCoefficient(AuditorMetric):
    name: str = "mcc"
    label: str = "Matthews Correlation Coefficient"
    inputs: list[str] = ["tp", "tn", "fp", "fn"]

    def data_call(self, data: pd.DataFrame) -> float:
        tp = data["tp"].sum()
        tn = data["tn"].sum()
        fp = data["fp"].sum()
        fn = data["fn"].sum()

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if denominator == 0:
            return 0.0
        return numerator / denominator


class FBetaScore(AuditorMetric):
    name: str = "fbeta"
    label: str = "F-beta Score"
    inputs: list[str] = ["precision", "recall"]

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.name = f"f{beta:.1f}".replace(".", "_")  # e.g., "f0_5" or "f2_0"
        self.label = f"F{beta:.1f} Score"

    def data_call(self, data: pd.DataFrame) -> float:
        precision = Precision().data_call(data)
        recall = Recall().data_call(data)
        beta_sq = self.beta**2

        if precision + recall == 0:
            return 0.0

        return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
