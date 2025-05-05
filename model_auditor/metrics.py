from typing import Protocol, Union, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class AuditorMetric(Protocol):
    name: str
    label: str
    inputs: list[str]
    ci_eligible: bool

    def data_call(self, data: pd.DataFrame) -> Union[float, int]:
        """
        method called on a dataframe to calculate a metric
        """
        raise NotImplementedError


class Sensitivity(AuditorMetric):
    name: str = "sensitivity"
    label: str = "Sensitivity"
    inputs: list[str] = ["tp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        n_tp: int = data["tp"].sum()
        n_fn: int = data["fn"].sum()
        return n_tp / (n_tp + n_fn + eps)


class Specificity(AuditorMetric):
    name: str = "specificity"
    label: str = "Specificity"
    inputs: list[str] = ["tn", "fp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        n_tn: int = data["tn"].sum()
        n_fp: int = data["fp"].sum()
        return n_tn / (n_tn + n_fp + eps)


class Precision(AuditorMetric):
    name: str = "precision"
    label: str = "Precision"
    inputs: list[str] = ["tp", "fp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        n_tp: int = data["tp"].sum()
        n_fp: int = data["fp"].sum()
        return n_tp / (n_tp + n_fp + eps)


class Recall(AuditorMetric):
    name: str = "recall"
    label: str = "Recall"
    inputs: list[str] = ["tp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        n_tp: int = data["tp"].sum()
        n_fn: int = data["fn"].sum()
        return n_tp / (n_tp + n_fn + eps)


class F1Score(AuditorMetric):
    name: str = "f1"
    label: str = "F1 Score"
    inputs: list[str] = ["tp", "fp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        # Recalculate to avoid dependency on ordering of metrics
        precision = Precision().data_call(data)
        recall = Recall().data_call(data)
        return 2 * (precision * recall) / (precision + recall + eps)


class AUROC(AuditorMetric):
    name: str = "auroc"
    label: str = "AUROC"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        try:
            return float(roc_auc_score(data["_truth"], data["_pred"]))
        except ValueError:
            return 0.0


class AUPRC(AuditorMetric):
    name: str = "auprc"
    label: str = "AUPRC"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        try:
            return float(average_precision_score(data["_truth"], data["_pred"]))
        except ValueError:
            return 0.0


class MatthewsCorrelationCoefficient(AuditorMetric):
    name: str = "mcc"
    label: str = "Matthews Correlation Coefficient"
    inputs: list[str] = ["tp", "tn", "fp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
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
    name: str = "fbeta"
    label: str = "F-beta Score"
    inputs: list[str] = ["precision", "recall"]
    ci_eligible: bool = True

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


class TPR(Sensitivity):
    name: str = "tpr"
    label: str = "TPR"


class TNR(Specificity):
    name: str = "tnr"
    label: str = "TNR"


class FPR(AuditorMetric):
    name: str = "fpr"
    label: str = "FPR"
    inputs: list[str] = ["fp", "tn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        n_fp: int = data["fp"].sum()
        n_tn: int = data["tn"].sum()
        return n_fp / (n_fp + n_tn + eps)


class FNR(AuditorMetric):
    name: str = "fnr"
    label: str = "FNR"
    inputs: list[str] = ["fn", "tp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame, eps: float = 1e-8) -> float:
        n_fn: int = data["fn"].sum()
        n_tp: int = data["tp"].sum()
        return n_fn / (n_fn + n_tp + eps)


class nData(AuditorMetric):
    name: str = "n"
    label: str = "N"
    inputs: list[str] = []
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        return len(data)


class nTP(AuditorMetric):
    name: str = "n_tp"
    label: str = "TP"
    inputs: list[str] = ['tp']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        return data['tp'].sum()
    

class nTN(AuditorMetric):
    name: str = "n_tn"
    label: str = "TN"
    inputs: list[str] = ['tn']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        return data['tn'].sum()
  

class nFP(AuditorMetric):
    name: str = "n_fp"
    label: str = "FP"
    inputs: list[str] = ['fp']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        return data['fp'].sum()
    

class nFN(AuditorMetric):
    name: str = "n_fn"
    label: str = "FN"
    inputs: list[str] = ['fn']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        return data['fn'].sum()


class nPositive(AuditorMetric):
    name: str = "n_pos"
    label: str = "Pos."
    inputs: list[str] = ['_truth']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        return (data['_truth'] == 1).astype(int).sum()
    

class nNegative(AuditorMetric):
    name: str = "n_neg"
    label: str = "Neg."
    inputs: list[str] = ['_truth']
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        return (data['_truth'] == 0).astype(int).sum()