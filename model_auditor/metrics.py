from typing import Protocol, Union, Optional
import pandas as pd


class AuditorMetricInput(Protocol):
    name: str
    label: str
    inputs: list[str]

    def row_call(self, row: pd.Series) -> Union[int, float]:
        """
        method called on each row of a dataframe to calculate a metric
        """
        raise NotImplementedError

    def data_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        method called on a dataframe to add a metric input column inplace
        """
        data[self.name] = data.apply(self.row_call, axis=1)
        return data


class AuditorMetric(Protocol):
    name: str
    label: str
    inputs: list[str]

    def data_call(self, data: pd.DataFrame) -> Union[float, int]:
        """
        method called on a dataframe to calculate a metric
        """
        raise NotImplementedError


class TruePositives(AuditorMetricInput):
    name: str = "tp"
    label: str = "TP"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        return ((row["_truth"] == 1.0) & (row["_binary_pred"] == 1.0)).astype(int)


class FalsePositives(AuditorMetricInput):
    name: str = "fp"
    label: str = "FP"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        return ((row["_truth"] == 1.0) & (row["_binary_pred"] == 1.0)).astype(int)


class TrueNegatives(AuditorMetricInput):
    name: str = "tn"
    label: str = "TN"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        return ((row["_truth"] == 1.0) & (row["_binary_pred"] == 1.0)).astype(int)


class FalseNegatives(AuditorMetricInput):
    name: str = "fn"
    label: str = "FN"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        return ((row["_truth"] == 1.0) & (row["_binary_pred"] == 1.0)).astype(int)


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
