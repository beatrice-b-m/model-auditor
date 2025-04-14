import pandas as pd
from typing import Protocol, Union, runtime_checkable


@runtime_checkable
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


class TruePositives(AuditorMetricInput):
    name: str = "tp"
    label: str = "TP"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        return int((row["_truth"] == 1.0) & (row["_binary_pred"] == 1.0))


class FalsePositives(AuditorMetricInput):
    name: str = "fp"
    label: str = "FP"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        return int((row["_truth"] == 0.0) & (row["_binary_pred"] == 1.0))


class TrueNegatives(AuditorMetricInput):
    name: str = "tn"
    label: str = "TN"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        return int((row["_truth"] == 0.0) & (row["_binary_pred"] == 0.0))


class FalseNegatives(AuditorMetricInput):
    name: str = "fn"
    label: str = "FN"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        return int((row["_truth"] == 1.0) & (row["_binary_pred"] == 0.0))
