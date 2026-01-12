"""Metric input calculations for confusion matrix components.

This module defines the AuditorMetricInput protocol and provides
implementations for calculating true positives, false positives,
true negatives, and false negatives from prediction data.

These metric inputs are computed row-wise and added as columns to the
DataFrame, which are then aggregated by the metric classes.
"""

import pandas as pd
from typing import Protocol, Union, runtime_checkable


@runtime_checkable
class AuditorMetricInput(Protocol):
    """Protocol defining the interface for metric input calculators.

    Metric inputs are intermediate values computed row-wise that are
    then used by metrics for aggregation (e.g., TP, FP, TN, FN).

    Attributes:
        name: Column name to use when adding this input to a DataFrame.
        label: Human-readable display name.
        inputs: List of column names required to compute this input.
    """
    name: str
    label: str
    inputs: list[str]

    def row_call(self, row: pd.Series) -> Union[int, float]:
        """Calculate the metric input value for a single row.

        Args:
            row: A single row from the DataFrame as a pandas Series.

        Returns:
            The computed input value (typically 0 or 1 for binary indicators).
        """
        raise NotImplementedError

    def data_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the row_call function to add a new column to the DataFrame.

        Args:
            data: DataFrame to transform.

        Returns:
            The same DataFrame with the new metric input column added.
        """
        data[self.name] = data.apply(self.row_call, axis=1)
        return data


class TruePositives(AuditorMetricInput):
    """Calculator for true positive indicators.

    Produces 1 for rows where both ground truth and prediction are positive,
    0 otherwise.
    """
    name: str = "tp"
    label: str = "TP"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        """Check if the row is a true positive.

        Args:
            row: Series with '_truth' and '_binary_pred' columns.

        Returns:
            1 if both truth and prediction are 1, else 0.
        """
        return int((row["_truth"] == 1.0) & (row["_binary_pred"] == 1.0))


class FalsePositives(AuditorMetricInput):
    """Calculator for false positive indicators.

    Produces 1 for rows where ground truth is negative but prediction is
    positive, 0 otherwise.
    """
    name: str = "fp"
    label: str = "FP"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        """Check if the row is a false positive.

        Args:
            row: Series with '_truth' and '_binary_pred' columns.

        Returns:
            1 if truth is 0 and prediction is 1, else 0.
        """
        return int((row["_truth"] == 0.0) & (row["_binary_pred"] == 1.0))


class TrueNegatives(AuditorMetricInput):
    """Calculator for true negative indicators.

    Produces 1 for rows where both ground truth and prediction are negative,
    0 otherwise.
    """
    name: str = "tn"
    label: str = "TN"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        """Check if the row is a true negative.

        Args:
            row: Series with '_truth' and '_binary_pred' columns.

        Returns:
            1 if both truth and prediction are 0, else 0.
        """
        return int((row["_truth"] == 0.0) & (row["_binary_pred"] == 0.0))


class FalseNegatives(AuditorMetricInput):
    """Calculator for false negative indicators.

    Produces 1 for rows where ground truth is positive but prediction is
    negative, 0 otherwise.
    """
    name: str = "fn"
    label: str = "FN"
    inputs: list[str] = ["_truth", "_binary_pred"]

    def row_call(self, row: pd.Series) -> int:
        """Check if the row is a false negative.

        Args:
            row: Series with '_truth' and '_binary_pred' columns.

        Returns:
            1 if truth is 1 and prediction is 0, else 0.
        """
        return int((row["_truth"] == 1.0) & (row["_binary_pred"] == 0.0))
