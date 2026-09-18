"""Metric input calculations for confusion matrix components.

This module defines the AuditorMetricInput protocol and provides
implementations for calculating true positives, false positives,
true negatives, and false negatives from prediction data.

Built-in transforms compute vectorized 0/1 indicators and mutate the supplied
DataFrame by adding or replacing a column. Auditor uses these on private slices.
Direct callers should pass a copy when they need to preserve their input.
row_call remains available for individual rows and custom input implementations.
"""

from typing import Protocol, Union, runtime_checkable

import pandas as pd


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
            The same DataFrame, mutated by adding or overwriting self.name.
            The default protocol implementation uses row-wise apply; built-in
            classes override it with vectorized transforms.

        Raises:
            KeyError: An input required by row_call is absent.
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

    def data_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add or replace the tp indicator column using vectorized comparisons.

        Args:
            data: DataFrame with nonmissing binary _truth and _binary_pred columns.
                Binary validity is not checked here; Auditor evaluation validates it.

        Returns:
            The same DataFrame object, mutated in place with integer 0/1 column tp.
            Row order and index are preserved; copy data first to avoid mutation.

        Raises:
            KeyError: A required input column is missing."""
        data[self.name] = (
            (data["_truth"] == 1.0) & (data["_binary_pred"] == 1.0)
        ).astype(int)
        return data


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

    def data_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add or replace the fp indicator column using vectorized comparisons.

        Args:
            data: DataFrame with nonmissing binary _truth and _binary_pred columns.
                Binary validity is not checked here; Auditor evaluation validates it.

        Returns:
            The same DataFrame object, mutated in place with integer 0/1 column fp.
            Row order and index are preserved; copy data first to avoid mutation.

        Raises:
            KeyError: A required input column is missing."""
        data[self.name] = (
            (data["_truth"] == 0.0) & (data["_binary_pred"] == 1.0)
        ).astype(int)
        return data


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

    def data_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add or replace the tn indicator column using vectorized comparisons.

        Args:
            data: DataFrame with nonmissing binary _truth and _binary_pred columns.
                Binary validity is not checked here; Auditor evaluation validates it.

        Returns:
            The same DataFrame object, mutated in place with integer 0/1 column tn.
            Row order and index are preserved; copy data first to avoid mutation.

        Raises:
            KeyError: A required input column is missing."""
        data[self.name] = (
            (data["_truth"] == 0.0) & (data["_binary_pred"] == 0.0)
        ).astype(int)
        return data


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

    def data_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add or replace the fn indicator column using vectorized comparisons.

        Args:
            data: DataFrame with nonmissing binary _truth and _binary_pred columns.
                Binary validity is not checked here; Auditor evaluation validates it.

        Returns:
            The same DataFrame object, mutated in place with integer 0/1 column fn.
            Row order and index are preserved; copy data first to avoid mutation.

        Raises:
            KeyError: A required input column is missing."""
        data[self.name] = (
            (data["_truth"] == 1.0) & (data["_binary_pred"] == 0.0)
        ).astype(int)
        return data
