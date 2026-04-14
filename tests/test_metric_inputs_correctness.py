"""Deterministic correctness tests for confusion-matrix input transforms."""

import pandas as pd

from model_auditor.metric_inputs import (
    FalseNegatives,
    FalsePositives,
    TrueNegatives,
    TruePositives,
)


BASE_DF = pd.DataFrame(
    {
        "_truth": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "_binary_pred": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    }
)

EXPECTED = {
    "tp": [1, 0, 0, 0, 1, 0],
    "fp": [0, 0, 1, 0, 0, 0],
    "tn": [0, 0, 0, 1, 0, 1],
    "fn": [0, 1, 0, 0, 0, 0],
}


def test_row_call_matches_expected_indicators():
    calculators = {
        "tp": TruePositives(),
        "fp": FalsePositives(),
        "tn": TrueNegatives(),
        "fn": FalseNegatives(),
    }

    for name, calculator in calculators.items():
        got = [calculator.row_call(row) for _, row in BASE_DF.iterrows()]
        assert got == EXPECTED[name]


def test_data_transform_creates_expected_columns_and_totals():
    calculators = [TruePositives(), FalsePositives(), TrueNegatives(), FalseNegatives()]
    transformed = BASE_DF.copy()

    for calculator in calculators:
        transformed = calculator.data_transform(transformed)

    for col, expected_values in EXPECTED.items():
        assert transformed[col].tolist() == expected_values

    assert transformed["tp"].sum() == 2
    assert transformed["fp"].sum() == 1
    assert transformed["tn"].sum() == 2
    assert transformed["fn"].sum() == 1


def test_each_row_maps_to_exactly_one_confusion_indicator():
    calculators = [TruePositives(), FalsePositives(), TrueNegatives(), FalseNegatives()]
    transformed = BASE_DF.copy()

    for calculator in calculators:
        transformed = calculator.data_transform(transformed)

    row_totals = transformed[["tp", "fp", "tn", "fn"]].sum(axis=1)
    assert row_totals.tolist() == [1, 1, 1, 1, 1, 1]
