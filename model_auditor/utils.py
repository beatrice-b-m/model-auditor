"""Utility functions for the model auditor package.

This module provides helper functions for dynamically collecting and
validating metric input classes.
"""

import importlib
import inspect
from typing import Type
from model_auditor.metric_inputs import AuditorMetricInput


def is_metric_input_valid(cls: type) -> bool:
    """Check if a class implements the AuditorMetricInput protocol.

    Validates that a class has all required attributes and methods
    to function as a metric input calculator.

    Args:
        cls: The class to validate.

    Returns:
        True if the class has name, label, inputs attributes and
        row_call, data_transform methods; False otherwise.
    """
    return (
        inspect.isclass(cls)
        and hasattr(cls, "name")
        and hasattr(cls, "label")
        and hasattr(cls, "inputs")
        and callable(getattr(cls, "row_call", None))
        and callable(getattr(cls, "data_transform", None))
    )


def collect_metric_inputs() -> dict[str, Type[AuditorMetricInput]]:
    """Dynamically collect all metric input classes from the metric_inputs module.

    Scans the metric_inputs module for classes that implement the
    AuditorMetricInput protocol and returns them as a dictionary.

    Returns:
        Dictionary mapping metric input names to their class types,
        e.g., {"tp": TruePositives, "fp": FalsePositives, ...}.
    """
    module = importlib.import_module("model_auditor.metric_inputs")

    input_classes = {
        cls.name: cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if is_metric_input_valid(cls) and cls is not AuditorMetricInput
    }

    return input_classes
