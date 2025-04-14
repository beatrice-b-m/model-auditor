import importlib
import inspect
from typing import Type
from model_auditor.metric_inputs import AuditorMetricInput


def is_metric_input_valid(cls: type) -> bool:
    return (
        inspect.isclass(cls)
        and hasattr(cls, "name")
        and hasattr(cls, "label")
        and hasattr(cls, "inputs")
        and callable(getattr(cls, "row_call", None))
        and callable(getattr(cls, "data_transform", None))
    )


def collect_metric_inputs() -> dict[str, Type[AuditorMetricInput]]:
    module = importlib.import_module("model_auditor.metric_inputs")

    input_classes = {
        cls.name: cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if is_metric_input_valid(cls) and cls is not AuditorMetricInput
    }

    return input_classes
