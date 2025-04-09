import importlib
import inspect
from typing import Type
from model_auditor.metric_inputs import AuditorMetricInput


def collect_input_metrics() -> dict[str, Type[AuditorMetricInput]]:
    module = importlib.import_module("model_auditor.metric_inputs")

    input_classes = {
        cls.name: cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, AuditorMetricInput) and cls is not AuditorMetricInput
    }

    return input_classes
