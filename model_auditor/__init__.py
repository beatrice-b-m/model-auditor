"""Evaluate fixed binary predictions overall and within subgroups.

Public root exports are Auditor and InferenceConfig. Metric instances live in
model_auditor.metrics; configuration and result dataclasses in
model_auditor.schemas. The plotting extra provides figure rendering and the
styling extra provides pandas styled tables; neither is needed for evaluation.

Example:
    >>> import pandas as pd
    >>> from model_auditor import Auditor
    >>> from model_auditor.metrics import Sensitivity
    >>> audit = Auditor(pd.DataFrame({"score": [0.9, 0.1], "truth": [1, 0]}))
    >>> audit.add_score("score", threshold=0.5)
    >>> audit.add_outcome("truth")
    >>> audit.set_metrics([Sensitivity()])
    >>> result = audit.evaluate_metrics("score", n_bootstraps=None)
    >>> result.to_numeric_dataframe()["estimate"].tolist()
    [1.0]"""

from model_auditor.core import Auditor
from model_auditor.schemas import InferenceConfig

__all__ = ["Auditor", "InferenceConfig"]
