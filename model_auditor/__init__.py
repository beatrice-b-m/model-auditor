"""Model Auditor: A library for evaluating ML models with stratified metrics.

This package provides tools for auditing machine learning model performance
across different subgroups, calculating metrics with confidence intervals,
and visualizing results hierarchically.

Example:
    Basic usage for evaluating a binary classifier::

        from model_auditor import Auditor
        from model_auditor.metrics import Sensitivity, Specificity, AUROC

        auditor = Auditor()
        auditor.add_data(df)
        auditor.add_feature(name="gender")
        auditor.add_score(name="prediction_score", threshold=0.5)
        auditor.add_outcome(name="label")
        auditor.set_metrics([Sensitivity(), Specificity(), AUROC()])
        results = auditor.evaluate(score_name="prediction_score")
"""

from model_auditor.core import Auditor