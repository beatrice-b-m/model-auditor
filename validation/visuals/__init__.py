"""Executable visual examples for galleries, tests, and documentation assets.

The collection provides deterministic synthetic data, named example functions,
and an explicit documentation subset.  ``python -m validation.visuals.gallery``
renders every example into ``artifacts/visuals/`` (ignored by Git).
"""

from validation.visuals.data import (
    DIFFICULT_FEATURES,
    DOC_FEATURES,
    SCORE_NAME,
    difficult_auditor,
    difficult_evaluation,
    example_auditor,
    example_evaluation,
)
from validation.visuals.examples import (
    DEVELOPER_EXAMPLES,
    DOCUMENTATION_EXAMPLES,
    EXAMPLES,
    Example,
    run_example,
)

__all__ = [
    "DEVELOPER_EXAMPLES",
    "DIFFICULT_FEATURES",
    "DOC_FEATURES",
    "DOCUMENTATION_EXAMPLES",
    "EXAMPLES",
    "SCORE_NAME",
    "Example",
    "difficult_auditor",
    "difficult_evaluation",
    "example_auditor",
    "example_evaluation",
    "run_example",
]
