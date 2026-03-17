"""Error metrics for confusion-matrix group analysis.

Defines the AuditorErrorMetric protocol and built-in implementations.
Currently provides RepresentationRatio, which quantifies how over- or
under-represented each feature level is within a given confusion-matrix group
(TP, TN, FP, or FN) relative to its prevalence in the full dataset.

Example::

    from model_auditor.error_metrics import RepresentationRatio

    # RepresentationRatio is the default metric used by Auditor.evaluate_errors().
    # To implement a custom error metric, follow the AuditorErrorMetric protocol:

    class MyErrorMetric:
        name = "my_metric"
        label = "My Metric"
        ci_eligible = True

        def compute(
            self, group_count: int, group_total: int,
            full_count: int, full_total: int
        ) -> float:
            ...
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class AuditorErrorMetric(Protocol):
    """Protocol for error metrics used in Auditor.evaluate_errors().

    An error metric receives pre-aggregated counts from one confusion-matrix
    group and the full dataset for a specific feature level, and returns a
    scalar value representing some property of that group/level combination.

    Attributes:
        name: Unique identifier used as the dict key in result objects.
        label: Human-readable display name.
        ci_eligible: Whether bootstrap confidence intervals apply to this metric.
    """

    name: str
    label: str
    ci_eligible: bool

    def compute(
        self,
        group_count: int,
        group_total: int,
        full_count: int,
        full_total: int,
    ) -> float:
        """Compute the metric from pre-aggregated counts.

        Args:
            group_count: Rows in the confusion group that belong to this level.
            group_total: Total rows in the confusion group.
            full_count: Rows in the full dataset that belong to this level.
            full_total: Total rows in the full dataset.

        Returns:
            Scalar metric value (may be NaN for undefined cases).
        """
        raise NotImplementedError


class RepresentationRatio:
    """Representation ratio: P(level | group) / P(level | full dataset).

    Quantifies whether a feature level is over- or under-represented in a
    confusion-matrix group relative to its prevalence in the full dataset.

    Interpretation:
        - 1.0: the level appears in the group at exactly its baseline rate.
        - > 1.0: over-represented in the group (more common in errors / correct
          predictions than in the data as a whole).
        - < 1.0: under-represented.

    Undefined cases:
        - full_count == 0 (level absent from the full dataset): returns NaN.
          This covers declared-but-unobserved categorical levels.
        - group_total == 0 (empty confusion group): group rate is treated as
          0.0, yielding ratio = 0.0 when full_count > 0.
    """

    name: str = "representation_ratio"
    label: str = "Representation Ratio"
    ci_eligible: bool = True

    def compute(
        self,
        group_count: int,
        group_total: int,
        full_count: int,
        full_total: int,
    ) -> float:
        """Compute the representation ratio.

        Args:
            group_count: Rows in the confusion group belonging to this level.
            group_total: Total rows in the confusion group.
            full_count: Rows in the full dataset belonging to this level.
            full_total: Total rows in the full dataset.

        Returns:
            Representation ratio, or NaN if the level is absent from the
            full dataset (full_count == 0 or full_total == 0).
        """
        # The baseline prevalence is undefined when the level has no rows in
        # the full dataset; the ratio cannot be computed.
        if full_total == 0 or full_count == 0:
            return float("nan")

        baseline = full_count / full_total

        # When the confusion group is empty the level has zero representation
        # in it, so the group rate is 0.
        group_rate = group_count / group_total if group_total > 0 else 0.0

        return group_rate / baseline
