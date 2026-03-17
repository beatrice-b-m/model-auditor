"""Error metrics for confusion-matrix group analysis.

Defines the AuditorErrorMetric protocol and built-in implementations.
Currently provides OddsRatio, which quantifies how over- or under-represented
each feature level is within a given confusion-matrix group (TP, TN, FP, or FN)
relative to all other levels combined.

Example::

    from model_auditor.error_metrics import OddsRatio

    # OddsRatio is the default metric used by Auditor.evaluate_errors().
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


class OddsRatio:
    """Canonical 2x2 odds ratio: odds(level in group) / odds(non-level in group).

    Compares how likely a feature level is to fall in a confusion-matrix group
    relative to all other levels combined.  Given a 2x2 contingency table:

        +-----------+------------+-------------+
        |           |  In group  | Not in group|
        +-----------+------------+-------------+
        |  Level    |     a      |      b      |
        |  Not Level|     c      |      d      |
        +-----------+------------+-------------+

    Where:
        a = group_count
        b = full_count - group_count
        c = group_total - group_count
        d = (full_total - full_count) - c

    Odds Ratio:
        OR = (a * d) / (b * c)

    Interpretation:
        - OR = 1: level has the same odds of group membership as all others.
        - OR > 1: level is over-represented in the group.
        - OR < 1: level is under-represented in the group.

    Undefined cases (return NaN):
        - full_count == 0: declared-but-unobserved level; no rows to compare.
        - full_total - full_count == 0: no comparator population (all rows are
          this level); the 2x2 table collapses and OR is undefined.
        - b * c == 0 and a * d == 0: both numerator and denominator are zero;
          the OR is indeterminate.

    Sparse-table arithmetic (denominator zero, numerator non-zero):
        - b == 0 (all level rows are in group) → OR = +inf.
        - c == 0 (no non-level rows in group) and a > 0 → OR = +inf.
        - a == 0 and b > 0 and c > 0 → OR = 0.0 (level absent from group).
    """

    name: str = "odds_ratio"
    label: str = "Odds Ratio"
    ci_eligible: bool = True

    def compute(
        self,
        group_count: int,
        group_total: int,
        full_count: int,
        full_total: int,
    ) -> float:
        """Compute the canonical 2x2 odds ratio.

        Args:
            group_count: Rows in the confusion group belonging to this level.
            group_total: Total rows in the confusion group.
            full_count: Rows in the full dataset belonging to this level.
            full_total: Total rows in the full dataset.

        Returns:
            Odds ratio, or NaN when the table is undefined (see class docstring).
        """
        # Declared-but-unobserved level: no 2x2 table can be formed.
        if full_count == 0:
            return float("nan")

        # No comparator population: all rows belong to this level.
        if full_total - full_count == 0:
            return float("nan")

        a = group_count                              # level ∩ group
        b = full_count - group_count                 # level ∩ not-group
        c = group_total - group_count                # not-level ∩ group
        d = (full_total - full_count) - c            # not-level ∩ not-group

        numerator = a * d
        denominator = b * c

        if denominator == 0:
            # 0/0 is indeterminate; anything/0 is infinite.
            if numerator == 0:
                return float("nan")
            return float("inf")

        return numerator / denominator
