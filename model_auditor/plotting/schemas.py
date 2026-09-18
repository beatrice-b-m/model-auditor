"""Data schemas for hierarchical plotting.

This module defines the data structures used for building hierarchical
visualizations, including the plotter data container and hierarchy
definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PlotterData:
    """Container for hierarchical plot data.

    Stores the parallel arrays needed to construct sunburst or treemap
    plots in libraries like Plotly.

    Attributes:
        labels: Original group values (any pandas group-key type) or string labels.
            New empty list by default; values are not coerced to strings.
        ids: Node IDs; new empty list by default. Uniqueness is the caller
            or compiler's responsibility; add() performs no validation.
        parents: Parent node IDs (empty string for root); new empty list by default.
        values: Integer row counts for sizing nodes; new empty list by default.
        colors: Scalar aggregation results or None; new empty list by default.
            Lists are mutable, parallel, and stored by reference if supplied.
            No JSON conversion is performed; encode nonstandard labels and
            NaN colors explicitly for strict JSON serialization.
    """

    labels: list[Any] = field(default_factory=list)
    """Original group values (any pandas group-key type) or string labels.
    New empty list by default; values are not coerced to strings."""
    ids: list[str] = field(default_factory=list)
    """Node IDs; new empty list by default. Uniqueness is the caller
    or compiler's responsibility; add() performs no validation."""
    parents: list[str] = field(default_factory=list)
    """Parent node IDs (empty string for root); new empty list by default."""
    values: list[int] = field(default_factory=list)
    """Integer row counts for sizing nodes; new empty list by default."""
    colors: list[Optional[float]] = field(default_factory=list)
    """Scalar aggregation results or None; new empty list by default.
    Lists are mutable, parallel, and stored by reference if supplied.
    No JSON conversion is performed; encode nonstandard labels and
    NaN colors explicitly for strict JSON serialization."""

    def add(
        self,
        label: Any,
        node_id: str,
        parent: str,
        value: int,
        color: Optional[float] = None,
    ) -> None:
        """Append one entry to every parallel list in place, without validation.

        Args:
            label: Display label for the node.
            node_id: Unique identifier for the node.
            parent: ID of the parent node (empty string for root).
            value: Numeric value for sizing the node.
            color: Optional color value for the node.
        """
        self.labels.append(label)
        self.ids.append(node_id)
        self.parents.append(parent)
        self.values.append(value)
        self.colors.append(color)


@dataclass
class HItem:
    """A single item within a hierarchy level.

    Represents a feature column that can optionally be filtered by a query.

    Attributes:
        name: Column name in the DataFrame.
        query: Optional pandas eval expression; default None always applies.
            An item applies only when the expression is true for every row
            in the current branch. It does not filter rows. Use only trusted
            expressions; pandas eval can execute arbitrary code.
    """

    name: str
    """Column name in the DataFrame."""
    query: Optional[str] = None
    """Optional pandas eval expression; default None always applies.
    An item applies only when the expression is true for every row
    in the current branch. It does not filter rows. Use only trusted
    expressions; pandas eval can execute arbitrary code."""


@dataclass
class HLevel:
    """A single level in the hierarchy containing one or more items.

    When multiple items are present at a level, their values are concatenated
    to form a composite feature level.

    Attributes:
        items: Ordered HItem objects; new empty list by default. Applicable
            items are joined with " & "; no applicable items ends the branch.
    """

    items: list[HItem] = field(default_factory=list)
    """Ordered HItem objects; new empty list by default. Applicable
    items are joined with " & "; no applicable items ends the branch."""


@dataclass
class Hierarchy:
    """Container for a complete feature hierarchy definition.

    Defines the structure of a hierarchical visualization as a sequence
    of levels, where each level contains one or more feature items.

    Attributes:
        levels: Ordered HLevel objects from root to leaf; new empty list
            by default. Supplied lists are retained by reference.

    Example:
        Creating a two-level hierarchy::

            hierarchy = Hierarchy(levels=[
                HLevel([HItem(name="region")]),
                HLevel([HItem(name="age_group")])
            ])
    """

    levels: list[HLevel] = field(default_factory=list)
    """Ordered HLevel objects from root to leaf; new empty list
    by default. Supplied lists are retained by reference."""
