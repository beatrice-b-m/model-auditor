"""Data schemas for hierarchical plotting.

This module defines the data structures used for building hierarchical
visualizations, including the plotter data container and hierarchy
definitions.
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class PlotterData:
    """Container for hierarchical plot data.

    Stores the parallel arrays needed to construct sunburst or treemap
    plots in libraries like Plotly.

    Attributes:
        labels: Display labels for each node.
        ids: Unique identifiers for each node.
        parents: Parent node IDs (empty string for root nodes).
        values: Numeric values (typically counts) for sizing nodes.
        colors: Optional color values for coloring nodes.
    """
    labels: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    parents: list = field(default_factory=list)
    values: list = field(default_factory=list)
    colors: list = field(default_factory=list)

    def add(self, label: str, id: str, parent: str, value: int, color: Optional[float] = None) -> None:
        """Add a node to the plotter data.

        Args:
            label: Display label for the node.
            id: Unique identifier for the node.
            parent: ID of the parent node (empty string for root).
            value: Numeric value for sizing the node.
            color: Optional color value for the node.
        """
        self.labels.append(label)
        self.ids.append(id)
        self.parents.append(parent)
        self.values.append(value)

        if color is not None:
            self.colors.append(color)


@dataclass
class HItem:
    """A single item within a hierarchy level.

    Represents a feature column that can optionally be filtered by a query.

    Attributes:
        name: Column name in the DataFrame.
        query: Optional pandas query string to filter when this item applies.
            If None, the item always applies.
    """

    name: str
    query: Optional[str] = None


@dataclass
class HLevel:
    """A single level in the hierarchy containing one or more items.

    When multiple items are present at a level, their values are concatenated
    to form a composite feature level.

    Attributes:
        items: List of HItem objects at this hierarchy level.
    """

    items: list[HItem] = field(default_factory=list)


@dataclass
class Hierarchy:
    """Container for a complete feature hierarchy definition.

    Defines the structure of a hierarchical visualization as a sequence
    of levels, where each level contains one or more feature items.

    Attributes:
        levels: Ordered list of HLevel objects from root to leaf.

    Example:
        Creating a two-level hierarchy::

            hierarchy = Hierarchy(levels=[
                HLevel([HItem(name="region")]),
                HLevel([HItem(name="age_group")])
            ])
    """

    levels: list[HLevel] = field(default_factory=list)
