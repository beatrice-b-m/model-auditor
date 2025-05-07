from typing import Optional
from dataclasses import dataclass, field


@dataclass
class PlotterData:
    labels: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    parents: list = field(default_factory=list)
    values: list = field(default_factory=list)
    colors: list = field(default_factory=list)

    def add(self, label: str, id: str, parent: str, value: int, color: Optional[float] = None) -> None:
        self.labels.append(label)
        self.ids.append(id)
        self.parents.append(parent)
        self.values.append(value)
        
        if color is not None:
            self.colors.append(color)


@dataclass
class HItem:
    """Hierarchy Item"""

    name: str
    query: Optional[str] = None


@dataclass
class HLevel:
    """Hierarchy level"""

    items: list[HItem] = field(default_factory=list)


@dataclass
class Hierarchy:
    """Hierarchy container"""

    levels: list[HLevel] = field(default_factory=list)
