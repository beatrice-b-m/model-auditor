"""Integration coverage for hierarchy compilation and custom aggregators."""

import pandas as pd
import pytest

from model_auditor.plotting import HierarchyPlotter
from model_auditor.plotting.schemas import Hierarchy, HItem, HLevel


def test_hierarchy_counts_colors_and_parent_links():
    data = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "group": ["A", "B", "A", "A"],
            "score": [0.2, 0.4, 0.6, 0.8],
        }
    )
    plotter = HierarchyPlotter()
    plotter.set_data(data)
    plotter.set_features(["region", "group"])
    plotter.set_score("score")
    plotter.set_aggregator("mean")
    result = plotter.compile("All")
    assert result.ids == ["All", "All$N", "All$N$A", "All$N$B", "All$S", "All$S$A"]
    assert result.values == [4, 2, 1, 1, 2, 2]
    assert result.parents == ["", "All", "All$N", "All$N", "All", "All$S"]
    assert result.colors == pytest.approx([0.5, 0.3, 0.2, 0.4, 0.7, 0.7])


def test_custom_aggregator_receives_whole_dataframe():
    plotter = HierarchyPlotter()
    plotter.set_data(pd.DataFrame({"group": ["A", "A", "B"], "score": [0.1, 0.2, 0.3]}))
    plotter.set_features(["group"])
    plotter.set_score("score")
    plotter.set_aggregator(lambda group: group["score"].sum() / len(group))
    assert plotter.compile("All").colors == pytest.approx([0.2, 0.15, 0.3])


def test_empty_hierarchy_returns_root():
    plotter = HierarchyPlotter()
    plotter.set_data(pd.DataFrame({"score": [0.1, 0.2]}))
    plotter.set_features([])
    plotter.set_score("score")
    plotter.set_aggregator("size")
    result = plotter.compile("All")
    assert result.ids == ["All"]
    assert result.colors == [2.0]


def test_composite_hierarchy_preserves_caller_data():
    data = pd.DataFrame({"a": ["A", "A"], "b": ["B", "C"]})
    before = data.copy(deep=True)
    plotter = HierarchyPlotter()
    plotter.set_data(data)
    plotter.set_features(Hierarchy([HLevel([HItem("a"), HItem("b")])]))
    with pytest.warns(UserWarning, match="No score"):
        result = plotter.compile("All")
    assert result.labels == ["All", "A & B", "A & C"]
    pd.testing.assert_frame_equal(data, before)
