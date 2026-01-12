"""Plotting subpackage for hierarchical visualizations.

This subpackage provides tools for creating hierarchical visualizations
(sunburst, treemap) of model evaluation results across different features
and subgroups.

Example:
    Creating a hierarchical plot::

        from model_auditor.plotting import HierarchyPlotter

        plotter = HierarchyPlotter()
        plotter.set_data(df)
        plotter.set_features(["region", "age_group"])
        plotter.set_score(name="risk_score")
        data = plotter.compile(container="All Data")
"""

from model_auditor.plotting.plotters import HierarchyPlotter
