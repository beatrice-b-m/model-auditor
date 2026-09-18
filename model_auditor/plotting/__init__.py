"""Hierarchy compilation and optional Matplotlib rendering.

HierarchyPlotter (the sole package export) compiles node arrays without importing
plotting libraries. For figures, use Auditor.plot_score_distributions() or
ScoreEvaluation.plot_metric_intervals(). The equivalent callable renderers live
in the distributions and intervals submodules and accept the owning object as
their first argument. They create figures without showing or closing them."""

from model_auditor.plotting.plotters import HierarchyPlotter

__all__ = ["HierarchyPlotter"]
