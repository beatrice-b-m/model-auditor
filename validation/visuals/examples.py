"""Named, executable visual examples and the documentation subset.

Each example stores the exact standalone script that generates its output.
The gallery and the opt-in pytest capture both ``exec`` that script, so the
recorded code can never drift from the rendered artifact.

``kind`` selects the artifact writer:

- ``"figure"`` — mapping of feature name to ``(Figure, Axes)`` from the
  interval or distribution plotters.
- ``"table"`` — a pandas ``Styler`` rendered to standalone HTML.
- ``"plotly"`` — a Plotly ``Figure`` rendered to self-contained HTML.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any

KIND_FIGURE = "figure"
KIND_TABLE = "table"
KIND_PLOTLY = "plotly"


@dataclass(frozen=True)
class Example:
    """One reproducible visual example."""

    slug: str
    title: str
    kind: str
    description: str
    caption: str
    code: str
    outputs: tuple[str, ...]
    expected_text: tuple[str, ...]
    documentation: bool = False


def _script(source: str) -> str:
    return dedent(source).strip() + "\n"


def run_example(example: Example) -> dict[str, Any]:
    """Execute ``example.code`` and return its declared output variables."""
    namespace: dict[str, Any] = {"__name__": "__visual_example__"}
    exec(compile(example.code, f"<visual:{example.slug}>", "exec"), namespace)
    missing = [name for name in example.outputs if name not in namespace]
    if missing:
        raise AssertionError(
            f"Example {example.slug!r} did not define outputs {missing!r}."
        )
    return {name: namespace[name] for name in example.outputs}


def _example(
    slug: str,
    title: str,
    kind: str,
    description: str,
    caption: str,
    code: str,
    outputs: tuple[str, ...],
    expected_text: tuple[str, ...],
    documentation: bool = False,
) -> Example:
    return Example(
        slug=slug,
        title=title,
        kind=kind,
        description=description,
        caption=caption,
        code=_script(code),
        outputs=outputs,
        expected_text=expected_text,
        documentation=documentation,
    )


INTERVALS_DEFAULT = _example(
    slug="intervals-default",
    title="Default interval plot",
    kind=KIND_FIGURE,
    description=(
        "Horizontal confidence interval plot for sensitivity across sex, age "
        "group, and site, with an Overall comparator and sample-size annotations."
    ),
    caption=(
        "Default interval rendering: one subplot per feature, an Overall "
        "comparator level, and pointwise confidence intervals."
    ),
    code="""
        from validation.visuals.data import example_evaluation

        evaluation = example_evaluation()
        plots = evaluation.plot_metric_intervals("sensitivity")
    """,
    outputs=("plots",),
    expected_text=(
        "Sensitivity",
        "Overall",
        "Female",
        "Male",
        "Age group",
        "Site",
    ),
    documentation=True,
)

DISTRIBUTIONS_DEFAULT = _example(
    slug="distributions-default",
    title="Default score distributions",
    kind=KIND_FIGURE,
    description=(
        "Density histograms of the risk score, one stacked subplot per level "
        "of sex, age group, and site."
    ),
    caption=(
        "Default score distributions share one bin grid per feature so levels "
        "are directly comparable."
    ),
    code="""
        from validation.visuals.data import SCORE_NAME, example_auditor

        auditor = example_auditor()
        plots = auditor.plot_score_distributions(SCORE_NAME)
    """,
    outputs=("plots",),
    expected_text=("Sex", "Age group", "Site", "Risk score", "Female", "Male"),
    documentation=True,
)

DISTRIBUTIONS_CLASS_SPLIT = _example(
    slug="distributions-class-split",
    title="Class-split score distributions",
    kind=KIND_FIGURE,
    description=(
        "Overlaid negative/positive histograms per level using the configured "
        "binary outcome (split_classes=True)."
    ),
    caption=(
        "Class-split distributions overlay the outcome classes within each "
        "level; each class is density-normalized separately."
    ),
    code="""
        from validation.visuals.data import SCORE_NAME, example_auditor

        auditor = example_auditor()
        plots = auditor.plot_score_distributions(
            SCORE_NAME,
            feature_names=["sex", "age_group"],
            split_classes=True,
        )
    """,
    outputs=("plots",),
    expected_text=("Negative", "Positive", "Female", "Male"),
    documentation=True,
)

PERFORMANCE_TABLES_NEUTRAL = _example(
    slug="performance-tables-neutral",
    title="Neutral performance table",
    kind=KIND_TABLE,
    description=(
        "Formatted subgroup metrics with the default neutral styling "
        "(rank=False); no cell carries performance color."
    ),
    caption=(
        "Neutral performance tables format estimates and intervals without "
        "implying better or worse performance."
    ),
    code="""
        from validation.visuals.data import example_evaluation

        evaluation = example_evaluation()
        styled = evaluation.style_dataframe(metric_labels=True)
    """,
    outputs=("styled",),
    expected_text=("Sensitivity", "Specificity", "F1 Score", "AUROC", "Female"),
    documentation=True,
)

PERFORMANCE_TABLES_RANKED = _example(
    slug="performance-tables-ranked",
    title="Ranked performance table",
    kind=KIND_TABLE,
    description=(
        "Descriptive within-feature tiers enabled with rank=True and explicit "
        "metric directions."
    ),
    caption=(
        "Ranked tables add relative tier coloring inside each feature; tiers "
        "are descriptive and do not express significance."
    ),
    code="""
        from validation.visuals.data import example_evaluation

        evaluation = example_evaluation()
        styled = evaluation.style_dataframe(metric_labels=True, rank=True)
    """,
    outputs=("styled",),
    expected_text=("Sensitivity", "Specificity", "F1 Score", "AUROC"),
    documentation=True,
)

ERROR_TABLES = _example(
    slug="error-tables",
    title="Error enrichment table",
    kind=KIND_TABLE,
    description=(
        "Neutral cross-group confusion-membership enrichment table with odds "
        "ratios and inline confidence intervals."
    ),
    caption=(
        "Error tables fold OR confidence intervals into the estimate cell and "
        "stay neutral: enrichment has no universal better/worse direction."
    ),
    code="""
        from validation.visuals.data import example_error_evaluation

        evaluation = example_error_evaluation()
        styled = evaluation.style_dataframe(metric_labels=True)
    """,
    outputs=("styled",),
    expected_text=("Odds Ratio", "% overall", "TP", "TN", "FP", "FN"),
    documentation=True,
)

HIERARCHY_PLOTLY = _example(
    slug="hierarchy-plotly",
    title="Hierarchy treemap",
    kind=KIND_PLOTLY,
    description=(
        "Plotly rendering of HierarchyPlotter output: a mean-risk treemap "
        "nested by sex then site."
    ),
    caption=(
        "Compiled hierarchy arrays render directly in Plotly; cell area is "
        "level support and cell color is the configured aggregate."
    ),
    code="""
        import plotly.express as px

        from model_auditor.plotting import HierarchyPlotter
        from validation.visuals.data import SCORE_NAME, example_frame

        plotter = HierarchyPlotter()
        plotter.set_data(example_frame())
        plotter.set_features(["sex", "site"])
        plotter.set_score(SCORE_NAME, label="Risk score")
        plotter.set_aggregator("mean")
        hierarchy = plotter.compile("All patients")

        figure = px.treemap(
            names=hierarchy.labels,
            ids=hierarchy.ids,
            parents=hierarchy.parents,
            values=hierarchy.values,
            color=hierarchy.colors,
            color_continuous_scale="Blues",
        )
    """,
    outputs=("figure",),
    expected_text=("All patients", "Female", "Male"),
    documentation=True,
)

# ---------------------------------------------------------------------------
# Developer collection: difficult rendering cases
# ---------------------------------------------------------------------------

ROTATED_LONG_LABELS = _example(
    slug="rotated-long-labels",
    title="Rotated plot with long labels",
    kind=KIND_FIGURE,
    description=(
        "Vertical interval plot over four long service-region labels with "
        "sample-size annotations."
    ),
    caption="Rotated interval plots must keep annotations legible for long labels.",
    code="""
        from validation.visuals.data import difficult_evaluation

        evaluation = difficult_evaluation()
        plots = evaluation.plot_metric_intervals(
            "sensitivity",
            feature_names=["service_region"],
            rotate_plots=True,
        )
    """,
    outputs=("plots",),
    expected_text=("Sensitivity", "Northeast Metropolitan Service Area"),
)

ROTATED_DENSE_LEVELS = _example(
    slug="rotated-dense-levels",
    title="Rotated plot with many levels",
    kind=KIND_FIGURE,
    description=(
        "Vertical interval plot across ten clinic levels, where adjacent "
        "annotations would otherwise overlap."
    ),
    caption="Dense rotated plots expose annotation collisions.",
    code="""
        from validation.visuals.data import difficult_evaluation

        evaluation = difficult_evaluation()
        plots = evaluation.plot_metric_intervals(
            "specificity",
            feature_names=["clinic"],
            rotate_plots=True,
        )
    """,
    outputs=("plots",),
    expected_text=("Specificity", "Clinic 01", "Clinic 10"),
)

UNDEFINED_LEVELS = _example(
    slug="undefined-levels",
    title="Undefined and unobserved levels",
    kind=KIND_FIGURE,
    description=(
        "Interval plot over a categorical feature with an unobserved declared "
        "level; omitted levels are named with a readable reason."
    ),
    caption="Omitted levels stay visible with a human-readable reason.",
    code="""
        from validation.visuals.data import difficult_evaluation

        evaluation = difficult_evaluation()
        plots = evaluation.plot_metric_intervals(
            "auroc",
            feature_names=["referral"],
        )
    """,
    outputs=("plots",),
    expected_text=("AUROC", "Not drawn", "Declined", "estimate undefined"),
)

MANY_LEVEL_DISTRIBUTIONS = _example(
    slug="many-level-distributions",
    title="Distribution stack with many levels",
    kind=KIND_FIGURE,
    description=(
        "Ten stacked clinic density histograms, exercising figure height and "
        "shared bin grids."
    ),
    caption="Distribution stacks scale vertically with the level count.",
    code="""
        from validation.visuals.data import SCORE_NAME, difficult_auditor

        auditor = difficult_auditor()
        plots = auditor.plot_score_distributions(
            SCORE_NAME,
            feature_names=["clinic"],
        )
    """,
    outputs=("plots",),
    expected_text=("Clinic 01", "Clinic 10", "Risk score"),
)

WIDE_DIFFICULT_ERROR_TABLE = _example(
    slug="wide-difficult-error-table",
    title="Wide error table with long labels",
    kind=KIND_TABLE,
    description=(
        "Wide cross-group enrichment table over long service-region labels "
        "and ten clinic levels."
    ),
    caption="Wide tables must stay readable at documentation widths.",
    code="""
        from validation.visuals.data import difficult_error_evaluation

        evaluation = difficult_error_evaluation()
        styled = evaluation.style_dataframe(metric_labels=True)
    """,
    outputs=("styled",),
    expected_text=("Odds Ratio", "Northeast Metropolitan Service Area", "Clinic 10"),
)

DOCUMENTATION_EXAMPLES: tuple[Example, ...] = (
    INTERVALS_DEFAULT,
    DISTRIBUTIONS_DEFAULT,
    DISTRIBUTIONS_CLASS_SPLIT,
    PERFORMANCE_TABLES_NEUTRAL,
    PERFORMANCE_TABLES_RANKED,
    ERROR_TABLES,
    HIERARCHY_PLOTLY,
)

DEVELOPER_EXAMPLES: tuple[Example, ...] = (
    ROTATED_LONG_LABELS,
    ROTATED_DENSE_LEVELS,
    UNDEFINED_LEVELS,
    MANY_LEVEL_DISTRIBUTIONS,
    WIDE_DIFFICULT_ERROR_TABLE,
)

EXAMPLES: tuple[Example, ...] = DOCUMENTATION_EXAMPLES + DEVELOPER_EXAMPLES
