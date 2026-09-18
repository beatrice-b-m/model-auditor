"""User-facing documentation, examples and explicit forwarding contracts."""

import ast
import doctest
import importlib
import inspect
import pydoc
import re
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from model_auditor import Auditor, InferenceConfig
from model_auditor.plotting.distributions import plot_score_distributions
from model_auditor.plotting.intervals import plot_metric_intervals
from model_auditor.schemas import ScoreEvaluation

# Deliberately enumerate supported modules; imported helpers are not new exports.
MODULES = (
    "model_auditor",
    "model_auditor.core",
    "model_auditor.schemas",
    "model_auditor.metrics",
    "model_auditor.error_metrics",
    "model_auditor.metric_inputs",
    "model_auditor.utils",
    "model_auditor.plotting",
    "model_auditor.plotting.schemas",
    "model_auditor.plotting.plotters",
    "model_auditor.plotting.distributions",
    "model_auditor.plotting.intervals",
)


def public_nodes(tree):
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (
            not node.name.startswith("_") or node.name == "__init__"
        ):
            yield node
            if isinstance(node, ast.ClassDef):
                yield from public_nodes(node)


@pytest.mark.parametrize("name", MODULES)
def test_public_callables_document_parameters_and_annotations(name):
    module = importlib.import_module(name)
    tree = ast.parse(Path(module.__file__).read_text())
    for node in public_nodes(tree):
        description = ast.get_docstring(node)
        assert description, f"{name}.{node.name} needs a docstring"
        if not isinstance(node, ast.FunctionDef):
            continue
        assert node.returns is not None, f"{name}.{node.name} return annotation"
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for arg in args:
            if arg.arg in {"self", "cls"}:
                continue
            assert arg.annotation is not None, f"{name}.{node.name}.{arg.arg} type"
            assert re.search(
                rf"^\s+{arg.arg}(?:\s+\([^)]*\))?:", description, re.MULTILINE
            ), f"{name}.{node.name}.{arg.arg} parameter help"


@pytest.mark.parametrize("name", MODULES)
def test_inline_examples_execute(name):
    module = importlib.import_module(name)
    failures, _ = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert failures == 0


@pytest.mark.parametrize(
    "method, renderer",
    [
        (Auditor.plot_score_distributions, plot_score_distributions),
        (ScoreEvaluation.plot_metric_intervals, plot_metric_intervals),
    ],
)
def test_forwarded_plot_arguments_are_discoverable(method, renderer):
    public = inspect.signature(method)
    implementation = inspect.signature(renderer)
    # First argument is self on the method and an explicit owner on the renderer.
    # Optional plotting types are imported only for checking these annotations.
    namespace = {
        "Auditor": Auditor,
        "ScoreEvaluation": ScoreEvaluation,
        "Figure": Figure,
        "Axes": Axes,
    }

    def resolved_parameters(function, signature):
        hints = get_type_hints(function, globalns={**function.__globals__, **namespace})
        return [
            p.replace(annotation=hints[p.name])
            for p in list(signature.parameters.values())[1:]
        ]

    assert resolved_parameters(method, public) == resolved_parameters(
        renderer, implementation
    )
    assert all(
        parameter.kind != inspect.Parameter.VAR_KEYWORD
        for parameter in public.parameters.values()
    )
    assert "Callers own" in inspect.getdoc(method) or "callers" in inspect.getdoc(
        method
    )


@pytest.mark.parametrize(
    "module_name", ["model_auditor.schemas", "model_auditor.plotting.schemas"]
)
def test_dataclass_fields_have_source_hover_documentation(module_name):
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text())
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for index, field in enumerate(cls.body):
            if not isinstance(field, ast.AnnAssign):
                continue
            following = cls.body[index + 1]
            assert isinstance(following, ast.Expr)
            assert isinstance(following.value, ast.Constant)
            assert isinstance(following.value.value, str)
            assert following.value.value.strip(), f"{cls.name}.{field.target.id}"


def test_help_and_exports_work_without_loading_optional_renderers():
    assert "min_valid_fraction" in pydoc.render_doc(InferenceConfig)
    assert "n_bootstraps" in pydoc.render_doc(Auditor.evaluate_metrics)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, model_auditor; "
            "assert model_auditor.__all__ == ['Auditor', 'InferenceConfig']; "
            "assert not any(name.split('.')[0] in {'matplotlib', 'plotly', 'jinja2'} "
            "for name in sys.modules)",
        ],
        check=True,
    )
