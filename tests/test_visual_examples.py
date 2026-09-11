"""Opt-in capture of the visual example collection.

Ordinary runs stay fast by skipping everything here; pass ``--visuals`` to
render every example into ``artifacts/visuals/pytest/`` using the same
executable scripts as the gallery.  Outputs are saved before figure cleanup
and each artifact is checked for required content and non-empty bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validation.visuals import render
from validation.visuals.examples import (
    DOCUMENTATION_EXAMPLES,
    EXAMPLES,
    Example,
    run_example,
)

pytestmark = pytest.mark.visuals


def _visuals_enabled(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--visuals"))


def test_example_catalog_is_consistent():
    """The collection has unique slugs, valid kinds, and a curated subset."""
    assert EXAMPLES, "the visual example collection must not be empty"
    assert DOCUMENTATION_EXAMPLES, "the documentation subset must be explicit"
    slugs = [example.slug for example in EXAMPLES]
    assert len(slugs) == len(set(slugs)), "example slugs must be unique"
    for example in EXAMPLES:
        assert example.kind in {"figure", "table", "plotly"}
        assert example.outputs, f"{example.slug} must declare outputs"
        assert example.expected_text, f"{example.slug} must declare expected text"
    documentation_slugs = {example.slug for example in DOCUMENTATION_EXAMPLES}
    assert documentation_slugs <= set(slugs)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda item: item.slug)
def test_visual_example(example: Example, request: pytest.FixtureRequest):
    if not _visuals_enabled(request):
        pytest.skip("pass --visuals to render example artifacts")

    render.configure_matplotlib()
    outputs = run_example(example)
    render.check_example(example, outputs)

    output_dir = Path(request.config.getoption("--visuals-dir")) / example.slug
    artifacts = render.render_outputs(example, outputs, output_dir)
    assert artifacts, f"{example.slug} produced no artifacts"
    for artifact in artifacts:
        path = output_dir / artifact["file"]
        assert path.exists(), f"missing artifact {path}"
        assert path.stat().st_size > 0, f"empty artifact {path}"
