"""Artifact writers and content checks for the visual example collection.

Rendering stays deterministic: fixed DPI, an explicit font family, a fixed
viewport, and pinned dependencies in CI.  The writers do not change package
styling; they only make the existing output reproducible and inspectable.
"""

from __future__ import annotations

import hashlib
import html
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; import before pyplot

import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from validation.visuals.examples import (  # noqa: E402
    KIND_FIGURE,
    KIND_PLOTLY,
    KIND_TABLE,
    Example,
)

FONT_FAMILY = "DejaVu Sans"
FIGURE_DPI = 100
VIEWPORT_WIDTH = 1000
VIEWPORT_HEIGHT = 750


def configure_matplotlib() -> None:
    """Pin rendering settings that affect artifact reproducibility."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.facecolor": "white",
            "font.family": FONT_FAMILY,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slugify(value: str) -> str:
    cleaned = [c.lower() if c.isalnum() else "-" for c in str(value)]
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "level"


def _iter_figures(
    name: str, output: Any
) -> Iterable[tuple[str, matplotlib.figure.Figure]]:
    """Yield ``(artifact_name, figure)`` pairs from a plot output."""
    if isinstance(output, matplotlib.figure.Figure):
        yield name, output
        return
    if isinstance(output, dict):
        for key, value in output.items():
            if (
                isinstance(value, tuple)
                and value
                and isinstance(value[0], matplotlib.figure.Figure)
            ):
                yield f"{name}-{_slugify(key)}", value[0]


def collect_figure_text(figure: matplotlib.figure.Figure) -> str:
    """Gather every visible text element from a figure for content checks."""
    chunks: list[str] = [figure.get_suptitle()]
    for text in figure.texts:
        chunks.append(text.get_text())
    for axes in figure.axes:
        chunks.append(axes.get_title())
        chunks.append(axes.get_xlabel())
        chunks.append(axes.get_ylabel())
        chunks.extend(label.get_text() for label in axes.get_xticklabels())
        chunks.extend(label.get_text() for label in axes.get_yticklabels())
        chunks.extend(text.get_text() for text in axes.texts)
        legend = axes.get_legend()
        if legend is not None:
            chunks.extend(text.get_text() for text in legend.get_texts())
    return "\n".join(chunks)


def render_outputs(
    example: Example, outputs: dict[str, Any], directory: Path
) -> list[dict[str, Any]]:
    """Write every artifact for one example; return manifest entries."""
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    if example.kind == KIND_FIGURE:
        for name, output in outputs.items():
            for artifact_name, figure in _iter_figures(name, output):
                png_path = directory / f"{artifact_name}.png"
                figure.savefig(png_path, facecolor="white")
                plt.close(figure)
                artifacts.append(
                    {"name": artifact_name, "type": "image", "file": png_path.name}
                )
        return artifacts

    if example.kind == KIND_TABLE:
        for name, styler in outputs.items():
            # A stable ID changes selectors, not the Styler's presentation.
            body = styler.to_html(table_uuid=f"{example.slug}-{name}")
            document = (
                "<!doctype html>\n<html><head><meta charset='utf-8'>"
                f"<title>{html.escape(example.title)}</title>"
                "</head>"
                f"<body>{body}</body></html>\n"
            )
            path = directory / f"{name}.html"
            path.write_text(document, encoding="utf-8")
            artifacts.append({"name": name, "type": "html", "file": path.name})
        return artifacts

    if example.kind == KIND_PLOTLY:
        for name, figure in outputs.items():
            document = figure.to_html(
                include_plotlyjs=True, full_html=True, div_id=f"{example.slug}-{name}"
            )
            path = directory / f"{name}.html"
            path.write_text(document, encoding="utf-8")
            artifacts.append({"name": name, "type": "html", "file": path.name})
        return artifacts

    raise ValueError(f"Unknown example kind: {example.kind!r}")


def collect_output_text(example: Example, outputs: dict[str, Any]) -> str:
    """Gather searchable text from an example's outputs for content checks."""
    if example.kind == KIND_FIGURE:
        chunks = []
        for name, output in outputs.items():
            for _, figure in _iter_figures(name, output):
                chunks.append(collect_figure_text(figure))
        return "\n".join(chunks)
    if example.kind == KIND_TABLE:
        return "\n".join(styler.to_html() for styler in outputs.values())
    if example.kind == KIND_PLOTLY:
        return "\n".join(figure.to_json() for figure in outputs.values())
    raise ValueError(f"Unknown example kind: {example.kind!r}")


def missing_expected_text(example: Example, outputs: dict[str, Any]) -> list[str]:
    """Return expected strings absent from the rendered output."""
    text = collect_output_text(example, outputs)
    return [item for item in example.expected_text if item not in text]


def check_example(example: Example, outputs: dict[str, Any]) -> None:
    """Raise when an example output is missing required content."""
    missing = missing_expected_text(example, outputs)
    if missing:
        raise AssertionError(
            f"Example {example.slug!r} is missing expected text: {missing!r}"
        )


# ---------------------------------------------------------------------------
# Optional browser rendering checks (Playwright is not a library dependency)
# ---------------------------------------------------------------------------


def playwright_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("playwright.sync_api") is not None
    except ModuleNotFoundError:
        return False


def _new_browser_page(view_width: int, view_height: int):
    import importlib

    sync_api = importlib.import_module("playwright.sync_api")

    playwright = sync_api.sync_playwright().start()
    try:
        browser = playwright.chromium.launch()
    except Exception:
        playwright.stop()
        raise
    page = browser.new_page(viewport={"width": view_width, "height": view_height})
    return playwright, browser, page


def verify_html_rendering(
    path: Path,
    expected_text: Iterable[str] = (),
    screenshot: Path | None = None,
    view_width: int = VIEWPORT_WIDTH,
    view_height: int = VIEWPORT_HEIGHT,
    full_page: bool = True,
    kind: str = KIND_TABLE,
    timeout_ms: int = 8000,
) -> str:
    """Load an HTML artifact in Chromium, optionally screenshot it, and return its text.

    Requires Playwright; callers should gate on :func:`playwright_available`.
    Raises ``AssertionError`` when expected text is absent after rendering.
    """
    playwright, browser, page = _new_browser_page(view_width, view_height)
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "requestfailed", lambda request: errors.append(f"Request failed: {request.url}")
    )
    try:
        page.goto(path.resolve().as_uri(), wait_until="load")
        if errors:
            raise AssertionError(f"{path.name}: browser errors: {errors}")
        if kind == KIND_TABLE:
            surface = page.locator("table").first
            text_selector = "th, td"
        elif kind == KIND_PLOTLY:
            surface = page.locator(".plotly-graph-div svg.main-svg").first
            text_selector = "text"
            # An empty chart container is not a rendered trace.
            page.locator("svg.main-svg .trace, svg.main-svg .slice").first.wait_for(
                state="visible", timeout=timeout_ms
            )
        else:
            raise ValueError(f"Unsupported browser artifact kind: {kind!r}")
        surface.wait_for(state="visible", timeout=timeout_ms)
        page.evaluate("document.fonts.ready")
        content = "\n".join(
            element.text_content() or ""
            for element in surface.locator(text_selector).all()
            if element.is_visible()
        )
        if not content.strip():
            raise AssertionError(f"{path.name}: empty rendered surface")
        missing = [item for item in expected_text if item not in content]
        if missing:
            raise AssertionError(f"{path.name} is missing rendered text: {missing!r}")
        if errors:
            raise AssertionError(f"{path.name}: browser errors: {errors}")
        if screenshot is not None:
            page.screenshot(path=str(screenshot), full_page=full_page)
        return content
    finally:
        browser.close()
        playwright.stop()
