"""Generate the local visual gallery under ``artifacts/visuals/``.

Run from the repository root::

    python -m validation.visuals.gallery
    python -m validation.visuals.gallery --documentation-only
    python -m validation.visuals.gallery --examples intervals-default

The gallery is a developer review aid: outputs are written outside Git, the
index links every artifact to the exact generating script, and a manifest
records content hashes for provenance checks in CI and release bundles.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from validation.visuals import render
from validation.visuals.examples import (
    DOCUMENTATION_EXAMPLES,
    EXAMPLES,
    Example,
    run_example,
)

DEFAULT_OUTPUT = Path("artifacts/visuals")


@dataclass
class ExampleResult:
    example: Example
    directory: Path
    artifacts: list[dict[str, Any]]


def _write_index(results: Sequence[ExampleResult], output_dir: Path) -> None:
    sections: list[str] = []
    for result in results:
        example = result.example
        embeds: list[str] = []
        for artifact in result.artifacts:
            relative = f"{example.slug}/{artifact['file']}"
            if artifact["type"] == "image":
                embeds.append(
                    f'<figure><img src="{relative}" alt="{html.escape(example.title)}">'
                    f"</figure>"
                )
            elif example.kind == "plotly":
                embeds.append(
                    f'<iframe src="{relative}" title="{html.escape(example.title)}" '
                    'width="1000" height="750"></iframe>'
                )
            else:
                embeds.append(
                    f'<iframe src="{relative}" title="{html.escape(example.title)}" '
                    'width="1000" height="420"></iframe>'
                )
            if artifact.get("screenshot"):
                shot = f"{example.slug}/{artifact['screenshot']}"
                embeds.append(
                    f'<figure><img src="{shot}" '
                    f'alt="{html.escape(example.title)} screenshot"></figure>'
                )
        badge = "documentation" if example.documentation else "developer"
        sections.append(
            f'<section id="{example.slug}">'
            f"<h2>{html.escape(example.title)} <small>[{badge}]</small></h2>"
            f"<p>{html.escape(example.description)}</p>"
            f"<p><em>{html.escape(example.caption)}</em></p>"
            f"{''.join(embeds)}"
            f"<h3>Generating code</h3><pre><code>"
            f"{html.escape(example.code)}</code></pre>"
            "</section>"
        )
    document = (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        "<title>Model Auditor visual gallery</title>"
        "<style>body{font-family:DejaVu Sans,Helvetica,Arial,sans-serif;"
        "margin:24px;max-width:1040px;color:#111}"
        "section{margin-bottom:48px;border-top:1px solid #ddd;padding-top:16px}"
        "img,iframe{max-width:100%;border:1px solid #eee}"
        "pre{background:#f6f8fa;padding:12px;overflow:auto}"
        "small{color:#666;font-weight:normal}</style></head><body>"
        "<h1>Model Auditor visual gallery</h1>"
        "<p>Generated review artifacts. This directory is ignored by Git.</p>"
        + "".join(sections)
        + "</body></html>\n"
    )
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _manifest_entry(result: ExampleResult) -> dict[str, Any]:
    outputs = []
    for artifact in result.artifacts:
        path = result.directory / artifact["file"]
        entry = {
            "name": artifact["name"],
            "type": artifact["type"],
            "file": f"{result.example.slug}/{artifact['file']}",
            "sha256": render.sha256_file(path),
        }
        if artifact.get("screenshot"):
            shot = result.directory / artifact["screenshot"]
            entry["screenshot"] = f"{result.example.slug}/{artifact['screenshot']}"
            entry["screenshot_sha256"] = render.sha256_file(shot)
        outputs.append(entry)
    return {
        "slug": result.example.slug,
        "title": result.example.title,
        "kind": result.example.kind,
        "documentation": result.example.documentation,
        "description": result.example.description,
        "caption": result.example.caption,
        "code_sha256": render.sha256_text(result.example.code),
        "outputs": outputs,
    }


def _versions() -> dict[str, str]:
    import matplotlib

    versions = {"matplotlib": matplotlib.__version__}
    try:
        import plotly

        versions["plotly"] = plotly.__version__
    except ImportError:  # pragma: no cover - plotly is in the plotting extra
        versions["plotly"] = "unavailable"
    try:
        import pandas

        versions["pandas"] = pandas.__version__
    except ImportError:  # pragma: no cover - pandas is a core dependency
        versions["pandas"] = "unavailable"
    return versions


def build_gallery(
    examples: Sequence[Example],
    output_dir: Path = DEFAULT_OUTPUT,
    screenshots: bool = False,
    verify_browser: bool = True,
    require_browser: bool = False,
) -> dict[str, Any]:
    """Render every example, write the index and manifest, and return the manifest."""
    render.configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    use_browser = screenshots or verify_browser
    if use_browser and not render.playwright_available():
        if screenshots or require_browser:
            raise RuntimeError(
                "Browser rendering was requested but Playwright is not installed."
            )
        use_browser = False

    results: list[ExampleResult] = []
    for example in examples:
        outputs = run_example(example)
        render.check_example(example, outputs)
        directory = output_dir / example.slug
        artifacts = render.render_outputs(example, outputs, directory)
        if use_browser:
            for artifact in artifacts:
                if artifact["type"] != "html":
                    continue
                path = directory / artifact["file"]
                shot_name = f"{artifact['name']}-screenshot.png"
                shot_path = directory / shot_name if screenshots else None
                render.verify_html_rendering(
                    path, example.expected_text, screenshot=shot_path, kind=example.kind
                )
                if shot_path is not None:
                    artifact["screenshot"] = shot_name
        results.append(ExampleResult(example, directory, artifacts))

    rendered = [result.example.slug for result in results]
    expected = [example.slug for example in examples]
    if rendered != expected:  # pragma: no cover - defensive completeness gate
        raise RuntimeError(
            f"Gallery completeness check failed: rendered {rendered!r}, "
            f"expected {expected!r}."
        )
    if require_browser and not use_browser:  # pragma: no cover - CI gate
        raise RuntimeError("Browser rendering verification is required but disabled.")

    _write_index(results, output_dir)
    manifest = {
        "generator": "validation.visuals.gallery",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "versions": _versions(),
        "font": render.FONT_FAMILY,
        "figure_dpi": render.FIGURE_DPI,
        "viewport": {
            "width": render.VIEWPORT_WIDTH,
            "height": render.VIEWPORT_HEIGHT,
        },
        "browser_verified": use_browser,
        "examples": [_manifest_entry(result) for result in results],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--documentation-only",
        action="store_true",
        help="render only the curated documentation subset",
    )
    parser.add_argument(
        "--examples",
        default=None,
        help="comma-separated example slugs to render",
    )
    parser.add_argument(
        "--screenshots",
        action="store_true",
        help="capture browser screenshots (requires Playwright)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="skip browser rendering verification",
    )
    parser.add_argument(
        "--require-browser",
        action="store_true",
        help="fail when browser rendering verification is unavailable",
    )
    return parser.parse_args(argv)


def _select_examples(args: argparse.Namespace) -> tuple[Example, ...]:
    selected = DOCUMENTATION_EXAMPLES if args.documentation_only else EXAMPLES
    if args.examples:
        wanted = [slug.strip() for slug in args.examples.split(",") if slug.strip()]
        known = {example.slug for example in EXAMPLES}
        unknown = [slug for slug in wanted if slug not in known]
        if unknown:
            raise SystemExit(f"Unknown example slugs: {unknown!r}")
        selected = tuple(example for example in selected if example.slug in wanted)
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    examples = _select_examples(args)
    manifest = build_gallery(
        examples,
        output_dir=args.output,
        screenshots=args.screenshots,
        verify_browser=not args.no_browser,
        require_browser=args.require_browser,
    )
    print(
        f"Rendered {len(manifest['examples'])} examples to {args.output}; "
        f"browser_verified={manifest['browser_verified']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
