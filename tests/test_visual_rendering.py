"""Capture must preserve default styling and reject broken browser output."""

from unittest.mock import patch

import pandas as pd
import pytest

from validation.visuals import render
from validation.visuals.examples import PERFORMANCE_TABLES_NEUTRAL


def test_playwright_is_optional_when_parent_package_is_absent():
    with patch.dict("sys.modules", {"playwright": None}):
        assert not render.playwright_available()


def test_table_wrapper_preserves_styler_html_without_extra_css(tmp_path):
    styler = pd.DataFrame({"Value": [1, 2]}).style
    example = PERFORMANCE_TABLES_NEUTRAL
    render.render_outputs(example, {"styled": styler}, tmp_path)
    html = (tmp_path / "styled.html").read_text()
    expected = styler.to_html(table_uuid=f"{example.slug}-styled")
    assert f"<body>{expected}</body>" in html
    assert "<style>" not in html.split("<body>")[0]
    render.render_outputs(example, {"styled": styler}, tmp_path)
    assert (tmp_path / "styled.html").read_text() == html


@pytest.mark.parametrize(
    "body",
    [
        '<script>throw new Error("render failed"); const data = "Expected";</script>',
        '<script type="application/json">{"label":"Expected"}</script>',
        '<table style="display:none"><tr><td>Expected</td></tr></table>',
        '<table><tr><td>Other</td></tr></table><script>const data="Expected";</script>',
        "<table></table>",
    ],
)
def test_browser_rejects_failed_or_invisible_output(tmp_path, request, body):
    if not request.config.getoption("--visuals"):
        pytest.skip("pass --visuals to run browser rendering regressions")
    path = tmp_path / "broken.html"
    path.write_text(f"<html><body>{body}</body></html>")
    from playwright.sync_api import TimeoutError as BrowserTimeout

    with pytest.raises((AssertionError, BrowserTimeout)):
        render.verify_html_rendering(path, ["Expected"], timeout_ms=500)


def test_browser_accepts_visible_table_and_captures_it(tmp_path, request):
    if not request.config.getoption("--visuals"):
        pytest.skip("pass --visuals to run browser rendering regressions")
    path = tmp_path / "table.html"
    path.write_text("<table><tr><th>Expected</th><td>42</td></tr></table>")
    screenshot = tmp_path / "table.png"
    text = render.verify_html_rendering(path, ["Expected", "42"], screenshot)
    assert "42" in text
    assert screenshot.stat().st_size > 0
