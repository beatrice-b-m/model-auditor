"""Reject incomplete or altered galleries before publication."""

import json

import pytest

from validation.visuals.examples import PERFORMANCE_TABLES_NEUTRAL
from validation.visuals.gallery import build_gallery
from validation.visuals.manifest import verify_bundle


@pytest.fixture
def bundle(tmp_path):
    build_gallery([PERFORMANCE_TABLES_NEUTRAL], tmp_path, verify_browser=False)
    return tmp_path


def _change_manifest(bundle, change):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def test_complete_bundle_includes_verifiable_executable_inputs(bundle):
    assert verify_bundle(bundle, [PERFORMANCE_TABLES_NEUTRAL.slug]) == []
    assert (bundle / "sources/validation/visuals/data.py").is_file()
    assert (bundle / PERFORMANCE_TABLES_NEUTRAL.slug / "example.py").is_file()


@pytest.mark.parametrize("outputs", [[{}], [], [None]])
def test_incomplete_output_records_are_rejected(bundle, outputs):
    _change_manifest(bundle, lambda m: m["examples"][0].update(outputs=outputs))
    assert verify_bundle(bundle)


@pytest.mark.parametrize(
    "relative",
    [
        "performance-tables-neutral/styled.html",
        "performance-tables-neutral/example.py",
        "sources/validation/visuals/data.py",
        "sources/validation/visuals/catalog.json",
        "index.html",
    ],
)
def test_altered_output_code_or_input_is_rejected(bundle, relative):
    (bundle / relative).write_text("altered")
    assert any("hash mismatch" in p for p in verify_bundle(bundle))


def test_missing_screenshot_is_rejected_when_capture_requires_it(bundle):
    _change_manifest(bundle, lambda m: m.update(screenshots_required=True))
    assert verify_bundle(bundle)


def test_release_requires_complete_documentation_set_and_browser_verification(bundle):
    _change_manifest(bundle, lambda m: m.update(release={}))
    problems = verify_bundle(bundle)
    assert any("required examples" in p for p in problems)
    assert any("browser-verified" in p for p in problems)


def test_unknown_required_example_is_rejected(bundle):
    assert verify_bundle(bundle, ["missing-example"])


def test_duplicate_outputs_are_rejected(bundle):
    _change_manifest(
        bundle,
        lambda m: m["examples"][0]["outputs"].append(m["examples"][0]["outputs"][0]),
    )
    assert verify_bundle(bundle)
