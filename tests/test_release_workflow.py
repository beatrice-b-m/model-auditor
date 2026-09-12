"""Exercise the release workflow's checksum and upload-directory commands."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/publish.yml"


def _step(name):
    match = re.search(
        rf"- name: {re.escape(name)}\n\s+run: ([^\n]+)",
        WORKFLOW.read_text(),
    )
    assert match, f"missing shell step: {name}"
    return match.group(1)


def _run(command, directory):
    if not shutil.which("sha256sum"):
        if not shutil.which("shasum"):
            pytest.skip("a SHA-256 command is required")
        command = 'sha256sum() { shasum -a 256 "$@"; }; ' + command
    return subprocess.run(
        ["bash", "-ec", command], cwd=directory, capture_output=True, text=True
    )


def test_downloaded_wheel_verifies_and_only_wheel_is_published(tmp_path):
    build = tmp_path / "build"
    dist = build / "dist"
    dist.mkdir(parents=True)
    filename = "model_auditor-1.0.0-py3-none-any.whl"
    (dist / filename).write_bytes(b"wheel artifact")
    assert _run(_step("Record wheel hash"), build).returncode == 0

    # Simulate download-artifact restoring the artifact under a new workspace.
    publish = tmp_path / "publish"
    shutil.copytree(dist, publish / "dist")
    result = _run(_step("Verify wheel hash"), publish)
    assert result.returncode == 0, result.stderr
    result = _run(_step("Prepare distributions for publication"), publish)
    assert result.returncode == 0, result.stderr
    assert [path.name for path in (publish / "packages").iterdir()] == [filename]
    assert "packages-dir: packages/" in WORKFLOW.read_text()

    (publish / "dist" / filename).write_bytes(b"corrupted artifact")
    assert _run(_step("Verify wheel hash"), publish).returncode != 0
