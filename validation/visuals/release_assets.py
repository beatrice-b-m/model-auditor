"""Generate the verified documentation asset bundle for a release.

Run from an isolated rendering environment that has the built wheel installed
and cannot import the library checkout::

    python -m validation.visuals.release_assets \\
        --output release-assets \\
        --wheel dist/model_auditor-<version>-py3-none-any.whl \\
        --tag v<version> --commit <sha>

The bundle contains the curated documentation examples rendered from the
installed wheel, screenshots, and a manifest recording the tag, commit, wheel
hash, example inputs, rendering environment, and every output hash.  A failed
generation or verification raises, so package publication can be blocked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Sequence

from validation.visuals import gallery, render
from validation.visuals.data import (
    INFERENCE_SEED,
    N_BOOTSTRAPS,
    OUTCOME_NAME,
    SCORE_NAME,
    SEED,
    THRESHOLD,
)
from validation.visuals.examples import DOCUMENTATION_EXAMPLES
from validation.visuals.manifest import hash_file, verify_bundle


def _assert_wheel_import() -> Path:
    """Ensure ``model_auditor`` resolves to an installed package, not a checkout."""
    import model_auditor

    module_path = Path(model_auditor.__file__).resolve()
    if (Path.cwd() / "model_auditor").resolve() == module_path.parent:
        raise RuntimeError(
            "model_auditor resolved to the repository checkout; the isolated "
            "rendering environment must import the installed wheel only."
        )
    return module_path


def _repository_slug() -> str | None:
    """Derive ``owner/repo`` from the installed distribution's Homepage URL."""
    from importlib.metadata import metadata

    for entry in metadata("model-auditor").get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        if label.strip().lower() == "homepage":
            return url.strip().removeprefix("https://github.com/").removesuffix(".git")
    return None


def _requirements_hash() -> str | None:
    requirements = Path(__file__).with_name("requirements-render.txt")
    return hash_file(requirements) if requirements.exists() else None


def build_bundle(
    output: Path,
    wheel: Path,
    tag: str,
    commit: str,
    expect_version: str | None = None,
) -> dict:
    """Render and verify the curated documentation bundle; return its manifest."""
    module_path = _assert_wheel_import()
    package_version = installed_version("model-auditor")
    if expect_version is not None and package_version != expect_version:
        raise RuntimeError(
            f"Installed model-auditor version {package_version!r} does not match "
            f"the expected release version {expect_version!r}."
        )

    manifest = gallery.build_gallery(
        DOCUMENTATION_EXAMPLES,
        output_dir=output,
        screenshots=True,
        verify_browser=True,
        require_browser=True,
    )
    manifest["release"] = {
        "repository": _repository_slug(),
        "tag": tag,
        "commit": commit,
        "package_version": package_version,
        "wheel_filename": wheel.name,
        "wheel_sha256": hash_file(wheel),
        "module_path": str(module_path),
        "requirements_render_sha256": _requirements_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "example_inputs": {
            "seed": SEED,
            "score": SCORE_NAME,
            "outcome": OUTCOME_NAME,
            "threshold": THRESHOLD,
            "n_bootstraps": N_BOOTSTRAPS,
            "inference_seed": INFERENCE_SEED,
        },
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    problems = verify_bundle(
        output, [example.slug for example in DOCUMENTATION_EXAMPLES]
    )
    if problems:
        raise RuntimeError("Release bundle verification failed: " + "; ".join(problems))
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--expect-version", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.wheel.exists():
        raise SystemExit(f"Wheel not found: {args.wheel}")
    render.configure_matplotlib()
    manifest = build_bundle(
        args.output,
        args.wheel,
        args.tag,
        args.commit,
        expect_version=args.expect_version,
    )
    print(
        f"Release bundle for {manifest['release']['tag']} written to {args.output} "
        f"({len(manifest['examples'])} examples)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
