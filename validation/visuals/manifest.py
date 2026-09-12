"""Read and verify visual artifact manifests.

Manifest integrity is checked by re-hashing every referenced file.  The same
verifier protects CI gallery uploads, release asset bundles, and documentation
imports.  Run standalone with::

    python -m validation.visuals.manifest artifacts/visuals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Sequence

MANIFEST_NAME = "manifest.json"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(directory: Path) -> dict:
    """Load ``manifest.json`` from a gallery or bundle directory."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"No {MANIFEST_NAME} found in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_file(directory: Path, relative, expected, problems: list[str]) -> None:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not (directory / relative).resolve().is_relative_to(directory.resolve())
    ):
        problems.append(f"invalid artifact path: {relative!r}")
        return
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        problems.append(f"{relative}: invalid SHA-256")
        return
    path = directory / relative
    if not path.is_file():
        problems.append(f"missing artifact: {relative}")
    elif hash_file(path) != expected:
        problems.append(f"hash mismatch: {relative}")


def verify_bundle(directory: Path, require_examples: Sequence[str] = ()) -> list[str]:
    """Check the catalog, executable inputs, and every required output."""
    problems: list[str] = []
    try:
        manifest = load_manifest(directory)
    except (OSError, json.JSONDecodeError) as exc:
        return [str(exc)]
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        return ["unsupported manifest schema_version"]
    _verify_file(
        directory, manifest.get("index_file"), manifest.get("index_sha256"), problems
    )
    sources = manifest.get("source_files")
    if not isinstance(sources, list) or not sources:
        return ["missing executable source_files"]
    source_paths: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            problems.append("invalid source entry")
            continue
        relative = source.get("file")
        _verify_file(directory, relative, source.get("sha256"), problems)
        if isinstance(relative, str):
            if relative in source_paths:
                problems.append(f"duplicate source: {relative}")
            source_paths.add(relative)
    catalog_file = manifest.get("catalog_file")
    if not isinstance(catalog_file, str) or catalog_file not in source_paths:
        problems.append("catalog_file must reference a verified source file")
    if problems:
        return problems
    try:
        catalog = json.loads((directory / catalog_file).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read catalog: {exc}"]
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        return ["unsupported catalog schema_version"]
    specifications = catalog.get("examples")
    if not isinstance(specifications, dict) or not specifications:
        return ["catalog declares no examples"]
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        return ["manifest declares no examples"]
    require_screenshots = manifest.get("screenshots_required") is True
    if "release" in manifest:
        require_screenshots = True
        if manifest.get("browser_verified") is not True:
            problems.append("release bundle must be browser-verified")
        require_examples = [
            *require_examples,
            *(
                slug
                for slug, spec in specifications.items()
                if spec.get("documentation")
            ),
        ]
    seen: set[str] = set()
    for example in examples:
        if not isinstance(example, dict):
            problems.append("invalid example entry")
            continue
        slug = example.get("slug")
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9-]+", slug):
            problems.append("invalid example slug")
            continue
        if slug in seen:
            problems.append(f"duplicate example: {slug}")
        seen.add(slug)
        spec = specifications.get(slug)
        if not isinstance(spec, dict):
            problems.append(f"{slug}: absent from catalog")
            continue
        if any(example.get(key) != spec.get(key) for key in ("kind", "documentation")):
            problems.append(f"{slug}: metadata disagrees with catalog")
        _verify_file(
            directory, example.get("code_file"), example.get("code_sha256"), problems
        )
        expected_files = spec.get("files")
        if not isinstance(expected_files, list) or not expected_files:
            problems.append(f"{slug}: catalog declares no outputs")
            continue
        expected_files = {f"{slug}/{name}" for name in expected_files}
        outputs = example.get("outputs")
        if not isinstance(outputs, list):
            problems.append(f"{slug}: missing outputs")
            continue
        actual_files = []
        for output in outputs:
            if not isinstance(output, dict):
                problems.append(f"{slug}: invalid output entry")
                continue
            relative = output.get("file")
            _verify_file(directory, relative, output.get("sha256"), problems)
            if isinstance(relative, str):
                actual_files.append(relative)
            expected_type = "image" if spec.get("kind") == "figure" else "html"
            if output.get("type") != expected_type:
                problems.append(f"{slug}: incorrect output type")
            if output.get("screenshot") or (
                require_screenshots and expected_type == "html"
            ):
                _verify_file(
                    directory,
                    output.get("screenshot"),
                    output.get("screenshot_sha256"),
                    problems,
                )
        if set(actual_files) != expected_files or len(actual_files) != len(
            expected_files
        ):
            problems.append(f"{slug}: outputs disagree with catalog")
    missing = sorted(set(require_examples) - seen)
    if missing:
        problems.append(f"missing required examples: {missing!r}")
    return problems


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--require",
        default=None,
        help="comma-separated example slugs that must be present",
    )
    parser.add_argument(
        "--require-documentation",
        action="store_true",
        help="require every curated documentation example slug",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    require: tuple[str, ...] = ()
    if args.require:
        require = tuple(
            slug.strip() for slug in args.require.split(",") if slug.strip()
        )
    if args.require_documentation:
        from validation.visuals.examples import DOCUMENTATION_EXAMPLES

        require = tuple(
            dict.fromkeys(
                [*require, *(example.slug for example in DOCUMENTATION_EXAMPLES)]
            )
        )
    problems = verify_bundle(args.directory, require)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print(f"Verified bundle at {args.directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
