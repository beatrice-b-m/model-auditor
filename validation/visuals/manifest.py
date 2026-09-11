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


def verify_bundle(directory: Path, require_examples: Sequence[str] = ()) -> list[str]:
    """Return a list of problems found in the bundle (empty when valid)."""
    problems: list[str] = []
    try:
        manifest = load_manifest(directory)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return [str(exc)]

    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        return [f"{MANIFEST_NAME} declares no examples"]

    seen: set[str] = set()
    for example in examples:
        slug = example.get("slug")
        if not slug:
            problems.append("example entry without a slug")
            continue
        seen.add(slug)
        code_hash = example.get("code_sha256")
        if not code_hash:
            problems.append(f"{slug}: missing code_sha256")
        outputs = example.get("outputs") or []
        if not outputs:
            problems.append(f"{slug}: no outputs recorded")
        for output in outputs:
            for file_key, hash_key in (
                ("file", "sha256"),
                ("screenshot", "screenshot_sha256"),
            ):
                relative = output.get(file_key)
                if not relative:
                    continue
                path = directory / relative
                if not path.exists():
                    problems.append(f"{slug}: missing {relative}")
                    continue
                expected = output.get(hash_key)
                if not expected:
                    problems.append(f"{slug}: {relative} has no recorded hash")
                    continue
                actual = hash_file(path)
                if actual != expected:
                    problems.append(
                        f"{slug}: {relative} hash mismatch "
                        f"(expected {expected[:12]}, got {actual[:12]})"
                    )

    missing = [slug for slug in require_examples if slug not in seen]
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
