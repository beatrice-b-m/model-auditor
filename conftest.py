"""Shared pytest configuration for the Model Auditor test suite."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser) -> None:
    group = parser.getgroup("visuals")
    group.addoption(
        "--visuals",
        action="store_true",
        default=False,
        help="render the visual example collection and write artifacts",
    )
    group.addoption(
        "--visuals-dir",
        action="store",
        default="artifacts/visuals/pytest",
        help="directory for opt-in visual example artifacts",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "visuals: renders and saves a visual example artifact (opt-in via --visuals)",
    )
