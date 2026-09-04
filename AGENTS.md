# Repository instructions

Model Auditor is a synchronous Python library for subgroup evaluation of binary classifiers. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, module boundaries, statistical contracts, and release/documentation workflow.

- Install development dependencies with `python -m pip install -e '.[dev]'` in a virtual environment.
- Verify changes with `python -m ruff check .`, `python -m ruff format --check .`, and `python -m pytest -q`. Run `python -m build` for packaging changes.
- Keep the `Auditor` API in `core.py` and public dataclasses in `schemas.py`. Put numerical helpers, styling, and plotting in their existing implementation modules.
- Preserve public imports, result formats, categorical ordering, and statistical definitions unless the task explicitly changes them. Add regression coverage for behavior changes.
- Keep confusion inputs vectorized and optional plotting/styling dependencies out of core imports. Do not mutate caller-owned DataFrames.
- Use Python 3.10-compatible type hints and the repository's Ruff configuration. Prefer explicit state and small helpers over additional inheritance or framework layers.
- Keep repository documentation short and developer-focused. The separate [documentation repository](https://github.com/beatrice-b-m/model-auditor-docs) owns tutorials and reference pages and tracks a stable release; do not sync it to unreleased changes automatically.
