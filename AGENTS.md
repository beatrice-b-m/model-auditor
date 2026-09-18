# Repository instructions

Model Auditor is a synchronous Python library for subgroup evaluation of binary classifiers. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, module boundaries, statistical contracts, and release/documentation workflow.

- Install development dependencies with `python -m pip install -e '.[dev]'` in a virtual environment.
- Verify changes with `python -m ruff check .`, `python -m ruff format --check .`, and `python -m pytest -q`. Run `python -m build` for packaging changes.
- Keep the `Auditor` API in `core.py` and public dataclasses in `schemas.py`. Put numerical helpers, styling, and plotting in their existing implementation modules.
- Preserve public imports, result formats, categorical ordering, and statistical definitions unless the task explicitly changes them. Add regression coverage for behavior changes.
- Keep confusion inputs vectorized and optional plotting/styling dependencies out of core imports. Do not mutate caller-owned DataFrames.
- Use Python 3.10-compatible type hints and the repository's Ruff configuration. Prefer explicit state and small helpers over additional inheritance or framework layers.
- Keep repository documentation short and developer-focused. The separate [documentation repository](https://github.com/beatrice-b-m/model-auditor-docs) owns tutorials and reference pages and tracks a stable release; do not sync it to unreleased changes automatically.

## Inline API documentation

- Update docstrings and type annotations alongside every public API change,
  including exports, supported submodule APIs, public members, and returned objects.
- Use existing Google-style docstrings (`Args`, `Returns`, `Raises`,
  `Attributes`, and `Example`), covering purpose, parameters and defaults,
  returns, relevant exceptions, and runnable examples where useful. Document
  exported entry points and forwarding methods, not only implementation helpers.
- Explain applicable contracts: missing values, units, ordering, mutation,
  side effects, ownership, limits, and serialization. Keep essential guidance
  in docstrings and consistent with README.md, CONTRIBUTING.md, and the
  release-matched documentation site under the synchronization policy above.
- Provide precise public types and statically discoverable signatures,
  including decorator-added controls and forwarded options. Keep private
  arguments out of supported interfaces and optional dependencies out of core
  runtime imports; retain the packaged `py.typed` marker.
- Execute changed examples and run applicable documentation and typing checks.
  For interface changes, check signatures, completions, and result navigation,
  distinguishing static/editor checks from runtime introspection. Bring touched
  public interfaces into compliance. See CONTRIBUTING.md for validation commands.

## Git commits

Track all changes in reasonably scoped Git commits. Keep each commit focused on
one coherent change, use a descriptive message, and stage files selectively.
Commit completed work before handing it back to the user.
