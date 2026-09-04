# Working on Model Auditor

This repository contains the Python library and its tests. User guides and API reference live in [model-auditor-docs](https://github.com/beatrice-b-m/model-auditor-docs).

## Setup and checks

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run from the repository root:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
```

Use `python -m ruff format .` to format changes. CI runs the tests on Python 3.10, 3.11, and 3.12, plus lint, formatting, and package-build checks. Matplotlib tests use the noninteractive Agg backend.

The `plotting` extra installs Matplotlib and Plotly; `styling` installs Jinja2. Core evaluation must remain usable without these extras. The `dev` extra includes both plus the verification tools.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `core.py` | Public `Auditor` API, configuration, shared input preparation, orchestration |
| `_thresholds.py` | Scalar/conditional threshold validation and binary predictions |
| `_evaluation.py` | Subgroup aggregation, category ordering, bootstrap calculations |
| `metrics.py`, `error_metrics.py` | Metric protocols and numerical definitions |
| `metric_inputs.py`, `utils.py` | Vectorized confusion indicators and input discovery |
| `schemas.py` | Configuration/result dataclasses and DataFrame exports; public presentation methods delegate |
| `_styling.py` | Table formatting and relative performance colors |
| `plotting/intervals.py`, `plotting/distributions.py` | Optional Matplotlib renderers |
| `plotting/plotters.py`, `plotting/schemas.py` | Hierarchy compilation and Plotly-compatible arrays |
| `tests/` | Metric oracles, evaluation integration, regressions, and plotting/table behavior |

Keep numerical evaluation independent of rendering. Internal helpers receive data and configuration explicitly. Preserve public import paths and method signatures when reorganizing implementations; underscore-prefixed modules are internal.

## Behavior to preserve deliberately

- `add_data` copies the caller's DataFrame. Evaluations use private slices. Replacing data requires registering the outcome again.
- Predictions use `score >= threshold`. Call-time thresholds override the configured scalar or `ConditionalThreshold`. Missing conditional levels require a default; null levels also require a default.
- Evaluation rejects empty data, missing/nonbinary truth, nonfinite scores, duplicate metric names, reserved feature names, and invalid bootstrap counts instead of silently producing misleading results.
- Noncategorical evaluation levels are sorted after string conversion. Categorical levels follow declared order, including unobserved NaN placeholders. Distribution plots retain first-appearance order for noncategorical levels and omit unobserved categories.
- A metric implements `name`, `label`, `inputs`, `ci_eligible`, and `data_call(data)`. Available input columns are `_truth`, `_pred`, `_binary_pred`, and the discovered confusion indicators. Names must be unique. Built-in indicator transforms are vectorized; `row_call` remains available for individual rows and custom implementations.
- Bootstrap resampling uses pandas sampling and NumPy's global random state. Metric point estimates come from the original sample. Error-analysis point estimates currently use the bootstrap mean when intervals are enabled, including infinite odds ratios from sparse tables. Changing these statistical conventions requires a separate, explicit API decision.
- Metric DataFrame exports contain formatted strings; error exports contain numeric values and MultiIndex columns. Preserve these contracts unless introducing an explicit alternative. Styling tiers currently infer count/direction metadata from metric names or labels.
- `split_classes=True` now overlays histograms using a configured binary outcome. The default is `False`, preserving the previous combined histogram output; the old `True` default was an unimplemented no-op.
- Youden optimization selects a finite observed threshold, so its output can be passed directly to evaluation. Interval plots skip nonfinite/reversed bounds and can render a point estimate outside its percentile interval.
- Hierarchy custom aggregators receive a whole group DataFrame; string aggregators operate on the score Series.

Add tests for externally observable behavior, mathematical edge cases, and regressions. Compare metric results against hand-computed values or an independent implementation. Avoid tests that merely restate private implementation steps. For bootstrap equivalence, seed NumPy immediately before each calculation or supply controlled resamples in tests.

## Documentation and releases

Keep the README to installation, a runnable quick start, and links. Keep this guide and AGENTS.md focused on repository work; do not grow a second user manual or tutorial notebook here. Preserve API docstrings beside the implementation.

Versioning comes from Git tags through setuptools-scm. The release workflow runs validation before building and publishing to PyPI through OIDC; fetch full Git history so versions resolve correctly.

For a release, review public behavior changes against the documentation site's `docs-source.json` provenance, then synchronize the affected site pages to that release. Do not publish unreleased checkout behavior as the site's stable API. In particular, the optional dependency extras, stricter validation, and class-splitting behavior added here need inclusion in the next documentation synchronization.
