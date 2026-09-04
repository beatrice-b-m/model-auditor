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
| `_evaluation.py`, `_comparisons.py` | Subgroup aggregation, support, interval policies, shared-resample contrasts |
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
- Ambiguous string representations of distinct subgroup values are rejected. Missing-feature policy is explicit (`exclude`, `include`, or `error`); exclusions are recorded. Noncategorical evaluation levels are sorted after string conversion. Categorical levels follow declared order, including unobserved NaN placeholders. Distribution plots retain first-appearance order for noncategorical levels and omit unobserved categories.
- A metric implements `name`, `label`, `inputs`, `ci_eligible`, and `data_call(data)`. Available input columns are `_truth`, `_pred`, `_binary_pred`, and the discovered confusion indicators. Names must be unique. Built-in indicator transforms are vectorized; `row_call` remains available for individual rows and custom implementations.
- Point estimates always use the original sample, including enrichment ORs. Undefined ratios return NaN. IID auto inference uses Wilson binomial-rate intervals (or opt-in Clopper-Pearson) and conditional exact OR intervals; other metrics/designs use diagnosed percentile resampling. Degenerate or insufficient bootstrap distributions do not receive intervals. Local random generators, confidence level, resampling unit, and missingness are configured by `InferenceConfig`.
- Legacy metric DataFrame exports remain formatted strings and error exports numeric/MultiIndex. `to_numeric_dataframe()` adds unrounded long-form results, support, interval diagnostics and provenance. Tables are neutral by default; optional metric ranks use explicit direction metadata within each feature. Enrichment ORs have no performance coloring.
- `split_classes=True` now overlays histograms using a configured binary outcome. The default is `False`, preserving the previous combined histogram output; the old `True` default was an unimplemented no-op.
- Youden selects a finite observed threshold; target optimization also considers a finite all-negative endpoint when representable. Both require both truth classes and explicitly identify tuning-data performance. Interval plots identify undrawn/undefined levels and can render point estimates outside percentile intervals.
- Hierarchy custom aggregators receive a whole group DataFrame; string aggregators operate on the score Series.

Add tests for externally observable behavior, mathematical edge cases, and regressions. Compare metric results against hand-computed values or an independent implementation. Avoid tests that merely restate private implementation steps. Use an explicit local seed for reproducibility. Tests must distinguish formula correctness from interval coverage; `validation/coverage.py` is an opt-in simulation report with Monte Carlo error, interval width and failure rates.

## Statistical scope

The core evaluates fixed binary predictions: ranking, decisions, probability accuracy/calibration, and paired model or reference-group contrasts. Inference is pointwise and conditional on supplied predictions and policies. Cluster resampling changes the sampling unit, not the row-weighted estimand. No training/refitting uncertainty, selection correction, simultaneous testing, survey weights, temporal blocks, multiclass, survival, or causal fairness claims are implied. Reject or document unsupported designs rather than silently treating them as IID.

Changes in this statistical revision intentionally replace zero-denominator zeros, bootstrap-mean ORs, global RNG behavior, and default performance coloring. Preserve public aliases and positional arguments, but update regression tests for these authorized contract changes. `n_bootstraps=None` disables all intervals; positive counts request intervals, with analytic methods avoiding unnecessary resampling. Release these changes with migration notes.

## Documentation and releases

Keep the README to installation, a runnable quick start, and links. Keep this guide and AGENTS.md focused on repository work; do not grow a second user manual or tutorial notebook here. Preserve API docstrings beside the implementation.

Versioning comes from Git tags through setuptools-scm. The release workflow runs validation before building and publishing to PyPI through OIDC; fetch full Git history so versions resolve correctly.

For a release, review public behavior changes against the documentation site's `docs-source.json` provenance, then synchronize the affected site pages to that release. Do not publish unreleased checkout behavior as the site's stable API. In particular, the optional dependency extras, stricter validation, and class-splitting behavior added here need inclusion in the next documentation synchronization.
