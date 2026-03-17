# Repository Guidelines

## Project Overview

model-auditor is a Python library for evaluating machine learning model performance across subgroups with stratified metrics, bootstrap confidence intervals, and hierarchical visualizations. The library is designed for fairness auditing, subgroup analysis, and comprehensive model evaluation workflows.

**Key capabilities:**
- Stratified evaluation across categorical features
- Bootstrap confidence intervals for all supported metrics
- Comprehensive classification metrics (AUROC, AUPRC, F1, MCC, sensitivity, specificity, etc.)
- Threshold optimization via Youden index
- Hierarchical visualizations (sunburst, treemap) compatible with Plotly
- Protocol-based architecture for custom metrics

## Architecture & Data Flow

### High-Level Structure

```
Auditor (orchestrator)
    ├─ Data Layer (DataFrame + config)
    │   ├─ AuditorFeature (stratification features)
    │   ├─ AuditorScore (model scores + thresholds)
    │   └─ AuditorOutcome (ground truth with optional mapping)
    ├─ Metric Layer (AuditorMetric protocol)
    │   ├─ Classification metrics (AUROC, AUPRC, F1, MCC, etc.)
    │   ├─ Rate metrics (TPR, TNR, FPR, FNR)
    │   └─ Count metrics (N, TP, TN, FP, FN, etc.)
    ├─ Input Layer (AuditorMetricInput protocol)
    │   └─ Confusion matrix calculators (TP, FP, TN, FN)
    └─ Result Layer (hierarchical dataclasses)
        └─ ScoreEvaluation → FeatureEvaluation → LevelEvaluation → LevelMetric
```

### Evaluation Pipeline

1. **Data Ingestion**: `add_data(df)` stores the base DataFrame
2. **Configuration**: `add_feature()`, `add_score()`, `add_outcome()` define evaluation parameters
3. **Metric Setup**: `set_metrics()` specifies which metrics to compute
4. **Input Collection**: `_collect_inputs()` discovers required metric inputs (TP, FP, etc.)
5. **Data Preparation**: `_apply_inputs()` computes row-wise metric inputs, `_binarize()` thresholds scores
6. **Evaluation Loop**:
   - For each feature: `_evaluate_feature()` stratifies data by feature levels
   - For each level: compute metrics via `metric.data_call()`
   - Optionally: `_evaluate_confidence_interval()` runs bootstrap resampling
7. **Result Assembly**: Populates `ScoreEvaluation` hierarchy

### Two-Stage Metric Computation

**Stage 1 - Row-wise inputs** (via `AuditorMetricInput`):
- `TruePositives`, `FalsePositives`, `TrueNegatives`, `FalseNegatives`
- Compare `_truth` column vs `_binary_pred` column row-by-row
- Add computed columns to DataFrame

**Stage 2 - Aggregated metrics** (via `AuditorMetric`):
- Metrics call `data_call(data)` on the DataFrame with input columns
- Example: `Sensitivity()` sums `tp` and `fn` columns, computes `tp / (tp + fn)`

### Bootstrap Confidence Intervals

- **Method**: Resampling with replacement (n_bootstraps iterations)
- **CI bounds**: 2.5th and 97.5th percentiles via `np.nanpercentile`
- **Eligibility**: Metrics set `ci_eligible = True` (count metrics excluded)
- **Trigger**: Set `n_bootstraps` parameter in `evaluate()`; `None` disables CIs

## Key Directories

```
model_auditor/
├── __init__.py          # Public API: exports Auditor class
├── core.py              # Auditor class - main orchestration logic
├── metrics.py           # All metric implementations (AuditorMetric protocol)
├── metric_inputs.py     # Row-wise input calculators (AuditorMetricInput protocol)
├── schemas.py           # Dataclasses for results and configuration
├── utils.py             # Metric input discovery and validation
└── plotting/
    ├── __init__.py      # Exports HierarchyPlotter
    ├── plotters.py      # HierarchyPlotter class
    └── schemas.py       # PlotterData, Hierarchy, HLevel, HItem
```

## Development Commands

### Installation
```bash
pip install -e .
```

### Build
```bash
python -m build
```

### Dependencies (from pyproject.toml)
- `pandas >= 2.2` - DataFrame operations and data structures
- `numpy >= 2.1` - Numerical computations and CI calculation
- `scikit-learn >= 1.5` - ROC/AUC metrics
- `tqdm >= 4.0` - Progress bars during evaluation

### Publishing
- Automated via GitHub Actions (`.github/workflows/publish.yml`)
- Triggered by creating a release
- Uses OIDC authentication for PyPI

### Python Version
- Requires Python 3.10 or higher
- Tested on 3.10, 3.11, 3.12

## Code Conventions & Common Patterns

### Type System
- **Type hints throughout**: All functions use modern Python 3.10+ type hints
- **Union types**: `Union[float, int]` for numeric results, `Optional[T]` for nullable values
- **Dataclasses**: Used for all configuration and result schemas
- **Protocols**: `AuditorMetric` and `AuditorMetricInput` define extensible interfaces

### Protocol-Based Architecture

**AuditorMetric protocol** (in `metrics.py`):
```python
@runtime_checkable
class AuditorMetric(Protocol):
    name: str              # Unique metric identifier
    label: str             # Display label
    inputs: list[str]      # Required DataFrame columns
    ci_eligible: bool      # Whether bootstrap CIs apply

    def data_call(self, data: pd.DataFrame) -> Union[float, int]:
        """Compute metric on input data."""
        ...
```

**AuditorMetricInput protocol** (in `metric_inputs.py`):
```python
@runtime_checkable
class AuditorMetricInput(Protocol):
    name: str              # Input name (e.g., "tp")
    label: str             # Display label
    inputs: list[str]      # Required DataFrame columns

    def row_call(self, row: pd.Series) -> int:
        """Compute row-wise input value."""
        ...

    def data_transform(self, data: pd.DataFrame, col_name: str) -> None:
        """Add computed column to DataFrame."""
        ...
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `Auditor`, `Sensitivity`, `LevelMetric`)
- **Methods**: snake_case (e.g., `add_data`, `evaluate`, `_collect_inputs`)
- **Private methods**: Prefix with `_` (e.g., `_evaluate_feature`, `_apply_inputs`)
- **Constants**: UPPER_SNAKE_CASE (rare; mostly not used)
- **Dataclass fields**: snake_case (e.g., `score`, `interval`, `ci_eligible`)

### Error Handling
- **Defensive validation**: Early checks in public methods (e.g., `evaluate()` validates required inputs)
- **Explicit error messages**: Clear descriptions of what's missing or invalid
- **No silent failures**: Missing data or configuration raises exceptions

### Async Patterns
- **No async code**: Library is synchronous throughout
- **Progress bars**: Uses `tqdm` for long-running operations (bootstrap iterations)

### Configuration Patterns
- **Builder-style API**: Methods return `self` (though not consistently chained)
- **Optional parameters**: Many methods have sensible defaults (e.g., `threshold` in `add_score()`)
- **Late validation**: Configuration validated at evaluation time, not setup time

### State Management
- **Centralized state**: `Auditor` class holds all state (data, features, scores, outcomes, metrics)
- **Immutable results**: Evaluation returns new `ScoreEvaluation` objects; original data unchanged
- **No global state**: All state encapsulated in class instances

## Important Files

### Entry Points
- `model_auditor/__init__.py` - Main package entry point, exports `Auditor` class
- `model_auditor/core.py` - Core `Auditor` class (436 lines)

### Key Modules
- `model_auditor/metrics.py` - All metric implementations (~450 lines)
- `model_auditor/metric_inputs.py` - Row-wise input calculators (~160 lines)
- `model_auditor/schemas.py` - Result and configuration dataclasses (~310 lines)
- `model_auditor/utils.py` - Metric input discovery and validation (~50 lines)

### Configuration Files
- `pyproject.toml` - Project metadata, dependencies, build configuration
- `.github/workflows/publish.yml` - Automated PyPI publishing workflow

### Examples
- `example.ipynb` - Comprehensive Jupyter notebook demonstrating full workflow
- `README.md` - User-facing documentation with quick-start examples

### Protocols and Base Structures
- `AuditorMetric` - Protocol for metric implementations (metrics.py)
- `AuditorMetricInput` - Protocol for metric input calculators (metric_inputs.py)
- `LevelMetric` - Atomic metric result with optional CI (schemas.py)
- `ScoreEvaluation` - Root result containing all features and scores (schemas.py)

## Runtime/Tooling Preferences

### Runtime Requirements
- **Python version**: 3.10 or higher (required for modern type hints)
- **No special runtime**: Pure Python library, no compiled extensions

### Package Manager
- **Build system**: setuptools with setuptools_scm (dynamic versioning from git)
- **Installation**: pip (`pip install -e .` for editable install)

### Development Tools
- **No linter configured**: No ruff, black, mypy, or similar tools in pyproject.toml
- **No test framework**: No pytest, unittest, or test infrastructure
- **Minimal setup**: Development workflow is manual (build and test via example.ipynb)

### Dependencies by Category
- **Data manipulation**: pandas, numpy
- **Metrics**: scikit-learn (roc_auc_score, precision_recall_curve)
- **Progress reporting**: tqdm
- **Visualization**: Plotly (user-supplied, not a direct dependency)

## Testing & QA

### Current State
- **No test suite**: No test files or test infrastructure present
- **Manual testing**: Example notebook (`example.ipynb`) serves as integration test
- **No CI testing**: GitHub Actions only handles publishing, not testing

### Manual Testing Workflow
1. Run cells in `example.ipynb` to verify functionality
2. Test custom metrics via notebook examples
3. Validate plotting outputs with Plotly integration

### Recommendations (Not Currently Implemented)
- **Unit tests**: pytest for metric computations, input calculators
- **Integration tests**: Full Auditor workflow with synthetic data
- **Type checking**: mypy to validate type hints
- **Coverage**: pytest-cov to measure test coverage

## Common Patterns for AI Assistants

### Adding a New Metric
1. Implement `AuditorMetric` protocol in `metrics.py`
2. Define `name`, `label`, `inputs`, `ci_eligible` attributes
3. Implement `data_call(self, data: pd.DataFrame) -> Union[float, int]`
4. Import and use via `auditor.set_metrics([NewMetric()])`

### Adding Custom Hierarchies
1. Import `Hierarchy`, `HLevel`, `HItem` from `model_auditor.plotting.schemas`
2. Define hierarchy as sequence of levels with optional query filters
3. Pass to `HierarchyPlotter.set_features(hierarchy)`

### Accessing Results
```python
results = auditor.evaluate_metrics(score_name="risk_score")
df = results.to_dataframe(n_decimals=3, metric_labels=True)
feature_df = results.features["gender"].to_dataframe()
level_df = results.features["gender"].levels["Male"].to_dataframe()
```

### Working with Confidence Intervals
- Enable: `evaluate(score_name="...", n_bootstraps=1000)`
- Disable: `evaluate(score_name="...", n_bootstraps=None)`
- Access: `results.features["f"].levels["l"].metrics["m"].interval`

### Bootstrap Resampling Details
- **Method**: Sampling with replacement from original data
- **Iterations**: Controlled by `n_bootstraps` parameter (typically 1000)
- **Computation**: Parallelizable (currently sequential in implementation)
- **Stability**: More iterations → narrower, more stable CI bounds
