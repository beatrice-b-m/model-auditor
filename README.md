# Model Auditor

A Python library for evaluating machine learning model performance across subgroups with support for stratified metrics, bootstrap confidence intervals, and hierarchical visualizations.

## Installation

```bash
pip install model-auditor
```

## Features

- **Stratified Evaluation**: Evaluate model metrics across different subgroups (e.g., by age, gender, region)
- **Bootstrap Confidence Intervals**: Calculate 95% confidence intervals for all supported metrics
- **Comprehensive Metrics**: Built-in support for classification metrics including:
  - Sensitivity, Specificity, Precision, Recall, F1 Score
  - AUROC, AUPRC
  - Matthews Correlation Coefficient (MCC)
  - F-beta Score (configurable beta)
  - TPR, TNR, FPR, FNR
  - Count metrics (N, TP, TN, FP, FN, Positive, Negative)
- **Threshold Optimization**: Automatic threshold selection using the Youden index
- **Hierarchical Visualization**: Generate data structures for sunburst/treemap plots
- **Extensible Design**: Protocol-based architecture for custom metrics

## Quick Start

```python
from model_auditor import Auditor
from model_auditor.metrics import Sensitivity, Specificity, AUROC, F1Score

# Initialize the auditor
auditor = Auditor()

# Add your data
auditor.add_data(df)

# Define stratification features
auditor.add_feature(name="age_group", label="Age Group")
auditor.add_feature(name="gender", label="Gender")

# Define the score column and threshold
auditor.add_score(name="risk_score", label="Risk Score", threshold=0.5)

# Define the outcome column
auditor.add_outcome(name="diagnosis", mapping={"positive": 1, "negative": 0})

# Set metrics to evaluate
auditor.set_metrics([
    Sensitivity(),
    Specificity(),
    AUROC(),
    F1Score()
])

# Run evaluation with bootstrap confidence intervals
results = auditor.evaluate(score_name="risk_score", n_bootstraps=1000)

# Convert results to a DataFrame
results_df = results.to_dataframe()
print(results_df)
```

## Threshold Optimization

Find the optimal decision threshold using the Youden index:

```python
auditor = Auditor()
auditor.add_data(df)
auditor.add_score(name="risk_score")
auditor.add_outcome(name="label")

# Find optimal threshold
optimal_threshold = auditor.optimize_score_threshold(score_name="risk_score")
# Output: Optimal threshold for 'risk_score' found at: 0.423
```

## Available Metrics

### Classification Metrics

| Metric | Class | Description |
|--------|-------|-------------|
| Sensitivity | `Sensitivity()` | TP / (TP + FN) |
| Specificity | `Specificity()` | TN / (TN + FP) |
| Precision | `Precision()` | TP / (TP + FP) |
| Recall | `Recall()` | TP / (TP + FN) |
| F1 Score | `F1Score()` | Harmonic mean of precision and recall |
| F-beta | `FBetaScore(beta=2.0)` | Weighted harmonic mean |
| MCC | `MatthewsCorrelationCoefficient()` | Matthews Correlation Coefficient |

### Ranking Metrics

| Metric | Class | Description |
|--------|-------|-------------|
| AUROC | `AUROC()` | Area Under ROC Curve |
| AUPRC | `AUPRC()` | Area Under Precision-Recall Curve |

### Rate Metrics

| Metric | Class | Description |
|--------|-------|-------------|
| TPR | `TPR()` | True Positive Rate |
| TNR | `TNR()` | True Negative Rate |
| FPR | `FPR()` | False Positive Rate |
| FNR | `FNR()` | False Negative Rate |

### Count Metrics

| Metric | Class | Description |
|--------|-------|-------------|
| N | `nData()` | Sample size |
| TP | `nTP()` | True positive count |
| TN | `nTN()` | True negative count |
| FP | `nFP()` | False positive count |
| FN | `nFN()` | False negative count |
| Positive | `nPositive()` | Positive class count |
| Negative | `nNegative()` | Negative class count |

## Custom Metrics

Create custom metrics by implementing the `AuditorMetric` protocol:

```python
from model_auditor.metrics import AuditorMetric
import pandas as pd

class AccuracyMetric(AuditorMetric):
    name = "accuracy"
    label = "Accuracy"
    inputs = ["tp", "tn", "fp", "fn"]
    ci_eligible = True

    def data_call(self, data: pd.DataFrame) -> float:
        tp = data["tp"].sum()
        tn = data["tn"].sum()
        fp = data["fp"].sum()
        fn = data["fn"].sum()
        return (tp + tn) / (tp + tn + fp + fn)

# Use with the auditor
auditor.set_metrics([AccuracyMetric(), Sensitivity()])
```

## Hierarchical Visualization

Generate data for hierarchical plots (sunburst, treemap):

```python
from model_auditor.plotting import HierarchyPlotter

plotter = HierarchyPlotter()
plotter.set_data(df)
plotter.set_features(["region", "age_group", "gender"])
plotter.set_score(name="risk_score")
plotter.set_aggregator("median")  # or "mean", or a custom function

# Compile plot data
plot_data = plotter.compile(container="All Patients")

# Use with Plotly
import plotly.graph_objects as go

fig = go.Figure(go.Sunburst(
    labels=plot_data.labels,
    ids=plot_data.ids,
    parents=plot_data.parents,
    values=plot_data.values,
    marker=dict(colors=plot_data.colors)
))
fig.show()
```

### Custom Hierarchies

Define complex hierarchies with conditional features:

```python
from model_auditor.plotting.schemas import Hierarchy, HLevel, HItem

hierarchy = Hierarchy(levels=[
    HLevel([HItem(name="region")]),
    HLevel([
        HItem(name="urban_category", query="region == 'Urban'"),
        HItem(name="rural_category", query="region == 'Rural'")
    ]),
    HLevel([HItem(name="age_group")])
])

plotter.set_features(hierarchy)
```

## Disabling Confidence Intervals

For faster evaluation without confidence intervals:

```python
results = auditor.evaluate(score_name="risk_score", n_bootstraps=None)
```

## Output Format

Results are returned as nested dataclass objects that can be converted to DataFrames:

```python
# Get results as DataFrame
df = results.to_dataframe(n_decimals=3, metric_labels=True)

# Access specific feature results
gender_results = results.features["gender"].to_dataframe()

# Access specific level results
male_results = results.features["gender"].levels["Male"].to_dataframe()
```

## License

MIT License

## Author

Beatrice BM
