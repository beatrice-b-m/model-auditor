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
- **Threshold Optimization**: Optimize thresholds via Youden index or target sensitivity/specificity constraints
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
results = auditor.evaluate_metrics(score_name="risk_score", n_bootstraps=1000)

# Convert results to a DataFrame
results_df = results.to_dataframe()
print(results_df)
```

## Threshold Optimization

Use Youden-index optimization when you want a single balanced operating point:

```python
auditor = Auditor()
auditor.add_data(df)
auditor.add_score(name="risk_score")
auditor.add_outcome(name="label")

youden_threshold = auditor.optimize_score_threshold(score_name="risk_score")
# Output: Optimal threshold for 'risk_score' found at: 0.423
```

Use target-based optimization when deployment requires a minimum sensitivity
or specificity:

```python
# Highest threshold that still satisfies sensitivity >= 0.70
sens_threshold = auditor.optimize_score_threshold_for_target(
    score_name="risk_score",
    target=0.70,
    metric="sensitivity",
)

# Lowest threshold that still satisfies specificity >= 0.90
spec_threshold = auditor.optimize_score_threshold_for_target(
    score_name="risk_score",
    target=0.90,
    metric="specificity",
)
```

If no **finite** threshold can satisfy the requested target, the method raises
`ValueError` and reports the achievable metric range across finite thresholds:

```python
try:
    auditor.optimize_score_threshold_for_target(
        score_name="risk_score",
        target=0.95,
        metric="specificity",
    )
except ValueError as exc:
    print(exc)
    # No finite threshold for score 'risk_score' can satisfy specificity >= ...
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
results = auditor.evaluate_metrics(score_name="risk_score", n_bootstraps=None)
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

## Score Interval Plots

Visualise bootstrap confidence intervals for any metric across feature levels
using `plot_metric_intervals`.  The method requires `matplotlib` and returns one
`(Figure, Axes)` tuple per feature:

```python
# Run evaluation with bootstrap CIs first.
results = auditor.evaluate_metrics(score_name="risk_score", n_bootstraps=1000)

# Plot every feature (one figure per feature).
plots = results.plot_metric_intervals(metric="sensitivity")

# Select a subset of features.
plots = results.plot_metric_intervals(
    metric="sensitivity",
    feature_names=["gender", "age_group"],
)

# Metric can be supplied by name or by label.
plots = results.plot_metric_intervals(metric="Sensitivity")   # label also works

# Display in a Jupyter notebook.
import matplotlib.pyplot as plt

for feature_name, (fig, ax) in plots.items():
    plt.show()
```

Each figure shows levels on the y-axis (in evaluation order) and the metric
value on the x-axis.  Horizontal whiskers span the 95% CI bounds.
Levels with no CI data (e.g. unobserved categorical placeholders, count
metrics) are automatically excluded from the plot.

```python
# Access the Axes object to customise the figure further.
fig, ax = plots["gender"]
ax.set_xlim(0.0, 1.0)
ax.axvline(0.8, linestyle="--", color="grey", label="Target")
ax.legend()
plt.tight_layout()
plt.show()
```


## Score Distribution Plots

Visualize raw score distributions stratified by feature levels — no `evaluate_metrics()` call required:

```python
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend if needed

# Plot score distributions for all registered features
plots = auditor.plot_score_distributions(score_name="risk_score")

# Each entry: (matplotlib.figure.Figure, numpy.ndarray of Axes)
fig, axes = plots["age_group"]
fig.savefig("age_group_distributions.png", dpi=150)

# Limit to a specific subset of features
plots = auditor.plot_score_distributions(
    score_name="risk_score",
    feature_names=["gender", "age_group"],
)

# Raw counts instead of density
plots = auditor.plot_score_distributions(
    score_name="risk_score",
    density=False,
)

# Custom binning
plots = auditor.plot_score_distributions(
    score_name="risk_score",
    bins=50,
)
```

Each figure contains one histogram subplot per feature level. All subplots share the same x-axis and use identical bin edges, enabling direct visual comparison of score distributions across subgroups. Bins are computed from the entire feature slice rather than per-level, so relative spread and overlap are preserved.


## Controlling Feature Level Order

By default, feature levels appear in the order they were encountered in the
data.  To control the row order in exported DataFrames, assign the feature
column a `pd.Categorical` dtype with an explicit `categories` list before
passing the data to the auditor:

```python
import pandas as pd
from model_auditor import Auditor
from model_auditor.metrics import Sensitivity, Specificity

# Declare the desired display order for the 'age_group' column.
# Categories not present in the data still appear as rows (with NaN values).
df["age_group"] = pd.Categorical(
    df["age_group"],
    categories=["<30", "30-50", "50-70", ">70"],
    ordered=True,
)

auditor = Auditor()
auditor.add_data(df)
auditor.add_feature(name="age_group")
auditor.add_score(name="risk_score", threshold=0.5)
auditor.add_outcome(name="outcome")
auditor.set_metrics([Sensitivity(), Specificity()])

results = auditor.evaluate_metrics(score_name="risk_score", n_bootstraps=None)

# Rows appear in the declared order: <30, 30-50, 50-70, >70.
# If no rows belong to a declared category (e.g. '>70' is absent from the
# data), that category still appears as a row with NaN metric values.
df_out = results.features["age_group"].to_dataframe()
```

The same order is preserved in `style_dataframe()` and in the score-level
`ScoreEvaluation.to_dataframe()` / `ScoreEvaluation.style_dataframe()` exports.
Non-categorical feature columns are unaffected.



## Error Analysis

Use `evaluate_errors()` to understand which subgroups are over- or
under-represented within each confusion-matrix group (TP, TN, FP, FN).
For every feature level the *canonical 2×2 odds ratio* (OR) is computed:

    OR(level, group) = (a × d) / (b × c)

Where `a = count(level ∩ group)`, `b = count(level ∩ not-group)`,
`c = count(not-level ∩ group)`, `d = count(not-level ∩ not-group)`.

OR = 1 means the level has the same odds of appearing in that confusion group
as all other levels combined.  OR > 1 indicates over-representation;
OR < 1 under-representation.

```python
# No additional metric setup required — evaluate_errors() uses OddsRatio by default.
error_results = auditor.evaluate_errors(score_name="risk_score", n_bootstraps=1000)

# Convert to a wide analysis-ready DataFrame.
# Rows: MultiIndex(feature, level)
# Columns: MultiIndex(section, metric)
#   Overall section: N, % overall, N_pos, N_neg, Pos %
#   Per group (TP/TN/FP/FN): N, % overall, % group, odds_ratio,
#                             odds_ratio_ci_lower, odds_ratio_ci_upper
df = error_results.to_dataframe()
print(df)

# Use metric_labels=True for human-readable column names:
df_labels = error_results.to_dataframe(metric_labels=True)

# Per-group deep inspection is still available:
tp_age = error_results.groups["tp"].features["age_group"]
print(tp_age.to_dataframe())

# Styled wide view: OR cells include inline CI when bootstraps were used,
# and FP/FN tier colouring is inverted (higher OR = worse in error groups).
display(error_results.style_dataframe(n_decimals=3, metric_labels=True))
```

## License
## Notebook Styling

For Jupyter notebooks, `style_dataframe(...)` returns a pandas `Styler` that colours cells by relative performance tier within each metric column.

```python
# Colour all levels in a feature by relative tier (default: performance metrics only)
display(results.features['age_group'].style_dataframe(n_decimals=3, metric_labels=True))

# Also colour count columns (N, TP, TN, …)
display(results.features['gender'].style_dataframe(include_count_metrics=True))

# Opt into custom colours
display(results.style_dataframe(
    low_color="#ffd6d6",
    medium_color="#fff9c4",
    high_color="#d0f0d0",
))
```

### Tier assignment

| Tier | Default colour | Meaning |
|------|---------------|----------|
| High | `#d4edda` (green) | Top third of values in the column |
| Medium | `#fff3cd` (yellow) | Middle third |
| Low | `#f8d7da` (red) | Bottom third |

Tiers are computed **per metric column** across all rows in the table. Lower-is-better metrics (`fpr`, `fnr`) are inverted: a lower value receives the high (green) tier.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_decimals` | `3` | Decimal places for numeric display |
| `metric_labels` | `False` | Use metric labels as column headers instead of names |
| `include_count_metrics` | `False` | Also style count columns (N, TP, TN, FP, FN, Pos., Neg.) |
| `low_color` | `"#f8d7da"` | Background colour for low-tier cells |
| `medium_color` | `"#fff3cd"` | Background colour for medium-tier cells |
| `high_color` | `"#d4edda"` | Background colour for high-tier cells |


MIT License

## Author

Beatrice BM
