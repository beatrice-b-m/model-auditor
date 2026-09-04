# Model Auditor

Evaluate fixed binary-model predictions across subgroups: discrimination, decision performance, probability calibration, and paired comparisons, with explicit statistical assumptions and reproducible results.

[Documentation](https://model-auditor-docs.beatricebm.workers.dev) · [PyPI](https://pypi.org/project/model-auditor/) · [Issues](https://github.com/beatrice-b-m/model-auditor/issues)

## Installation

Requires Python 3.10 or later.

```bash
pip install model-auditor
```

Optional plotting and styled-table dependencies:

```bash
pip install 'model-auditor[plotting,styling]'
```

## Quick start

```python
import pandas as pd

from model_auditor import Auditor, InferenceConfig
from model_auditor.metrics import AUROC, Sensitivity, Specificity, nData

# Replace this small example with your own observations.
df = pd.DataFrame({
    "region": ["North", "North", "North", "South", "South", "South"],
    "risk_score": [0.9, 0.6, 0.2, 0.8, 0.4, 0.1],
    "outcome": [1, 0, 0, 1, 1, 0],
})

auditor = Auditor(data=df)
auditor.add_feature(name="region", label="Region")
auditor.add_score(name="risk_score", threshold=0.5)
auditor.add_outcome(name="outcome")
auditor.set_metrics([nData(), Sensitivity(), Specificity(), AUROC()])

results = auditor.evaluate_metrics(
    "risk_score", inference=InferenceConfig(random_state=42), cohort="held-out"
)
print(results.to_numeric_dataframe())
```

Results include original-sample estimates, denominators, exclusions, and interval diagnostics. Undefined metrics are NaN. IID rates use approximate Wilson intervals (`rate_interval="exact"` selects conservative binomial intervals); enrichment ORs use conditional exact intervals. Other metrics use diagnosed bootstrap intervals. Set `n_bootstraps=None` for descriptive estimates only.

Use `compare_scores()` for paired models, `compare_groups()` for reference-group metric contrasts, `evaluate_calibration()` for probability diagnostics, and `add_intersection()` for joint subgroups. `InferenceConfig(resampling="cluster", cluster="patient_id", random_state=42)` resamples whole subjects while retaining row-weighted estimates.

Intervals are pointwise and condition on fixed predictions and thresholds. Select models/thresholds on separate data; repeated observations require an appropriate sampling unit. Enrichment odds ratios describe confusion-group membership, not class-conditional error-rate disparities or causal fairness. Multiclass, survival, survey-weighted inference, and model-refitting validation are outside this package's current scope.

## Learn more

The [documentation website](https://model-auditor-docs.beatricebm.workers.dev) contains the guides, examples, metric definitions, API reference, and statistical conventions. Its [source repository](https://github.com/beatrice-b-m/model-auditor-docs) tracks the latest stable release; this checkout may contain unreleased changes.

For work on the library itself, see [CONTRIBUTING.md](CONTRIBUTING.md). Agent instructions are in [AGENTS.md](AGENTS.md).
