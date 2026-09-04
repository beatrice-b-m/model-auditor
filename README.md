# Model Auditor

Audit binary-classification performance across subgroups with stratified metrics, bootstrap confidence intervals, configurable thresholds, error analysis, and visualizations.

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

from model_auditor import Auditor
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

results = auditor.evaluate_metrics("risk_score", n_bootstraps=None)
print(results.to_dataframe(metric_labels=True))
```

Set `n_bootstraps=1000` to estimate 95% confidence intervals for eligible metrics; count metrics do not receive intervals. Results include an overall baseline and each configured subgroup.

## Learn more

The [documentation website](https://model-auditor-docs.beatricebm.workers.dev) contains the guides, examples, metric definitions, API reference, and statistical conventions. Its [source repository](https://github.com/beatrice-b-m/model-auditor-docs) tracks the latest stable release; this checkout may contain unreleased changes.

For work on the library itself, see [CONTRIBUTING.md](CONTRIBUTING.md). Agent instructions are in [AGENTS.md](AGENTS.md).
