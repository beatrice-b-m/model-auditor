"""Seeded bootstrap-output snapshot guarding resampling refactors.

The snapshot records every interval, resample count, status and hypothesis field
for a fixed grid of evaluations. Performance work on the bootstrap loops must
leave it unchanged. Regenerate it only when results change on purpose:

    python -m validation.bootstrap_snapshot
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model_auditor import Auditor, InferenceConfig
from model_auditor.metrics import (
    AUROC,
    FPR,
    Precision,
    Sensitivity,
    Specificity,
    nTP,
)

SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "bootstrap_snapshot.json"
)
N_BOOTSTRAPS = 30
# Low enough to keep the test fast; min_resamples is lowered to match.
_BASE = {"random_state": 3, "min_resamples": 20}


def snapshot_data(n: int = 240, seed: int = 7) -> pd.DataFrame:
    """Synthetic cohort with edge cases for every resampling path.

    Level "s4" of ``site`` has only negatives (undefined sensitivity) and level
    "s3" sits in a single cluster. ``tier`` is categorical with an unobserved
    category.
    """
    rng = np.random.default_rng(seed)
    site = rng.choice(["s1", "s2", "s3", "s4"], n, p=[0.4, 0.3, 0.15, 0.15])
    y = rng.integers(0, 2, n)
    y[site == "s4"] = 0
    new = np.clip(0.35 * y + rng.uniform(size=n) * 0.7, 0, 1)
    ref = np.clip(0.3 * y + rng.uniform(size=n) * 0.7, 0, 1)
    cluster = rng.integers(0, 60, n)
    cluster[site == "s3"] = 999
    tier = pd.Categorical(
        rng.choice(["low", "high"], n), categories=["low", "mid", "high"]
    )
    return pd.DataFrame(
        {
            "y": y,
            "new": new,
            "new_copy": new.copy(),
            "ref": ref,
            "site": site,
            "tier": tier,
            "cluster": cluster,
            "cluster_str": np.char.add("c", cluster.astype(str)),
        }
    )


def snapshot_auditor() -> Auditor:
    auditor = Auditor(
        data=snapshot_data(),
        metrics=[Sensitivity(), Specificity(), FPR(), Precision(), AUROC(), nTP()],
    )
    auditor.add_outcome("y")
    for score in ("new", "new_copy", "ref"):
        auditor.add_score(score, threshold=0.5)
    auditor.add_feature("site")
    auditor.add_feature("tier")
    return auditor


CONFIGS = {
    "iid": InferenceConfig(**_BASE),
    "iid_bootstrap": InferenceConfig(**_BASE, method="bootstrap"),
    "stratified": InferenceConfig(**_BASE, resampling="stratified"),
    "cluster": InferenceConfig(**_BASE, resampling="cluster", cluster="cluster"),
    "cluster_str": InferenceConfig(
        **_BASE, resampling="cluster", cluster="cluster_str"
    ),
}

# Hypothesis tests and the identical-model case run on these configs only.
TESTED_CONFIGS = ("iid", "cluster")

TESTS = {
    "none": {},
    "noninferiority": {
        "test": "noninferiority",
        "test_metrics": ["sensitivity", "specificity", "fpr", "auroc"],
        "margin": 0.1,
        "alpha": 0.05,
    },
    "superiority": {
        "test": "superiority",
        "test_metrics": ["sensitivity", "fpr"],
        "alpha": 0.05,
    },
    "equivalence": {
        "test": "equivalence",
        "test_metrics": ["specificity", "precision"],
        "margin": 0.15,
        "alpha": 0.05,
    },
}


def _encode(value):
    """JSON-safe, bit-exact encoding of result values."""
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return {"float": value.hex() if math.isfinite(value) else repr(value)}
    return value


def _levels(evaluation) -> dict:
    return {
        feature_name: {
            level_name: {
                "support": _encode(level.support),
                "metrics": {
                    name: _encode(metric) for name, metric in level.metrics.items()
                },
            }
            for level_name, level in feature.levels.items()
        }
        for feature_name, feature in evaluation.features.items()
    }


def build_snapshot() -> dict:
    """Evaluate the full grid and return its encoded results."""
    auditor = snapshot_auditor()
    out: dict = {}
    for config_name, config in CONFIGS.items():
        shared = {"n_bootstraps": N_BOOTSTRAPS, "inference": config}
        out[f"evaluate_metrics/{config_name}"] = _levels(
            auditor.evaluate_metrics("new", **shared)
        )
        errors = auditor.evaluate_errors("new", **shared)
        out[f"evaluate_errors/{config_name}"] = {
            group: _levels(evaluation) for group, evaluation in errors.groups.items()
        }
        out[f"evaluate_errors_support/{config_name}"] = _encode(errors.support_data)
        tests = TESTS if config_name in TESTED_CONFIGS else {"none": TESTS["none"]}
        for test_name, test in tests.items():
            out[f"compare_scores/{config_name}/{test_name}"] = _levels(
                auditor.compare_scores("new", "ref", **shared, **test)
            )
            out[f"compare_groups/{config_name}/{test_name}"] = _levels(
                auditor.compare_groups("new", "site", "s1", **shared, **test)
            )
        if config_name in TESTED_CONFIGS:
            out[f"compare_scores_identical/{config_name}"] = _levels(
                auditor.compare_scores("new", "new_copy", **shared)
            )
        out[f"compare_groups_ratio/{config_name}"] = _levels(
            auditor.compare_groups("new", "site", "s1", contrast="ratio", **shared)
        )
    return out


def main() -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(build_snapshot(), indent=0, sort_keys=True) + "\n"
    )
    print(f"Wrote {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
