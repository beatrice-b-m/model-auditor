"""Opt-in coverage experiments for fixed binary predictions (not a CI gate).

Run: python validation/coverage.py --trials 100 --bootstraps 200
Reports unconditional coverage, Monte Carlo SE, interval availability and width.
These selected data-generating processes do not establish universal coverage.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from model_auditor._evaluation import evaluate_error_feature, evaluate_level
from model_auditor.error_metrics import OddsRatio
from model_auditor.metrics import AUROC, Sensitivity
from model_auditor.schemas import AuditorFeature, InferenceConfig


def run(trials, bootstraps):
    rng = np.random.default_rng(20260904)
    results = {
        key: []
        for key in (
            "iid_sensitivity",
            "cluster_sensitivity",
            "binormal_auc",
            "sparse_null_or",
        )
    }
    for i in range(trials):
        config = InferenceConfig(random_state=i)
        success = rng.binomial(1, 0.95, size=10)
        data = pd.DataFrame({"_truth": 1, "tp": success, "fn": 1 - success})
        m = evaluate_level([Sensitivity()], data, "", bootstraps, config).metrics[
            "sensitivity"
        ]
        results["iid_sensitivity"].append((0.95, m.interval))

        success = rng.binomial(1, 0.8, size=30)
        data = pd.DataFrame(
            {"subject": np.arange(30), "_truth": 1, "tp": success, "fn": 1 - success}
        )
        data = data.loc[data.index.repeat(5)]
        clustered = InferenceConfig(
            random_state=i, resampling="cluster", cluster="subject"
        )
        m = evaluate_level([Sensitivity()], data, "", bootstraps, clustered).metrics[
            "sensitivity"
        ]
        results["cluster_sensitivity"].append((0.8, m.interval))

        data = pd.DataFrame(
            {
                "_truth": [0] * 50 + [1] * 50,
                "_pred": np.r_[rng.normal(0, 1, 50), rng.normal(1, 1, 50)],
            }
        )
        m = evaluate_level([AUROC()], data, "", bootstraps, config).metrics["auroc"]
        results["binormal_auc"].append((float(norm.cdf(1 / np.sqrt(2))), m.interval))

        cells = rng.multinomial(20, [0.25] * 4)
        data = pd.DataFrame(
            {
                "g": ["A"] * int(cells[0] + cells[1])
                + ["B"] * int(cells[2] + cells[3]),
                "fp": [1] * int(cells[0])
                + [0] * int(cells[1])
                + [1] * int(cells[2])
                + [0] * int(cells[3]),
                "_truth": [0] * 20,
            }
        )
        result, _ = evaluate_error_feature(
            data, "fp", AuditorFeature("g"), OddsRatio(), bootstraps, 20, config
        )
        metric = result.levels.get("A")
        results["sparse_null_or"].append(
            (1.0, metric.metrics["odds_ratio"].interval if metric else None)
        )
    summary = {}
    for name, observations in results.items():
        available = [ci for _, ci in observations if ci is not None]
        coverage = (
            sum(
                ci is not None and ci[0] <= truth <= ci[1] for truth, ci in observations
            )
            / trials
        )
        widths = [ci[1] - ci[0] for ci in available if np.isfinite(ci).all()]
        summary[name] = {
            "trials": trials,
            "coverage_all_trials": coverage,
            "monte_carlo_se": float(np.sqrt(coverage * (1 - coverage) / trials)),
            "interval_availability": len(available) / trials,
            "median_finite_width": float(np.median(widths)) if widths else None,
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--bootstraps", type=int, default=200)
    args = parser.parse_args()
    if args.trials < 1 or args.bootstraps < 100:
        parser.error("Use positive trials and at least 100 bootstraps.")
    print(json.dumps(run(args.trials, args.bootstraps), indent=2))
