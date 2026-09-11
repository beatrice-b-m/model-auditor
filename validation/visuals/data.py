"""Deterministic synthetic data for the visual example collection.

The frames built here are synthetic and carry no clinical or fairness meaning.
Fixed seeds and fixed level counts keep every generated artifact reproducible
across machines and CI runs.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from model_auditor import Auditor
from model_auditor.metrics import AUROC, F1Score, Sensitivity, Specificity
from model_auditor.schemas import (
    AuditorFeature,
    InferenceConfig,
    ScoreEvaluation,
)

SEED = 20260601
SCORE_NAME = "risk_score"
SCORE_LABEL = "Risk score"
OUTCOME_NAME = "label"
THRESHOLD = 0.5
N_BOOTSTRAPS = 400
INFERENCE_SEED = 17

AGE_LEVELS = ("<40", "40-64", "65+")
DIFFICULT_AGE_LEVELS = AGE_LEVELS

DOC_FEATURES = (
    AuditorFeature(name="sex", label="Sex"),
    AuditorFeature(name="age_group", label="Age group"),
    AuditorFeature(name="site", label="Site"),
)

DIFFICULT_FEATURES = (
    AuditorFeature(name="service_region", label="Service region"),
    AuditorFeature(name="clinic", label="Clinic"),
    AuditorFeature(name="referral", label="Referral pathway"),
)

SERVICE_REGIONS = (
    "Northeast Metropolitan Service Area",
    "Pacific Northwest Coastal Region",
    "Upper Midwest Rural District",
    "Southern Gulf Catchment",
)

# Clinic 10 receives two rows; its per-level metrics are frequently undefined,
# which exercises the omitted-level rendering path.
CLINIC_NAMES = tuple(f"Clinic {index:02d}" for index in range(1, 11))
CLINIC_COUNTS = (46, 40, 34, 28, 22, 16, 12, 8, 4, 2)

# "Declined" is declared but never observed, producing an unobserved
# categorical placeholder level in evaluation output.
REFERRAL_LEVELS = ("Self-referral", "Primary care", "Specialist", "Declined")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def example_frame(n: int = 240) -> pd.DataFrame:
    """Return the seeded stratification frame used by documentation examples."""
    rng = np.random.default_rng(SEED)
    sex = rng.choice(("Female", "Male"), size=n, p=(0.55, 0.45))
    age = rng.choice(AGE_LEVELS, size=n, p=(0.35, 0.40, 0.25))
    site = rng.choice(("A", "B", "C", "D"), size=n, p=(0.30, 0.26, 0.24, 0.20))

    latent = (
        1.10 * (sex == "Female")
        + 0.70 * (age == "65+")
        + 0.25 * (age == "40-64")
        - 0.30 * (site == "D")
        + rng.normal(0.0, 1.0, size=n)
    )
    label = (latent > 0.5).astype(int)
    subgroup_shift = (
        0.30 * (sex == "Male") - 0.20 * (age == "<40") + 0.15 * (site == "B")
    )
    risk_score = _sigmoid(
        1.10 * latent - 0.60 + subgroup_shift + rng.normal(0.0, 0.90, size=n)
    )

    return pd.DataFrame(
        {
            "sex": sex,
            "age_group": pd.Categorical(age, categories=list(AGE_LEVELS), ordered=True),
            "site": site,
            SCORE_NAME: risk_score,
            OUTCOME_NAME: label,
        }
    )


def _assign(counts: tuple[int, ...], names: tuple[str, ...], n: int) -> np.ndarray:
    """Repeat each name to its requested count, then clip or pad to ``n`` rows."""
    assigned = np.concatenate(
        [np.full(int(count), name) for count, name in zip(counts, names)]
    )
    if len(assigned) >= n:
        return assigned[:n]
    filler = np.full(n - len(assigned), names[0])
    return np.concatenate([assigned, filler])


def difficult_frame(n: int = 212) -> pd.DataFrame:
    """Return the seeded frame for difficult rendering cases.

    Includes long labels, ten clinic levels (two with tiny support), an
    unobserved categorical placeholder, and missing feature values.
    """
    rng = np.random.default_rng(SEED + 101)
    service_region = _assign((86, 63, 42, 21), SERVICE_REGIONS, n)
    clinic = _assign(CLINIC_COUNTS, CLINIC_NAMES, n)
    referral = rng.choice(REFERRAL_LEVELS[:3], size=n, p=(0.30, 0.50, 0.20))
    rng.shuffle(service_region)
    rng.shuffle(clinic)

    region_weight = np.select(
        [service_region == name for name in SERVICE_REGIONS],
        [0.5, 0.1, -0.2, -0.4],
        default=0.0,
    )
    clinic_weight = np.array([0.05 * (CLINIC_NAMES.index(name) % 3) for name in clinic])
    latent = (
        region_weight
        + clinic_weight
        + 0.6 * (referral == "Specialist")
        - 0.3 * (referral == "Self-referral")
        + rng.normal(0.0, 1.0, size=n)
    )
    label = (latent > 0.15).astype(int)
    risk_score = _sigmoid(1.4 * latent - 0.2 + rng.normal(0.0, 0.7, size=n))

    frame = pd.DataFrame(
        {
            "service_region": service_region,
            "clinic": clinic,
            "referral": pd.Categorical(referral, categories=list(REFERRAL_LEVELS)),
            SCORE_NAME: risk_score,
            OUTCOME_NAME: label,
        }
    )
    # Preserve an explicit missing-value path without disturbing level counts.
    frame.loc[frame.index[::53], "service_region"] = np.nan
    return frame


def _build_auditor(
    frame: pd.DataFrame, features: tuple[AuditorFeature, ...]
) -> Auditor:
    auditor = Auditor()
    auditor.add_data(frame)
    for feature in features:
        auditor.add_feature(feature.name, feature.label)
    auditor.add_score(SCORE_NAME, SCORE_LABEL, threshold=THRESHOLD)
    auditor.add_outcome(OUTCOME_NAME)
    auditor.set_metrics([Sensitivity(), Specificity(), F1Score(), AUROC()])
    return auditor


@lru_cache(maxsize=None)
def example_auditor() -> Auditor:
    """Configured auditor over :func:`example_frame`."""
    return _build_auditor(example_frame(), DOC_FEATURES)


@lru_cache(maxsize=None)
def difficult_auditor() -> Auditor:
    """Configured auditor over :func:`difficult_frame`."""
    return _build_auditor(difficult_frame(), DIFFICULT_FEATURES)


@lru_cache(maxsize=None)
def example_evaluation() -> ScoreEvaluation:
    """Seeded interval evaluation used by the documentation examples."""
    return example_auditor().evaluate_metrics(
        SCORE_NAME,
        n_bootstraps=N_BOOTSTRAPS,
        inference=InferenceConfig(random_state=INFERENCE_SEED),
    )


@lru_cache(maxsize=None)
def difficult_evaluation() -> ScoreEvaluation:
    """Seeded interval evaluation used by the difficult rendering examples."""
    return difficult_auditor().evaluate_metrics(
        SCORE_NAME,
        n_bootstraps=N_BOOTSTRAPS,
        inference=InferenceConfig(random_state=INFERENCE_SEED),
    )


@lru_cache(maxsize=None)
def example_error_evaluation():
    """Seeded confusion-membership enrichment evaluation."""
    return example_auditor().evaluate_errors(
        SCORE_NAME,
        n_bootstraps=N_BOOTSTRAPS,
        inference=InferenceConfig(random_state=INFERENCE_SEED),
    )


@lru_cache(maxsize=None)
def difficult_error_evaluation():
    """Seeded enrichment evaluation over the difficult frame."""
    return difficult_auditor().evaluate_errors(
        SCORE_NAME,
        n_bootstraps=N_BOOTSTRAPS,
        inference=InferenceConfig(random_state=INFERENCE_SEED),
    )
