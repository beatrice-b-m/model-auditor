"""Resampling work scales with levels, not with resamples or metrics."""

import numpy as np
import pandas as pd
import pytest

from model_auditor import Auditor, InferenceConfig
from model_auditor.metrics import AUROC, FPR, Sensitivity

N_BOOTSTRAPS = 40


def make_auditor():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "new": rng.uniform(size=n),
            "ref": rng.uniform(size=n),
            "g": rng.choice(["a", "b"], n),
            "cluster": rng.integers(0, 30, n),
        }
    )
    auditor = Auditor(data=df, metrics=[Sensitivity(), FPR(), AUROC()])
    auditor.add_outcome("y")
    auditor.add_score("new", threshold=0.5)
    auditor.add_score("ref", threshold=0.5)
    auditor.add_feature("g")
    return auditor


@pytest.fixture
def cluster_groupbys(monkeypatch):
    """Count DataFrame.groupby calls keyed on the cluster column."""
    calls = []
    original = pd.DataFrame.groupby

    def counting(self, by=None, *args, **kwargs):
        if isinstance(by, str) and by == "cluster":
            calls.append(len(self))
        return original(self, by, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", counting)
    return calls


CLUSTER = InferenceConfig(random_state=1, resampling="cluster", cluster="cluster")


def test_evaluate_metrics_groups_clusters_once_per_level(cluster_groupbys):
    make_auditor().evaluate_metrics("new", n_bootstraps=N_BOOTSTRAPS, inference=CLUSTER)
    # Overall plus two levels of g.
    assert len(cluster_groupbys) == 3


def test_compare_scores_groups_clusters_once_per_level(cluster_groupbys):
    make_auditor().compare_scores(
        "new", "ref", n_bootstraps=N_BOOTSTRAPS, inference=CLUSTER
    )
    assert len(cluster_groupbys) == 3


def test_compare_groups_groups_clusters_once_per_call(cluster_groupbys):
    make_auditor().compare_groups(
        "new", "g", "a", n_bootstraps=N_BOOTSTRAPS, inference=CLUSTER
    )
    assert len(cluster_groupbys) == 1


@pytest.fixture
def draws(monkeypatch):
    """Count resamples drawn by contrast evaluation."""
    from model_auditor import _comparisons

    calls = []
    original = _comparisons.make_sampler

    def counting(data, config):
        sample = original(data, config)

        def draw(rng):
            calls.append(len(data))
            return sample(rng)

        return draw

    monkeypatch.setattr(_comparisons, "make_sampler", counting)
    return calls


@pytest.mark.parametrize("n_metrics", [1, 3])
def test_compare_scores_draws_once_per_level_for_all_metrics(draws, n_metrics):
    auditor = make_auditor()
    auditor.set_metrics([Sensitivity(), FPR(), AUROC()][:n_metrics])
    auditor.compare_scores("new", "ref", n_bootstraps=N_BOOTSTRAPS, inference=CLUSTER)
    assert len(draws) == 3 * N_BOOTSTRAPS


@pytest.mark.parametrize("n_metrics", [1, 3])
def test_compare_groups_draws_once_per_call_for_all_levels(draws, n_metrics):
    rng = np.random.default_rng(2)
    auditor = make_auditor()
    auditor.data["g"] = rng.choice(["a", "b", "c", "d"], len(auditor.data))
    auditor.set_metrics([Sensitivity(), FPR(), AUROC()][:n_metrics])
    auditor.compare_groups(
        "new", "g", "a", n_bootstraps=N_BOOTSTRAPS, inference=CLUSTER
    )
    assert draws == [len(auditor.data)] * N_BOOTSTRAPS


@pytest.fixture
def error_draws(monkeypatch):
    """Count resamples drawn by error evaluation."""
    from model_auditor import _evaluation

    calls = []
    original = _evaluation.make_index_sampler

    def counting(data, config):
        sample = original(data, config)

        def draw(rng):
            calls.append(len(data))
            return sample(rng)

        return draw

    monkeypatch.setattr(_evaluation, "make_index_sampler", counting)
    return calls


def test_evaluate_errors_draws_once_per_feature_for_all_groups(error_draws):
    make_auditor().evaluate_errors("new", n_bootstraps=N_BOOTSTRAPS, inference=CLUSTER)
    # Only g resamples (overall has no "rest", so its OR is undefined), and
    # its draws are shared by TP, TN, FP and FN.
    assert len(error_draws) == N_BOOTSTRAPS


def test_evaluate_errors_groups_clusters_once_per_feature(cluster_groupbys):
    make_auditor().evaluate_errors("new", n_bootstraps=N_BOOTSTRAPS, inference=CLUSTER)
    assert len(cluster_groupbys) == 1


@pytest.mark.parametrize(
    "config",
    [
        InferenceConfig(random_state=5),
        InferenceConfig(random_state=5, resampling="stratified"),
        CLUSTER,
    ],
    ids=["iid", "stratified", "cluster"],
)
def test_index_sampler_matches_frame_sampler(config):
    from model_auditor._evaluation import make_index_sampler, make_sampler

    data = make_auditor().data.assign(_truth=lambda d: d["y"])
    positions = make_index_sampler(data, config)
    frames = make_sampler(data, config)
    index_rng = np.random.default_rng(config.random_state)
    frame_rng = np.random.default_rng(config.random_state)
    for _ in range(5):
        expected = data.iloc[positions(index_rng)]
        pd.testing.assert_frame_equal(frames(frame_rng), expected)
