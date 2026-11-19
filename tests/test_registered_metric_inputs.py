"""Registered custom metric inputs (issue #5, part 1)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, norm

from model_auditor import Auditor, InferenceConfig
from model_auditor.metric_inputs import AuditorMetricInput
from model_auditor.utils import is_metric_input_valid


class DetectedEvent:
    """Predicted positive and a secondary event occurred; no row_call."""

    name = "detected_event"
    label = "Detected event"
    inputs = ["_binary_pred", "event"]

    def data_transform(self, data):
        data[self.name] = ((data["_binary_pred"] == 1) & (data["event"] == 1)).astype(
            int
        )
        return data


class DetectedEventRate:
    name = "detected_event_rate"
    label = "Detected event rate"
    inputs = ["detected_event"]
    ci_eligible = True
    direction = "higher"
    binomial_indicator = "detected_event"

    def data_call(self, data):
        return float(data["detected_event"].mean())


class ScaledDetectedEventRate(DetectedEventRate):
    name = "detected_events_per_1000"
    scale = 1000


class TruePositiveEvent:
    """Chained: registered input built on a built-in (tp)."""

    name = "tp_event"
    label = "TP with event"
    inputs = ["tp", "event"]

    def data_transform(self, data):
        data[self.name] = (data["tp"] * data["event"]).astype(int)
        return data


class TruePositiveEventFlag:
    """Chained: registered input built on another registered input."""

    name = "tp_event_flag"
    label = "TP event flag"
    inputs = ["tp_event"]

    def data_transform(self, data):
        data[self.name] = data["tp_event"]
        return data


class TruePositiveEventRate:
    name = "tp_event_rate"
    label = "TP event rate"
    inputs = ["tp_event_flag"]
    ci_eligible = True
    direction = "higher"
    binomial_indicator = "tp_event_flag"

    def data_call(self, data):
        return float(data["tp_event_flag"].mean())


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "y": y,
            "s": np.clip(0.3 * y + rng.uniform(0, 0.7, n), 0, 1),
            "r": np.clip(0.1 * y + rng.uniform(0, 0.9, n), 0, 1),
            "event": rng.integers(0, 2, n),
            "g": rng.choice(["a", "b", "c"], n),
            "cluster": rng.integers(0, 50, n),
        }
    )


def make_auditor(df, metrics=None):
    a = Auditor(data=df)
    a.add_outcome("y")
    a.add_score("s", threshold=0.5)
    a.add_score("r", threshold=0.4)
    a.add_feature("g")
    a.register_metric_input(DetectedEvent)
    a.set_metrics(metrics or [DetectedEventRate()])
    return a


def hand_rate(df, score, threshold, mask=None):
    rows = df if mask is None else df[mask]
    detected = (rows[score] >= threshold) & (rows["event"] == 1)
    return int(detected.sum()), len(rows)


def hand_wilson(k, n, level=0.95):
    z = norm.ppf((1 + level) / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - radius), min(1.0, centre + radius)


def test_detected_event_rate_gets_wilson_and_exact_intervals():
    df = make_data()
    a = make_auditor(df)
    wilson = a.evaluate_metrics("s", n_bootstraps=100)
    exact = a.evaluate_metrics(
        "s", n_bootstraps=100, inference=InferenceConfig(rate_interval="exact")
    )
    for level_name, level in wilson.features["g"].levels.items():
        k, n = hand_rate(df, "s", 0.5, df["g"] == level_name)
        metric = level.metrics["detected_event_rate"]
        assert metric.score == pytest.approx(k / n)
        assert metric.denominator == n
        assert metric.interval_method == "wilson"
        assert metric.interval == pytest.approx(hand_wilson(k, n))
        ci = binomtest(k, n).proportion_ci(method="exact")
        exact_metric = (
            exact.features["g"].levels[level_name].metrics["detected_event_rate"]
        )
        assert exact_metric.interval_method == "clopper_pearson"
        assert exact_metric.interval == pytest.approx((ci.low, ci.high))


@pytest.mark.parametrize(
    "inference",
    [
        InferenceConfig(random_state=1, resampling="stratified"),
        InferenceConfig(random_state=1, resampling="cluster", cluster="cluster"),
    ],
)
def test_registered_input_under_percentile_resampling(inference):
    df = make_data()
    result = make_auditor(df).evaluate_metrics(
        "s", n_bootstraps=100, inference=inference
    )
    metric = result.features["overall"].levels["Overall"].metrics["detected_event_rate"]
    k, n = hand_rate(df, "s", 0.5)
    assert metric.score == pytest.approx(k / n)
    assert metric.interval_method == f"{inference.resampling}_percentile"
    assert metric.interval[0] <= k / n <= metric.interval[1]


def test_scaled_rate_on_registered_indicator():
    df = make_data()
    a = make_auditor(df, [DetectedEventRate(), ScaledDetectedEventRate()])
    level = (
        a.evaluate_metrics("s", n_bootstraps=50).features["overall"].levels["Overall"]
    )
    raw, scaled = (
        level.metrics["detected_event_rate"],
        level.metrics["detected_events_per_1000"],
    )
    assert scaled.score == pytest.approx(raw.score * 1000)
    assert scaled.interval == pytest.approx(tuple(b * 1000 for b in raw.interval))
    assert scaled.denominator == raw.denominator


@pytest.mark.parametrize("resampling", ["iid", "stratified", "cluster"])
def test_compare_scores_recomputes_input_for_reference(resampling):
    df = make_data()
    config = InferenceConfig(
        random_state=2,
        resampling=resampling,
        cluster="cluster" if resampling == "cluster" else None,
    )
    result = make_auditor(df).compare_scores(
        "s", "r", n_bootstraps=100, inference=config
    )
    for level_name, level in result.features["g"].levels.items():
        mask = df["g"] == level_name
        k_left, n = hand_rate(df, "s", 0.5, mask)
        k_right, _ = hand_rate(df, "r", 0.4, mask)
        metric = level.metrics["detected_event_rate"]
        assert metric.score == pytest.approx((k_left - k_right) / n)
        assert metric.interval is not None


def test_compare_groups_uses_registered_input():
    df = make_data()
    result = make_auditor(df).compare_groups("s", "g", "a", n_bootstraps=100)
    k_ref, n_ref = hand_rate(df, "s", 0.5, df["g"] == "a")
    for level_name, level in result.features["g"].levels.items():
        k, n = hand_rate(df, "s", 0.5, df["g"] == level_name)
        metric = level.metrics["detected_event_rate"]
        assert metric.score == pytest.approx(k / n - k_ref / n_ref)
        assert metric.interval is not None


def test_chained_inputs_resolve_in_dependency_order():
    df = make_data()
    a = Auditor(data=df)
    a.add_outcome("y")
    a.add_score("s", threshold=0.5)
    # Register the dependent first: order comes from inputs, not registration.
    a.register_metric_input(TruePositiveEventFlag)
    a.register_metric_input(TruePositiveEvent())
    a.set_metrics([TruePositiveEventRate()])
    metric = (
        a.evaluate_metrics("s", n_bootstraps=10)
        .features["overall"]
        .levels["Overall"]
        .metrics["tp_event_rate"]
    )
    expected = ((df["s"] >= 0.5) & (df["y"] == 1) & (df["event"] == 1)).mean()
    assert metric.score == pytest.approx(expected)
    assert metric.interval_method == "wilson"


def test_metadata_lists_registered_inputs_only_when_present():
    df = make_data()
    result = make_auditor(df).evaluate_metrics("s", n_bootstraps=None)
    assert result.metadata["metric_inputs"] == [
        {"name": "detected_event", "class": f"{__name__}.DetectedEvent"}
    ]
    from model_auditor.metrics import Sensitivity

    plain = Auditor(data=df, metrics=[Sensitivity()])
    plain.add_outcome("y")
    plain.add_score("s", threshold=0.5)
    assert "metric_inputs" not in plain.evaluate_metrics("s").metadata


def test_registered_view_is_read_only_and_caller_data_is_unchanged():
    df = make_data()
    before = df.copy()
    a = make_auditor(df)
    a.evaluate_metrics("s", n_bootstraps=None)
    a.compare_scores("s", "r", n_bootstraps=None)
    pd.testing.assert_frame_equal(df, before)
    assert "detected_event" not in a.data
    view = a.registered_metric_inputs
    assert list(view) == ["detected_event"]
    with pytest.raises(TypeError):
        view["other"] = DetectedEvent()  # type: ignore[index]


def test_threshold_is_required_through_registered_input():
    df = make_data()
    a = Auditor(data=df)
    a.add_outcome("y")
    a.add_score("s")
    a.register_metric_input(DetectedEvent)
    a.set_metrics([DetectedEventRate()])
    with pytest.raises(ValueError, match="threshold"):
        a.evaluate_metrics("s")


def test_row_call_is_optional():
    assert is_metric_input_valid(DetectedEvent)
    assert not is_metric_input_valid(DetectedEvent())

    class Subclassed(AuditorMetricInput):
        name, label, inputs = "sub", "Sub", ["_truth"]

        def data_transform(self, data):
            data[self.name] = data["_truth"]
            return data

    assert is_metric_input_valid(Subclassed)


def test_unknown_input_name_raises():
    class Broken(DetectedEventRate):
        inputs = ["missing_input"]

    a = make_auditor(make_data(), [Broken()])
    with pytest.raises(ValueError, match="Unknown metric input 'missing_input'"):
        a.evaluate_metrics("s")


def test_unknown_dependency_of_registered_input_raises():
    class NeedsMissing(DetectedEvent):
        name = "needs_missing"
        inputs = ["nowhere"]

    class Rate(DetectedEventRate):
        inputs = ["needs_missing"]

    a = make_auditor(make_data(), [Rate()])
    a.register_metric_input(NeedsMissing)
    with pytest.raises(ValueError, match="Unknown metric input 'nowhere'"):
        a.evaluate_metrics("s")


def named(name, inputs=("_truth",)):
    class Input:
        label = "x"

        def data_transform(self, data):
            data[self.name] = 1
            return data

    Input.name = name
    Input.inputs = list(inputs)
    return Input


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("_truth", "reserved"),
        ("_binary_pred", "reserved"),
        ("overall", "reserved"),
        ("__reference_score", "reserved"),
        ("tp", "built-in"),
        ("event", "existing data column"),
        ("detected_event", "already registered"),
    ],
)
def test_name_collisions_raise_at_registration(name, message):
    a = make_auditor(make_data())
    with pytest.raises(ValueError, match=message):
        a.register_metric_input(named(name))


def test_data_column_collision_after_add_data_raises_at_evaluation():
    df = make_data()
    a = make_auditor(df)
    a.add_data(df.assign(detected_event=0))
    a.add_outcome("y")
    with pytest.raises(ValueError, match="matches a data column"):
        a.evaluate_metrics("s")


def test_cluster_cannot_be_a_registered_input():
    a = make_auditor(make_data())
    with pytest.raises(ValueError, match="reserved evaluation column"):
        a.evaluate_metrics(
            "s",
            inference=InferenceConfig(resampling="cluster", cluster="detected_event"),
        )


def test_malformed_inputs_raise():
    class NoTransform:
        name, label, inputs = "x", "X", ["_truth"]

    with pytest.raises(ValueError, match="data_transform"):
        Auditor().register_metric_input(NoTransform)
    with pytest.raises(ValueError, match="inputs"):
        Auditor().register_metric_input(named("y", inputs=[1]))
    with pytest.raises(ValueError, match="name"):
        Auditor().register_metric_input(named(""))


def test_cycle_raises_naming_the_cycle():
    a = Auditor()
    a.register_metric_input(named("first", ["second"]))
    with pytest.raises(ValueError, match="dependency cycle: .*'first'.*'second'"):
        a.register_metric_input(named("second", ["first"]))
    with pytest.raises(ValueError, match="dependency cycle"):
        a.register_metric_input(named("self_loop", ["self_loop"]))


def test_transform_returning_wrong_shape_raises():
    class Dropper(DetectedEvent):
        def data_transform(self, data):
            return data.iloc[:1].assign(detected_event=1)

    df = make_data()
    a = Auditor(data=df, metrics=[DetectedEventRate()])
    a.add_outcome("y")
    a.add_score("s", threshold=0.5)
    a.register_metric_input(Dropper)
    with pytest.raises(ValueError, match="same rows"):
        a.evaluate_metrics("s")
