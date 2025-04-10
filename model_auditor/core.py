from typing import Optional, Type, Union, Callable
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

from model_auditor.metric_inputs import AuditorMetricInput
from model_auditor.metrics import AuditorMetric
from model_auditor.schemas import AuditorFeature, AuditorScore, AuditorOutcome
from model_auditor.utils import collect_input_metrics


class Auditor:
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        features: Optional[list[AuditorFeature]] = None,
        scores: Optional[list[AuditorScore]] = None,
        outcome: Optional[AuditorOutcome] = None,
    ) -> None:
        # initialize data
        self.data: Optional[pd.DataFrame] = None if data is None else data.copy()

        # initialize features
        self.features: dict[str, AuditorFeature] = dict()
        if features is not None:
            for feature in features:
                self.add_feature(**vars(feature))

        # initialize scores
        self.scores: dict[str, AuditorScore] = dict()
        if scores is not None:
            for score in scores:
                self.add_score(**vars(score))

        # initialize outcome
        if outcome is not None:
            self.add_outcome(**vars(outcome))

        # initialize attrs for later
        self._inputs: list[Type[AuditorMetricInput]] = list()
        self._evaluations: list = list()

    def add_data(self, data: pd.DataFrame) -> None:
        self.data = data.copy()

    def add_feature(
        self, name: str, label: Optional[str] = None, levels: Optional[list[any]] = None
    ) -> None:
        feature = AuditorFeature(
            name=name,
            label=label,
            levels=levels,
        )
        self.features[feature.name] = feature

    def add_score(
        self, name: str, label: Optional[str] = None, threshold: Optional[float] = None
    ) -> None:
        score = AuditorScore(
            name=name,
            label=label,
            threshold=threshold,
        )
        self.scores[score.name] = score

    def add_outcome(self, name: str, mapping: Optional[dict[any, int]] = None) -> None:
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")

        if mapping is not None:
            self.data["_truth"] = self.data[name].map(mapping)
        else:
            self.data["_truth"] = self.data[name]

    def optimize_score_threshold(self, score_name: str) -> float:
        if len(self.scores) == 0:
            raise ValueError("Please define at least one score first")
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")
        elif "_truth" not in self.data.columns.tolist():
            raise ValueError(
                "Please define an outcome variable data with .add_outcome() first"
            )

        # throws an error if the score has not been defined
        score: AuditorScore = self.scores[score_name]
        score_list: list[float] = self.data[score.name].astype(float).tolist()

        # otherwise the target score will be the single item in the list
        truth_list: list[float] = self.data["_truth"].astype(float).tolist()

        # calculate optimal threshold
        fpr, tpr, thresholds = roc_curve(truth_list, score_list)
        idx: int = np.argmax(tpr - fpr).astype(int)
        optimal_threshold: float = thresholds[idx]

        print(
            f"Optimal threshold for '{score.name}' found at: {optimal_threshold}"
        )
        return optimal_threshold

    def define_metrics(self, metrics: list[AuditorMetric]) -> None:
        self.metrics: list[AuditorMetric] = metrics

    def collect_inputs(self) -> None:
        inputs_set: set[str] = set()
        for metric in self.metrics:
            inputs_set.update(metric.inputs)

        inputs_dict: dict[str, Type[AuditorMetricInput]] = collect_input_metrics()

        # reinit self._inputs and add all necessary inputs to it
        self._inputs: list[Type[AuditorMetricInput]] = list()
        for input_name in list(inputs_set):
            self._inputs.append(inputs_dict[input_name])

    def apply_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        for input_type in self._inputs:
            metric_input = input_type()
            data: pd.DataFrame = metric_input.data_transform(data)

        return data

    def evaluate_score(
        self, score_name: str, threshold: Optional[float] = None
    ) -> None:
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")

        # get score
        score: AuditorScore = self.scores[score_name]

        if (threshold is None) & (score.threshold is None):
            raise ValueError(
                "Threshold must be defined in score object or passed to .evaluate_score()"
            )
        elif threshold is None:
            threshold = score.threshold

        # copy a slice of the dataframe with the score col and the outcome column
        data_slice: pd.DataFrame = self.data.loc[:, ["_truth", score.name]]
        data_slice = self.apply_inputs(data=data_slice)

        for feature in self.features.values():
            feature_metrics: dict[str, dict[str, Union[float, int]]] = (
                self.evaluate_score_feature(data=data_slice, feature=feature)
            )

            # RESUME HERE !

    def evaluate_score_feature(
        self, data: pd.DataFrame, feature: AuditorFeature
    ) -> dict[str, dict[str, Union[float, int]]]:
        # cast feature levels to string
        data[feature.name] = data[feature.name].astype(str)

        # then group the df by the feature and get all metrics for each
        feature_groups = data.groupby(feature.name)

        # e.g. {"f1": {'levelA': 0.2, 'levelB': 0.4}, ... }
        feature_metrics: dict[str, dict[str, Union[float, int]]] = dict()
        for metric in self.metrics:
            # e.g. {'levelA': 0.2, 'levelB': 0.4}
            feature_level_metrics: dict[str, Union[float, int]] = feature_groups.apply(
                metric.data_call
            ).to_dict()

            feature_metrics[metric.name] = feature_level_metrics

        return feature_metrics
