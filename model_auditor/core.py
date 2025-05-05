from typing import Optional, Type, Union, Callable
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_curve
from tqdm.auto import tqdm

from model_auditor.metric_inputs import AuditorMetricInput
from model_auditor.metrics import AuditorMetric
from model_auditor.schemas import (
    AuditorFeature,
    AuditorScore,
    AuditorOutcome,
    FeatureEvaluation,
    ScoreEvaluation,
)
from model_auditor.utils import collect_metric_inputs


class Auditor:
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        features: Optional[list[AuditorFeature]] = None,
        scores: Optional[list[AuditorScore]] = None,
        outcome: Optional[AuditorOutcome] = None,
        metrics: Optional[list[AuditorMetric]] = None,
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

        # initialize metrics
        self.metrics: list[AuditorScore] = list()
        if metrics is not None:
            self.metrics = metrics

        # initialize attrs for later
        self._inputs: list[Type[AuditorMetricInput]] = list()
        self._evaluations: list = list()

    def add_data(self, data: pd.DataFrame) -> None:
        """
        Method to add a dataframe to the auditor

        Args:
            data (pd.DataFrame): Full dataframe which will be subset for subgroup evaluation
        """
        self.data = data.copy()

    def add_feature(
        self, name: str, label: Optional[str] = None, levels: Optional[list[any]] = None
    ) -> None:
        """
        Method to add a feature to the auditor. Equivalent to a grouping variable in
        packages like tableone, the score variable will be stratified by this feature

        Args:
            name (str): Column name for the feature.
            label (Optional[str], optional): Optional label for the feature. Defaults to None.
            levels (Optional[list[any]], optional): Valid levels to consider for the feature,
            by default (when set to None) all levels will be considered. Defaults to None.
            [Not currently implemented]
        """
        feature = AuditorFeature(
            name=name,
            label=label,
            levels=levels,
        )
        self.features[feature.name] = feature

    def add_score(
        self, name: str, label: Optional[str] = None, threshold: Optional[float] = None
    ) -> None:
        """
        Method to add a score to the auditor. Expects a continuous feature which will
        be used to calculate metrics and confidence intervals

        Args:
            name (str): Column name for the score.
            label (Optional[str], optional): Optional label for the score. Defaults to None.
            threshold (Optional[float], optional): Threshold used to binarize the score column.
            Defaults to None and can be optimized using the Youden index or updated separately later.
        """
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
        """
        Method to optimize the decision threshold for a score based on the Youden index.

        Args:
            score_name (str): Name of the target score

        Raises:
            ValueError: If no scores have been defined with .add_score() first
            ValueError: If no data has been added with .add_data() first
            ValueError: If no outcome variable has been defined with .add_outcome() first

        Returns:
            float: Optimal threshold identified
        """
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

        print(f"Optimal threshold for '{score.name}' found at: {optimal_threshold}")
        return optimal_threshold

    def set_metrics(self, metrics: list[AuditorMetric]) -> None:
        """
        Method to define the metrics the auditor will use during evaluation of score variables.

        Args:
            metrics (list[AuditorMetric]): A list of metrics classes following the AuditorMetric
            protocol (pre-made metrics listed in model_auditor.metrics)
        """
        self.metrics: list[AuditorMetric] = metrics

    def _collect_inputs(self) -> None:
        """
        Collects the minimum set of metric inputs necessary for evaluation
        (based on the metrics defined in self.metrics with the .define_metrics() method)
        """
        inputs_set: set[str] = set()
        for metric in self.metrics:
            inputs_set.update(metric.inputs)

        inputs_dict: dict[str, Type[AuditorMetricInput]] = collect_metric_inputs()

        # reinit self._inputs and add all necessary inputs to it
        self._inputs: list[Type[AuditorMetricInput]] = list()
        for input_name in list(inputs_set):
            if input_name not in ["_truth", "_pred"]:
                self._inputs.append(inputs_dict[input_name])

    def _apply_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method to apply the metric input functions (collected with .collect_inputs())
        to the target data to prepare it for metric calculation

        Args:
            data (pd.DataFrame): Dataframe to add input columns to

        Returns:
            pd.DataFrame: Transformed dataframe with metric input columns
        """
        for input_type in self._inputs:
            metric_input = input_type()
            data: pd.DataFrame = metric_input.data_transform(data)

        return data

    def _binarize(self, score_data: pd.Series, threshold: float) -> pd.Series:
        return (score_data >= threshold).astype(int)

    def evaluate(self, score_name: str, threshold: Optional[float] = None, n_bootstraps: Optional[int] = 1000):
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")

        if len(self.metrics) == 0:
            raise ValueError(
                "Please define at least one metric with .set_metrics() first"
            )

        # get score
        score: AuditorScore = self.scores[score_name]

        if (threshold is None) & (score.threshold is None):
            raise ValueError(
                "Threshold must be defined in score object or passed to .evaluate_score()"
            )
        elif threshold is None:
            threshold = score.threshold

        # collect metric inputs to prep for evaluation
        self._collect_inputs()

        # get the list of columns to retain in the data
        column_list: list[str] = [*self.features.keys(), "_truth"]

        # copy a slice of the dataframe
        data_slice: pd.DataFrame = self.data.loc[:, column_list]
        data_slice["_pred"] = self.data[score.name]
        data_slice["_binary_pred"] = self._binarize(score_data=data_slice["_pred"], threshold=threshold)  # type: ignore
        data_slice = self._apply_inputs(data=data_slice)

        # create an 'Overall' feature which will be used to calculate metrics on the full data
        data_slice["overall"] = "Overall"
        eval_features: dict[str, AuditorFeature] = {
            "overall": AuditorFeature(
                name="overall",
                label="Overall",
            )
        }
        eval_features.update(**self.features)

        score_eval: ScoreEvaluation = ScoreEvaluation(
            name=score.name,
            label=score.label if score.label is not None else score.name,
        )
        with tqdm(
            eval_features.values(), position=0, leave=True, desc="Features"
        ) as pbar:
            for feature in pbar:
                pbar.set_postfix({"name": feature.name})

                # e.g. {"f1": {'levelA': 0.2, 'levelB': 0.4}, ... }
                feature_eval: FeatureEvaluation = self._evaluate_feature(
                    data=data_slice, feature=feature, n_bootstraps=n_bootstraps
                )
                score_eval.features[feature.name] = feature_eval

        return score_eval

    def _evaluate_feature(
        self, data: pd.DataFrame, feature: AuditorFeature, n_bootstraps: Optional[int]
    ) -> FeatureEvaluation:
        with tqdm(range(2), position=1, desc="Stages", leave=False) as feature_pbar:
            feature_pbar.set_postfix({"stage": "Evaluating metrics"})

            # cast feature levels to string
            data[feature.name] = data[feature.name].astype(str)

            # then group the df by this feature (so each group contains one
            # unique level of the data) and get all metrics for each
            feature_groups = data.groupby(feature.name)

            # e.g. {"f1": {'levelA': 0.2, 'levelB': 0.4}, ... }
            feature_eval: FeatureEvaluation = FeatureEvaluation(
                name=feature.name,
                label=feature.label if feature.label is not None else feature.name,
            )
            for metric in tqdm(self.metrics, position=2, desc="Metrics", leave=False):
                # gets a dict with the current metric calculated for levels of the feature
                # e.g. {levelA: 0.5, levelB: 0.5}
                level_eval_dict = feature_groups.apply(metric.data_call).to_dict()

                feature_eval.update(
                    metric_name=metric.name,
                    metric_label=metric.label,
                    data=level_eval_dict,
                )

            feature_pbar.update(1)
            feature_pbar.set_postfix({"stage": "Evaluating intervals"})
            # if calculating confidence intervals, do that here
            if n_bootstraps is not None:
                for level_name, level_data in tqdm(
                    feature_groups, position=2, desc="Bootstrap Levels", leave=False
                ):
                    # calculate confidence intervals for eligible metrics for the current feature level
                    level_metric_intervals: dict[str, tuple[float, float]] = (
                        self._evaluate_confidence_interval(data=level_data, n_bootstraps=n_bootstraps)
                    )
                    # register the calculated intervals
                    feature_eval.update_intervals(
                        level_name=str(level_name),
                        metric_intervals=level_metric_intervals,
                    )
            feature_pbar.update(1)

        return feature_eval

    def _evaluate_confidence_interval(
        self, data: pd.DataFrame, n_bootstraps: int
    ) -> dict[str, tuple[float, float]]:
        n: int = len(data)

        bootstrap_results: dict[str, NDArray[np.float64]] = dict()
        for metric in self.metrics:
            if metric.ci_eligible:
                bootstrap_results[metric.name] = np.empty(shape=(n_bootstraps), dtype=np.float64)

        # sample n_bootstrap times with replacement
        for i in range(n_bootstraps):
            boot_data: pd.DataFrame = data.sample(n, replace=True)

            # calculate metrics on current bootstrap data
            for metric in self.metrics:
                if metric.ci_eligible:
                    bootstrap_results[metric.name][i] = metric.data_call(boot_data)

        metric_intervals: dict[str, tuple[float, float]] = dict()
        for metric_name, bootstrap_array in bootstrap_results.items():
            # get 95% confidence bounds for metric
            lower, upper = np.percentile(bootstrap_array, [2.5, 97.5])
            metric_intervals[metric_name] = (lower, upper)

        return metric_intervals
