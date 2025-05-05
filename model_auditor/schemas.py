from typing import Optional, Union
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class LevelMetric:
    """
    Object to store the evaluation results for one metric of one level of a feature.
    (for example, AUC for one category of finding)

    Args:
        name (str): Name of the current feature level metric
        score (Union[float, int]): Score for the current feature level metric
        interval (tuple[float, float], optional): Optional lower and upper confidence
        bounds for the current feature level metric (defaults to None)
    """

    name: str
    label: str
    score: Union[float, int]
    interval: Optional[tuple[float, float]] = None


@dataclass
class LevelEvaluation:
    """
    Object to store the evaluation results for one level of a feature
    (for example, all metrics for one category of finding).

    Args:
        name (str): Name of the current feature level
        metrics (dict[str, LevelMetric]): Metrics for the current feature level
        (defaults to an empty dict)
    """

    name: str
    metrics: dict[str, LevelMetric] = field(default_factory=dict)

    def update(self, metric_name: str, metric_label: str, metric_score: float) -> None:
        self.metrics[metric_name] = LevelMetric(
            name=metric_name, label=metric_label, score=metric_score
        )

    def update_intervals(self, metric_intervals: dict[str, tuple[float, float]]):
        for metric_name, confidence_interval in metric_intervals.items():
            self.metrics[metric_name].interval = confidence_interval

    def to_dataframe(self, n_decimals: int = 3, add_index: bool = False, metric_labels: bool = False):
        metric_data: dict[str, str] = dict()
        for metric in self.metrics.values():
            # get the key name for the current metric (label if metric_labels is True)
            metric_key: str = metric.label if metric_labels else metric.name
            
            if metric.interval is not None:
                metric_data[metric_key] = (
                    f"{metric.score:.{n_decimals}f} ({metric.interval[0]:.{n_decimals}f}, {metric.interval[1]:.{n_decimals}f})"
                )
            elif isinstance(metric.score, float):
                metric_data[metric_key] = f"{metric.score:.{n_decimals}f}"
            else:
                # integer scores (default to comma delimited for now)
                metric_data[metric_key] = f"{metric.score:,}"

        return pd.DataFrame(metric_data, index=[self.name])


@dataclass
class FeatureEvaluation:
    """
    Object to store the evaluation results for one feature type
    (for example, metrics associated with different types of findings)

    Args:
        name (str): Name of the current feature
        name (str): Label for the current feature
        levels (dict[str, LevelEvaluation]): Levels of the current feature
        (defaults to an empty dict)
    """

    name: str
    label: str
    levels: dict[str, LevelEvaluation] = field(default_factory=dict)

    def update(
        self, metric_name: str, metric_label: str, data: dict[str, float]
    ) -> None:
        # expects a dict for one metric type: {'levelA': 0.5, 'levelB': 0.5}
        # and maps them to child level metric dicts
        for level_name, level_metric in data.items():
            # try to get the level item and instantiate a new one if it doesn't exist yet
            level_eval: LevelEvaluation = self.levels.get(
                level_name, LevelEvaluation(name=level_name)
            )
            # update the metrics for that level eval object and save it back to the dict
            level_eval.update(
                metric_name=metric_name,
                metric_label=metric_label,
                metric_score=level_metric,
            )
            self.levels[level_name] = level_eval

    def update_intervals(
        self, level_name: str, metric_intervals: dict[str, tuple[float, float]]
    ):
        self.levels[level_name].update_intervals(metric_intervals=metric_intervals)

    def to_dataframe(
        self, n_decimals: int = 3, add_index: bool = False, metric_labels: bool = False
    ) -> pd.DataFrame:
        data: list[pd.DataFrame] = []
        for level_data in self.levels.values():
            data.append(level_data.to_dataframe(n_decimals=n_decimals, metric_labels=metric_labels))

        if add_index:
            return pd.concat({self.label: pd.concat(data, axis=0)})
        else:
            return pd.concat(data, axis=0)


@dataclass
class ScoreEvaluation:
    name: str
    label: str
    features: dict[str, FeatureEvaluation] = field(default_factory=dict)

    def to_dataframe(
        self, n_decimals: int = 3, add_index: bool = False, metric_labels: bool = False
    ) -> pd.DataFrame:
        data: list[pd.DataFrame] = []
        for feature_data in self.features.values():
            data.append(
                feature_data.to_dataframe(n_decimals=n_decimals, add_index=True, metric_labels=metric_labels)
            )

        if add_index:
            return pd.concat({self.label: pd.concat(data, axis=0)})
        else:
            return pd.concat(data, axis=0)


@dataclass
class AuditorFeature:
    name: str
    label: Optional[str] = None
    levels: Optional[list[any]] = None


@dataclass
class AuditorScore:
    name: str
    label: Optional[str] = None
    threshold: Optional[float] = None


@dataclass
class AuditorOutcome:
    name: str
    mapping: Optional[dict[any, int]] = None
