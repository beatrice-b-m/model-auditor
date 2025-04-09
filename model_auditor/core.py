from typing import Optional, Union, Callable
from model_auditor.schemas import AuditorFeature, AuditorScore, AuditorOutcome
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


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
        if self.data == None:
            raise ValueError("Please add data with .add_data() first")

        if mapping is not None:
            self.data["_outcome"] = self.data[name].map(mapping)
        else:
            self.data["_outcome"] = self.data[name]

    def optimize_score_threshold(self, score_name: str) -> float:
        if len(self.scores) == 0:
            raise ValueError("Please define at least one score first")
        if self.data is None:
            raise ValueError("Please add data with .add_data() first")
        elif "_outcome" not in self.data.columns.tolist():
            raise ValueError(
                "Please define an outcome variable data with .add_outcome() first"
            )

        # throws an error if the score has not been defined
        score: AuditorScore = self.scores[score_name]
        score_list: list[float] = self.data[score.name].astype(float).tolist()

        # otherwise the target score will be the single item in the list
        truth_list: list[float] = self.data["_outcome"].astype(float).tolist()

        fpr, tpr, thresholds = roc_curve(truth_list, score_list)
        idx: int = np.argmax(tpr - fpr).astype(int)
        optimal_threshold: float = thresholds[idx]

        print(
            f"Optimal threshold for '{score.label}' [{score.name}] found at: {optimal_threshold}"
        )
        return optimal_threshold
