from typing import Optional, Union, Callable
import pandas as pd

from model_auditor.schemas import AuditorScore, AuditorOutcome
from model_auditor.plotting.schemas import Hierarchy, HLevel, HItem, PlotterData


class HierarchyPlotter:
    def __init__(self) -> None:
        self.features: Optional[Hierarchy] = None  # type: ignore
        self.data: Optional[pd.DataFrame] = None
        self.aggregator: Union[str, Callable] = "median"

        self.score: Optional[AuditorScore] = None
        self.outcome: Optional[AuditorOutcome] = None

    def set_data(self, data: pd.DataFrame) -> None:
        """Set data for the plotter

        Args:
            data (pd.DataFrame): Data used to build the plot
        """
        self.data = data

    def set_features(self, features: Union[Hierarchy, list[str]]) -> None:
        """Set the feature hierarchy for the plotter

        Args:
            features (Union[Hierarchy, list[str]]): Expects a list of strings
            (column names corresponding to the data provided) or a predefined
            custom Hierarchy object

        Raises:
            ValueError: Raised if something other than a list of Hierarchy
            object was passed
        """
        # flat hierarchy
        if isinstance(features, list):
            self.features: Hierarchy = Hierarchy()
            for feature in features:
                self.features.levels.append(HLevel([HItem(name=feature)]))
        # complex/custom hierarchy
        elif isinstance(features, Hierarchy):
            self.features = features
        else:
            raise ValueError(
                "unrecognized type for features, please pass a list of strings or a predefined Hierarchy() object"
            )

    def set_aggregator(self, method: Union[str, Callable]) -> None:
        """Sets the aggregator used to color the plot cells

        Args:
            method (Union[str, Callable]): Expects a string corresponding to a
            predefined aggregator for the .agg() pandas method, or a function
            that takes the score column as a series and outputs some float
        """
        self.aggregator = method

    def set_score(
        self, name: str, label: Optional[str] = None, threshold: Optional[float] = None
    ) -> None:
        """Sets the score column used by the plotter

        Args:
            name (str): Name of the score column
            label (Optional[str], optional): Label of the score column. Defaults to None
            (plot will just use the column name).
            threshold (Optional[float], optional): Threshold to binarize the score column.
            Defaults to None (currently unused).
        """
        self.score = AuditorScore(
            name=name, label=label if label is not None else name, threshold=threshold
        )

    def compile(self, container: str) -> PlotterData:
        """Compiles the data for the plotter based on the defined HierarchyPlotter parameters

        Args:
            container (str): Name of the plot container trace

        Raises:
            ValueError: If features have not been set with .set_features() first
            ValueError: If a score has not been set with .set_score() first

        Returns:
            PlotterData: Returns the formatted plotter data TODO: wrap this internally
        """
        if self.features is None:
            raise ValueError("Please set features with .set_features() first")

        datasource = self._prepare_datasource()
        data = PlotterData()

        if self.score is not None:
            if isinstance(self.aggregator, str):
                container_agg: float = (
                    datasource[self.score.name].agg(self.aggregator).item()
                )
            else:
                container_agg: float = self.aggregator(datasource)

            data.add(
                label=container,
                id=container,
                parent="",
                value=len(datasource),
                color=container_agg,
            )

        else:
            print('No score specified')
            data.add(
                label=container,
                id=container,
                parent="",
                value=len(datasource),
            )


        return self._recursive_record(
            data=data, datasource=datasource, parent_id=container, idx=0
        )

    def _recursive_record(
        self, data: PlotterData, datasource: pd.DataFrame, parent_id: str, idx: int
    ):
        """Recursive internal function used to compile data for the plotter"""
        level: HLevel = self.features.levels[idx]
        # init a list to track valid features for this level
        level_features: list[HItem] = []
        for item in level.items:
            # if the item has no query then include it
            if item.query is None:
                level_features.append(item)

            # otherwise, include it if the feature query evaluates to true for the *entire* datasource
            elif all(datasource.eval(item.query).tolist()): # type: ignore
                level_features.append(item)

        # if this level has only 1 valid item, consider it the feature
        if len(level_features) == 1:
            feature = level_features[0]

        # otherwise, if this level has >1 valid item, concatenate them into a temp derived feature
        elif len(level_features) > 1:
            datasource.loc[:, '_temp_feature'] = (
                datasource[[i.name for i in level_features]]
                    .apply(lambda row: " & ".join(row.values.astype(str)), axis=1)
            )

            feature = HItem(name="_temp_feature")

        # if this level has 0 valid items, return
        else:
            return data
            
        count_dict: dict[str, int] = (
            datasource.groupby(feature.name, as_index=True, observed=False)
            .size()
            .to_dict()
        )

        if self.score is not None:
            # group the df by the current feature and get its frequency and agg metric
            assert isinstance(
                self.score, AuditorScore
            )  # handled by the wrapper but here for type hinting


            if isinstance(self.aggregator, str):
                # built-in aggregators
                agg_dict: dict[str, float] = (
                    datasource.groupby(feature.name, as_index=True, observed=False)[self.score.name]
                    .agg(self.aggregator)
                    .to_dict()
                )

            else:
                # custom aggregators (pass entire df here instead of just the score series)
                agg_dict: dict = (
                    datasource.groupby(feature.name, as_index=True, observed=False)
                    .apply(self.aggregator)
                    .to_dict()
                )

        # extract the count dict keys to get the levels for the current feature
        feature_levels: list[str] = list(count_dict.keys())

        # format the parent_id + feature levels into trace identifiers
        id_dict: dict[str, str] = {
            feature_level: f"{parent_id}${feature_level}"
            for feature_level in count_dict.keys()
        }

        for feature_level in feature_levels:
            # add the current feature data
            if self.score is not None:
                data.add(
                    label=feature_level,
                    id=id_dict[feature_level],
                    parent=parent_id,
                    value=count_dict[feature_level],
                    color=agg_dict[feature_level],
                )
                
            else:
                data.add(
                    label=feature_level,
                    id=id_dict[feature_level],
                    parent=parent_id,
                    value=count_dict[feature_level],
                )

            # if this isn't the last feature in the stack, get the subset of data for this feature and recurse
            if idx < (len(self.features.levels) - 1):
                data = self._recursive_record(
                    data=data,
                    datasource=datasource.loc[datasource[feature.name] == feature_level, :].copy(),
                    parent_id=id_dict[feature_level],
                    idx=idx + 1,
                )

        return data

    def _prepare_datasource(self) -> pd.DataFrame:
        """Internal function used to prepare the datasource for plotting

        Raises:
            ValueError: If data has not been added with .set_data() first

        Returns:
            pd.DataFrame: Prepared data source
        """
        if self.data is None:
            raise ValueError("Please set data with .set_data() first")

        data = self.data.copy()
        if self.score is not None:
            data["_pred"] = self.score.name
        else:
            print("no score set")

        if self.outcome is not None:
            data["_outcome"] = self.outcome.name
        else:
            print("no outcome set")

        return data
