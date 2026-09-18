"""Hierarchical plotter for visualizing stratified data.

This module provides the HierarchyPlotter class for creating hierarchical
visualizations (suitable for sunburst or treemap charts) from stratified
data.
"""

import warnings
from typing import Callable, Optional, Union

import pandas as pd

from model_auditor.plotting.schemas import Hierarchy, HItem, HLevel, PlotterData
from model_auditor.schemas import AuditorScore, ThresholdSpec


class HierarchyPlotter:
    """Compile parallel node arrays for sunburst or treemap visualizations.

    Attributes:
        features: Hierarchy configuration or None before set_features().
        data: Source DataFrame or None. set_data() retains a reference; compile()
            copies it, so caller edits before compilation are visible.
        aggregator: pandas score-Series aggregation name or callable receiving a
            whole group DataFrame and returning a scalar; default "median".
        score: Optional AuditorScore. Without it, colors are None and compile warns.

    Notes:
        This compiler creates no figure and does not require Plotly or Matplotlib.
        Supplied callbacks should not modify their data. Hierarchy configurations
        are shared by reference; node arrays are new on every compile().

    Example:
        >>> import pandas as pd
        >>> from model_auditor.plotting import HierarchyPlotter
        >>> plotter = HierarchyPlotter()
        >>> plotter.set_data(pd.DataFrame({"region": ["N", "S"], "risk": [0.2, 0.8]}))
        >>> plotter.set_features(["region"])
        >>> plotter.set_score("risk")
        >>> plotter.compile("All").values
        [2, 1, 1]"""

    def __init__(self) -> None:
        """Initialize the HierarchyPlotter with default settings."""
        self.features: Optional[Hierarchy] = None
        self.data: Optional[pd.DataFrame] = None
        self.aggregator: Union[str, Callable[[pd.DataFrame], float]] = "median"

        self.score: Optional[AuditorScore] = None

    def set_data(self, data: pd.DataFrame) -> None:
        """Retain the source DataFrame by reference for later compilation.

        Args:
            data: Source rows. Edits made before compile() are visible to the plotter.
                Compilation uses a pandas copy and does not add columns to this frame."""
        self.data = data

    def set_features(self, features: Union[Hierarchy, list[str]]) -> None:
        """Replace the ordered hierarchy used for compilation.

        Args:
            features: List of column names, converted to one item per level, or a
                Hierarchy object retained by reference. An empty hierarchy yields
                only the container node.

        Raises:
            ValueError: features is neither a list nor a Hierarchy."""
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

    def set_aggregator(
        self, method: Union[str, Callable[[pd.DataFrame], float]]
    ) -> None:
        """Set the scalar statistic used for node colors; initial default is "median".

        Args:
            method: pandas Series aggregation name such as "mean", "median" or "size",
                or a callable receiving the whole group DataFrame (including all source
                columns) and returning a numeric scalar. Called for root and child nodes.
                Missing-score behavior follows pandas or the callable. Validation and
                callback exceptions occur at compile time; no copy of method is made."""
        self.aggregator = method

    def set_score(
        self,
        name: str,
        label: Optional[str] = None,
        threshold: Optional[ThresholdSpec] = None,
    ) -> None:
        """Sets the score column used by the plotter

        Args:
            name (str): Name of the score column
            label (Optional[str], optional): Label of the score column. Defaults to None
            (plot will just use the column name).
            threshold (Optional[ThresholdSpec], optional): Scalar or conditional
            threshold retained as metadata only; it does not binarize scores.
            Defaults to None.
        """
        self.score = AuditorScore(
            name=name, label=label if label is not None else name, threshold=threshold
        )

    def compile(self, container: str) -> PlotterData:
        """Compile root-first node arrays without creating or displaying a figure.

        Args:
            container: Root label and ID; descendant IDs join it to level values with
                "$". Choose labels without ambiguous "$" paths to avoid ID collisions.

        Returns:
            New PlotterData with parallel arrays in depth-first order. Values are row
            counts; colors use the configured aggregation, or None without a score.
            pandas group ordering applies, including unobserved categorical nodes.
            Null single-feature levels are omitted; composite items stringify values
            joined by " & ", including missing components. Root counts include all rows.

        Raises:
            ValueError: Data or features have not been configured, or pandas rejects
                a query/aggregation. Callback exceptions propagate.
            KeyError: A referenced data column is absent.

        Notes:
            Emits UserWarning if no score is set. An empty hierarchy returns the root.
            Copies the source DataFrame; no source columns are changed. HItem queries
            choose whether an item applies to an entire branch, rather than filtering
            rows. This is a visualization hierarchy, not Auditor subgroup evaluation."""
        if self.features is None:
            raise ValueError("Please set features with .set_features() first")

        datasource = self._prepare_datasource()
        data = PlotterData()

        if self.score is not None:
            if isinstance(self.aggregator, str):
                container_agg: float = float(
                    datasource[self.score.name].agg(self.aggregator)
                )
            else:
                container_agg: float = self.aggregator(datasource)

            data.add(
                label=container,
                node_id=container,
                parent="",
                value=len(datasource),
                color=container_agg,
            )

        else:
            warnings.warn("No score specified. Color values will not be computed.")
            data.add(
                label=container,
                node_id=container,
                parent="",
                value=len(datasource),
            )

        return self._recursive_record(
            data=data, datasource=datasource, parent_id=container, idx=0
        )

    def _recursive_record(
        self, data: PlotterData, datasource: pd.DataFrame, parent_id: str, idx: int
    ) -> PlotterData:
        """Recursively build plotter data by traversing the hierarchy levels.

        Args:
            data: PlotterData object to append nodes to.
            datasource: DataFrame subset for the current hierarchy branch.
            parent_id: ID of the parent node in the hierarchy.
            idx: Current index in the hierarchy levels.

        Returns:
            Updated PlotterData with nodes for current and child levels.
        """
        assert self.features is not None  # validated by compile()
        if idx >= len(self.features.levels):
            return data
        level: HLevel = self.features.levels[idx]
        # init a list to track valid features for this level
        level_features: list[HItem] = []
        for item in level.items:
            # if the item has no query then include it
            if item.query is None:
                level_features.append(item)

            # otherwise, include it if the feature query evaluates to true for the *entire* datasource
            elif all(datasource.eval(item.query).tolist()):  # type: ignore
                level_features.append(item)

        # if this level has only 1 valid item, consider it the feature
        if len(level_features) == 1:
            feature = level_features[0]

        # otherwise, if this level has >1 valid item, concatenate them into a temp derived feature
        elif len(level_features) > 1:
            datasource.loc[:, "_temp_feature"] = datasource[
                [i.name for i in level_features]
            ].apply(lambda row: " & ".join(row.values.astype(str)), axis=1)

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
            if not isinstance(self.score, AuditorScore):
                raise TypeError(f"Expected AuditorScore, got {type(self.score)}")

            if isinstance(self.aggregator, str):
                # built-in aggregators
                agg_dict: dict[str, float] = (
                    datasource.groupby(feature.name, as_index=True, observed=False)[
                        self.score.name
                    ]
                    .agg(self.aggregator)
                    .to_dict()
                )

            else:
                # custom aggregators (pass entire df here instead of just the score series)
                agg_dict = {
                    key: self.aggregator(group)
                    for key, group in datasource.groupby(feature.name, observed=False)
                }

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
                    node_id=id_dict[feature_level],
                    parent=parent_id,
                    value=count_dict[feature_level],
                    color=agg_dict[feature_level],
                )

            else:
                data.add(
                    label=feature_level,
                    node_id=id_dict[feature_level],
                    parent=parent_id,
                    value=count_dict[feature_level],
                )

            # if this isn't the last feature in the stack, get the subset of data for this feature and recurse
            if idx < (len(self.features.levels) - 1):
                data = self._recursive_record(
                    data=data,
                    datasource=datasource.loc[
                        datasource[feature.name] == feature_level, :
                    ].copy(),
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
        return self.data.copy()
