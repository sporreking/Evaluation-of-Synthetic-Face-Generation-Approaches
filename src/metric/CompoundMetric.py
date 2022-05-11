from __future__ import annotations
from src.dataset.Dataset import Dataset
from src.metric.SampleMetricManager import SampleMetricManager
from src.population.Population import Population
from src.core.Setupable import Setupable

import abc
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.metric.CompoundMetricManager import CompoundMetricManager


class CompoundMetric(Setupable, metaclass=abc.ABCMeta):
    def __init__(
        self,
        name: str,
        compound_metric_manager: CompoundMetricManager,
        smm: SampleMetricManager = None,
    ):
        """
        Constructor for the abstract CompoundMetric class.

        Args:
            name (str): Name of the metric.
            cmm (CompoundMetricManager): Manager used by metrics. Population and dataset is derived
                from this manager.
            smm (SampleMetricManager, optional): Manager used by per-sample metrics. Defaults to None.
        """
        self._dataset = compound_metric_manager.get_dataset()
        self._population = compound_metric_manager.get_population()
        self._name = name
        self._smm = smm
        self._cmm = compound_metric_manager
        pass

    def get_dataset(self) -> Dataset:
        """
        Returns the dedicated dataset of this compound metric. This Dataset
        was assigned upon the metrics's instantiation, inherited by the compound
        metric manager.

        Returns:
            Dataset: The dataset of this manager.
        """
        return self._dataset

    def get_population(self) -> Population:
        """
        Returns the dedicated population of this compound metric. This Population
        was assigned upon the metrics's instantiation, inherited by the compound
        metric manager.

        Returns:
            Population: The population of this manager.
        """
        return self._population

    def get_name(self) -> str:
        """
        Should return the name of this compound metric.

        Returns:
            str: The name of this compound metric.
        """
        return self._name

    @abc.abstractmethod
    def calc(self, **parameters: Any) -> Any:
        """
        Should calculate a compound metric for a set of samples.

        Args:
            filter_bit (int, optional): Filter bit used to select a subset of the
                population. Filter bit is defined by the order in FilterRegistry. For example,
                the first filter corresponds to filter bit 1. Defaults to 1 (IdentityFilter).
            **parameters (Any): Arbitrary metric parameters.

        Returns:
            Any: The value of the metric.
        """
        pass

    @abc.abstractmethod
    def get(self, calc_if_missing: bool = False, **parameters: Any) -> Any:
        """
        Get the metric.

        Note that if `calc_if_missing` is True, then a `filter_bit` probably should be
        specified, see documentation for `calc()`.

        Args:
            calc_if_missing (bool, optional): If True should calculate the metric if missing
                by calling `calc()`. Defaults to False.
            **parameters (Any): Arbitrary parameters required by the metrics.
                These are only used if `calc_if_missing=True`, and are all forwarded
                to `calc()`.

        Returns:
            Any: The value of the metric. Should return None if the metric is missing
                and `calc_if_missing=False`.
        """
        pass

    @abc.abstractmethod
    def print_result(self) -> None:
        """
        Should print the result of the metric. Should only be called
        after metric has been calculated through `calc()`.
        """
        pass

    @abc.abstractmethod
    def plot_result(self) -> None:
        """
        Should plot the result of the metric. Should only be called
        after metric has been calculated through `calc()`.
        """
        pass
