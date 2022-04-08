import abc
from src.dataset.Dataset import Dataset
import pandas as pd
import numpy as np
from typing import Any
from src.metric.SampleMetricManager import SampleMetricManager
from src.population.Population import Population


class CompoundMetric(metaclass=abc.ABCMeta):
    def __init__(
        self,
        name: str,
        dataset: Dataset,
        population: Population,
        smm: SampleMetricManager = None,
    ):
        """
        Constructor for the abstract CompoundMetric class.

        Args:
            name (str): Name of the metric.
            dataset (Dataset): Dataset used to compute the metric.
            population (Population): Population used to compute the metric.
            smm (SampleMetricManager, optional): Manager used by per-sample metrics. Defaults to None.
        """
        self._ds = dataset
        self._pop = population
        self._name = name
        self._smm = smm
        pass

    def get_dataset(self) -> Dataset:
        """
        Should return the Dataset of this compound metric.

        Returns:
            str: The name of this compound metric.
        """
        return self._ds

    def get_population(self) -> Population:
        """
        Should return the Population of this compound metric.

        Returns:
            str: The name of this compound metric.
        """
        return self._pop

    def get_name(self) -> str:
        """
        Should return the name of this compound metric.

        Returns:
            str: The name of this compound metric.
        """
        return self._name

    @abc.abstractmethod
    def setup(self, mode: str) -> None:
        """
        Should train/prepare all auxiliary models required for performing
        a calculation with `calc()`.

            mode (str): The mode of the setup. The available modes
                depend on the the metric.
        """
        pass

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """
        Should return `True` if the `setup()` has been completed, i.e.,
        if all auxiliary models are at disposal when calling `calc()`.

        Returns:
            bool: `True` if the metric is set up.
        """
        pass

    @abc.abstractmethod
    def calc(self, **parameters: Any) -> Any:
        """
        Should calculate a compound metric for a set of samples.

        Args:
            **parameters (Any): Arbitrary metric parameters.

        Returns:
            Any: The value of the metric.
        """
        pass

    @abc.abstractmethod
    def get(self, calc_if_missing: bool = False, **parameters: Any) -> Any:
        """
        Get the metric.

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
