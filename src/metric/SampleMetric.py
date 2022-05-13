from __future__ import annotations
import abc
import pandas as pd
import numpy as np

from typing import Any, TYPE_CHECKING

from src.core.Setupable import Setupable
from src.dataset.Dataset import Dataset

if TYPE_CHECKING:
    from src.population.Population import Population
    from src.metric.SampleMetricManager import SampleMetricManager


class SampleMetric(Setupable, metaclass=abc.ABCMeta):
    def __init__(self, smm: SampleMetricManager):
        """
        Constructs a new sample metric.

        The metric will compute metrics for the population of the given manager,
        and will also be able to utilize its dataset.

        Args:
            smm (SampleMetricManager): The manager to which this sample metric belongs.
                The dataset and population of this sample metric will be derived
                from the manager.
        """
        self._smm = smm
        self._dataset = smm.get_dataset()
        self._population = smm.get_population()

    @staticmethod
    @abc.abstractmethod
    def get_name() -> str:
        """
        Should return the name of this sample metric description.

        Returns:
            str: The name of this sample metric description.
        """
        pass

    def get_dataset(self) -> Dataset:
        """
        Returns the dataset associated with this sample metric.

        Returns:
            Dataset: The dataset of this sample metric.
        """
        return self._dataset

    def get_population(self) -> Population:
        """
        Returns the population on which this sample metrics operates.

        Returns:
            Population: The population of this sample metric.
        """
        return self._population

    @abc.abstractmethod
    def calc(self, ids: pd.Index, **parameters: Any) -> np.ndarray:
        """
        Should calculate a per-sample metric for the samples in the population
        to which the specified `ids` map.

        Args:
            ids (pd.Index): The indices of the samples to calculate for.
            **parameters (Any): Arbitrary metric parameters.

        Returns:
            np.ndarray: One calculated value for each sample, i.e., the vector
                should have a single axis with length equal to the number of
                input samples.
        """
        pass
