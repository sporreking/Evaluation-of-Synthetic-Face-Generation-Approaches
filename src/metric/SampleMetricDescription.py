import abc
import pandas as pd
import numpy as np

from typing import Any

from src.dataset.Dataset import Dataset


class SampleMetricDescription(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def get_name() -> str:
        """
        Should return the name of this sample metric description.

        Returns:
            str: The name of this sample metric description.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def setup(dataset: Dataset, mode: str) -> None:
        """
        Should train/prepare all auxiliary models required for performing
        a calculation with `calc()`.

        Args:
            dataset (Dataset): The dataset that should be used to train
                and prepare the metric.
            mode (str): The mode of the setup. The available modes
                depend on the the metric.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def is_ready(dataset: Dataset) -> bool:
        """
        Should return `True` if the `setup()` has been completed, i.e.,
        if all auxiliary models are at disposal when calling `calc()`.

        Args:
            dataset (Dataset): The dataset that is supposed to have been
                used for the setup. Metrics may require setups on a
                per-dataset basis.

        Returns:
            bool: `True` if the metric is set up.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def calc(data: pd.DataFrame, dataset: Dataset, **parameters: Any) -> np.ndarray:
        """
        Should calculate a per-sample metric for a set of samples.

        Args:
            data (pd.DataFrame): The samples to calculate for, specified
                on a per-row basis. The format should match that of a
                Population's data frame.
            dataset (Dataset): The dataset that was used to train the
                models that were used to generate the samples in `data`.
            **parameters (Any): Arbitrary metric parameters.

        Returns:
            np.ndarray: One floating point score for each sample, i.e.,
                the dimensions of the vector is equal to the number of
                input samples.
        """
        pass
