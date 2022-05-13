from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Type
from src.filter.Filter import Filter
from src.core.Setupable import SetupMode
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from src.metric.SampleMetric import SampleMetric
    from src.metric.SampleMetricManager import SampleMetricManager


class SampleMetricFilter(Filter, metaclass=abc.ABCMeta):
    """
    Abstract static class outlining the general interface for different types
    of sample metric filters.
    """

    def __new__(cls):
        raise RuntimeError(f"{cls.__name__} should not be instantiated.")

    @classmethod
    def sample_metric_reg_setup_modes(cls) -> dict[str, SetupMode]:
        """
        Returns setup modes for all relevent sample metrics.

        # ! Note that if setup modes for sample metrics have the same name
        # then only one of them (the last added) will be kept.

        Returns:
            dict[str, SetupMode]: The setup modes.
        """
        setup_modes = {}
        # Merge all setup modes
        for sm in cls.get_used_sample_metrics():
            setup_modes |= sm.reg_setup_modes()
        return setup_modes

    @staticmethod
    def display_bad_images(passed_indices: pd.index, smm: SampleMetricManager):
        """
        Displays the bad images removed by a filter.

        Note that this function blocks calling process until all
        image windows have been shut down.

        Args:
            passed_indices (pd.index): Indices of which samples passed.
            smm (SampleMetricManager): Manager which population will be inferred from.
        """
        pop_data = smm.get_population().get_data()
        for i in pop_data.index.difference(passed_indices):
            m = Image.open(
                smm.get_population()
                .get_data(i)[smm.get_population().COLUMN_URI]
                .values[0]
            )
            plt.imshow(np.asarray(m))
            plt.show(block=True)

    @staticmethod
    @abc.abstractmethod
    def get_used_sample_metrics() -> list[Type[SampleMetric]]:
        """
        Returns a list of the used sample metrics.

        Returns:
            list[SampleMetric]: The sample metrics.
        """
