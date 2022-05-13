from __future__ import annotations
import pandas as pd
import abc
from typing import TYPE_CHECKING, Type
from src.core.Setupable import Setupable

if TYPE_CHECKING:
    from src.filter.FilterRegistry import FilterRegistry
    from src.metric.SampleMetricManager import SampleMetricManager


class Filter(Setupable, metaclass=abc.ABCMeta):
    """
    Abstract static class outlining the general interface for different types
    of filters.
    """

    def __new__(cls):
        raise RuntimeError(f"{cls.__name__} should not be instantiated.")

    @classmethod
    def get_bit(cls, filter_registry: Type[FilterRegistry]):
        """
        Returns the bit associated with the filter.

        The order of the bits are analogous to the order of filters in the `filter_registry`.

        Args:
            filter_registry (Type[FilterRegistry]): The filter registry to infer ordering from.

        Returns:
            int: The bit representing this filter.
        """
        filters = filter_registry.get_resources()
        for id, filter in enumerate(filters):
            if cls == filter:
                return 1 << id

    @staticmethod
    @abc.abstractmethod
    def get_name():
        """
        Returns the name of the filter.

        Returns:
            str: name of the filter.
        """

    @staticmethod
    @abc.abstractmethod
    def apply(smm: SampleMetricManager) -> pd.index:
        """
        Should apply filter to population and return the indices of the samples
        which passed.

        Args:
            smm (SampleMetricManager): Filtering will be based on the metrics contained by this manager.
                Population will also be derived from this manager.

        Returns:
            pd.index: The indices of the samples which passed the filter.
        """
