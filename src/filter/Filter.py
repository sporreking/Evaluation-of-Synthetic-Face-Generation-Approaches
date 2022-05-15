from __future__ import annotations
import pandas as pd
import abc
from typing import TYPE_CHECKING, Type, Any
from src.core.Setupable import Setupable

if TYPE_CHECKING:
    from src.filter.FilterRegistry import FilterRegistry
    from src.metric.SampleMetricManager import SampleMetricManager


class Filter(Setupable, metaclass=abc.ABCMeta):
    """
    Abstract static class outlining the general interface for different types
    of filters.
    """

    def __init__(self, _setup_only: bool = False, smm: SampleMetricManager = None):
        """
        Filters are supposed to be used statically, but instances are required
        for performing setups. To ensure that the instance is only used for
        this purpose, the flag `_setup_only` must be set to `True` for the
        instantiation to work without causing an exception.

        Args:
            _setup_only (bool, mandatory): Must be True. Defaults to False.
            smm (SampleMetricManager, mandatory): A sample metric manager to
                use for the setup.

        Raises:
            RuntimeError: If the `_setup_only` flag was not set to `True`.
            ValueError: If `smm=None` (the default value).
        """
        if not _setup_only:
            raise RuntimeError(
                "Filters may only be instantiated for the purpose of performing setups!"
            )

        if smm is None:
            raise ValueError("No sample metric manager was provided!")

        self._smm = smm

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
            str: The name of the filter.
        """

    @staticmethod
    @abc.abstractmethod
    def apply(smm: SampleMetricManager, **parameters: Any) -> pd.index:
        """
        Should apply filter to population and return the indices of the samples
        which passed.

        Args:
            smm (SampleMetricManager): Filtering will be based on the metrics contained by this manager.
                Population will also be derived from this manager.
            **parameters (Any): Arbitrary parameters.
        Returns:
            pd.index: The indices of the samples which passed the filter.
        """
