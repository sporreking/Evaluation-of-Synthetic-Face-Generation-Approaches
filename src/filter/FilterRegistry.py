from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.filter.Filter import Filter

from src.filter.IdentityFilter import IdentityFilter, IDENTITY_FILTER_NAME
from src.filter.DatasetSimilarityFilter import (
    DatasetSimilarityFilter,
    DATASET_SIMILARITY_FILTER_NAME,
)
from src.filter.PopulationSimilarityFilter import (
    PopulationSimilarityFilter,
    POPULATION_SIMILARITY_FILTER_NAME,
)

from src.core.Registry import Registry


class FilterRegistry(Registry):
    """
    Registry used for initialization and storing of all subclasses of the
    filter class.

    * If more filters are implemented they must be manually added
    to the internal storage (`_FILTERS`) of this class.
    """

    _FILTERS = {
        IDENTITY_FILTER_NAME: IdentityFilter,
        DATASET_SIMILARITY_FILTER_NAME: DatasetSimilarityFilter,
        POPULATION_SIMILARITY_FILTER_NAME: PopulationSimilarityFilter,
    }

    @staticmethod
    def get_names() -> list[str]:
        return list(FilterRegistry._FILTERS.keys())

    @staticmethod
    def get_resource(name: str) -> type[Filter]:
        return FilterRegistry._FILTERS[name]

    @staticmethod
    def get_resources() -> list[type[Filter]]:
        return list(FilterRegistry._FILTERS.values())
