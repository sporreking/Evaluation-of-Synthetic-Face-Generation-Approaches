from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.filter.Filter import Filter

from src.filter.IdentityFilter import IdentityFilter, IDENTITY_FILTER_NAME

from src.core.Registry import Registry


class FilterRegistry(Registry):
    """
    Registry used for initialization and storing of all subclasses of the
    filter class.

    Note that order of filter corresponds what bit in the bitmap
    that filter is associated with.

    * If more filters are implemented they must be manually added
    to the internal storage (`_FILTERS`) of this class.
    """

    _FILTERS = {IDENTITY_FILTER_NAME: IdentityFilter}

    @staticmethod
    def get_names() -> list[str]:
        return list(FilterRegistry._FILTERS.keys())

    @staticmethod
    def get_resource(name: str) -> type[Filter]:
        return FilterRegistry._FILTERS[name]

    @staticmethod
    def get_resources() -> list[type[Filter]]:
        return list(FilterRegistry._FILTERS.values())
