from __future__ import annotations
from typing import TYPE_CHECKING
from src.filter.Filter import Filter

if TYPE_CHECKING:
    from src.metric.SampleMetricManager import SampleMetricManager

IDENTITY_FILTER_NAME = "IDENTITY"


class IdentityFilter(Filter):
    """
    Implementation of an identity filter.

    This filter lets every sample pass.
    """

    def apply(smm: SampleMetricManager):
        # Every sample passes
        return smm.get_population().get_data().index

    def get_name():
        return IDENTITY_FILTER_NAME
