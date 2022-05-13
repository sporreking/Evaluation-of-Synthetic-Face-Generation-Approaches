from __future__ import annotations
from typing import TYPE_CHECKING
from src.filter.Filter import Filter
from src.core.Setupable import SetupMode

if TYPE_CHECKING:
    from src.metric.SampleMetricManager import SampleMetricManager

IDENTITY_FILTER_NAME = "IDENTITY"


class IdentityFilter(Filter):
    """
    Implementation of an identity filter.

    This filter lets every sample pass.
    """

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        return {}

    def apply(smm: SampleMetricManager):
        # Every sample passes
        return smm.get_population().get_data().index

    def get_name():
        return IDENTITY_FILTER_NAME
