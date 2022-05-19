from __future__ import annotations
from typing import TYPE_CHECKING, Type, Any
from src.filter.SampleMetricFilter import SampleMetricFilter
from src.core.Setupable import SetupMode
from src.metric.DatasetSimilaritySampleMetric import DatasetSimilaritySampleMetric
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.metric.SampleMetric import SampleMetric
    from src.metric.SampleMetricManager import SampleMetricManager

DATASET_SIMILARITY_FILTER_NAME = "DatasetSimilarity"


class DatasetSimilarityFilter(SampleMetricFilter):
    """
    Implementation of an dataset similarity filter.

    This filter measures a similarity score between samples in a population and a dataset.
    Each sample in the population gets a similarity score based on the "closest"
    sample in the dataset. The idea is to filter out look-a-likes, i.e., when it's
    obvious that the generator has produced images which is copied directly from the dataset,
    those should be removed to mitigate identity leakage.
    """

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        return self.sample_metric_reg_setup_modes()

    @staticmethod
    def get_bit() -> int:
        return 2

    @staticmethod
    def apply(smm: SampleMetricManager, **parameters: Any) -> pd.index:
        """
        Applies filter on population contained in given sample metric manager.

        Args:
            smm (SampleMetricManager): Manager which population will be inferred from.
                Filtering will be applied to this population.
            quantile (float, optional): Statistical quantile used for determining threshold.
                For example, a 0.99 quantile means that the 1% worst scores will be filtered out.
                Defaults to 0.99.
            threshold (float, optional): If not none, this threshold will be used instead of
                inferring threshold from `quantile`. For example, all samples above `threshold` will
                be filtered out. Defaults to None.
            display_bad_images (bool, optional): Displays. Defaults to False.

        Returns:
            pd.index: The indices of the passing samples.
        """
        # Fetch parameters
        quantile = parameters["quantile"] if "quantile" in parameters else 0.99
        threshold = parameters["threshold"] if "threshold" in parameters else None
        display_bad_images = (
            parameters["display_bad_images"]
            if "display_bad_images" in parameters
            else False
        )

        # Calc metric value
        smm.calc(
            [DatasetSimilaritySampleMetric.get_name()],
        )
        ds_sim = smm.get([DatasetSimilaritySampleMetric.get_name()]).astype(float)

        # Get threshold
        threshold = (
            float(ds_sim.quantile(quantile, interpolation="higher"))
            if threshold is None
            else threshold
        )

        # Filter ind based on ds_simshow_removed_images
        passed_ind = ds_sim.index[np.where(ds_sim < threshold)[0]]

        # Show bad images
        if display_bad_images:
            DatasetSimilarityFilter.display_bad_images(passed_ind, smm)

        return passed_ind

    @staticmethod
    def get_name():
        return DATASET_SIMILARITY_FILTER_NAME

    def get_used_sample_metrics(self) -> list[Type[SampleMetric]]:
        return [DatasetSimilaritySampleMetric]
