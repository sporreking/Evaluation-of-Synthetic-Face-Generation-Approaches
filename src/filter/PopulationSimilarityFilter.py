from __future__ import annotations
from typing import TYPE_CHECKING, Type, Any
from src.filter.SampleMetricFilter import SampleMetricFilter
from src.core.Setupable import SetupMode
from src.metric.PopulationSimilaritySampleMetric import PopulationSimilaritySampleMetric
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.metric.SampleMetric import SampleMetric
    from src.metric.SampleMetricManager import SampleMetricManager

POPULATION_SIMILARITY_FILTER_NAME = "PopulationSimilarity"


class PopulationSimilarityFilter(SampleMetricFilter):
    """
    Implementation of an population similarity filter.

    This filter measures a similarity score between samples in a population.
    Each sample in the population gets a similarity score based on the "closest"
    sample in the population. The idea is to filter out look-a-likes, i.e., when it's
    obvious that the generator has produced images which is copied directly from the population,
    those should be removed to mitigate identity leakage.
    """

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        return self.sample_metric_reg_setup_modes()

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

        # Get metric value
        pop_sim = smm.get(
            [PopulationSimilaritySampleMetric.get_name()], calc_if_missing=True
        )

        # Parse population similarity
        pairs = []
        skip = []
        for i in pop_sim.index:
            # Skip pairs
            if i in skip:
                continue

            # Get indices to find pairs
            sim = pop_sim.loc[i].item()
            closest_to_i = int(sim[0][0])

            # Skip already checked
            if not isinstance(pop_sim.loc[closest_to_i].item(), float):
                closest_to_closest_to_i = int(pop_sim.loc[closest_to_i].item()[0][0])

                # Check if pair is closest to each other
                if closest_to_closest_to_i == i:
                    pairs.append((i, closest_to_closest_to_i))
                    skip.append(closest_to_closest_to_i)
                    continue

            # Get distance
            pop_sim.loc[i] = sim[0][1]

        # Fix pair similarity
        if pairs:
            for i, j in pairs:
                sim_i = pop_sim.loc[i].item()
                sim_j = pop_sim.loc[j].item()

                # Let the pair sample with second highest similarity keep
                # it's similarity to the other pair sample.
                if sim_i[1][1] > sim_j[1][1]:
                    # i gets a higher similarity
                    pop_sim.loc[i] = sim_i[0][1]
                    pop_sim.loc[j] = sim_j[1][1]
                else:
                    # j gets higher similarity
                    pop_sim.loc[i] = sim_i[1][1]
                    pop_sim.loc[j] = sim_j[0][1]

        # Get threshold
        pop_sim = pop_sim.astype(float)
        threshold = (
            float(pop_sim.quantile(quantile, interpolation="higher"))
            if threshold is None
            else threshold
        )

        # Filter ind based on ds_simshow_removed_images
        passed_ind = pop_sim.index[np.where(pop_sim < threshold)[0]]

        # Show bad images
        if display_bad_images:
            # TODO Show bad images together with their closest sample
            PopulationSimilarityFilter.display_bad_images(passed_ind, smm)

        return passed_ind

    def get_name():
        return POPULATION_SIMILARITY_FILTER_NAME

    def get_used_sample_metrics(self) -> list[Type[SampleMetric]]:
        return [PopulationSimilaritySampleMetric]
