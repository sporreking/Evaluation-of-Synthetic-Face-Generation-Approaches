from src.metric.SampleMetric import SampleMetric
from src.metric.SampleMetricManager import SampleMetricManager
from src.population.Population import Population
import src.metric.MatchingScore as MS
from src.core.Setupable import SetupMode

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

POPULATION_SIMILARITY_NAME = "PopulationSimilarity"

class PopulationSimilaritySampleMetric(SampleMetric):
    """
    Sample metric used for calculating similarity between samples
    and their own population. This similarity is defined by extension
    from the Partial FC backbone network, provided by MatchingScore.py;
    the similarity score between a sample and its population is defined as the
    similarity score between the sample and the image in the population that
    is most similar to that sample (excluding itself) according to the Partial
    FC backbone.
    """

    def __init__(self, smm: SampleMetricManager):
        super().__init__(smm)

    @staticmethod
    def get_name() -> str:
        return POPULATION_SIMILARITY_NAME

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        return {
            "download_backbone": SetupMode(
                lambda _: MS.setup,
                MS.is_ready,
                lambda: "Partial FC backbone downloaded.",
                required_modes=[],
            ),
        }

    def calc(self, ids: pd.Index, **parameters: Any) -> np.ndarray:
        """
        Calculates the similarity between the specified samples and their own
        population. For each sample associated with the specified `ids`, a
        numpy array will be derived, with `k` rows and 2 columns. The rows
        represent the k most similar images from its own population (excluding
        itself) in falling order. The first column contains IDs of the samples,
        and the second column contains the similarity scores.

        Args:
            ids (pd.Index): The indices of the samples to calculate for.
            k (int, optional): The number of most-similar neighbors to find.
                Defaults to 2.

        Returns:
            np.ndarray: A numpy array containing nested numpy arrays for each sample
                specified through `ids`. Each per-sample array contains similarity
                scores and IDs of the `k` most similar samples (see aforementioned
                format).
        """

        # Get parameter
        k = parameters["k"] if "k" in parameters else 2  #! Default (should match docstring)

        # Get data
        df = self._population.get_data()

        # Fetch samples to calculate for
        uris = list(df[Population.COLUMN_URI])

        # Project population images
        sample_projections = MS.project_images(uris)

        # Target population (to calculate scores for)
        target_projections = sample_projections[[df.index.get_loc(id) for id in ids]]

        # Derive similarity scores and find largest per sample
        output = np.zeros(target_projections.shape[0], dtype=np.ndarray)
        for i in tqdm(
            range(target_projections.shape[0]), desc="Calculating similarity scores"
        ):
            similarities = target_projections[i, :].dot(sample_projections.T).flatten()
            similarities[ids[i]] = -float("inf")  # Do not check similarity with self
            raw_ids = np.argsort(similarities)[-k:][::-1]
            val = np.array(
                [[df.index[raw_id], similarities[raw_id]] for raw_id in raw_ids]
            )
            output[i] = val

        # Return similarity scores between samples and dataset
        return output
