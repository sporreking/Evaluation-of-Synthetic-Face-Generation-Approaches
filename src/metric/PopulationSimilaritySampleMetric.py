from __future__ import annotations
from src.metric.SampleMetric import SampleMetric
import src.metric.MatchingScore as MS
from src.core.Setupable import SetupMode
from typing import Any, TYPE_CHECKING, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from src.metric.SampleMetricManager import SampleMetricManager
    from src.population.Population import Population


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
                True,
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
            filter_bit (int): Filter bit used for filtering the population.
            k (int, optional): The number of most-similar neighbors to find.
                Defaults to 2.

        Returns:
            np.ndarray: A numpy array containing nested numpy arrays for each sample
                specified through `ids`. Each per-sample array contains similarity
                scores and IDs of the `k` most similar samples (see aforementioned
                format).
        """
        filter_bit = parameters["filter_bit"] if "filter_bit" in parameters else 1

        # Get parameter
        k = (
            parameters["k"] if "k" in parameters else 2
        )  #! Default (should match docstring)

        # Get data
        df = self._population.get_filtered_data(filter_bit)

        # Update ids according to filter
        ids = [id for id in ids if id in df.index]

        # Fetch samples to calculate for
        uris = list(df[self._population.COLUMN_URI])

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
            similarities[df.index.get_loc(ids[i])] = -float(
                "inf"
            )  # Do not check similarity with self
            raw_ids = np.argsort(similarities)[-k:][::-1]
            val = np.array(
                [[df.index[raw_id], similarities[raw_id]] for raw_id in raw_ids]
            )
            output[i] = [{filter_bit: val}]

        # Update
        data_to_be_updated = self._smm.get(
            [self.get_name()], ids=ids, skip_if_missing=True
        )
        for id in data_to_be_updated.index:
            i = df.index.get_loc(id)
            output[i] = [data_to_be_updated.loc[id].values[0][0] | output[i][0]]

        # Return similarity scores between sample and their population
        return output
