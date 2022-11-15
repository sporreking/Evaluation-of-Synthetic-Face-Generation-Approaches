from __future__ import annotations
import src.metric.MatchingScore as MS
from src.core.Setupable import SetupMode
from typing import Any, TYPE_CHECKING
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.metric.SampleMetric import SampleMetric

if TYPE_CHECKING:
    from src.metric.SampleMetricManager import SampleMetricManager
    from src.dataset.Dataset import Dataset

DATASET_SIMILARITY_NAME = "DatasetSimilarity"


class DatasetSimilaritySampleMetric(SampleMetric):
    """
    Sample metric used for calculating similarity between samples
    and a dataset. This similarity is defined by extension from the
    Partial FC backbone network, provided by MatchingScore.py; the
    similarity score between a sample and a dataset is defined as the
    similarity score between the sample and the image in the dataset that
    is most similar to that sample according to the Partial FC backbone.
    """

    def __init__(self, smm: SampleMetricManager):
        super().__init__(smm)

    @staticmethod
    def get_name() -> str:
        return DATASET_SIMILARITY_NAME

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        ds = self._dataset
        ds_name = ds.get_name(ds.get_resolution())

        def _check_ds_projection():
            try:
                MS.load_projected_images(file_name_suffix=ds_name)
                return True
            except FileNotFoundError:
                return False

        return {
            "download_backbone": SetupMode(
                True,
                lambda _: MS.setup,
                MS.is_ready,
                lambda: "Partial FC backbone downloaded.",
                required_modes=[],
            ),
            "project_dataset": SetupMode(
                True,
                lambda _: MS.project_images(
                    ds.get_image_paths(), file_name_suffix=ds_name
                ),
                _check_ds_projection,
                lambda: f"{len(ds)} samples were projected for '{ds_name}'.",
                required_modes=["download_backbone"],
            ),
        }

    def calc(self, ids: pd.Index, **parameters: Any) -> np.ndarray:
        """
        Calculates the similarity between the specified samples and their closest
        neighbor in the dataset.

        Args:
            ids (pd.Index): The indices of the samples to calculate for.

        Returns:
            np.ndarray: One similarity score per requested sample, defined as the
                similarity betwen the sample in question and its closest neighbor
                in the dataset, i.e., a one-dimensional vector of the same length
                as `ids`.
        """

        # Fetch samples to calculate for
        uris = list(self._population.get_data(ids)[self._population.COLUMN_URI])

        # Load dataset projections
        dataset_projections = MS.load_projected_images(
            self._dataset.get_name(self._dataset.get_resolution())
        )

        # Project sample images
        sample_projections = None
        try:
            sample_projections = MS.load_projected_images(
                self._smm.get_population().get_name()
            )
        except FileNotFoundError:
            sample_projections = MS.project_images(
                uris, file_name_suffix=self._smm.get_population().get_name()
            )

        # Derive similarity scores and find largest per sample
        output = np.zeros(sample_projections.shape[0])
        for i in tqdm(
            range(sample_projections.shape[0]), desc="Calculating similarity scores"
        ):
            similarities = sample_projections[i, :].dot(dataset_projections.T).flatten()
            output[i] = np.sort(similarities)[-1]

        # Return similarity scores between samples and dataset
        return output

    @staticmethod
    def dataset_vs_dataset_similarity(dataset: Dataset):
        # Load dataset projections
        dataset_projections = MS.load_projected_images(
            dataset.get_name(dataset.get_resolution())
        )

        # Derive similarity scores and find largest per sample
        output = np.zeros(dataset_projections.shape[0])
        for i in tqdm(
            range(dataset_projections.shape[0]), desc="Dataset vs dataset similarity"
        ):
            # TODO: Make this using multiprocessing i.e. Pool and Map
            similarities = (
                dataset_projections[i, :].dot(dataset_projections.T).flatten()
            )
            similarities[i] = -float("inf")  # Do not check similarity with self
            output[i] = np.sort(similarities)[-1]

        # Return similarity scores between dataset and dataset
        return output
