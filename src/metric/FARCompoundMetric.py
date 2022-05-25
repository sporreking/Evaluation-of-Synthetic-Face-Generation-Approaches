from __future__ import annotations
from src.metric.CompoundMetric import CompoundMetric
from src.metric.SampleMetricManager import SampleMetricManager
from src.core.Setupable import SetupMode
from src.metric.CompoundMetricManager import CompoundMetricManager
from typing import Any, TYPE_CHECKING, Union
from src.population.Population import Population
from src.metric.DatasetSimilaritySampleMetric import DatasetSimilaritySampleMetric
from src.metric.PopulationSimilaritySampleMetric import PopulationSimilaritySampleMetric
import numpy as np
from src.util.AuxUtil import get_file_jar
import src.metric.MatchingScore as MS
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

if TYPE_CHECKING:
    from src.compound_model.CompoundModelFactory import CompoundModelFactoryContext

FAR_NAME = "FAR"
DS_VS_DS_FILE_NAME = (
    lambda ds_name: "_".join([ds_name, "vs", ds_name, "similarity"]) + ".npy"
)

# SETUP NAMES
DS_VS_DS_SETUP_NAME = "DATASET_VS_DATASET"

# Threshold constants
T_MIN = 0
T_MAX = 10
T_STEPS = 50


class FARCompoundMetric(CompoundMetric):
    def __init__(
        self,
        cmm: CompoundMetricManager,
        smm: SampleMetricManager,
    ):
        super(FARCompoundMetric, self).__init__(FAR_NAME, cmm, smm)

        # Define threshold linspace
        self._thresholds = self._get_threshold_space()

        # Init storage structure for this metric
        self._file_jar = get_file_jar()
        self._far_ds_vs_ds = None
        self._far = None
        self._filter_name_dict = dict()

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        """
        Return a dictionary declaring all available setup modes.

        The dictionary consist of setup mode names, mapped to SetupMode objects.

        Raises:
            AssertionError: When DatasetSimilaritySampleMetric does not exist in
                the sample metric manager provided on construction.
            AssertionError: When PopulationSimilaritySampleMetric does not exist in
                the sample metric manager provided on construction.

        Returns:
            dict[str, SetupMode]: All setup modes, and their functionality.
        """
        ds_setup_modes = None
        pop_setup_modes = None
        for sm in self._smm.get_metric_instances():
            if isinstance(sm, DatasetSimilaritySampleMetric):
                ds_setup_modes = sm.reg_setup_modes()
            elif isinstance(sm, PopulationSimilaritySampleMetric):
                pop_setup_modes = sm.reg_setup_modes()
        assert ds_setup_modes is not None
        assert pop_setup_modes is not None

        # Merge sample metric setup modes
        sample_metric_setup_modes = pop_setup_modes | ds_setup_modes

        # Define setup mode exclusive for FAR
        ds_vs_ds_setup_mode = {
            DS_VS_DS_SETUP_NAME: SetupMode(
                True,
                lambda _: self._setup_dataset_vs_dataset(),
                self._is_dataset_vs_dataset_ready,
                required_modes=[setup_name for setup_name in ds_setup_modes.keys()],
            )
        }

        # Merge all
        return sample_metric_setup_modes | ds_vs_ds_setup_mode

    def _setup_dataset_vs_dataset(self):
        # Calculate the dataset vs dataset similarity
        sim = DatasetSimilaritySampleMetric.dataset_vs_dataset_similarity(self._dataset)

        # Store result on disk
        self._file_jar.store_file(
            DS_VS_DS_FILE_NAME(self._dataset.get_name(self._dataset.get_resolution())),
            lambda p: np.save(p, sim),
        )

    def _is_dataset_vs_dataset_ready(self):
        return self._file_jar.has_file(
            DS_VS_DS_FILE_NAME(self._dataset.get_name(self._dataset.get_resolution()))
        )

    def _calc_far(self, similarities: np.ndarray) -> np.ndarray:
        """
        Calculates the FAR as a function of thresholds

        Assumes all samples are different identities

        If a similarity is above the threshold, then this sample is considered the same
        identity with its neighbor, by assumption samples are not same identities.
        Thus, it is misclassified when they are the same identity.
        """

        return np.mean(
            similarities.reshape(-1, 1) > self._thresholds.reshape(1, -1),
            0,
        )

    def _get_threshold_space(self) -> np.ndarray:
        # TODO: Find a cool way to inferr max/min
        return np.linspace(T_MIN, T_MAX, T_STEPS)

    def calc(self, filter_bit: int = 1, **parameters: Any) -> Any:
        """
        Calculates the false acceptance rate as a function of similarity thresholds.

        Requires a context of other models and populations, this is given from the
        compound metric manager, be sure to create a `CompoundModelFactoryContext` before
        calculating FAR.

        Args:
            filter_bit (int, optional): Filter bit used for filtering. Defaults to 1.

        Raises:
            FileNotFoundError: When setup with dataset vs dataset has not been performed.
            AssertionError: When context of other populations are not available.

        Returns:
            Any: FAR as a function of similarity threshold, formatted as a list with containing a
                dict with all the FAR combinations (dataset vs dataset, population vs dataset,
                population vs population).
        """

        # Get context
        context = self._cmm.get_context()
        assert context is not None

        # Dataset vs dataset (Symmetrical)
        if self._far_ds_vs_ds is None:
            similarities = self._file_jar.get_file(
                DS_VS_DS_FILE_NAME(
                    self._dataset.get_name(self._dataset.get_resolution())
                ),
                np.load,
            )
            if similarities is None:
                raise FileNotFoundError(
                    "Dataset vs dataset similarity not found, run setup first!"
                )

            self._far_ds_vs_ds = self._calc_far(similarities)

        # Population variables
        all_ids = self.get_population().get_data().index
        filtered_ids = list(
            all_ids[self.get_population().get_filtering_indices(filter_bit)]
        )

        # Population vs dataset
        similarities = self._smm.get(
            [DatasetSimilaritySampleMetric.get_name()],
            filtered_ids,
            skip_if_missing=True,
            **parameters,
        ).to_numpy()
        if similarities.shape[0] == len(filtered_ids):
            far_pop_vs_ds = self._calc_far(similarities)
        else:
            far_pop_vs_ds = self._population_vs_dataset_far(filter_bit)

        # This population vs populations/filter_bits

        # Get context
        populations_vs = list(context.populations.values())
        filter_bits_vs = list(context.filter_bits.values())
        far_pop_vs_pops = self._population_vs_populations_far(
            filter_bit, filter_bits_vs, populations_vs
        )

        # Format result
        far = dict()
        cmf_name = self._population.get_name()
        ds_name = self._dataset.get_name(self._dataset.get_resolution())
        far[cmf_name] = {
            (
                ds_name,
                None,
            ): far_pop_vs_ds
        } | far_pop_vs_pops
        far[ds_name] = {ds_name: self._far_ds_vs_ds}

        # Save results
        self._far = [far]
        self._cmf_name = cmf_name
        self._far_context = context

        return [far]

    def get(self, calc_if_missing: bool = False, **parameters: Any) -> Any:
        # Check if metric already calculated
        filter_bit = parameters["filter_bit"] if "filter_bit" in parameters else 1
        if self._far is not None:
            if self._far[0][1] == filter_bit:
                if self._far_context == self._cmm.get_context():
                    return self._far

        # Check if calculate when missing
        if calc_if_missing:
            return self.calc(**parameters)

        else:
            return None

    def print_result(self) -> None:
        for name, d in self._far.items():
            for t_name, far in d.items():
                print(f"{name}_vs_{t_name}: ", far)

    def has_plot(self) -> bool:
        return True

    def plot_result(self) -> None:
        # TODO: DO NOT USE MANAGER AFTER STORAGE REWORK
        val = self._cmm.get(self._name)

        def _parse_pop_name(name: tuple[str, int]):
            cm_name = name[0][4:].split("_")
            context = self._cmm.get_context()
            for filter_bit, filter_name in [
                val for inner in context.filter_bits.values() for val in inner
            ]:
                if name[1] == filter_bit:
                    correct_filter_name = filter_name
            return f"({cm_name[0]}, {cm_name[1]}, {correct_filter_name})"

        for name, d in val[0].items():
            for t_name, far in d.items():
                # ds vs ds
                if type(name) == str:
                    name1 = name
                    name2 = name1
                    style = "--"
                # pop vs ds
                elif t_name[1] is None:
                    name1 = _parse_pop_name(name)
                    name2 = t_name[0]
                    style = "-."
                # pop vs pop
                else:
                    name1 = _parse_pop_name(name)
                    name2 = _parse_pop_name(t_name)
                    style = ":"

                # self vs self
                if name == t_name and type(name) != str:
                    style = "-"

                plt.plot(self._thresholds, far, style, label=f"{name1} vs {name2}")
        plt.title("FAR as a Function of Similarity Score Threshold")
        plt.xlabel("Similarity Score Threshold")
        plt.ylabel("FAR (False Acceptance Rate)")
        plt.yscale("log")
        plt.legend()
        plt.show(block=True)

    def _parse_context(
        self, context: CompoundModelFactoryContext
    ) -> tuple[list[Population], list[list[int]]]:
        populations = [population for population in context.populations.values()]
        filter_bits = [
            filter_bits_pop for filter_bits_pop in context.filter_bits.values()
        ]
        return populations, filter_bits

    def _get_projected_images(
        self, uris: list[Union[str, Path]], name: str
    ) -> np.ndarray:
        try:
            projections = MS.load_projected_images(name)
            if projections.shape[0] != len(uris):
                raise FileNotFoundError("Not all images was projected.")
        except:
            projections = MS.project_images(uris, file_name_suffix=name)

        return projections

    def _population_vs_dataset_far(
        self, filter_bit: int
    ) -> dict[tuple[str, int], np.ndarray]:
        pop_data = self._population.get_data()

        # Fetch samples to calculate for
        uris = list(pop_data[self._population.COLUMN_URI])

        # Project self population
        unfiltered_sample_projections = self._get_projected_images(
            uris, self._population.get_name()
        )
        filtering_inds = self._population.get_filtering_indices(filter_bit)
        sample_projections = unfiltered_sample_projections[filtering_inds]

        # Load dataset projections
        dataset_projections = MS.load_projected_images(
            self._dataset.get_name(self._dataset.get_resolution())
        )

        # Derive similarity scores and find largest per sample
        output = np.zeros(sample_projections.shape[0])
        for i in tqdm(
            range(sample_projections.shape[0]), desc="Population vs dataset similarity"
        ):
            similarities = sample_projections[i, :].dot(dataset_projections.T).flatten()
            output[i] = np.sort(similarities)[-1]

        # Return similarity scores between samples and dataset
        return output

    def _population_vs_populations_far(
        self,
        filter_bit: int,
        filter_bits_vs: list[list[tuple[int, str]]],
        populations_vs: list[Population],
    ) -> dict[tuple[str, int], np.ndarray]:

        # Fetch samples to calculate for
        uris = list(self._population.get_data()[self._population.COLUMN_URI])

        # Get/project self population
        unfiltered_target_projections = self._get_projected_images(
            uris, self._population.get_name()
        )

        # Filter projections
        filtering_inds = self._population.get_filtering_indices(filter_bit)
        target_projections = unfiltered_target_projections[filtering_inds]
        target_pop_data = self._population.get_filtered_data(filter_bit)
        target_pop_name = self._population.get_name()

        results = dict()
        # Loop through all populations
        for i, population in tqdm(
            enumerate(populations_vs),
            position=0,
            desc="Populations vs population similarity",
            total=len(populations_vs),
        ):
            pop_name = population.get_name()
            pop_data = population.get_data()
            uris_unfiltered_vs = list(pop_data[population.COLUMN_URI])

            # Get/project population images
            sample_unfiltered_projections_vs = self._get_projected_images(
                uris_unfiltered_vs, pop_name
            )

            # Loop through all filter
            for filter_bit_vs in tqdm(
                filter_bits_vs[i],
                position=1,
                desc="Similarity score per filter",
                leave=False,
                total=len(filter_bits_vs[i]),
            ):
                # Save filter bit and name connection
                self._filter_name_dict[filter_bit_vs[0]] = filter_bit_vs[1]
                filter_bit_vs = filter_bit_vs[0]

                # Filter population projections
                filtering = population.get_filtering_indices(filter_bit_vs)
                sample_projections_vs = sample_unfiltered_projections_vs[filtering]

                # Derive similarity scores and find largest per sample
                output = np.zeros(target_projections.shape[0], dtype=np.ndarray)
                for i in tqdm(
                    range(target_projections.shape[0]),
                    position=1,
                    desc="Calculating similarity scores",
                    leave=False,
                ):
                    similarities = (
                        target_projections[i, :].dot(sample_projections_vs.T).flatten()
                    )

                    # Check if similarity score for samples can contain similarity to itself
                    if pop_name == target_pop_name:
                        id = target_pop_data.index[i]
                        sample_index = pop_data[population.COLUMN_URI].index[filtering]
                        if id in sample_index:
                            # Do not check similarity with self
                            similarities[sample_index.get_loc(id)] = -float("inf")

                    output[i] = np.sort(similarities)[-1]

                # Calc far to save memory
                results[(pop_name, filter_bit_vs)] = self._calc_far(output)

        return results
