from pathlib import Path
import pandas as pd
import numpy as np

from src.util.FileJar import FileJar


class Population:
    """
    This class is used for representing synthetic populations,
    and filtered sub-populations.
    """

    # Static
    POPULATION_ROOT_DIR = Path("population/")
    POPULATION_DATA_FILE_NAME = "data.pkl"

    _ATTRIB_START_INDEX = 4

    def __init__(self, name: str):
        """
        Constructs a new population. If a population with the specified name
        already exists on disk, it will be loaded.

        Args:
            name (str): The name of this population.
        """
        self._name = name
        self._file_jar = FileJar(
            Population.POPULATION_ROOT_DIR / name, create_root_dir=True
        )
        self._data = self._file_jar.get_file(
            Population.POPULATION_DATA_FILE_NAME, pd.read_pickle
        )

    def _create_dataframe(self, attributes: list[str]) -> None:
        self._data = pd.DataFrame(
            columns=["latent_code", "latent_seed", "uri", "filter_bitmap", *attributes]
        )

    def _save_to_disk(self) -> None:
        self._file_jar.store_file(
            Population.POPULATION_DATA_FILE_NAME, self._data.to_pickle
        )

    def get_name(self) -> str:
        """
        Returns the name of this population.

        Returns:
            str: The name of this population.
        """
        return self._name

    def get_attributes(self) -> list[str]:
        """
        Returns the attributes used for creating this population. Decided by the first
        set of samples added to this population.

        Returns:
            list[str]: The attributes of this population.
        """
        return list(self._data.columns[Population._ATTRIB_START_INDEX :])

    def num_samples(self) -> int:
        """
        Returns the number of samples contained by this population.

        Returns:
            int: The number of samples contained by this population.
        """
        return self._data.shape[0]

    def add_all(
        self,
        latent_codes: np.ndarray,
        latent_seeds: np.ndarray,
        uris: list[str],
        filter_bitmaps: list[int],
        append: bool = True,
        save_to_disk: bool = True,
        **attributes: np.ndarray,
    ) -> None:
        """
        Adds new samples to this population. Note that the dimensions of all parameters
        must match.

        Args:
            latent_codes (np.ndarray): The latent codes used for generating the samples.
                The codes are interpreted on a per-row basis.
            latent_seeds (np.ndarray): The latent codes before they were manipulated with
                the `attributes` input to become `latent_codes`. The codes are interpreted
                on a per-row basis.
            uris (list[str]): Paths leading to the sample images.
            filter_bitmaps (list[int]): Bitmaps describing what filters the samples have
                passed. The order of the bits are analogous to the order of filter
                appearence in the FilterRegistry.
            append (bool, optional): If `True`, the samples will be appended onto the
                existing population. Otherwise, all current samples will be replaced.
                Defaults to True.
            save_to_disk (bool, optional): If `True`, the population will be saved to
                disk when the samples have been added. Defaults to True.
            **attributes (np.ndarray): The attributes used for manipulating `latent_seeds`
                into `latent_codes`. If the samples being added are the first ones of this
                population, these will define what attributes are available, i.e., if
                other attributes are added later on, they will be denied.

        Raises:
            ValueError: If the dimensions of the input do no match, or if an invalid
                attribute was provided.
        """

        # Sanity check (dimensions)
        if (
            latent_codes.shape[0] != latent_seeds.shape[0]
            or latent_seeds.shape[0] != len(uris)
            or len(uris) != len(filter_bitmaps)
            or any(len(filter_bitmaps) != len(v) for v in attributes.values())
        ):
            raise ValueError("Input dimensions do not match!")

        # Sanity check (attributes)
        for a in attributes.keys():
            if a not in self._data.columns:
                raise ValueError(
                    f"Invalid attribute '{a}'. Must be one of: {self.get_attributes()}"
                )

        # Create DataFrame if non-existent
        if self._data is None or not append:
            self._create_dataframe(attributes.keys())

        # Add all new samples
        self._data = pd.concat(
            [
                self._data,
                pd.DataFrame(
                    {
                        "latent_code": [
                            latent_codes[i, :] for i in range(latent_codes.shape[0])
                        ],
                        "latent_seed": [
                            latent_seeds[i, :] for i in range(latent_seeds.shape[0])
                        ],
                        "uri": uris,
                        "filter_bitmap": filter_bitmaps,
                        **attributes,
                    }
                ),
            ],
            ignore_index=True,
            axis=0,
        )

        # Save to disk if applicable
        if save_to_disk:
            self._save_to_disk()

    def add(
        self,
        latent_code: np.ndarray,
        latent_seed: np.ndarray,
        uri: str,
        filter_bitmap: int = 0,
        append: bool = True,
        save_to_disk: bool = True,
        **attributes: float,
    ) -> None:
        """
        Adds a new sample to this population.

        Args:
            latent_code (np.ndarray): The latent code used for generating this sample.
            latent_seed (np.ndarray): The latent code before it was manipulated with
                the `attributes` input to become `latent_code`.
            uri (str): A path leading to the sample image.
            filter_bitmap (int, optional): A bitmap describing what filters this
                sample has passed. The order of the bits are analogous to the order of
                filter appearence in the FilterRegistry. Defaults to 0.
            append (bool, optional): If `True`, this sample will be appended onto the
                existing population. Otherwise, all current samples will be replaced.
                    Defaults to True.
            save_to_disk (bool, optional): If `True`, the population will be saved to
                disk when the sample has been added. Defaults to True.
            **attributes (float): The attributes used for manipulating `latent_seed`
                into `latent_code`. If the sample being added is the first one of this
                population, these will define what attributes are available, i.e., if
                other attributes are added later on, they will be denied.

        Raises:
            ValueError: If an invalid attribute was provided.
        """

        # Sanity check (attributes)
        for a in attributes.keys():
            if a not in self._data.columns:
                raise ValueError(
                    f"Invalid attribute '{a}'. Must be one of: {self.get_attributes()}"
                )

        # Create DataFrame if non-existent
        if self._data is None or not append:
            self._create_dataframe(attributes.keys())

        # Add new sample
        self._data = pd.concat(
            [
                self._data,
                pd.DataFrame(
                    {
                        "latent_code": [latent_code],
                        "latent_seed": [latent_seed],
                        "uri": [uri],
                        "filter_bitmap": [filter_bitmap],
                        **{k: [v] for k, v in attributes.items()},
                    }
                ),
            ],
            ignore_index=True,
            axis=0,
        )

        # Save to disk if applicable
        if save_to_disk:
            self._save_to_disk()

    def remove(self, index: list[int] | int) -> None:
        """
        Removes the sample with the specified `index`.

        Args:
            index (int): The index of the sample to remove.
        """
        self._data = self._data.drop(index)

    def __getitem__(self, key: int | slice) -> pd.DataFrame:
        return self.get_data(key)

    def get_data(self, indices: list[int] | int = None) -> pd.DataFrame:
        """
        Returns the data of the samples with the specified `indices`.

        Args:
            indices (list[int] | int): The indices of the samples to fetch, or
                None if all samples should be fetched. Default is None.

        Returns:
            pd.DataFrame: The fetched samples.
        """
        if indices is None:
            return self._data
        else:
            return self._data.iloc[[indices] if type(indices) == int else indices]

    def get_filtered_data(self, mask: int) -> pd.DataFrame:
        """
        Returns the samples that pass the specified bitmap `mask`.

        Args:
            mask (int): A bitmap with '1' for all filters that the samples must pass.
                The order of the bits are analogous to the order of filters
                in the FilterRegistry.

        Returns:
            pd.DataFrame: The filtered samples.
        """
        return self._data[[(bm & mask) == mask for bm in self._data["filter_bitmap"]]]
