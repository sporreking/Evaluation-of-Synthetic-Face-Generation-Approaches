from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Type, TYPE_CHECKING
from src.util.FileJar import FileJar
from src.filter.Filter import Filter


if TYPE_CHECKING:
    from src.metric.SampleMetricManager import SampleMetricManager


class Population:
    """
    This class is used for representing synthetic populations,
    and filtered sub-populations.
    """

    # Static
    POPULATION_ROOT_DIR = Path("population/")
    POPULATION_DATA_FILE_NAME = "data.pkl"
    POPULATION_METADATA_DIRECTORY_NAME = "metadata"

    COLUMN_LATENT_CODE = "latent_code"
    COLUMN_LATENT_SEED = "latent_seed"
    COLUMN_URI = "uri"
    COLUMN_FILTER_BITMAP = "filter_bitmap"
    _ATTRIB_START_INDEX = 4

    def __init__(self, name: str):
        """
        Constructs a new population. If a population with the specified name
        already exists on disk, it will be loaded.

        Args:
            name (str): The name of this population.

        Raises:
            FileNotFoundError: If file exists with the name 'metadata' in the root directory.
        """
        self._name = name
        self._root_dir = Population.POPULATION_ROOT_DIR / name
        self._metadata_dir = self._root_dir / "metadata"
        self._file_jar = FileJar(self._root_dir, create_root_dir=True)
        self._create_metadata_directory()

        self._data = self._file_jar.get_file(
            Population.POPULATION_DATA_FILE_NAME, pd.read_pickle
        )

    def _create_metadata_directory(self) -> None:
        # Check that metadata_dir is a directory.
        if not self._metadata_dir.is_dir():
            if not self._metadata_dir.exists():
                self._metadata_dir.mkdir(parents=True)
            elif self._metadata_dir.is_file():
                raise FileNotFoundError(
                    "File named 'metadata' is not allowed in the population directory."
                )

    def _clear_metadata(self) -> None:
        # Remove all files in metadata directory
        [f.unlink() for f in self._metadata_dir.glob("*") if f.is_file()]

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
        return 0 if self._data is None else self._data.shape[0]

    def _move_images(
        self, uris: list[str], start_id: int, append: bool
    ) -> tuple[list[int], list[str]]:
        """Move image to population directory."""

        new_ids = []
        new_uris = []
        for uri in uris:

            # Derive source path
            source = Path(uri)

            # Check if source file exists
            if not source.exists():
                raise FileNotFoundError(f"Could not find file: '{source.absolute()}'")

            # Check if source file lies in population directory
            if source.parent == self._file_jar.get_root_dir():
                if source.stem.isdigit():
                    if int(source.stem) in self._data.index:
                        # Check if image-to-be-moved is already part of population
                        raise ValueError(
                            f"Cannot add/move image '{source.absolute()}' to population! "
                            + "The image is already used by another sample in the population."
                        )
                    else:
                        # Derive ID from file name
                        new_ids.append(int(source.stem))
                        new_uris.append(str(source))
                        continue

            # Derive destination path
            id = start_id
            dest = None
            while True:
                dest = self._file_jar.get_root_dir() / f"{id}{source.suffix}"
                if dest.exists():
                    if str(dest) in uris:
                        # The ID is occupied
                        id += 1
                    elif append:
                        # There is a blocking file
                        raise ValueError(
                            f"File '{dest.absolute()}' already exists! "
                            + "If existing files should be replaced, set 'append' to False."
                        )
                    else:
                        # The file should be replaced
                        break
                else:
                    # There is no obstruction
                    break
            start_id = id + 1

            # Move files and update IDs / URIs
            new_ids.append(id)
            new_uris.append(str(source.replace(dest)))

        return new_ids, new_uris

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

        If `save_to_disk=True`, all images specified by `uris` will be moved to this population's
        root directory before being stored. The files will be named after their IDs, e.g.,
        "0.png", "1.png", etc. Note that if later calls are made to either `add_all()` or `add()`,
        files that were added earlier will not be moved. It is therefore recommended to always use
        `save_to_disk=True`.

        If a referenced file is already in the population directory, this function will first
        try to derive an ID from its file name to associate the sample with, but if this is not
        possible the population will generate a new ID for it and move it as usual. If a file
        is named according to an ID, but if it's already in use by another sample, an exception
        will be raised.

        When an unoccupied ID has been decided upon for a new sample, but there is still an
        obstructing file with the same name in the population directory, the file will be
        replaced if `append=False`. If `append=True`, an exception will be raised instead. This
        scenario should only occur if there is "junk" in the population directory, i.e., if there
        are old samples that should be replaced (`append=False`). Otherwise, something is wrong.

        Note that all files in the metadata directory will be deleted.

        Args:
            latent_codes (np.ndarray): The latent codes used for generating the samples.
                The codes are interpreted on a per-row basis.
            latent_seeds (np.ndarray): The latent codes before they were manipulated with
                the `attributes` input to become `latent_codes`. The codes are interpreted
                on a per-row basis.
            uris (list[str]): Paths leading to the sample images.
            filter_bitmaps (list[int]): Bitmaps describing what filters the samples have
                passed.
            append (bool, optional): If `True`, the samples will be appended onto the
                existing population. Otherwise, all current samples will be replaced.
                Defaults to True.
            save_to_disk (bool, optional): If `True`, the population will be saved to
                disk when the samples have been added, and the files referenced by `uris`
                will be moved to the population directory. Defaults to True.
            **attributes (np.ndarray): The attributes used for manipulating `latent_seeds`
                into `latent_codes`. If the samples being added are the first ones of this
                population, these will define what attributes are available, i.e., if
                other attributes are added later on, they will be denied.

        Raises:
            ValueError: If the dimensions of the input do no match, or if an invalid
                attribute was provided.
            FileNotFoundError: If any file referenced by `uris` does not exist.
            ValueError: If any file referenced by `uris` is named according to an ID,
                but if it's already in use by another sample.
            ValueError: If `append=True` but there is an obstructing file in the population
                directory which is not in use by any sample. This on-disk scenario should
                only occur if there are old samples that should be replaced (`append=False`).
        """

        # Sanity check (dimensions)
        if (
            latent_codes.shape[0] != latent_seeds.shape[0]
            or latent_seeds.shape[0] != len(uris)
            or len(uris) != len(filter_bitmaps)
            or any(len(filter_bitmaps) != len(v) for v in attributes.values())
        ):
            raise ValueError("Input dimensions do not match!")

        # Create DataFrame if non-existent
        if self._data is None or not append:
            self._create_dataframe(attributes.keys())

        # Sanity check (attributes)
        for a in attributes.keys():
            if a not in self._data.columns:
                raise ValueError(
                    f"Invalid attribute '{a}'. Must be one of: {self.get_attributes()}"
                )

        # Move images if applicable
        start_index = np.max(self._data.index) + 1 if len(self._data) > 0 else 0
        ids = None
        if save_to_disk:
            ids, uris = self._move_images(uris, start_index, append)
        else:
            ids, uris = range(start_index, start_index + len(uris)), uris

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
                    },
                    index=ids,
                ),
            ],
            ignore_index=False,
            axis=0,
        )

        # Clear old metadata
        self._clear_metadata()

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

        If `save_to_disk=True`, the image specified by `uri` will be moved to this population's
        root directory before being stored. The file will be named according to its IDs, e.g.,
        "0.png", "1.png", etc. Note that if later calls are made to either `add_all()` or `add()`,
        files that were added earlier will not be moved. It is therefore recommended to always use
        `save_to_disk=True`.

        If the referenced file is already in the population directory, this function will first
        try to derive an ID from its file name to associate the sample with, but if this is not
        possible the population will generate a new ID for it and move it as usual. If the file
        is named according to an ID, but if it's already in use by another sample, an exception
        will be raised.

        When an unoccupied ID has been decided upon for the new sample, but there is still an
        obstructing file with the same name in the population directory, the file will be
        replaced if `append=False`. If `append=True`, an exception will be raised instead. This
        scenario should only occur if there is "junk" in the population directory, i.e., if there
        are old samples that should be replaced (`append=False`). Otherwise, something is wrong.

        Note that all files in the metadata directory will be deleted.

        Args:
            latent_code (np.ndarray): The latent code used for generating this sample.
            latent_seed (np.ndarray): The latent code before it was manipulated with
                the `attributes` input to become `latent_code`.
            uri (str): A path leading to the sample image.
            filter_bitmap (int, optional): A bitmap describing what filters this
                sample has passed. Defaults to 0.
            append (bool, optional): If `True`, this sample will be appended onto the
                existing population. Otherwise, all current samples will be replaced.
                    Defaults to True.
            save_to_disk (bool, optional): If `True`, the population will be saved to
                disk when the sample has been added, and the file referenced by `uri`
                will be moved to the population directory. Defaults to True.
            **attributes (float): The attributes used for manipulating `latent_seed`
                into `latent_code`. If the sample being added is the first one of this
                population, these will define what attributes are available, i.e., if
                other attributes are added later on, they will be denied.

        Raises:
            ValueError: If an invalid attribute was provided.
            FileNotFoundError: If the file referenced by `uri` does not exist.
            ValueError: If the file referenced by `uri` is named according to an ID,
                but if it's already in use by another sample.
            ValueError: If `append=True` but there is an obstructing file in the population
                directory which is not in use by any sample. This on-disk scenario should
                only occur if there are old samples that should be replaced (`append=False`).
        """

        # Create DataFrame if non-existent
        if self._data is None or not append:
            self._create_dataframe(attributes.keys())

        # Sanity check (attributes)
        for a in attributes.keys():
            if a not in self._data.columns:
                raise ValueError(
                    f"Invalid attribute '{a}'. Must be one of: {self.get_attributes()}"
                )

        # Move image if applicable
        start_index = np.max(self._data.index) + 1 if len(self._data) > 0 else 0
        ids, uris = None, None
        if save_to_disk:
            ids, uris = self._move_images([uri], start_index, append)
        else:
            ids, uris = [start_index], [uri]

        # Add new sample
        self._data = pd.concat(
            [
                self._data,
                pd.DataFrame(
                    {
                        "latent_code": [latent_code],
                        "latent_seed": [latent_seed],
                        "uri": uris,
                        "filter_bitmap": [filter_bitmap],
                        **{k: [v] for k, v in attributes.items()},
                    },
                    index=ids,
                ),
            ],
            ignore_index=False,
            axis=0,
        )

        # Clear old metadata
        self._clear_metadata()

        # Save to disk if applicable
        if save_to_disk:
            self._save_to_disk()

    def remove(self, index: Union[list[int], int]) -> None:
        """
        Removes the sample with the specified `index`.

        Args:
            index (int): The index of the sample to remove.
        """
        self._data = self._data.drop(index)

    def __getitem__(self, key: Union[int, slice]) -> pd.DataFrame:
        return self.get_data(key)

    def get_data(self, indices: Union[list[int], int] = None) -> pd.DataFrame:
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
            return self._data.loc[[indices] if type(indices) == int else indices]

    def get_filtered_data(self, mask: int) -> pd.DataFrame:
        """
        Returns the samples that pass the specified bitmap `mask`.

        Args:
            mask (int): A bitmap with '1' for all filters that the samples must pass.

        Returns:
            pd.DataFrame: The filtered samples.
        """
        return self._data[self.get_filtering_indices(mask)]

    def get_filtering_indices(self, mask: int) -> list[bool]:
        """
        Returns a boolean list corresponding to the indices of the data in this population.
        Filtering is done for the specified bitmap `mask`.

        For example:

        The boolean list `[True, False, True]` means that indices 0 and 2 have passed
        the filter.

        Args:
            mask (int): A bitmap with '1' for all filters that the samples must pass.

        Returns:
            list[bool]: The filtering indices as a boolean list, True if indices passed,
                False otherwise.
        """
        return [(bm & mask) == mask for bm in self._data["filter_bitmap"]]

    def apply_filter(
        self, filter: Type[Filter], smm: SampleMetricManager, save_to_disk: bool = True
    ) -> None:
        """
        Apply a filter on this population and update
        the bitmap accordingly.

        Args:
            filter (Type(Filter)): The filter to apply.
            smm (SampleMetricManager): The sample manager used to apply the
                filter. Population is inferred from the sample manager, which
                must be the same population as this class.
            save_to_disk (bool, optional): If `True`, the updated bitmap will be saved to disk.
                Defaults to True.

        Raises:
            AssertionError: When this population does not match the population in the sample
                metric manager specified by `smm`.
        """
        assert smm.get_population() is self

        # Update bitmap for the passing samples
        # Set filter bit to 1 in bitmap
        bit = filter.get_bit()
        ind = filter.apply(smm)
        self._data[self.COLUMN_FILTER_BITMAP][ind] = self._data[
            self.COLUMN_FILTER_BITMAP
        ][ind].apply(lambda bitmap: bit | bitmap)

        # Update bitmap for the non-passing samples
        # Set filter bit to 0 in bitmap
        fail_ind = self._data.index.difference(ind, sort=False)
        self._data[self.COLUMN_FILTER_BITMAP][fail_ind] = self._data[
            self.COLUMN_FILTER_BITMAP
        ][fail_ind].apply(lambda bitmap: ~bit & bitmap)

        if save_to_disk:
            # Save result to disk
            self._save_to_disk()
