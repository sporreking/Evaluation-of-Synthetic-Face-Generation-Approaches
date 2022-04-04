import abc
from pathlib import Path
from src.util.FileJar import FileJar
from src.dataset.TorchImageDataset import TorchImageDataset
import pandas as pd


class Dataset(metaclass=abc.ABCMeta):
    """
    Abstract class outlining the general interface for different types
    of datasets.
    """

    # Path to the directory containing all datasets directories.
    DS_DIR_PREFIX: Path = Path("dataset/")

    def __init__(self, resolution: int):
        """
        Constructor for the Dataset class.

        Args:
            name (str): The name of the dataset, coincident with the
                directory of the dataset.
            resolution (int): The native resolution of the dataset.

        Raises:
            FileNotFoundError: If dataset directory does not exist.
        """
        self._resolution = resolution

        # Check if path is directory
        if not self.get_path().is_dir():
            raise FileNotFoundError(f"Dataset directory {self.get_path()} not found!")

        # Call abstract method
        self._file_jar, self._labels = self.init_files()

    def __len__(self) -> int:
        return self.num_samples()

    def num_samples(self) -> int:
        """
        Returns the number of samples (images) contained by this dataset.

        Returns:
            int: The number of sample (images) contained by this dataset.
        """
        return len(self.get_image_paths())

    def get_labels(self) -> pd.DataFrame:
        """
        Returns labels of the dataset.

        Returns:
            pd.DataFrame: Contains ID and labels for the images in the dataset.
        """
        return self._labels

    @abc.abstractmethod
    def get_processed_labels(self) -> pd.DataFrame:
        """
        Should returns labels of the dataset where all labels
        are between 0 and 1.

        Returns:
            pd.DataFrame: Contains ID and labels for the images in the dataset.
        """

    def get_path(self) -> Path:
        """
        Returns the path to the dataset directory.

        Returns:
            Path: Path to the dataset directory.
        """
        return self.DS_DIR_PREFIX / self.get_name(self._resolution)

    def get_resolution(self) -> int:
        """
        Returns the resolution of the dataset.

        Returns:
            int: The resolution of the dataset.
        """
        return self._resolution

    @classmethod
    def get_name(cls, resolution: int) -> str:
        """
        Returns the name of the dataset with the given resolution.

        Returns:
            str: Name of the dataset with the given resolution.
        """
        return f"{cls.get_resolution_invariant_name()}_{resolution}"

    @staticmethod
    @abc.abstractmethod
    def get_resolution_invariant_name() -> str:
        """
        Should return the name of the dataset without dependence on resolution.

        Note that this function is used in `get_name(res: int)` for deriving the
        actual name.

        Returns:
            str: The name of the dataset without dependence on resolution.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def is_ready(resolution: int) -> bool:
        """
        Should return `True` if the dataset has been properly initialized (with
        respect to the specified resolution), i.e., if all of its auxiliary
        models have been trained.

        Args:
            resolution (int): The resolution to check for.

        Returns:
            bool: `True` if the dataset has been properly initialized.
        """
        pass

    @abc.abstractmethod
    def get_image_dir(self) -> Path:
        """
        Should return a path to the directory where all images of the dataset
        are located.

        Returns:
            Path: The directory where all images of the dataset are located.
        """
        pass

    @abc.abstractmethod
    def get_image_paths(self) -> list[Path]:
        """
        Should return a list of paths to the image locations of the dataset.

        Returns:
            list[Path]: The image location of every sample. The index of a path
                denotes the index of the sample with which it is associated.
        """
        pass

    def to_torch_dataset(
        self, transform, use_labels=True, attr: str = None, use_processed_labels=True
    ) -> TorchImageDataset:
        """
        Returns a new TorchImageDataset instance, used for representing a torch
        compatible version of this dataset.


        Args:
            transform (Any): Transformations to apply to the data before using
                it for training. Transforms are found in the `torchvision.transforms` module.
            use_labels (bool, optional): True if labels should be used. Defaults to True.
            attr (str, optional): Attribute used,
                if attr is not None, use_labels must be true. Defaults to None.
            use_processed_labels (bool, optional): Uses processed labels if True.
                Processed labels have their values between 0-1. Defaults to True.

        Returns:
            TorchImageDataset: TorchImageDataset: A torch compatible version of this dataset.
        """
        if use_labels:
            return (
                TorchImageDataset(
                    self.get_image_paths(), transform, self.get_processed_labels(), attr
                )
                if use_processed_labels
                else TorchImageDataset(
                    self.get_image_paths(), transform, self.get_labels(), attr
                )
            )

        else:
            return TorchImageDataset(self.get_image_paths(), transform, attr=attr)

    @abc.abstractmethod
    def init_files(self) -> tuple[FileJar, pd.DataFrame]:
        """
        Extract labels from the dataset and saves them to a
        FileJar as well as a pd.DataFrame.
        Note that the use of FileJar allows for saving the dataframe such
        that it can be loaded again next time without any extraction needed.

        Returns:
            tuple[FileJar, pd.DataFrame]: FileJar with save paths and a pd.DataFrame containing labels.
        """
        pass
