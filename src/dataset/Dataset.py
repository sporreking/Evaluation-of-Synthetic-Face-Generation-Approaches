import abc
from pathlib import Path
from src.util.FileJar import FileJar
import pandas as pd


class Dataset(metaclass=abc.ABCMeta):
    """
    Abstract class outlining the general interface for different types
    of datasets.
    """

    # Path to the directory containing all datasets directories.
    DS_DIR_PREFIX: Path = Path("dataset/")

    def __init__(self, name: str):
        """
        Constructor for the Dataset class.

        Args:
            name (str): The name of the dataset, coincident with the
                directory of the dataset.

        Raises:
            FileNotFoundError: If dataset directory does not exist.
        """
        self._name = name
        self._ds_dir = self.DS_DIR_PREFIX / name

        # Check if path is directory
        if not self._ds_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory {self._ds_dir} not found!")

        # Call abstract method
        self._file_jar, self._labels = self.init_files()

    def get_labels(self) -> pd.DataFrame:
        """
        Returns labels of the dataset.

        Returns:
            pd.DataFrame: Contains ID and labels for the images in the dataset.
        """
        return self._labels

    def get_path(self) -> Path:
        """
        Returns the path to the dataset directory.

        Returns:
            Path: Path to the dataset directory.
        """
        return self._ds_dir

    def get_name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: Name of the dataset.
        """
        return self._name

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
