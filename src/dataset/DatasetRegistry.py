from src.dataset.FFHQDataset import FFHQDataset
from src.dataset.Dataset import Dataset


class DatasetRegistry:
    """
    Static class used for initialization and storing of all subclasses of the
    Dataset class.
    """

    _DATASETS = None

    @staticmethod
    def init_registry():
        """
        Static method to initialize `DATASETS` with Dataset subclasses.
        Calls the constructor for each dataset with predefined parameters.

        If more dataset are implemented they should be added in this function.
        """
        DatasetRegistry._DATASETS = dict(FFHQ_256=FFHQDataset(256))

    @staticmethod
    def get_names() -> list[str]:
        """
        Returns all the names (keys) in the registry.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """
        return list(DatasetRegistry._DATASETS.keys())

    @staticmethod
    def get_dataset(name: str) -> Dataset:
        """
        Returns a dataset with the given `name` from the registry.

        Args:
            name (str): Name of the dataset.

        Returns:
            Dataset: Dataset with the given `name`.
        """
        return DatasetRegistry._DATASETS[name]

    @staticmethod
    def get_datasets() -> list[Dataset]:
        """
        Returns all datasets from the registry.

        Returns:
            list[Dataset]: All datasets from the registry.
        """
        return list(DatasetRegistry._DATASETS.values())
