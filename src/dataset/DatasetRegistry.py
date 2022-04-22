from src.dataset.FFHQDataset import FFHQDataset, FFHQ_DEF_NAME
from src.dataset.Dataset import Dataset
from src.core.Registry import Registry


class DatasetRegistry(Registry):
    """
    Static class implementing the abstract Registry class
    used for initialization and storing of all subclasses of the
    Dataset class.

    * If more Datasets are implemented they must be manually added
    to the internal storage (`_DATASETS`) of this class.
    """

    _DATASETS = {FFHQ_DEF_NAME: FFHQDataset}

    @staticmethod
    def get_names() -> list[str]:
        """
        Returns all the names (keys) in the registry.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """
        return list(DatasetRegistry._DATASETS.keys())

    @staticmethod
    def get_resource(name: str) -> type[Dataset]:
        """
        Returns a dataset with the given `name` from the registry.

        Args:
            name (str): Name of the dataset.

        Returns:
            Dataset: Dataset with the given `name`.
        """
        return DatasetRegistry._DATASETS[name]

    @staticmethod
    def get_resources() -> list[type[Dataset]]:
        return list(DatasetRegistry._DATASETS.values())
