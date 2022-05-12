from src.dataset.FFHQDataset import FFHQDataset, FFHQ_NAME
from src.dataset.Dataset import Dataset
from src.core.Registry import Registry


class DatasetRegistry(Registry[type[Dataset]]):
    """
    Static class implementing the abstract Registry class
    used for initialization and storing of all subclasses of the
    Dataset class.

    * If more Datasets are implemented they must be manually added
    to the internal storage (`_DATASETS`) of this class. Available
    resolution must also be added per dataset (to `_RESOLUTIONS`).
    """

    _DATASETS = {FFHQ_NAME: FFHQDataset}
    _RESOLUTIONS = {FFHQ_NAME: [256]}

    @staticmethod
    def get_names() -> list[str]:
        return list(DatasetRegistry._DATASETS.keys())

    @staticmethod
    def get_resource(name: str) -> type[Dataset]:
        return DatasetRegistry._DATASETS[name]

    @staticmethod
    def get_resources() -> list[type[Dataset]]:
        return list(DatasetRegistry._DATASETS.values())

    @staticmethod
    def get_available_resolutions(name: str) -> list[int]:
        """
        Returns available resolution for the dataset associated with given `name`.

        Args:
            name (str): Name of the dataset.

        Returns:
            list[int]: The available resolutions of the specified dataset.
        """
        return DatasetRegistry._RESOLUTIONS[name]
