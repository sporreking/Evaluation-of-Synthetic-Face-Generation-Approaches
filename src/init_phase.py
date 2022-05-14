from src.environment.EnvironmentManager import EnvironmentManager as EM
from src.dataset.DatasetRegistry import DatasetRegistry
from src.dataset.Dataset import Dataset
from pathlib import Path
import os


def init_phase() -> None:
    """
    Create conda environments and download resources.
    """
    # Initialize environments
    for environment_name in _get_environment_names():
        EM.setup(environment_name)

    # Download resources and setup directories
    os.system("python environment/download.py")
    os.system("python dataset/download.py")


def init_done() -> bool:
    """
    Check if this phase is ready enough for the next phase to be executed.

    Returns:
        bool: True if this phase is satisfied, otherwise False.
    """
    return all(  # Check environment setup
        (
            EM.is_setup(environment_name)
            and (EM.ENV_ROOT_DIR / environment_name).is_dir()
        )
        for environment_name in _get_environment_names()
    ) and all(  # Check dataset setup
        all(directory.is_dir() for directory in directories)
        for directories in [
            (
                Dataset.DS_DIR_PREFIX / f"{ds_name}_LABELS",
                *(
                    Dataset.DS_DIR_PREFIX / f"{ds_name}_{res}"
                    for res in DatasetRegistry.get_available_resolutions(ds_name)
                ),
            )
            for ds_name in DatasetRegistry.get_names()
        ]
    )


def _get_environment_names() -> list[str]:
    pathlist = Path("environment").glob("env_*.yml")
    environment_names = []
    for path in pathlist:
        # Remove prefix and ext.
        environment_names.append(path.stem[4:])
    return environment_names
