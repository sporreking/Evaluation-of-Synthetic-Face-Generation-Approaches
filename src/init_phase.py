from src.environment.EnvironmentManager import EnvironmentManager as EM
from pathlib import Path
import os


def init_phase():
    """
    Create conda environments and download resources.
    """
    # Initialize environments
    for environment_name in _get_environment_names():
        EM.setup(environment_name)

    # Download resources and setup directories
    os.system("python environment/download.py")
    os.system("python dataset/download.py")


def _get_environment_names():
    pathlist = Path("environment").glob("env_*.yml")
    environment_names = []
    for path in pathlist:
        # Remove prefix and ext.
        environment_names.append(path.stem[4:])
    return environment_names
