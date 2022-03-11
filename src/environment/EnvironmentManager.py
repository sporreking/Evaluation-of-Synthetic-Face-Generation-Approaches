from pathlib import Path
from os import system


class EnvironmentManager:
    """
    Static class used for creating conda environments, and
    for launching system processes in them.
    """

    ENV_ROOT_DIR = Path("environment/")
    AGENT_FILE_PREFIX = "agent"
    ENV_FILE_PREFIX = "env"
    CONDA_ENV_PREFIX = "sfg"

    @staticmethod
    def setup(env_name: str) -> bool:
        """
        This function will look for a `environment/env_{env_name}.yml` file describing
        a conda environment. The environment will then be created on the system.

        Args:
            env_name (str): The name of the environment to load.

        Raises:
            FileNotFoundError: If the environment file could not be found.

        Returns:
            bool: `True` if the environment was successfully created, or `False` if the
                environment already exists or if an error occurred.
        """

        # Find the environment file
        env_file = EnvironmentManager.ENV_ROOT_DIR / Path(
            f"{EnvironmentManager.ENV_FILE_PREFIX}_{env_name}.yml"
        )

        # Sanity check
        if not env_file.is_file():
            raise FileNotFoundError(f"Could not find the environment file: {env_file}")

        # Start the system process
        print(f"======== creating environment '{env_name}' ========")
        res = system(
            f"conda env create -f {env_file} -n {EnvironmentManager.CONDA_ENV_PREFIX}_{env_name}"
        )
        print(f"==============================={'=' * len(env_name)}==========")

        return res == 0

    @staticmethod
    def run(env_name: str, *args, **kwargs) -> int:
        """
        This function will start the agent of the specified environment in a new system process.
        For this to work, there must be a file `environment/agent_{env_name}.py` which will
        be launched in the specified conda environment. Note that the environment must be
        created through `EnvironmentManager.setup(env_name)` first.

        All trailing arguments and keyworks arguments will be passed to the agent in a CLI format.

        Args:
            env_name (str): The name of the environment to run.

        Raises:
            FileNotFoundError: If the agent file could not be found.

        Returns:
            bool: `True` if the agent process exited successfully.
        """

        # Find agent file
        agent_file = EnvironmentManager.ENV_ROOT_DIR / Path(
            f"{EnvironmentManager.AGENT_FILE_PREFIX}_{env_name}.py"
        )

        # Sanity check
        if not agent_file.is_file():
            raise FileNotFoundError(f"Could not find the agent file: {agent_file}")

        # Compile arguments
        args_str = " ".join([str(arg) for arg in args])
        kwargs_str = " ".join([f"--{str(k)} '{str(v)}'" for k, v in kwargs.items()])

        # Start the conda process
        print(f"======== {agent_file} ========")
        res = system(
            f"conda run -n {EnvironmentManager.CONDA_ENV_PREFIX}_{env_name} "
            + f"python {agent_file} {args_str} {kwargs_str}"
        )
        print(f"========={'=' * len(str(agent_file))}=========")

        return res == 0