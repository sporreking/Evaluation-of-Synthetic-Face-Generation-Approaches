from pathlib import Path
from sys import platform
from os import system
import socket
import struct
import numpy as np
import threading

DEFAULT_LATENT_CODE_PORT = 6969


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
    def run(agent_name: str, env_name: str = None, *args, **kwargs) -> bool:
        """
        This function will start the specified agent(named `agent_name`) of the specified
        environment in a new system process.For this to work,
        there must be a file `environment/agent_{agent_name}.py` which will be launched in
        the specified conda environment. Note that the environment must be created through
        `EnvironmentManager.setup(env_name)` first.

        All trailing arguments and keyworks arguments will be passed to the agent in a CLI format.

        Args:
            agent_name (str): The name of the file to run.
            env_name (str): The name of the environment to run. If env_name is None, `env_name` is
                set to `agent_name`. Defaults to None.
        Raises:
            FileNotFoundError: If the agent file could not be found.

        Returns:
            bool: `True` if the agent process exited successfully.
        """
        if env_name is None:
            env_name = agent_name

        # Find agent file
        agent_file = EnvironmentManager.ENV_ROOT_DIR / Path(
            f"{EnvironmentManager.AGENT_FILE_PREFIX}_{agent_name}.py"
        )

        # Sanity check
        if not agent_file.is_file():
            raise FileNotFoundError(f"Could not find the agent file: {agent_file}")

        # Compile arguments
        args_str = " ".join([str(arg) for arg in args])
        kwargs_str = " ".join([f'--{str(k)} "{str(v)}"' for k, v in kwargs.items()])

        # Start the conda process
        print(f"======== {agent_file} ({platform}) ========")
        res = None
        if platform == "win32":
            res = system(
                f"conda activate {EnvironmentManager.CONDA_ENV_PREFIX}_{env_name} && "
                + f"python {agent_file} {args_str} {kwargs_str}"
            )
        else:
            res = system(
                f"conda run -n {EnvironmentManager.CONDA_ENV_PREFIX}_{env_name} "
                + "--no-capture-output --live-stream "
                + f"python {agent_file} {args_str} {kwargs_str}"
            )
        print(f"========={'=' * (len(str(agent_file))+len(platform))}============")

        return res == 0

    @staticmethod
    def send_latent_codes(latent_codes: np.ndarray) -> None:
        """
        Send latent codes through socket via localhost using `DEFAULT_LATENT_CODE_PORT`.

        Args:
            latent_codes (np.ndarray): Latent codes to send.
        """
        threading.Thread(target=_latent_code_agent, args=(latent_codes,)).start()


def _latent_code_agent(latent_codes: np.ndarray) -> None:
    # Flatten and tranform it to list
    num_latent_codes = latent_codes.shape[0]
    latent_codes = latent_codes.flatten().tolist()

    # Setup socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("localhost", DEFAULT_LATENT_CODE_PORT))
    s.listen(1)

    # Accept incoming connections
    conn, addr = s.accept()

    # Send data
    conn.sendall(num_latent_codes.to_bytes(4, "big"))
    data = struct.pack(f"<{len(latent_codes)}d", *latent_codes)
    conn.sendall(data)

    # Close connection
    conn.close()
