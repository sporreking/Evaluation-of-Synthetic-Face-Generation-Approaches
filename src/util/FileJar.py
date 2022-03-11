from pathlib import Path
from typing import Any, Callable


class FileJar:
    """
    FileJar is a class that allows you to save/load files using a
    with the help of a dictionary mapping, managed by the FileJar.

    Raises:
        FileNotFoundError: During init if the directory does not exist.
    """

    def __init__(self, root_dir: Path, create_root_dir: bool = False):
        """
        Constructor for a new FileJar.

        Args:
            root_dir (Path): Path object to the root directory
                associated with the new FileJar.
            create_root_dir (bool): If `True`, this FileJar's root directory will
                be created if non-existent.

        Raises:
            FileNotFoundError: If `root_dir` is not a directory.
        """
        # Check that root_dir is a directory.
        if not root_dir.is_dir():
            if create_root_dir and not root_dir.exists():
                root_dir.mkdir(parents=True)
            else:
                raise FileNotFoundError(f"Directory {root_dir} not found!")

        self._root_dir = root_dir

    def get_file(self, name: str, load_func: Callable[[Path], Any]) -> Any:
        """
        Return the file associated with the given `name` using `load_func`.

        Args:
            name (str): Name of the file to retrieve.
            load_func (function): Function used to load the file, only argument must be
                the path where the file is located.

        Returns:
            Any: Return of `load_func`.
        """
        try:
            return load_func(self._root_dir / name)
        except Exception as e:
            print(e)
            return None

    def store_file(self, name: str, save_func: Callable[[Path], Any]) -> Any:
        """
        Store the file `name` in this FileJar's root directory using
        the provided `save_func`.

        Args:
            name (str): File name of the file to store.
            save_func (function): Function used to save the file, only argument must be
                the path where the file should be stored.

        Returns:
            Any: Return of `save_func`.
        """

        # Append name to root directory
        save_path = self._root_dir / name

        # Save file
        try:
            return save_func(save_path)
        except Exception as e:
            print(e)
            return None
