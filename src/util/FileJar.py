from pathlib import Path
from typing import Any


class FileJar:
    """
    FileJar is a class that allows you to save/load files using a
    with the help of a dictionary mapping, managed by the FileJar.

    Raises:
        FileNotFoundError: During init if the directory does not exist.
    """

    _root_dir: Path
    _jar: dict

    def __init__(self, root_dir: Path, jar: dict = None):
        """
        Constructor for a new FileJar.

        Args:
            root_dir (Path): Path object to the root directory
                associated with the new FileJar.
            jar (dict, optional): A dictionary containing name and save_paths.
                Defaults to None.

        Raises:
            FileNotFoundError: If `root_dir` is not a directory.
        """
        # Check that root_dir is a directory.
        if root_dir.is_dir():
            self._root_dir = root_dir
        else:
            raise FileNotFoundError(f"Directory {root_dir} not found!")

        # Init jar
        if jar is None:
            self._jar = dict()
        else:
            self._jar = jar

    def get_file(self, name: str, load_func) -> Any:
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
            return load_func(self._jar[name])
        except Exception as e:
            print(e)
            return None

    def store_file(self, name: str, save_func) -> Any:
        """
        Store the file `name` in `self.root_dir` using the provided `save_func`.

        Args:
            name (str): _description_
            save_func (function): Function used to save the file, only argument must be
                the path where the file should be stored.

        Returns:
            Any: Return of `save_func`.
        """

        # Append name to root directory
        save_path = self._root_dir / Path(name)

        # Add file to jar
        self._jar[name] = save_path

        # Save file
        try:
            return save_func(save_path)
        except Exception as e:
            print(e)
            return None
