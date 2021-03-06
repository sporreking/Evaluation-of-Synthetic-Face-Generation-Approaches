from pathlib import Path
from typing import Any, Callable, Generator, Union


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

    def has_file(self, file: Union[str, Path]) -> bool:
        """
        Checks whether the specified `file` exists in the root
        directory of this file jar or not.

        Args:
            file (Union[str, Path]): The file to look for.

        Returns:
            bool: `True` if the file exists, else `False`.
        """
        return self._root_dir / file in self.iterdir()

    def get_root_dir(self) -> Path:
        """
        Returns the root directory of this file jar.

        Returns:
            Path: The root directory of this file jar.
        """
        return self._root_dir

    def get_file(self, name: str, load_func: Callable[[Path], Any]) -> Any:
        """
        Return the file associated with the given `name` using `load_func`.
        Note that if an exception is raised in `load_func`, `None` will be
        returned instead.

        Args:
            name (str): Name of the file to retrieve.
            load_func (function): Function used to load the file, only argument must be
                the path where the file is located.

        Returns:
            Any: Return of `load_func`, or `None` if an exception is raised from it.
        """
        try:
            return load_func(self._root_dir / name)
        except Exception as e:
            return None

    def store_file(self, name: str, save_func: Callable[[Path], Any]) -> Any:
        """
        Store the file `name` in this FileJar's root directory using
        the provided `save_func`.

        Args:
            name (str): File name of the file to store.
            save_func (function): Function used to save the file, only argument must be
                the path where the file should be stored.

        Raises:
            Exception: Potential propagation from `save_func`.

        Returns:
            Any: Return of `save_func`.
        """

        # Append name to root directory
        save_path = self._root_dir / name

        # Save file
        return save_func(save_path)

    def iterdir(self) -> Generator[Path, None, None]:
        """
        Returns an iterator over all files in this FileJar's root directory.

        Returns:
            Generator[Path, None, None]: An iterator over all files in the root directory.
        """
        return self._root_dir.iterdir()
