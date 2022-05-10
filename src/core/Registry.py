import abc
from typing import Any, Union, Tuple


class Registry(metaclass=abc.ABCMeta):
    """
    Abstract class outlining the general interface for different types
    of registries of resources (Datasets, Generators, Controllers etc).
    """

    def __new__(cls):
        raise RuntimeError(f"{cls.__name__} should not be instantiated.")

    @staticmethod
    @abc.abstractmethod
    def get_names() -> list[Union[str, Tuple]]:
        """
        Returns all the names (keys) in the registry. In the case of
        nestled resource storing structure, the multiple keys are stored in
        tuples of strings instead of string. The names may be used to access
        resources through the function `get_resource`.

        Returns:
            list[Union[str, Tuple]]: List of all names (keys) in the registry. Tuple if
                nestled resource storing structure, e.g., [("ASAD", "unetgan"),...]
                such that resource can be accessed using  `get_resource(("ASAD", "unetgan"))`.
        """

    @staticmethod
    @abc.abstractmethod
    def get_resource(name: Union[str, Tuple[str]]) -> Any:
        """
        Returns a resource with the given `name` from the registry.

        For more information see the documentation of `get_names()`.

        Args:
            name (Union[str, Tuple[str]]): Name of the dataset. Tuple of strings if nestled
                storing structure.

        Returns:
            Dataset: Dataset with the given `name`.
        """

    @staticmethod
    @abc.abstractmethod
    def get_resources() -> list[Any]:
        """
        Returns all resources from the registry.

        Returns:
            list[Any]: All resources from the registry.
        """
