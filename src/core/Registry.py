import abc
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T], metaclass=abc.ABCMeta):
    """
    Abstract class outlining the general interface for different types
    of registries of resources (Datasets, Generators, Controllers etc).
    """

    def __new__(cls):
        raise RuntimeError(f"{cls.__name__} should not be instantiated.")

    @staticmethod
    @abc.abstractmethod
    def get_names() -> list[str]:
        """
        Returns all the names (keys) in the registry. The names may be used to access
        resources through `get_resource(name)`.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """

    @staticmethod
    @abc.abstractmethod
    def get_resource(name: str) -> T:  # TODO: REMOVE NESTED SHIT
        """
        Returns the resource associated with the given `name` from the registry.

        Args:
            name (str): Name of the resource to fetch.

        Returns:
            T: The resource associated with the given `name`.
        """

    @staticmethod
    @abc.abstractmethod
    def get_resources() -> list[T]:
        """
        Returns all resources from the registry.

        Returns:
            list[T]: All resources from the registry.
        """
