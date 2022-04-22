import abc
import numpy as np
from typing import List

from src.dataset.Dataset import Dataset


class Generator(metaclass=abc.ABCMeta):
    """
    An abstract implementation of a generator.
    """

    def __init__(self, name: str, dataset: Dataset):
        """
        Constructs a new generator.

        Args:
            name (str): The name of the generator.
            dataset (Dataset): The dataset to associate the generator with.
        """
        self._name = name
        self._dataset = dataset

    def get_name(self) -> str:
        """
        Returns the name of this generator.

        Returns:
            str: The name of this generator.
        """
        return self._name

    def get_dataset(self) -> Dataset:
        """
        Returns the dataset associated with this generator.

        Returns:
            Dataset: the dataset associated with this generator.
        """
        return self._dataset

    def random_latent_code(self, n: int = 1) -> np.ndarray:
        """
        Samples `n` random latent codes based on the statistics of this generator.

        Raises:
            ValueError: If the dimensions of `latent_space_mean()` and
                `latent_space_std()` do not match.

        Args:
            n (int): The number of latent codes to sample.
                Default to 1.

        Returns:
            np.ndarray: The new random latent code.
        """
        mean = self.latent_space_mean()
        std = self.latent_space_std()

        # Sanity check
        if mean.shape != std.shape:
            raise ValueError(
                "The number of means and standard deviations do not match! "
                + "The implementation of this generator is faulty."
            )

        return np.random.normal(loc=mean, scale=std, size=(n, std.shape[0]))

    @abc.abstractmethod
    def latent_space_std(self) -> np.ndarray:
        """
        Should return the standard deviation of each dimension in the latent space.
        The distribution is assumed to be Gaussian.

        Returns:
            np.ndarray: The standard deviations of the latent space distribution.
        """
        pass

    @abc.abstractmethod
    def latent_space_mean(self) -> np.ndarray:
        """
        Should return the mean of each dimension in the latent space.
        The distribution is assumed to be Gaussian.

        Returns:
            np.ndarray: The means of the latent space distribution.
        """
        pass

    @abc.abstractmethod
    def generate(self, latent_codes: np.ndarray) -> List[str]:
        """
        Should generate images for the specified latent codes. This will most likely
        involve the launch of a separate agent process using the EnvironmentManager,
        allowing external generators to be launched in appropriate conda environments.

        Args:
            latent_codes (np.ndarray): The latent codes to generate images for.

        Returns:
            list[str]: A list of URIs to the generate images, in the same order as the
                specified latent code input.
        """
        pass
