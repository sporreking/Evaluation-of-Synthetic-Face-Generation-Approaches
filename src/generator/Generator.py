import abc
import numpy as np


class Generator(metaclass=abc.ABCMeta):
    """
    An abstract implementation of a generator.
    """

    def __init__(self, name: str):
        """
        Constructs a new generator.

        Args:
            name (str): The name of the generator.
        """
        self._name = name

    def get_name(self) -> str:
        """
        Returns the name of this generator.

        Returns:
            str: The name of this generator.
        """
        return self._name

    def random_latent_code(self) -> np.ndarray:
        """
        Samples a random latent code based on the statistics of this generator.

        Raises:
            ValueError: If the dimensions of `latent_space_mean()` and
                `latent_space_std()` do not match.

        Returns:
            np.ndarray: The new random latent code.
        """
        mean = self.latent_space_mean()
        std = self.latent_space_std()

        # Sanity check
        if mean.shape != std.shape:
            raise ValueError(
                "The number of means and standard deviations do not match! "
                + "The implementaiton of this generator is faulty."
            )

        return np.random.normal(loc=mean, scale=std)

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
    def generate(self, latent_codes: list[np.ndarray]) -> list[str]:
        """
        Should generate images for the specified latent codes. This will most likely
        involve the launch of a separate agent process using the EnvironmentManager,
        allowing external generators to be launched in appropriate conda environments.

        Args:
            latent_codes (list[np.ndarray]): The latent codes to generate images for.

        Returns:
            list[str]: A list of URIs to the generate images, in the same order as the
                specified latent code input.
        """
        pass
