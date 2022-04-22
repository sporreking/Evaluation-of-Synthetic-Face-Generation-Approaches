from src.generator.Generator import Generator
from src.dataset.Dataset import Dataset
import numpy as np
from src.environment.EnvironmentManager import EnvironmentManager as em
from typing import List
from os import path

UNETGAN_NAME = "unetgan"


class UNetGANGenerator(Generator):
    _DIM_Z = 128
    _PATH_TO_IMAGES = "environment/unetgan/out/sample/"
    _IMAGE_EXT = ".png"

    def __init__(self, dataset: Dataset):
        """
        Constructs a new UNetGANGenerator.

        Args:
            dataset (Dataset): The dataset to associate the generator with.
        """
        super().__init__(UNETGAN_NAME, dataset)

    def latent_space_std(self) -> np.ndarray:
        """
        Return the standard deviation of each dimension in the latent space.

        Returns:
            np.ndarray: The standard deviations of the latent space distribution.
        """
        return np.ones(self._DIM_Z)

    def latent_space_mean(self) -> np.ndarray:
        """
        Return the mean of each dimension in the latent space.

        Returns:
            np.ndarray: The means of the latent space distribution.
        """
        return np.zeros(self._DIM_Z)

    def generate(self, latent_codes: np.ndarray) -> List[str]:
        """
        Generate images for the specified latent codes. This will most likely
        involve the launch of a separate agent process using the EnvironmentManager,
        allowing external generators to be launched in appropriate conda environments.

        Args:
            latent_codes (np.ndarray): The latent codes to generate images for.

        Returns:
            list[str]: A list of URIs to the generate images, in the same order as the
                specified latent code input.
        """
        # Send codes via socket
        em.send_latent_codes(latent_codes)

        # Run U-NetGAN
        em.run(UNETGAN_NAME)

        # Construct list of URIs
        uris = [
            self._PATH_TO_IMAGES + str(i) + self._IMAGE_EXT
            for i in range(latent_codes.shape[0])
        ]

        # Confirm that images have been generated
        if any([not path.isfile(uri) for uri in uris]):
            raise FileNotFoundError("Could not find image(s) from Generator!")
        else:
            return uris
