from src.generator.Generator import Generator
from src.dataset.Dataset import Dataset
from src.environment.EnvironmentManager import EnvironmentManager as EM

from typing import List
from os import path

import numpy as np

STYLEGAN2ADA_NAME = "stylegan2ada"


class StyleGAN2ADAGenerator(Generator):
    _DIM_Z = 512
    _PATH_TO_IMAGES = "environment/stylegan2ada/out/"
    _IMAGE_EXT = ".png"
    _W_FILE = "w.npy"

    def __init__(self, dataset: Dataset):
        super().__init__(STYLEGAN2ADA_NAME, dataset)

    def latent_space_std(self) -> np.ndarray:
        return np.ones(self._DIM_Z)

    def latent_space_mean(self) -> np.ndarray:
        return np.zeros(self._DIM_Z)

    def _launch_generator(self, latent_codes: np.ndarray, mode: str):

        # Send codes via socket
        EM.send_latent_codes(latent_codes)

        # Run StyleGAN2-ADA
        if not EM.run(self.get_name(), mode=mode):
            raise RuntimeError("Agent failed! Could not generate images.")

    def preprocess_latent_code(self, latent_codes: np.ndarray) -> np.ndarray:
        """
        Performs preprocessing of `latent_codes` such that they may be used by
        the `generate()` method.

        For StyleGAN2-ADA, every latent code in `latent_codes` is expected to be in the
        Z-space, and the output of this method will be their projection onto the W-domain.

        Args:
            latent_codes (np.ndarray): The latent codes to project, specified
                on a per-row basis. The values should be from the Z-domain, preferably
                drawn from the distribution described by `latent_space_std()` and
                `latent_space_mean()`.

        Returns:
            np.ndarray: The projection of the latent codes onto W, specified on a per-row basis.

        Raises:
            FileNotFoundError: If the Z -> W projection output file was not found.
        """

        # Project latent codes
        self._launch_generator(latent_codes, "p")

        # Check if successful
        projection_output_file = self._PATH_TO_IMAGES + self._W_FILE
        if not path.isfile(projection_output_file):
            raise FileNotFoundError(
                f"Could not find projection output file: '{projection_output_file}'"
            )

        # Load projection
        return np.load(projection_output_file)

    def generate(self, latent_codes: np.ndarray) -> List[str]:
        """
        Generate images for the specified latent codes.

        Every latent code in `latent_codes` is expected to be in the W-domain.

        Args:
            latent_codes (np.ndarray): The latent codes to generate images for. Make
                sure that the codes have been properly preprocessed with
                `preprocess_latent_code()` method, such that they are in W-space. Note
                that `random_latent_code()` performs preprocessing automatically, i.e.,
                the output from that method MUST NOT be processed again.

        Returns:
            list[str]: A list of URIs to the generate images, in the same order as the
                specified latent code input.

        Raises:
            FileNotFoundError: If any of the expected image output files were not found.
        """

        # Generate images from latent codes (assuming W-domain)
        self._launch_generator(latent_codes, "w")

        # Construct list of URIs
        uris = [
            self._PATH_TO_IMAGES + str(i) + self._IMAGE_EXT
            for i in range(latent_codes.shape[0])
        ]
        if any(not path.isfile(uri) for uri in uris):
            raise FileNotFoundError("Could not find image(s) from Generator!")
        else:
            return uris
