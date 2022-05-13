from src.controller.Controller import Controller
from src.generator.Generator import Generator
from src.core.Setupable import SetupMode
import numpy as np
from typing import List, Tuple

IDENTITY_NAME = "IDENTITY"


class IdentityController(Controller):
    """
    Subclass to Controller, implements the identity controller.

    Note that the identity controller is not a real controller in the sense that
    it does not do anything to the latent codes. Its just a wrapper around
    the generator.
    """

    def __init__(self, gen: Generator, attributes: list[str] = None):
        """
        Constructs a new IdentityController

        Args:
            gen (Generator): The generator associated with the controller.
            attributes (list[str], optional): Used for calculating metrics.
                Not used for image generation. Default is None.

        """
        super().__init__(IDENTITY_NAME, gen)

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        return {}

    def parse_native_input(self, input: dict[str, np.ndarray] = None) -> None:
        """
        No parsing required. Not used in the identity controller.
        """
        pass

    def generate_native(
        self, latent_codes: np.ndarray, native_input: np.ndarray = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Generates the latent codes using the associated generator `self._gen`.

        Note that no modifications are made to the latent codes and `native_input` is ignored.

        Args:
            latent_codes (np.ndarray): Latent codes to be generated.
            native_input (np.ndarray, optional): Not used in the identity controller. Defaults to None.

        Returns:
            Tuple[List[str], np.ndarray]: URIs for the generated images and the latent codes (not manipulated in this case)
        """
        # Generate images based solely on the latent codes
        return self._gen.generate(latent_codes), latent_codes
