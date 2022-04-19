from src.controller.Controller import Controller
from src.generator.Generator import Generator
import numpy as np

IDENTITY_NAME = "IDENTITY"


class IdentityController(Controller):
    """
    Subclass to Controller, implements the identity controller.

    Note that the identity controller is not a real controller in the sense that
    it does not do anything to the latent codes. Its just a wrapper around
    the generator.
    """

    def __init__(self, gen: Generator):
        """
        Constructor to the IdentityController

        Args:
            gen (Generator): The generator associated with the controller.

        """
        super().__init__(IDENTITY_NAME, gen)

    def is_ready(self) -> bool:
        return True

    def is_setup_dependent_on_generator(self) -> bool:
        return False

    def setup(self, mode: str = None) -> None:
        """
        No setup required. Not used in the identity controller.
        """
        pass

    def parse_native_input(self, input: dict[str, np.ndarray] = None) -> None:
        """
        No parsing required. Not used in the identity controller.
        """
        pass

    def generate_native(
        self, latent_codes: np.ndarray, native_input: np.ndarray = None
    ) -> list[str]:
        """
        Generates the latent codes using the associated generator ´self._gen´.

        Note that no modifications are made to the latent codes and `native_input` is ignored.

        Args:
            latent_codes (np.ndarray): Latent codes to be generated.
            native_input (np.ndarray, optional): Not used in the identity controller. Defaults to None.

        Returns:
            list[str]: URIs for the generated images.
        """
        # Generate images based solely on the latent codes
        return self._gen.generate(latent_codes)