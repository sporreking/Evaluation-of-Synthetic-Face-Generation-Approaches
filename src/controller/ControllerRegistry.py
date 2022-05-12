from src.generator.GeneratorRegistry import GeneratorRegistry
from src.generator.UNetGANGenerator import UNETGAN_NAME
from src.generator.StyleGAN2ADAGenerator import STYLEGAN2ADA_NAME
from src.generator.StyleSwinGenerator import STYLESWIN_NAME
from src.controller.ASADController import ASADController, ASAD_NAME
from src.controller.IdentityController import IdentityController, IDENTITY_NAME
from src.controller.InterFaceGANController import (
    InterFaceGANController,
    INTERFACEGAN_NAME,
)
from src.controller.Controller import Controller
from typing import List
from src.core.Registry import Registry


class ControllerRegistry(Registry[type[Controller]]):
    """
    Static class used for initialization and storing of all subclasses of the
    Controller class.

    * If more controllers are implemented they should be added to the internal
    storage (`_CONTROLLERS`) of this class. If a controller is only compatible
    for certain generators, these should be added to `_COMPATIBLE_GENERATORS`.
    """

    _CONTROLLERS = {
        IDENTITY_NAME: IdentityController,
        ASAD_NAME: ASADController,
        INTERFACEGAN_NAME: InterFaceGANController,
    }

    _COMPATIBLE_GENERATORS = {  #! Omit controllers if all generators are compatible
        ASAD_NAME: [
            UNETGAN_NAME,
            STYLEGAN2ADA_NAME,
            STYLESWIN_NAME,
        ]
    }

    @staticmethod
    def get_names() -> List[str]:
        return list(ControllerRegistry._CONTROLLERS.keys())

    @staticmethod
    def get_resource(name: str) -> type[Controller]:
        return ControllerRegistry._CONTROLLERS[name]

    @staticmethod
    def get_resources() -> List[type[Controller]]:
        return list(ControllerRegistry._CONTROLLERS.values())

    @staticmethod
    def get_compatible_generator_names(name: str) -> list[str]:
        """
        Returns all compatible generators for the controller associated with given `name`.

        Args:
            name (str): Name of the controller.

        Returns:
            list[int]: The names of all compatible generators.
        """
        return (
            ControllerRegistry._COMPATIBLE_GENERATORS[name]
            if name in ControllerRegistry._COMPATIBLE_GENERATORS
            else GeneratorRegistry.get_names()
        )
