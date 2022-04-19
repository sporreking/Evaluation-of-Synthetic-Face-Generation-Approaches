from src.generator.UNetGANGenerator import UNETGAN_NAME
from src.controller.ASADController import ASADController, ASAD_NAME
from src.controller.IdentityController import IdentityController, IDENTITY_NAME
from src.controller.Controller import Controller
from typing import List, Tuple
from src.core.Registry import Registry


class ControllerRegistry(Registry):
    """
    Static class used for initialization and storing of all subclasses of the
    Controller class.

    * If more controllers (or controller/generator pairs) are implemented
    and compatible they should be added to the internal
    storage (`_CONTROLLERS`) of this class.
    """

    _CONTROLLERS = {
        ASAD_NAME: {UNETGAN_NAME: ASADController},
        IDENTITY_NAME: {UNETGAN_NAME: IdentityController},
    }

    @staticmethod
    def get_names() -> List[Tuple[str]]:
        """
        Returns all the names in the registry.

        Returns:
            List[Tuple[str]]: List of all names (contr, gen) in the registry.
        """
        d = ControllerRegistry._CONTROLLERS
        return [(contr, gen) for contr in d.keys() for gen in d[contr].keys()]

    @staticmethod
    def get_controller_names() -> List[str]:
        """
        Returns all the names (keys) in the registry.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """
        return list(ControllerRegistry._CONTROLLERS.keys())

    @staticmethod
    def get_generator_names(controller_name: str) -> List[str]:
        """
        Returns all the names (keys) in the registry.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """
        return list(ControllerRegistry._CONTROLLERS[controller_name].keys())

    @staticmethod
    def get_resource(name: str, gen_name: str) -> Controller:
        """
        Returns a controller with the given `name` and `gen_name` from the registry.

        Args:
            name (str): Name of the controller.
            gen_name (str): Name of the generator.

        Returns:
            Dataset: controller with the given `name`.
        """
        return ControllerRegistry._CONTROLLERS[name][gen_name]

    @staticmethod
    def get_resources() -> List[Controller]:
        """
        Returns all controllers from the registry.

        Returns:
            list[Controller]: All controllers from the registry.
        """
        return [
            controller
            for controller_d in ControllerRegistry._CONTROLLERS.values()
            for controller in controller_d.values()
        ]
