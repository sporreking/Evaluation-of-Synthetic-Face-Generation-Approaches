from src.generator.UNetGANGenerator import UNetGANGenerator, UNETGAN_NAME
from src.generator.StyleGAN2ADAGenerator import StyleGAN2ADAGenerator, STYLEGAN2ADA_NAME
from src.generator.StyleSwinGenerator import StyleSwinGenerator, STYLESWIN_NAME
from src.generator.Generator import Generator
from typing import List
from src.core.Registry import Registry


class GeneratorRegistry(Registry):
    """
    Static class implementing the abstract Registry class
    used for initialization and storing of all subclasses of the
    Generator class.

    * If more Generators are implemented they must be manually added
    to the internal storage (`_GENERATORS`) of this class.
    """

    _GENERATORS = {
        UNETGAN_NAME: UNetGANGenerator,
        STYLEGAN2ADA_NAME: StyleGAN2ADAGenerator,
        STYLESWIN_NAME: StyleSwinGenerator,
    }

    @staticmethod
    def get_names() -> List[str]:
        """
        Returns all the names (keys) in the registry.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """
        return list(GeneratorRegistry._GENERATORS.keys())

    @staticmethod
    def get_resource(name: str) -> type[Generator]:
        """
        Returns a generator with the given `name` from the registry.

        Args:
            name (str): Name of the generator.

        Returns:
            Generator: generator with the given `name`.
        """
        return GeneratorRegistry._GENERATORS[name]

    @staticmethod
    def get_resources() -> List[type[Generator]]:
        return list(GeneratorRegistry._GENERATORS.values())
