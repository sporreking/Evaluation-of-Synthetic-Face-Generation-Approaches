import abc
from src.generator import Generator
import numpy as np
from typing import Any, Dict, List


class Controller(metaclass=abc.ABCMeta):
    """
    An abstract implementation of a controller.
    """

    def __init__(self, name: str, gen: Generator):
        """
        Constructs a new controller.

        Args:
            name (str): The name of the controller.
            gen (Generator): The generator associated with the controller.
        """
        self._name = name
        self._gen = gen

    def get_name(self) -> str:
        """
        Returns the name of this controller.

        Returns:
            str: The name of this controller.
        """
        return self._name

    def get_generator(self) -> str:
        """
        Returns the generator of this controller.

        Returns:
            str: The generator of this controller.
        """
        return self._gen

    @abc.abstractmethod
    def is_ready() -> bool:
        """
        Should return True if setup is ready.

        Returns:
            bool: True if setup is done.
        """

    @abc.abstractmethod
    def is_setup_dependent_on_generator() -> bool:
        """
        Should return True if setup is dependent on generator.

        Returns:
            bool: True if setup is dependent on the generator.
        """

    @abc.abstractmethod
    def setup(self) -> None:
        """
        Should setup the controller auxiliary models and necessary files.
        """
        pass

    @abc.abstractmethod
    def parse_native_input(self, input: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Should parse input from standardized representation to native representation.

        Standardized representation refers to the float range of [-1,1].

        For example:

        Let input be defined as:
        `input_example = {"gender" : np.ndarray[-1, 1, -0.5], "age": np.ndarray[1,0,-1]}`
        then input for image 1 should then be parsed as an image with gender = -1
        and age = 1. Thus, `input example` defines parameters for 3 images (length of the array).

        Note that a value of 0 means that no changes for that parameter will be done for that image.

        Args:
            input (dict[str, np.ndarray]): Standardized input representation.
                Where key = parameter name, and the value = array of values
                (in float range of [-1,1]) of that parameter. Each value in the
                array represents one generated image. For example, image 1 should
                be based on value indexed 1 in the array for each parameter in input.keys().

        Returns:
            np.ndarray[dict[str, Any]]: 2D array containing dictionary with
                parameter names and their native values. Axis 0 corresponds to the number
                of images to generate/manipulate, while axis 1 denotes different sequential
                manipulation steps that should be applied. Each entry in the array
                corresponds to a single manipulation specified by a dictionary, with parameters
                on native format.
        """
        pass

    @abc.abstractmethod
    def generate_native(
        self, latent_codes: np.ndarray, native_input: np.ndarray
    ) -> List[str]:
        """
        Should manipulate `latent_codes` according to `native_input` and then generate the manipulated images
        given the manipulated latent codes. Generation should be done by calling the associated Generator.

        Args:
            latent_codes (np.ndarray): Latent codes to be manipulated according to `native_input` and
                then generated. Axis 0 in `latent_codes` corresponds to the number of images to be
                generated and should coincide length-wise with Axis 0 in `native_input`. For example:

                `latent_codes[0,:]` is latent code z0 and `native_input[0,:]` then defines the desired
                manipulations to that latent code.

            native_input (np.ndarray[dict[str, Any]]): 2D array containing dictionary with
                parameter names and their native values. Axis 0 corresponds to the number
                of images to generate/manipulate, while axis 1 denotes different sequential
                manipulation steps that should be applied. Each entry in the array
                corresponds to a single manipulation specified by a dictionary, with parameters
                on native format.

        Returns:
            list[str]: URIs for the generated images.
        """
