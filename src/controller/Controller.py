import abc
from src.generator.Generator import Generator
import numpy as np
from typing import Any, Dict, List, Union, Tuple
import random
from src.core.Setupable import Setupable


class Controller(Setupable, metaclass=abc.ABCMeta):
    """
    An abstract implementation of a controller.
    """

    def __init__(self, name: str, gen: Generator, attributes: list[str] = None):
        """
        Constructs a new Controller.

        Args:
            name (str): The name of the controller.
            gen (Generator): The generator associated with the controller.
            attributes (list[str], optional): Attributes from the dataset to be used.
                The dataset is derived from the generator. May be set to `None` if the
                controller is unsupervised. Default is None.
        """

        # Check attribute names
        if attributes is not None and any("_" in a for a in attributes):
            raise ValueError("Attribute names may not contain underscores!")

        self._name = name
        self._gen = gen
        self._ds = gen.get_dataset()
        self._attrs = attributes

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

    def sample_random_input(
        self, n: int, attr_sample_mode: Union[int, str]
    ) -> Dict[str, np.ndarray]:
        """
        Samples input from a uniform distribution.

        Args:
            n (int): Number of samples to generate standardized inputs for.
            attr_sample_mode (Union[int, str]): The number of random attributes to choose, or "random"
                which randomizes the number of random attributes for each sample.

        Returns:
            Dict[str,np.ndarray]: `n` samples in standardized representation to be used
                in as input to `parse_native_input()`.
        """
        num_attr = len(self._attrs)
        columns = []
        for i in range(n):
            # Define what attributes to use for this column
            if attr_sample_mode == "random":
                attrs_subset = random.sample(
                    self._attrs, np.random.randint(0, num_attr)
                )
            elif type(attr_sample_mode) == int:
                assert attr_sample_mode <= num_attr
                attrs_subset = random.sample(self._attrs, num_attr)

            # Generate column input (1 sample)
            columns.append(
                [
                    np.random.uniform(-1, 1) if attr in attrs_subset else 0
                    for attr in self._attrs
                ]
            )

        # Reshape and create dict
        arr = np.array(columns).T
        return {attr: arr[i, :].reshape(n) for attr in enumerate(self._attrs)}

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
    ) -> Tuple[List[str], np.ndarray]:
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
            Tuple[List[str], np.ndarray]: URIs for the generated images and the manipulated latent codes.
        """
