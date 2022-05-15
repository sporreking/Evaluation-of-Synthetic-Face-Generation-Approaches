# Inspired by https://github.com/genforce/interfacegan

from src.controller.Controller import Controller
from src.generator.Generator import Generator
import numpy as np
from src.controller.SupportVectorClassifier import (
    get_svc_names,
    get_missing_model_names,
    setup_population,
    setup_auxillary,
    setup_labels,
    train_svc,
    get_svc,
    is_population_ready,
    get_population_name,
    get_population_info,
    is_labels_ready,
)
from src.controller.IdentityController import IdentityController
from src.controller.ASADController import ASADController
from typing import List, Tuple
from sklearn.svm import SVC
from src.core.Setupable import SetupMode
from src.util.AuxUtil import load_aux_best

INTERFACEGAN_NAME = "InterFaceGAN"
STEP_MAX = 3  # * 5 max before bad results in paper
USE_PROJECTED = True

# Setup name constants
SETUP_POPULATION_NAME = "SETUP_POPULATION"
SETUP_LABELS_NAME = lambda attr: f"SETUP_LABELS_{attr}"
SETUP_SVC_PREFIX = "TRAIN_SVC"


class InterFaceGANController(Controller):
    """
    Subclass to Controller, implements InterFaceGAN.
    """

    def __init__(self, gen: Generator, attrs: list[str]):
        """
        Constructor to the InterFaceGAN

        Args:
            gen (Generator): The generator associated with the controller.
            attrs (list[str]): Attributes to be used.
        """
        super().__init__(INTERFACEGAN_NAME, gen, attrs)

        # For saving support vector classifiers
        self._svcs = {}

    def _setup_info_func(self, model_name: str) -> str:
        info = load_aux_best(model_name)
        return (
            f"Best model was trained for {info.epoch + info.batch / info.num_batches_per_epoch:.3f} epochs | "
            + f"Training loss = {info.train_loss:.6f} | Validation loss = {info.valid_loss:.6f}"
        )

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        # To get classifier models
        asad_controller = ASADController(self._gen, self._attrs)

        # Controller associated with generating
        # the population.
        controller = IdentityController(self._gen)

        # QoL
        mn_cls = lambda attr, omit_name=True: asad_controller.get_model_name(
            attr, True, omit_name
        )
        gen_name = self._gen.get_name()

        # TODO: Implement 'improve' functionality

        # Construct modes
        return {
            SETUP_POPULATION_NAME: SetupMode(
                False,
                lambda _: self._setup_pop(),
                lambda pop_name=get_population_name(
                    gen_name, controller.get_name()
                ): is_population_ready(pop_name),
                lambda pop_name=get_population_name(
                    gen_name, controller.get_name()
                ): get_population_info(pop_name),
            ),
            **{
                mn_cls(attr): SetupMode(
                    False,
                    lambda _, batch_size, epochs, attr=attr: self._setup_auxillary_classifier(
                        attr, batch_size, epochs
                    ),
                    lambda attr=attr: load_aux_best(mn_cls(attr, False)) is not None,
                    lambda attr=attr: self._setup_info_func(mn_cls(attr, False)),
                    required_modes=[],
                    batch_size=48,
                    epochs=40,
                )
                for attr in self._attrs
            },
            **{
                SETUP_LABELS_NAME(attr): SetupMode(
                    False,
                    lambda _, batch_size, epochs, attr=attr: self._setup_training_labels(
                        attr, batch_size, epochs
                    ),
                    lambda attr=attr, pop_name=get_population_name(
                        gen_name, controller.get_name()
                    ): is_labels_ready(attr, pop_name),
                    required_modes=[SETUP_POPULATION_NAME, mn_cls(attr)],
                    batch_size=48,
                    epochs=40,
                )
                for attr in self._attrs
            },
            **{
                (SETUP_SVC_PREFIX + "_" + attr): SetupMode(
                    True,
                    lambda _, attr=attr: self._train_svc(attr),
                    lambda attr=attr: get_svc(attr, gen_name, controller.get_name())
                    is not None,
                    required_modes=[SETUP_LABELS_NAME(attr)],
                )
                for attr in self._attrs
            },
        }

    def _setup_pop(self) -> None:
        """
        Setup a population, population used for training SVCs.
        Setup already existing population or if necessary constructs a new one
        and populate it.
        """
        setup_population(IdentityController(self._gen))

    def _setup_auxillary_classifier(
        self, attr: str, batch_size: int = 64, epochs: int = 40
    ) -> None:
        """
        Train auxiliary classifier (borrowed from ASADController) if
        necessary.

        Trained classifiers are saved according to ASADController.
        """
        setup_auxillary(attr, self.get_generator().get_dataset(), batch_size, epochs)

    def _setup_training_labels(
        self, attr: str, batch_size: int = 64, epochs: int = 40
    ) -> None:
        """
        Setup labels given a auxiliary classifier (Will train one if missing.)

        See documentation for `_setup_auxillary_classifier` and the ASADController
        class for more information about the classifier.
        """
        controller = IdentityController(
            self.get_generator()
        )  # Dummy controller, see setup_pop docs

        setup_labels(controller, attr, batch_size, epochs)

    def _train_svc(self, attr: str) -> None:
        """
        Train a support vector classifier to predict attributes on a population.
        """
        controller = IdentityController(
            self._gen
        )  # Dummy controller, see setup_labels docs

        svc = train_svc(controller, attr)

        # Save/Update model
        self._svcs[attr] = svc

    def parse_native_input(self, input: dict[str, np.ndarray]) -> np.ndarray:
        """
        Parse input from standardized representation to native representation.

        Standardized representation refers to the float range of [-1,1].

        For InterFaceGAN, the representation does not coincide with the float range of the native
        representation. Thus, values are transformed from [-1,1] to [`STEP_MAX`,-`STEP_MAX`].


        For example:

        Let input be defined as:
        `input_example = {"gender" : np.ndarray[-1, 1, -0.5], "age": np.ndarray[1,0,-1]}`
        then input for image 1 should then be parsed as an image with gender = -1 * `STEP_MAX`
        and age = 1 * `STEP_MAX`. Thus, `input example` defines parameters for 3 images (length of the array).

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
                manipulation steps that should be applied, however, in InterFaceGAN all
                manipulation can be done in one step. Thus the length of axis 1 should be 1.
                Each entry in the array corresponds to a single manipulation specified by a
                dictionary, with parameters on native format.
        """
        attrs = list(input.keys())
        vals = list(input.values())

        nr_images = len(vals[0])
        nr_attrs = len(attrs)

        parsed_input = np.empty((nr_images, 1), object)
        for i in range(nr_images):
            parsed_input[i, 0] = {
                attrs[j]: vals[j][i] * STEP_MAX for j in range(nr_attrs)
            }

        return parsed_input

    def generate_native(
        self,
        latent_codes: np.ndarray,
        native_input: np.ndarray,
        use_projected: bool = USE_PROJECTED,
    ) -> Tuple[List[str], np.ndarray]:
        gen_name = self._gen.get_name()

        projected_boundaries_dict = {}

        # Go through each latent code
        for i in range(latent_codes.shape[0]):
            d = native_input[i, 0]
            attrs = list(d.keys())

            boundaries_shape = (len(attrs), latent_codes.shape[1])
            boundaries = np.zeros(boundaries_shape)

            # Go through each attribute
            for j, attr in enumerate(attrs):
                # Get svcs
                if attr in self._svcs:
                    svc = self._svcs[attr]
                else:
                    svc = get_svc(attr, gen_name)
                    self._svcs[attr] = svc

                # Save boundaries
                boundary = self._get_boundary(svc)
                boundaries[
                    j,
                ] = boundary

                if not use_projected or boundaries_shape[0] == 1:
                    # Use boundaries to manipulate latent codes
                    latent_codes[i,] += (
                        d[attr] * boundary
                    )

            # Use conditioned/projected boundaries to manipulate latent codes
            if use_projected and boundaries_shape[0] != 1:
                sorted_attrs = sorted(attrs)
                attr_key = "_".join(sorted_attrs)

                # Check if projection already been done before
                if attr_key in projected_boundaries_dict:
                    projected_boundaries = projected_boundaries_dict[attr_key]
                else:
                    # Project each boundary conditioned on all other boundaries
                    projected_boundaries = np.zeros(boundaries_shape)
                    for j in range(projected_boundaries.shape[0]):
                        projected_boundaries[j] = self._project_boundary(
                            boundaries[
                                j,
                            ],
                            np.delete(boundaries, j, axis=0),
                        )

                # Go through each attribute
                for j, attr in enumerate(attrs):
                    latent_codes[i,] += (
                        d[attr]
                        * boundaries[
                            j,
                        ]
                    )

        # Generate images based on updated latent codes
        return self._gen.generate(latent_codes), latent_codes

    def _get_svc_names(self) -> list[str]:
        return get_svc_names(self._attrs, self._gen.get_name())

    def _missing_models(self) -> list[str]:
        return get_missing_model_names(self._get_svc_names())

    def _get_boundary(self, svc: SVC) -> np.ndarray:
        """
        Returns the normalized boundary of the SVC.
        """
        boundary_coefs = svc.coef_.flatten().astype(np.float32)
        return boundary_coefs / np.linalg.norm(boundary_coefs)

    def _project_boundary(self, boundary: np.ndarray, boundaries: np.ndarray):
        """
        Returns a normalized boundary conditioned for other `boundaries`, resulting in a
        boundary which allows for change with the associated attribute without (in theory)
        change to the other attributes associated with the other boundaries.

        Both `boundary` and `boundaries` should be normalized.

        Boundary should be a row vector of same length as the latent space, `boundaries` should be a
        collection of row vectors (same length as `boundary`), one for each attribute to condition for.
        """
        A = np.vstack([boundaries, boundary]).T

        # From Strang, G. Introduction to linear algebra.
        def _qr_factorization(A):
            m, n = A.shape
            Q = np.zeros((m, n))
            R = np.zeros((n, n))

            for j in range(n):
                v = A[:, j]

                for i in range(j - 1):
                    q = Q[:, i]
                    R[i, j] = q.dot(v)
                    v = v - R[i, j] * q

                norm = np.linalg.norm(v)
                Q[:, j] = v / norm
                R[j, j] = norm
            return Q, R

        # gram schmidt by qr factorization
        Q, _ = _qr_factorization(A)
        boundary = Q[:, -1].T

        # Return normalized boundary
        return boundary / np.linalg.norm(boundary)

    def _setup_info_func(self, model_name: str):
        info = load_aux_best(model_name)
        return (
            f"Best model was trained for {info.epoch + info.batch / info.num_batches_per_epoch:.3f} epochs | "
            + f"Training loss = {info.train_loss:.6f} | Validation loss = {info.valid_loss:.6f}"
        )
