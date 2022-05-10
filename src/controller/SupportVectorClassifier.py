from pathlib import Path
import joblib
from src.util.FileJar import FileJar
from src.controller.IdentityController import IdentityController, IDENTITY_NAME
from src.population.Population import Population
from src.controller.Controller import Controller
from src.controller.ASADController import ASADController
from src.generator.Generator import Generator
from src.generator.UNetGANGenerator import UNetGANGenerator
from src.dataset.Dataset import Dataset
from src.dataset.TorchImageDataset import TorchImageDataset
import torchvision.transforms as T
import numpy as np
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm
import src.util.CudaUtil as CU
import math
from typing import Tuple, Union, Any
import torch
import time

#! Setup function should be run in order:
#! setup_population -> setup_auxillary -> setup_labels -> train_svc
# File handling constants
SVC_PREFIX = "SVC"
SVC_DIR = Path() / "auxiliary"
SVC_FILE_EXT = ".joblib"
LABELS_EXT = ".npy"
LABELS_DIR = Population.POPULATION_METADATA_DIRECTORY_NAME + "/"

# Model constans
MIN_TRAIN_SIZE = 500000  # 500k used in interfacegan paper, 200k in styleganpaper
CLS_CONFIDENCE = 1 / 5  # CLS_CONFIDENCE * MIN_TRAIN_SIZE = nr training samples


def _get_svc_name(
    attr: str, gen_name: str, controller_name: str = IDENTITY_NAME
) -> str:
    return "_".join((SVC_PREFIX, controller_name, gen_name, attr))


def get_svc_names(
    attrs: list[str], gen_name: str, controller_name: str = IDENTITY_NAME
) -> list[str]:
    """
    Returns the names of all support vector classifiers given the parameters.

    Args:
        attrs (list[str]): List of attributes.
        gen_name (str): The name of the generator which produced the training samples.
        controller_name (str, optional): The name of the controller which produced the training samples.
            Use Default `IDENTITY_NAME` if controller didnt perform any manipulations to the samples.
            Defaults to IDENTITY_NAME.

    Returns:
        list[str]: SVC model names.
    """
    return [_get_svc_name(attr, gen_name, controller_name) for attr in attrs]


def get_attr_from_svc_name(name: str) -> str:
    """
    Returns the attribute associated with the given SVC name.

    Args:
        name (str): SVC model name.

    Returns:
        str: The attribute associated with the given SVC name.
    """
    return name.split("_")[-1]


def get_missing_model_names(all_svc_names: list[str]) -> list[str]:
    """
    Returns the names of all the missing support vector classifiers.

    Args:
        all_svc_names (list[str]): Name of all SVC names.

    Returns:
        list[str]: Names of the missing SVCs.
    """
    file_jar = FileJar(SVC_DIR)

    return [
        n
        for n in all_svc_names
        if file_jar.get_file(n + SVC_FILE_EXT, joblib.load) is None
    ]


def get_svc(
    attr: str, gen_name: str, controller_name: str = IDENTITY_NAME
) -> Union[SVC, None]:
    """
    Returns the SVC associated with the given parameters.

    Args:
        attr (str): Attribute which SVC is a predictor for.
        gen_name (str): Generator associated with the training of the SVC.
        controller_name (str, optional): Controller associated with the training of the SVC.
            Use Default `IDENTITY_NAME` if controller didnt perform any manipulations to the samples.
            Defaults to IDENTITY_NAME.

    Returns:
        Union[SVC, None]: The model if it was found, otherwise None.
    """
    file_jar = FileJar(SVC_DIR)

    return file_jar.get_file(
        _get_svc_name(attr, gen_name, controller_name) + SVC_FILE_EXT, joblib.load
    )


def get_population_name(gen_name: str, controller_name: str = IDENTITY_NAME) -> str:
    """
    Returns the name of the population associated with the given Controller.

    Args:
        gen_name (str): Generator associated with the population.
        controller_name (str, optional): Controller associated with the population.
            Use Default `IDENTITY_NAME` if controller didnt perform any manipulations to the samples.
            Defaults to IDENTITY_NAME.

    Returns:
        str: The name of the population.
    """
    return "_".join((SVC_PREFIX, controller_name, gen_name))


def get_population_info(pop_name: str) -> str:
    """
    Returns information about the population. (Number of samples)

    Args:
        pop_name (str): Name of the population.

    Returns:
        str: The information about the population.
    """
    return f"Population has {Population(pop_name).num_samples()} samples."


def is_population_ready(pop_name: str) -> bool:
    """
    Checks if the population is ready by comparing the
    number of samples to `MIN_TRAIN_SIZE`.

    Args:
        pop_name (str): Name of the population.

    Returns:
        bool: True if ready.
    """
    return Population(pop_name).num_samples() >= MIN_TRAIN_SIZE


def setup_population(controller: Controller) -> None:
    """
    Setup a population given by a Controller, population used for training
    SVCs. Setup already existing population or if necessary constructs a new one.

    If population is not populated, then `_populate_population()` will be called.

    Args:
        controller (Controller): Used for defining the name of the population.
        use dummy IdentityController(with the correct Generator) if no manipulations are needed.
    """
    # Get name of the population
    pop_name = get_population_name(
        controller.get_generator().get_name(), controller.get_name()
    )

    # Check population, populate if necessary
    if not is_population_ready(pop_name):
        _populate_population(Population(pop_name), controller)


def setup_auxillary(
    attr: str, ds: Dataset, batch_size: int = 64, epochs: int = 40
) -> None:
    """
    Train auxiliary classifier (borrowed from ASADController) if
    necessary. Retraining not supported.

    Trained classifier are saved according to ASADController.
    Args:
        attr (str): Attribute of the auxiliary classifier.
        ds (Dataset): Training data used/to be used by the classifier.
        batch_size (int,optional): Batch size of the training. Defaults to 64.
        epochs (int,optional): Number of epochs. Defaults to 40.
    """

    # Train classifier if necessary
    #! classifier borrowed from ASADController
    # TODO: break away classifier from ASADController such that
    # TODO: it can more easily be used by other components.
    # Create dummy controller, using dummy generator
    asad = ASADController(UNetGANGenerator(ds), [attr])
    asad.get_classifier(attr, batch_size=batch_size, epochs=epochs)


def setup_labels(
    controller: Controller,
    attr: str,
    batch_size: int = 64,
    epochs: int = 40,
    pop_name: str = None,
) -> None:
    """
    Setup `attr`-labels given a auxiliary classifier (Will train one if missing.)

    Labels of the population (defined by the controller) are predicted by
    the auxiliary classifier.

    Labels are saved (via FileJar) to disk in the root directory of the population.
    For example:

    '.../population_name/`attr`.npy'

    Args:
        controller (Controller): Controller associated with the use of the
            auxillary model.
        attr (str): The attribute of the labels, coincide with the attribute of
            the auxiliary classifier.
        batch_size (int,optional): Batch size used for predicting labels with classifier.
            (also used for training classifier if needed.) Defaults to 64.
        epochs (int,optional): Number of epochs used to train the classifier if needed. Defaults to 40.
        pop_name (str, optional): Name of the population to setup labels for. Defaults to None.
            If None, population name will be inferred from the controller.
    """
    # Get classifier, train classifier if necessary
    #! classifier borrowed from ASADController
    # TODO: break away classifier from ASADController such that
    # TODO: it can more easily be used by other components.
    # Create dummy controller, using dummy generator
    asad = ASADController(controller.get_generator(), [attr])
    cls_model = asad.get_classifier(attr, batch_size=batch_size, epochs=epochs)

    # Init population
    if pop_name is None:
        pop = Population(
            get_population_name(
                controller.get_generator().get_name(), controller.get_name()
            )
        )
    else:
        pop = Population(pop_name)

    n = pop.num_samples()

    # Loads classifier to GPU
    device = CU.get_default_device()
    cls_model = CU.to_device(cls_model, device)

    # Create torch image dataset from population
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    pop_data = pop.get_data()
    pop_ds = TorchImageDataset(
        pop_data["uri"].values.tolist(), T.Compose([T.ToTensor(), T.Normalize(*stats)])
    )

    # Create a dataloader
    pop_dl = DataLoader(pop_ds, batch_size, shuffle=False, pin_memory=True)

    # Predict labels
    all_labels = []
    with torch.no_grad():
        for images in tqdm(
            pop_dl, total=math.ceil(n / batch_size), desc="Classifying images..."
        ):
            # Send data to GPU
            images = CU.to_device(images, device)

            # Predict labels
            labels = cls_model(images)

            # Save labels
            all_labels.append(labels.cpu().numpy())

    # Concatenate all labels
    labels = np.concatenate(all_labels)

    # Save labels
    file_jar = FileJar(Population.POPULATION_ROOT_DIR / pop.get_name())
    file_jar.store_file(
        LABELS_DIR + attr + LABELS_EXT,
        lambda p: np.save(p, labels),
    )


def is_labels_ready(attr: str, pop_name: str) -> bool:
    """
    Checks if labels are ready.

    Args:
        attr (str): Attribute to check labels for.
        pop_name (str): Population to check labels for.

    Returns:
        bool: True if ready.
    """
    return get_labels(attr, pop_name) is not None


def get_labels(attr: str, pop_name: str) -> Union[None, np.ndarray]:
    """
    Gets the labels.

    Args:
        attr (str): Attribute to check labels for.
        pop_name (str): Population to check labels for.

    Returns:
        bool: True if ready.
    """
    file_jar = FileJar(Population.POPULATION_ROOT_DIR / pop_name)
    return file_jar.get_file(LABELS_DIR + attr + LABELS_EXT, np.load)


def train_svc(controller: Controller, attr: str) -> SVC:
    """
    Train a support vector classifier to predict `attr` on the population defined by the
    controller.

    Args:
        controller (Controller): Controller associated with the training data.
        attr (str): The attribute of the labels, coincide with the attribute of
            the auxiliary classifier.
    """
    # Init population
    pop = Population(
        get_population_name(
            controller.get_generator().get_name(), controller.get_name()
        )
    )

    # Get latent codes
    pop_data = pop.get_data()
    latent_codes = np.stack(pop_data["latent_code"].to_numpy())

    # Get labels
    labels = get_labels(attr, pop.get_name())

    # Fit svc to data
    return _fit_svc(
        latent_codes,
        labels,
        get_svc_names(
            [attr], controller.get_generator().get_name(), controller.get_name()
        )[0],
    )


def _fit_svc(latent_codes: np.ndarray, labels: np.ndarray, name: str) -> SVC:
    # Sanity check
    if latent_codes.shape[0] != labels.shape[0]:
        raise ValueError(
            f"The number of latent codes ({latent_codes.shape[0]}) must match "
            + f"the number of labels ({labels.shape[0]})."
        )

    # fit svc
    # Codes/labels filtering according to InterFaceGAN/Linear separability
    latent_codes, labels = _remove_uncertain_labels(latent_codes, labels)

    svc = SVC(kernel="linear", verbose=True)
    print("Training SVC...")
    start = time.time()
    svc.fit(latent_codes, labels)
    print(f"SVC training took {(time.time() - start)/60:.2f} minutes!")
    print("Training done!")

    # Save to disk
    file_jar = FileJar(SVC_DIR)
    file_jar.store_file(name + SVC_FILE_EXT, lambda p: joblib.dump(svc, p))
    return svc


def _remove_uncertain_labels(
    latent_codes: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    return (shuffled) `latent_codes`, `labels` with the
    number of samples multiplied by `CLS_CONFIDENCE`/2 highest/lowest confidence values.
    """

    # Sort by labels
    ind = np.argsort(labels, axis=0).flatten()

    # Get filter indices
    n = labels.shape[0]
    n_filter = math.floor(n * CLS_CONFIDENCE / 2)
    filtered_inds = np.concatenate((ind[:n_filter], ind[(n - n_filter) :]))

    # Filter latent codes
    latent_codes = latent_codes[
        filtered_inds,
    ]

    # Format labels
    labels = np.concatenate(
        [np.zeros(n_filter, dtype=np.int), np.ones(n_filter, dtype=np.int)], axis=0
    )

    # Shuffle data
    assert latent_codes.shape[0] == labels.shape[0]
    shuffle_inds = np.random.permutation(labels.shape[0])

    return (
        latent_codes[
            shuffle_inds,
        ],
        labels[shuffle_inds],
    )


def _populate_population(pop: Population, controller: Controller) -> None:
    """
    Populate a population given the generator of the given controller.
    """
    # TODO support append to population
    gen: Generator = controller.get_generator()

    # Generate latent codes
    latent_codes = gen.random_latent_code(MIN_TRAIN_SIZE)

    # Generate native input
    if isinstance(controller, IdentityController):
        # No manipulation of latent codes
        parsed_native_input = None
        controller_input = {}
    else:
        # Uniform sample to get input
        controller_input = controller.sample_random_input(
            MIN_TRAIN_SIZE, attr_sample_mode="random"
        )
        parsed_native_input = controller.parse_native_input(controller_input)

    # Generate samples, add samples to population
    uris, manipulated_latent_codes = controller.generate_native(
        latent_codes, parsed_native_input
    )
    #! Dummy values for columns not used
    pop.add_all(
        manipulated_latent_codes,
        latent_codes,
        uris,
        [0 for i in range(0, MIN_TRAIN_SIZE)],  # TODO Filter support
        append=False,
        save_to_disk=True,
        **controller_input,
    )
