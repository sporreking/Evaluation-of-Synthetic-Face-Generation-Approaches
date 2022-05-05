import src.util.AuxUtil as AU

from src.metric.iresnet import iresnet100, IResNet
import src.util.CudaUtil as CU

from typing import Union
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

import torch
import gdown

PARTIAL_FC_BACKBONE_PRETRAIN_URL = (  # * RestNet100 backbone from: https://github.com/deepinsight/insightface
    "https://drive.google.com/uc?id=1XNMRpB0MydK1stiljHoe4vKz4WfNCAIG"
)

PARTIAL_FC_BACKBONE_PRETRAIN_FILE_NAME = "partial_fc_r100.pth"

PARTIAL_FC_INPUT_RESOLUTION = 112
PARTIAL_FC_OUTPUT_DIM = 512

# Get file jar to store files in
_file_jar = AU.get_file_jar()


def is_ready() -> bool:
    """
    Checks whether the `setup()` has been performed.

    Returns:
        bool: `True` if the setup has been performed, otherwise `False`.
    """
    return _file_jar.has_file(PARTIAL_FC_BACKBONE_PRETRAIN_FILE_NAME)


def setup() -> None:
    """
    Sets up the Partial FC backbone such that the model may be fetched using
    `get()`. Primarily, the setup involves the download of pretrained weights
    from `PARTIAL_FC_BACKBONE_PRETRAIN_URL`.
    """
    # Download pretrained model
    _file_jar.store_file(
        PARTIAL_FC_BACKBONE_PRETRAIN_FILE_NAME,
        lambda p: gdown.download(PARTIAL_FC_BACKBONE_PRETRAIN_URL, str(p), quiet=False),
    )


def get() -> IResNet:
    """
    Fetches the Partial FC backbone network. Note that `setup()` must have been
    performed prior.

    The dot-products between image projections within this network are
    analogous to cosine similarity between the represented identities.

    Raises:
        RuntimeError: If the setup has not yet been performed.

    Returns:
        IResNet: The Partial FC backbone network.
    """
    state = _file_jar.get_file(PARTIAL_FC_BACKBONE_PRETRAIN_FILE_NAME, torch.load)

    if state is None:
        raise RuntimeError(
            "The setup for using matching scores has not yet been performed!"
        )

    net = iresnet100()
    net.load_state_dict(state, strict=False)
    return net.eval()


def _get_partial_fc_transform() -> T.Compose:
    """A composite transform for fitting images to backbone input layer."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Resize(112),
        ]
    )


def _projection_save_file_name(file_name_suffix: str) -> str:
    """Derives the full auxiliary file name from given suffix."""
    return f"partial_fc_proj_{file_name_suffix}.npy"


def load_projected_images(file_name_suffix: str) -> np.ndarray:
    """
    Loads the projections associated with the specified `file_name_suffix`.

    Args:
        file_name_suffix (str): The suffix of the projections to load.
            Note that images must have already been projected using the
            `project_images(file_name_suffix)` function.

    Raises:
        FileNotFoundError: If the projections have not yet been performed
            for the specified `file_name_suffix`.

    Returns:
        np.ndarray: The loaded projections.
    """
    file_name = _projection_save_file_name(file_name_suffix)
    projections = _file_jar.get_file(file_name, np.load)

    if projections is None:
        raise FileNotFoundError(
            f"Could not load projections using file name suffix '{file_name_suffix}', i.e., "
            + f"there was no auxiliary file: '{file_name}'."
        )

    return projections


def project_images(
    image_paths: list[Union[Path, str]], file_name_suffix: str = None
) -> np.ndarray:
    """
    Projects the given images through the Partial FC backbone network. Note that
    `setup()` must have been performed prior.

    If `file_name_suffix` is not `None`, a file will be saved so that it can be
    retrieved with `load_projected_images(file_name_suffix)`.

    Args:
        image_paths (list[Path | str]): Paths to the images that should be projected.
        file_name_suffix (str, optional): A suffix to append to the file name,
            or `None` if there should be no output file. Defaults to None.


    Raises:
        RuntimeError: If the setup has not yet been performed.

    Returns:
        np.ndarray: The projected images, on a row-wise basis, i.e.,
            the shape will be `(len(image_paths), PARTIAL_FC_OUTPUT_DIM)`.
    """
    # Get GPU
    device = CU.get_default_device()

    # Load Partial FC network
    backbone = CU.to_device(get(), device)

    # Get transform
    transform = _get_partial_fc_transform()

    # Load images and perform projection
    projections = np.zeros((len(image_paths), PARTIAL_FC_OUTPUT_DIM))
    with torch.no_grad():
        for i, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc="Projecting images with Partial FC backbone",
        ):
            img = CU.to_device(
                transform(Image.open(image_path).convert("RGB")).reshape(
                    1, 3, PARTIAL_FC_INPUT_RESOLUTION, PARTIAL_FC_INPUT_RESOLUTION
                ),
                device,
            )

            projections[i, :] = backbone(img).cpu().numpy()

    # Save file if applicable
    if file_name_suffix is not None:
        _file_jar.store_file(
            _projection_save_file_name(file_name_suffix),
            lambda p: np.save(p, projections),
        )

    return projections
