import src.util.AuxUtil as AU

from src.metric.iresnet import iresnet100, IResNet
import src.util.CudaUtil as CU
from src.util.AuxUtil import get_file_jar

from typing import Union
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

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
            T.CenterCrop(192),
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

    # Normalize projections
    projections /= np.linalg.norm(projections, axis=1).reshape((-1, 1))

    # Save file if applicable
    if file_name_suffix is not None:
        _file_jar.store_file(
            _projection_save_file_name(file_name_suffix),
            lambda p: np.save(p, projections),
        )

    return projections


def visualize_ffhq256_vs_ffhq256():
    """
    Visualizes ffhq256 similarity score with respect to itself.

    Also plots normal distributions fitted by a GMM. These normal distributions
    are used to infer a threshold value used for filtering.

    Raises:
        FileNotFoundError: When file is not found.
    """
    file_jar = get_file_jar()
    f_name = "FFHQ_256_vs_FFHQ_256_similarity.npy"
    similarity = file_jar.get_file(f_name, np.load)

    N = 1000
    if similarity is None:
        raise FileNotFoundError(f"{f_name} does not exist.")

    # Plot histogram
    plt.hist(similarity, bins=N, density=True)

    # Fit GMM
    n_comp = 4
    gm = GaussianMixture(
        n_components=n_comp,
        covariance_type="diag",
        tol=1e-8,
        verbose=1,
        random_state=0,
        max_iter=99999999,
    ).fit(similarity.reshape(-1, 1))

    # Construct linspace
    xmin, xmax = plt.xlim()
    space = np.linspace(xmin, xmax, N)

    # Plot all normal distributions
    p_tot = 0
    ps = []
    colors = ["aquamarine", "coral", "aquamarine", "coral"]
    for i in range(n_comp):
        std = np.sqrt(gm.covariances_[i][0])
        m = gm.means_[i][0]
        w = gm.weights_[i]
        p = norm.pdf(space, m, std) * w
        plt.plot(space, p, "--", color=colors[i])
        ps.append(p)
        p_tot += p

    # Plot the sum of the normal distributions
    plt.plot(space, p_tot, ":", color="black", linewidth=6)  # , "--", color="black")

    # Plot the normal distributions associated with unique identities
    plt.plot(
        space,
        (ps[0] + ps[2]),
        color="limegreen",
        linewidth=3,
        label="Unique identities",
    )

    # Plot the normal distributions associated with non-unique identities
    plt.plot(
        space,
        (ps[1] + ps[3]),
        color="red",
        linewidth=3,
        label="Same identities",
    )

    # Find and plot the intersection between the normals
    idx = int(np.argwhere(np.diff(np.sign((ps[0] + ps[2]) - (ps[1] + ps[3]))) != 0)[-1])
    x_threshold = space[idx]
    y_threshold = ((ps[0] + ps[2])[idx],)
    plt.axvline(x=x_threshold, ymin=0, ymax=0.6, ls=":", color="black", linewidth=2)
    plt.plot(x_threshold, y_threshold, marker="o", color="yellow", linewidth=6)

    # Make the plot look nice
    plt.legend()
    plt.xlim(0, 10)
    plt.annotate(f"{round(x_threshold,3)}", (x_threshold, 0.35), ha="center")
    plt.xlabel("Similarity score")
    plt.ylabel("Density")
    plt.title("FFHQ 256x256 similarity score histogram")
    plt.show()
