# * From: https://github.com/youngjung/improved-precision-and-recall-metric-pytorch (MIT license)
# * Fetched: 05 May 2022
# * Modifications: Compatibility with architecture and memory optimization.

import os
from functools import partial
from collections import namedtuple
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from src.dataset.Dataset import Dataset
from src.util.AuxUtil import get_file_jar
from src.util.CudaUtil import get_default_device, to_device
from typing import Union

# Define manifold
Manifold = namedtuple("Manifold", ["features", "radii"])

# Calculation constants
DIST_CALC_BATCH_SIZE_NAME = "dist_calc_batch_size"
DIST_CALC_BATCH_SIZE = 1000

# Model constants
BATCH_SIZE_NAME = "batch_size"
BATCH_SIZE = 32
K = 3


def get_ref_file_name(ds: Dataset, omit_extension: bool = False) -> str:
    """
    Returns the name of the manifold file.

    Args:
        ds (Dataset): The dataset to which the manifold file pertains.
        omit_extension (bool, optional): If True, the ".npz" suffix
            will be omitted. Defaults to False.

    Returns:
        str: The manifold name associated with the specified dataset.
    """
    return (
        f"{ds.get_name(ds.get_resolution())}_manifold{'' if omit_extension else '.npz'}"
    )


class ImprovedPrecisionRecall:
    def __init__(self, ds: Dataset, k: int = K):
        """
        Constructor for the (improved) precision and recall helper class.

        Args:
            ds (Dataset): Dataset to extract reference manifold from.
            k (int, optional): Number of neighbors to use for KNN (when calculating distances between samples).
                Defaults to `K`.
        """
        self.manifold_ref = None
        self.k = k
        self.file_jar = get_file_jar()
        self.ref_file_name = get_ref_file_name(ds)
        self.ds = ds
        self._cache_vgg16 = None

    @property
    def vgg16(self):
        if self._cache_vgg16 is None:
            print(
                "loading vgg16 for improved precision and recall...", end="", flush=True
            )
            self._cache_vgg16 = models.vgg16(pretrained=True).eval()
            print("done")
        return self._cache_vgg16

    @vgg16.setter
    def vgg16(self, value):
        self._cache_vgg16 = value

    def calc_recall(
        self,
        subject: Union[
            str,
            Path,
            np.ndarray,
            torch.Tensor,
            list[str],
            list[Path],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        batch_size: int = BATCH_SIZE,
        dist_calc_batch_size: int = DIST_CALC_BATCH_SIZE,
    ) -> float:
        """
        Compute recall for given subject.
        Dataset manifold should be precomputed by compute_manifold_ref()

        Args:
            subject (str | Path | np.ndarray | torch.Tensor | list[str | Path | np.ndarray | torch.Tensor]): Subject to
                which the calculation will be applied. Can be defined in many different formats:

                Path object (or file name) to directory containing images.

                Path object (or file name) to precalculated .npz file (containing the manifold).

                Images (with shape `N x C x H x W` as ndarray or tensor).

                List of images (each img with the shape `C x H x W` as ndarray or tensor)

                List of Path objects (or file names) to images.
            batch_size (int, optional): Batch size to use for calculating recall. Defaults to `BATCH_SIZE`.
            dist_calc_batch_size (int, optional): The number of samples from `X` to process at a time when
                calculating distances. Defaults to `DIST_CALC_BATCH_SIZE`.

        Returns:
            float: The recall of the given subject.
        """
        # Check manifold ref, get from local if None
        self._cache_manifold_ref_assert()

        manifold_subject = self._compute_manifold(
            subject, batch_size, dist_calc_batch_size
        )
        recall = self._compute_metric(
            manifold_subject,
            self.manifold_ref.features,
            dist_calc_batch_size,
            "computing recall...",
        )
        return recall

    def calc_precision(
        self,
        subject: Union[
            Path,
            str,
            np.ndarray,
            torch.Tensor,
            list[str],
            list[Path],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        batch_size: int = BATCH_SIZE,
        dist_calc_batch_size: int = DIST_CALC_BATCH_SIZE,
    ) -> float:
        """
        Compute precision for given subject
        Dataset manifold should be precomputed by compute_manifold_ref()

        Args:
            subject (str | Path | np.ndarray | torch.Tensor | list[str | Path | np.ndarray | torch.Tensor]): Subject to
                which the calculation will be applied. Can be defined in many different formats:

                Path object (or file name) to directory containing images.

                Path object (or file name) to precalculated .npz file (containing the manifold).

                Images (with shape `N x C x H x W` as ndarray or tensor).

                List of images (each img with the shape `C x H x W` as ndarray or tensor)

                List of Path objects (or file names) to images.
            batch_size (int, optional): Batch size to use for calculating precision. Defaults to `BATCH_SIZE`.
            dist_calc_batch_size (int, optional): The number of samples from `X` to process at a time when
                calculating distances. Defaults to `DIST_CALC_BATCH_SIZE`.

        Returns:
            float: The precision of the given subject.
        """
        # Check manifold ref, get from local if None
        self._cache_manifold_ref_assert()

        manifold_subject = self._compute_manifold(
            subject, batch_size, dist_calc_batch_size
        )
        precision = self._compute_metric(
            self.manifold_ref,
            manifold_subject.features,
            dist_calc_batch_size,
            "computing precision...",
        )
        return precision

    def compute_manifold_ref(
        self,
        batch_size: int = BATCH_SIZE,
        dist_calc_batch_size: int = DIST_CALC_BATCH_SIZE,
    ) -> None:
        """
        Computes the manifold for the associated dataset.

        Stores the result to file location defined by AuxUtil root directory.

        Args:
            batch_size (int, optional): Batch size to use for computing the manifold. Defaults to `BATCH_SIZE`.
            dist_calc_batch_size (int, optional): The number of samples from `X` to process at a time when
                calculating distances. Defaults to `DIST_CALC_BATCH_SIZE`.
        """

        self.manifold_ref = self._compute_manifold(
            self.ds.get_image_paths(), batch_size, dist_calc_batch_size
        )
        self.file_jar.store_file(
            self.ref_file_name,
            lambda p: np.savez_compressed(
                p, features=self.manifold_ref.features, radii=self.manifold_ref.radii
            ),
        )

    def _cache_manifold_ref_assert(self) -> bool:
        if self.manifold_ref is None:
            self.manifold_ref = self.get_precomputed_manifold()
        assert self.manifold_ref is not None, "call compute_manifold_ref() first"

    def get_precomputed_manifold(self) -> Manifold:
        """
        Getter for the precomputed manifold of the real dataset.

        Returns:
            Manifold: The manifold object containing precomputed features and radii.
        """
        npz = self.file_jar.get_file(self.ref_file_name, np.load)
        return Manifold(npz["features"], npz["radii"])

    def check_precalculated_manifold(self) -> bool:
        """
        Checks if manifold for the real dataset exists.

        Returns:
            bool: True if it exists, otherwise False.
        """
        return self.manifold_ref is not None or self.file_jar.has_file(
            self.ref_file_name
        )

    def _compute_manifold(
        self,
        input: Union[
            Path,
            str,
            np.ndarray,
            torch.Tensor,
            list[str],
            list[Path],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        batch_size: int = BATCH_SIZE,
        dist_calc_batch_size: int = DIST_CALC_BATCH_SIZE,
    ):
        """
        Compute manifold of given input.

        Args:
            input (str | Path | np.ndarray | torch.Tensor | list[str | Path | np.ndarray | torch.Tensor]): Subject to
                which the calculation will be applied. Can be defined in many different formats:

                Path object (or file name) to directory containing images.

                Path object (or file name) to precalculated .npz file (containing the manifold).

                Images (with shape `N x C x H x W` as ndarray or tensor).

                List of images (each img with the shape `C x H x W` as ndarray or tensor)

                List of Path objects (or file names) to images.
            batch_size (int, optional): Batch size to use for computing the manifold. Defaults to `BATCH_SIZE`.
            dist_calc_batch_size (int, optional): The number of samples from `X` to process at a time when
                calculating distances. Defaults to `DIST_CALC_BATCH_SIZE`.

        Returns:
            Manifold:  A manifold object containing features and radii.
        """
        # Path cast to str
        if isinstance(input, Path):
            input = str(input)
        elif isinstance(input, list) and isinstance(input[0], Path):
            input = [str(p) for p in input]

        # features
        if isinstance(input, str):
            if input.endswith(".npz"):  # input is precalculated file
                print("loading", input)
                f = np.load(input)
                feats = f["feature"]
                radii = f["radii"]
                f.close()
                return Manifold(feats, radii)
            else:  # input is dir
                feats = self._extract_features_from_files(input, batch_size)
        elif isinstance(input, torch.Tensor):
            feats = self._extract_features(input, batch_size)
        elif isinstance(input, np.ndarray):
            input = torch.Tensor(input)
            feats = self._extract_features(input, batch_size)
        elif isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.cat(input, dim=0)
                feats = self._extract_features(input, batch_size)
            elif isinstance(input[0], np.ndarray):
                input = np.concatenate(input, axis=0)
                input = torch.Tensor(input)
                feats = self._extract_features(input, batch_size)
            elif isinstance(input[0], str):  # input is list of fnames
                feats = self._extract_features_from_files(input, batch_size)
            else:
                raise TypeError
        else:
            print(type(input))
            raise TypeError

        # radii
        distances = self._compute_pairwise_distances(
            feats, dist_calc_batch_size=dist_calc_batch_size
        )
        radii = self._distances2radii(distances, k=self.k)
        return Manifold(feats, radii)

    def _extract_features(self, images: torch.Tensor, batch_size: int = BATCH_SIZE):
        """
        Extract features of vgg16-fc2 for all images.

        Args:
            images (torch.Tensor): Images on tensor format N x C x H x W.
            batch_size (int, optional): Batch size to use for feature extraction. Defaults to `BATCH_SIZE`.

        Returns:
            np.ndarray: Features of dimension (num images, dims).
        """
        desc = "extracting features of %d images" % images.size(0)
        num_batches = int(np.ceil(images.size(0) / batch_size))
        _, _, height, width = images.shape
        if height != 224 or width != 224:
            print("IPR: resizing %s to (224, 224)" % str((height, width)))
            resize = partial(F.interpolate, size=(224, 224))
        else:

            def resize(x):
                return x

        vgg16 = to_device(self.vgg16, get_default_device())
        features = []
        for bi in trange(num_batches, desc=desc):
            start = bi * batch_size
            end = start + batch_size
            batch = images[start:end]
            batch = resize(batch)
            before_fc = vgg16.features(batch.cuda())
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = vgg16.classifier[:4](before_fc)
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def _extract_features_from_files(
        self, path_or_fnames: Union[str, list[str]], batch_size: int = BATCH_SIZE
    ):
        """
        Extract features of vgg16-fc2 for all images in path.

        Args:
            path_or_fnames (str | list[str]]): Path to directory containing images or list of image paths.
            batch_size (int, optional): Batch size to use for feature extraction. Defaults to `BATCH_SIZE`.

        Returns:
            np.ndarray: Features of dimension (num images, dims).
        """

        dataloader = _get_custom_loader(path_or_fnames, batch_size=batch_size)
        num_found_images = len(dataloader.dataset)
        desc = "extracting features of %d images" % num_found_images
        vgg16 = to_device(self.vgg16, get_default_device())
        with torch.no_grad():
            features = []
            for batch in tqdm(dataloader, desc=desc):
                before_fc = vgg16.features(batch.cuda())
                before_fc = before_fc.view(-1, 7 * 7 * 512)
                feature = vgg16.classifier[:4](before_fc)
                features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def _compute_pairwise_distances(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        dist_calc_batch_size: int = DIST_CALC_BATCH_SIZE,
    ):  #! Warning, heavy on RAM!
        """
        Computes the pairwise distances between every sample in X and Y.

        Args:
            X (np.ndarray): First set of sample (shape: N x dim)
            Y (np.ndarray, optional): Second set of samples (shape: N x dim),
                or None if X should be checked against itself. Defaults to None.
            dist_calc_batch_size (int, optional): The number of samples from `X` to
                process at a time when calculating distances. Defaults to
                `DIST_CALC_BATCH_SIZE`.

        Returns:
            np.ndarray: N x N symmetric matrix with Euclidean distances.
        """

        # Get number of samples
        num_X = X.shape[0]
        if Y is None:
            num_Y = num_X
        else:
            num_Y = Y.shape[0]

        # Compute norms
        X = X.astype(np.float64)  # to prevent underflow
        X_norm_square = np.sum(X**2, axis=1, keepdims=True)
        if Y is None:
            Y_norm_square = X_norm_square
        else:
            Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)

        # Assign Y
        if Y is None:
            Y = X

        # Compute distances
        diff_square = np.zeros((num_X, num_Y))
        for i in tqdm(
            range(num_X // dist_calc_batch_size + 1), desc="computing squared distances"
        ):
            diff_square[
                i * dist_calc_batch_size : (i + 1) * dist_calc_batch_size, :
            ] = (
                X_norm_square[
                    i * dist_calc_batch_size : (i + 1) * dist_calc_batch_size, :
                ]
                - 2
                * np.dot(
                    X[i * dist_calc_batch_size : (i + 1) * dist_calc_batch_size, :], Y.T
                )
                + Y_norm_square.T
            )

        del X_norm_square, Y_norm_square, X, Y

        # check negative distance
        min_diff_square = diff_square.min()
        if min_diff_square < 0:
            idx = diff_square < 0
            diff_square[idx] = 0
            print(
                "WARNING: %d negative diff_squares found and set to zero, min_diff_square="
                % idx.sum(),
                min_diff_square,
            )

        del min_diff_square

        # Memory efficient square root
        for i in tqdm(
            range(num_X // dist_calc_batch_size + 1),
            "square-rooting to obtain final distances",
        ):
            diff_square[
                i * dist_calc_batch_size : (i + 1) * dist_calc_batch_size, :
            ] = np.sqrt(
                diff_square[
                    i * dist_calc_batch_size : (i + 1) * dist_calc_batch_size, :
                ]
            )
        return diff_square

    def _distances2radii(self, distances, k=K):
        num_features = distances.shape[0]
        radii = np.zeros(num_features)
        for i in range(num_features):
            radii[i] = self._get_kth_value(distances[i], k=k)
        return radii

    def _get_kth_value(self, np_array, k=K):
        kprime = k + 1  # kth NN should be (k+1)th because closest one is itself
        idx = np.argpartition(np_array, kprime)
        k_smallests = np_array[idx[:kprime]]
        kth_value = k_smallests.max()
        return kth_value

    def _compute_metric(
        self, manifold_ref, feats_subject, dist_calc_batch_size, desc=""
    ):
        num_subjects = feats_subject.shape[0]
        count = 0
        dist = self._compute_pairwise_distances(
            manifold_ref.features, feats_subject, dist_calc_batch_size
        )
        for i in trange(num_subjects, desc=desc):
            count += (dist[:, i] < manifold_ref.radii).any()
        return count / num_subjects


class ImageFolder(torchDataset):
    def __init__(self, root, transform=None):
        self.fnames = glob(os.path.join(root, "**", "*.jpg"), recursive=True) + glob(
            os.path.join(root, "**", "*.png"), recursive=True
        )

        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


class FileNames(torchDataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


def _get_custom_loader(
    image_dir_or_fnames,
    image_size=224,
    batch_size=BATCH_SIZE,
    num_workers=0
    # ? num_workers = 0 because of windows, more workers might work on linux
):
    transform = []
    transform.append(transforms.Resize([image_size, image_size]))
    transform.append(transforms.ToTensor())
    transform.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    transform = transforms.Compose(transform)

    if isinstance(image_dir_or_fnames, list):
        dataset = FileNames(image_dir_or_fnames, transform)
    elif isinstance(image_dir_or_fnames, str):
        dataset = ImageFolder(image_dir_or_fnames, transform=transform)
    else:
        raise TypeError

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader
