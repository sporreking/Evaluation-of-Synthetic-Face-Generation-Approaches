# Code regarding the calculation of PPL is inspired by:
"""Perceptual Path Length (PPL) from the paper "A Style-Based Generator
Architecture for Generative Adversarial Networks". The original
implementation by Karras et al. at
https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
and the pytorch implementation in StyleGan2-ada:
https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/perceptual_path_length.py
"""

from src.metric.CompoundMetric import CompoundMetric
from src.metric.SampleMetricManager import SampleMetricManager
from src.core.Setupable import SetupMode
from src.metric.CompoundMetricManager import CompoundMetricManager
from typing import Any, Union
import torch
from urllib import request
from tqdm import tqdm
import numpy as np
from src.util.CudaUtil import get_default_device, to_device
from src.util.FileJar import FileJar
from pathlib import Path
from src.generator.Generator import Generator
from src.dataset.TorchImageDataset import TorchImageDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.jit import ScriptModule
from src.util.AuxUtil import get_file_jar

PPL_NAME = "PerceptualPathLength"
VGG16_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
)
BATCH_SIZE = 64  #! BATCH_SIZE MUST BE EVEN! and >=2
VGG16_SETUP_NAME = "VGG16_SETUP"
VGG16_MODEL_NAME = "vgg16_zhang.pt"

# Metric constants
NUM_SAMPLES = 50000  # rec 100k stylegan paper, 50k in stylegan2-ada
EPS = 1e-4  # 1e-4 According to stylegan paper

#! BUG ALERT
# TODO getting high values, 10x what they should be
# Its probably not vgg16 model since ive tried downloading it myself
# and used a VGG16 from the python module LPIPS.

# When using W (stylegan2ada) the interpolation function is lerp, which is trivial to implement so thats
# probably not the problem.

# Tested to buffer random noise as done in the StyleGan github, didn't change anything major.

# Checked such that the right latent codes are compared, this should be right.

# The values are however scaling as in the stylegan paper: This implementation yields ~2k for W and ~4k for Z
# which corresponds to the papers values 200 for W and 400 for Z. (just a factor of 10 in difference)

# * Possible solutions:
# use EPS = 1e-3 instead, this makes the values reasonable, however, it is not the same as in the paper.
# This could be used to offset the unknown error causing problems since the scaling otherwise seems fine.


class PPLCompoundMetric(CompoundMetric):
    def __init__(
        self,
        cmm: CompoundMetricManager,
        smm: SampleMetricManager = None,
    ):
        """
        Constructor for PPLCompoundMetric class, subclass of the CompoundMetric class.

        Args:
            cmm (CompoundMetricManager): Manager used by metrics. Population, controller and dataset is derived
                from this manager.
            smm (SampleMetricManager, optional): Not used for this metric. Defaults to None.
        """
        super(PPLCompoundMetric, self).__init__(PPL_NAME, cmm, smm)

        # Init storage structure for ppl and vgg16
        self._ppl = None
        self._vgg16 = None

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        return {
            VGG16_SETUP_NAME: SetupMode(
                lambda _,: self._get_vgg16_feature_extractor(),
                lambda: self._get_local_vgg16_feature_extractor() is not None,
            )
        }

    def calc(self, **parameters: Any) -> Any:
        """
        Calculates the perceptual path length (PPL).

        Args:
            batch_size (int, optional): Batch size used to calculate the PPL.
                Note that batch size must be even!
                Defaults to `BATCH_SIZE`.
            use_crop (bool, optional): True if images should be cropped around the
                head (assumes centered images as in FFHQ etc). Defaults to True
        Raises:
            ValueError: If the resolution of the dataset is not 256x256.
            ValueError: If the batch size is less than or equal zero.

        Returns:
            Any: PPL of the population.
        """
        ## Check resolution
        if self.get_dataset().get_resolution() != 256:
            # TODO: General resolution support
            # TODO: would need to resize imgs down to 256x256 before vgg16
            raise ValueError("256x256 is the only supported resolution at the moment.")

        ## Fetch and check parameters
        if "batch_size" in parameters:
            batch_size = parameters["batch_size"]
            if batch_size % 2:
                # Make batch_size even
                batch_size += 1
            if batch_size <= 0:
                raise ValueError("batch_size must be larger than 1!")
        else:
            batch_size = BATCH_SIZE
        if "use_crop" in parameters:
            use_crop = parameters["use_crop"]
        else:
            use_crop = True

        ## Get latent codes from population
        pop_data = self._population.get_data()
        latent_codes = np.stack(pop_data["latent_code"].to_numpy())

        ## Interpolate to find latent codes pairs to test
        n_latent_codes = latent_codes.shape[0]
        N = n_latent_codes if n_latent_codes < NUM_SAMPLES else NUM_SAMPLES

        # Indices of random samples from the population.
        indices = np.random.choice(np.arange(n_latent_codes), size=N, replace=False)

        # Interpolation parameters
        t = np.random.uniform(0, 1 - EPS, size=N)

        gen: Generator = self._cmm._controller.get_generator()
        interpolated_latent_codes = np.zeros((N, latent_codes.shape[1]))
        for i in tqdm(range(0, N, 2), desc="Latent code interpolation"):
            code_1 = latent_codes[indices[i]]
            code_2 = latent_codes[indices[i + 1]]
            interpolated_latent_codes[i, :] = gen.interpolate(code_1, code_2, t[i])
            interpolated_latent_codes[i + 1, :] = gen.interpolate(
                code_1, code_2, t[i] + EPS
            )

        ## Generate images
        img_uris = gen.generate(interpolated_latent_codes)

        ## Preprocess before vgg16 feature extraction
        # Create torch image dataset
        # pixel values between 0 and 255
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        transforms = [
            T.ToTensor(),
            T.Normalize(*stats),
            T.Lambda(lambda img: (img + 1) * (255 / 2)),
        ]

        if use_crop:
            # According to StyleGan2-ada implementation
            c = self._dataset.get_resolution() // 8
            transforms.append(
                T.Lambda(lambda img, c=c: img[:, c * 3 : c * 7, c * 2 : c * 6])
            )

        img_ds = TorchImageDataset(
            img_uris,
            T.Compose(transforms),
        )

        # Create a dataloader

        pop_dl = DataLoader(img_ds, batch_size, shuffle=False, pin_memory=True)

        ## vgg16 feature extraction
        if self._vgg16 is None:
            vgg16 = self._get_vgg16_feature_extractor()
        else:
            vgg16 = self._vgg16

        distances = []
        with torch.no_grad():
            # vgg16 to GPU
            device = get_default_device()
            vgg16 = to_device(vgg16, device)
            for i, imgs in tqdm(
                enumerate(pop_dl),
                total=N // batch_size,
                desc="vgg16 distance calculation",
            ):
                # Extract features
                imgs = vgg16(
                    to_device(imgs, device), resize_images=False, return_lpips=True
                )

                # Calculate the pairwise distance
                dist = (
                    ((imgs[::2] - imgs[1::2]).square().sum(1) / EPS**2).cpu().numpy()
                )
                distances.append(dist)

        ## Filter distances
        distances = self._filter_outliers(np.concatenate(distances))

        ## Return PPL
        self._ppl = distances.mean()
        return self._ppl

    def get(self, calc_if_missing: bool = False, **parameters: Any) -> Any:
        # Check if metric already calculated
        if self._ppl is not None:
            return self._ppl

        # Check if calculate when missing
        elif calc_if_missing:
            return self.calc(**parameters)
        else:
            return None

    def print_result(self) -> None:
        print("Perceptual path length :", self._ppl)

    def plot_result(self) -> None:
        pass

    def _get_local_vgg16_feature_extractor(self) -> Union[ScriptModule, None]:
        vgg16 = get_file_jar().get_file(VGG16_MODEL_NAME, torch.jit.load)
        if vgg16 is not None:
            return vgg16.eval()
        return vgg16

    def _get_vgg16_feature_extractor(self) -> ScriptModule:
        vgg16 = self._get_local_vgg16_feature_extractor()
        if vgg16 is None:
            print("vgg16 not local, downloading...")
            # vgg16_zhang
            file_name = get_file_jar().get_root_dir() + VGG16_MODEL_NAME
            request.urlretrieve(VGG16_URL, file_name)
            vgg16 = torch.jit.load(file_name).eval()

        self._vgg16 = vgg16
        return vgg16

    def _filter_outliers(self, distances: np.ndarray) -> np.ndarray:
        # Filter and reject outliers. According to StyleGAN paper
        lo = np.percentile(distances, 1, interpolation="lower")
        hi = np.percentile(distances, 99, interpolation="higher")
        return np.extract(np.logical_and(lo <= distances, distances <= hi), distances)
