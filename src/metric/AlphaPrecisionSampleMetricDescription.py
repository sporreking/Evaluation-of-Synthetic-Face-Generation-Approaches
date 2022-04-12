import pandas as pd
import numpy as np
from tqdm import tqdm

from typing import Any

from src.dataset.Dataset import Dataset
from src.dataset.TorchImageDataset import TorchImageDataset
from src.metric.SampleMetricDescription import SampleMetricDescription
import src.metric.EvaluationEmbedding as EE
from src.util.CudaUtil import get_default_device, to_device
import src.util.ModelUtil as ModelUtil

import torch
import torchvision.transforms as T

#! Constants
PARAM_ALPHA_DEFAULT = 1.0
CALC_BATCH_SIZE = 10

SETUP_MODE_RESTART = "restart"  # Restart setup from beginning
SETUP_MODE_CONTINUE = "continue"  # Continue training of evaluation embedding
SETUP_MODE_PROJECT = "project"  # Skip training and project immediately


class AlphaPrecisionSampleMetricDescription(SampleMetricDescription):
    @staticmethod
    def get_name() -> str:
        return "AlphaPrecision"

    @staticmethod
    def setup(dataset: Dataset, mode: str = SETUP_MODE_CONTINUE) -> None:

        # Check if valid mode
        if mode not in (SETUP_MODE_RESTART, SETUP_MODE_CONTINUE, SETUP_MODE_PROJECT):
            raise ValueError(f"Invalid setup mode: '{mode}'")

        # Train evaluation embedding if applicable
        if mode != SETUP_MODE_PROJECT or EE.get() is None:

            # Sanity check for projection mode
            if mode == SETUP_MODE_PROJECT:
                print("Must train evaluation embedding before projection!")
                print("Changing mode to perform setup from scratch.")
                mode = SETUP_MODE_RESTART

            # Train network
            info = None
            if mode == SETUP_MODE_CONTINUE:
                info = ModelUtil.load_aux_best(EE.AUX_MODEL_NAME)
                if info is not None:
                    print(
                        f"Starting from (epoch, batch) = ({info.epoch}, {info.batch})"
                    )
            EE.train(dataset, info)

        # Project samples
        EE.project(dataset)

    @staticmethod
    def is_ready(dataset: Dataset) -> bool:
        return EE.get() is not None and EE.get_projections(dataset) is not None

    @staticmethod
    def calc(data: pd.DataFrame, dataset: Dataset, **parameters: Any) -> np.ndarray:
        ee = EE.get()  # TODO: DOCSTRING

        # Sanity check
        if ee is None:
            raise ValueError(
                "Evaluation embedding is not available! Make sure"
                + " that the setup has been performed."
            )

        # Load device
        device = get_default_device()

        # Send to device
        ee = to_device(ee, device)

        # Extract parameters
        alpha = parameters["alpha"] if "alpha" in parameters else PARAM_ALPHA_DEFAULT

        # Load population images
        pop_images = TorchImageDataset(
            [p for p in data.iloc[:, 2]],
            EE.get_inception_image_transform(),
        )

        # Create population loader
        pop_loader = torch.utils.data.DataLoader(
            pop_images, batch_size=CALC_BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Fetch dataset projection
        projections = EE.get_projections(dataset)

        # Compute radius
        print("Centering dataset projections...")
        centered_projections = (
            projections - ee.get_parameter(EE.PARAM_CENTER).cpu().detach().numpy()
        )

        print("Deriving dataset radii...")
        radii = np.linalg.norm(centered_projections, axis=1)

        r_alpha_hat = np.quantile(radii, alpha)

        print(f"r_alpha_hat: {r_alpha_hat}")

        # Get inception model
        inception = to_device(EE.get_inception_model(), device)

        # Compute output
        output = np.zeros(data.shape[0])
        outputv = np.zeros(data.shape[0])
        for i, image in tqdm(
            enumerate(pop_loader), total=len(pop_loader), desc="Computing metrics"
        ):
            image = inception(to_device(image, device))

            X_g_tilde = ee(image)

            dists_from_center = (
                torch.linalg.norm(X_g_tilde - ee.get_parameter(EE.PARAM_CENTER), axis=1)
                .cpu()
                .detach()
                .numpy()
            )

            output[(i * CALC_BATCH_SIZE) : ((i + 1) * CALC_BATCH_SIZE)] = (
                dists_from_center <= r_alpha_hat
            ).astype(int)
            outputv[
                (i * CALC_BATCH_SIZE) : ((i + 1) * CALC_BATCH_SIZE)
            ] = dists_from_center

        from PIL import Image
        import matplotlib.pyplot as plt

        print(np.sum(output))

        def disp_im(i: int):
            m = Image.open(data.iloc[i, 2])
            plt.imshow(np.asarray(m))
            plt.title(f"Index: {i} | Dist: {outputv[i]:.6f}")
            plt.show()

        for i in np.argsort(outputv)[::-1]:
            disp_im(i)

        return output
