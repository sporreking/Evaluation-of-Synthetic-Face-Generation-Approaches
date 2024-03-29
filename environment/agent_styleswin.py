from pathlib import Path
import sys

import torch
import torchvision.utils
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

from latent_code_receiver import receive_latent_codes

# * Add generator project root to python path
PROJECT_ROOT_DIR = Path().absolute()
GEN_ROOT_DIR = Path().absolute() / "environment" / "styleswin"
sys.path.append(str(PROJECT_ROOT_DIR))
sys.path.append(str(GEN_ROOT_DIR))


# * Imports from base project root
import src.util.CudaUtil as CU

# * Imports from generator project root
from models.generator import Generator

# * Launch modes
LAUNCH_MODE_PROJECT = "project"
LAUNCH_MODE_GENERATE = "generate"

# * Read system input
parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument(
    "--launch_mode",
    choices=[LAUNCH_MODE_PROJECT, LAUNCH_MODE_GENERATE],
    default=LAUNCH_MODE_GENERATE,
)

ARGS = parser.parse_args(sys.argv[1:] if __name__ == "__main__" else [])


# * Constants

# Launch mode
LAUNCH_MODE = ARGS.launch_mode

# Directories
OUT_DIR = GEN_ROOT_DIR / "out"
PRETRAIN_DIR = GEN_ROOT_DIR / "pretrain" / "FFHQ_256.pt"
OUT_W_FILE = OUT_DIR / "w.npy"

# Generator
BATCH_SIZE = ARGS.batch_size
LATENT_CODE_LENGTH = 512
G_CHANNEL_MULTIPLIER = 2
LR_MLP = 0.01
ENABLE_FULL_RESOLUTION = 8
USE_G_EMA = True  # False -> use "g"
LATENT_CODES_PER_PROJECTION = 100

# Load GPU if possible
device = CU.get_default_device()


def main():

    # Receive latent codes
    lc_data = receive_latent_codes(LATENT_CODE_LENGTH)
    latent_codes = torch.tensor(lc_data).reshape(
        (int(len(lc_data) / LATENT_CODE_LENGTH), LATENT_CODE_LENGTH)
    )

    # Create directories
    OUT_DIR.mkdir(exist_ok=True)

    if LAUNCH_MODE == LAUNCH_MODE_PROJECT:
        _project_codes(latent_codes)
    elif LAUNCH_MODE == LAUNCH_MODE_GENERATE:
        print("Assuming latent codes to be from the W-domain.")
        _generate_images(latent_codes)


def _project_codes(latent_codes: torch.Tensor):
    # Load mapping network
    M, _ = _load_generator()

    # Project latent codes
    projections = np.zeros(latent_codes.shape)
    with torch.no_grad():
        n_batches = int(np.ceil(latent_codes.shape[0] / LATENT_CODES_PER_PROJECTION))
        for i in tqdm(range(n_batches), desc="Projecting latent codes from Z to W"):
            projections[
                i * LATENT_CODES_PER_PROJECTION : (i + 1) * LATENT_CODES_PER_PROJECTION,
                :,
            ] = (
                M(
                    CU.to_device(
                        latent_codes[
                            i
                            * LATENT_CODES_PER_PROJECTION : (i + 1)
                            * LATENT_CODES_PER_PROJECTION,
                            :,
                        ],
                        device,
                    )
                )
                .cpu()
                .numpy()
            )
    print(f"Successfully projected {projections.shape[0]} latent codes!")

    # Save to disk
    print(f"Saving to file '{OUT_W_FILE}'...")
    np.save(OUT_W_FILE, projections)
    print("Done!")


def _generate_images(latent_codes: torch.Tensor):
    # Load generator
    _, generator = _load_generator()

    # Generate images
    with torch.no_grad():
        n_batches = int(np.ceil(latent_codes.shape[0] / BATCH_SIZE))
        for i in tqdm(range(n_batches), desc="Generating images"):
            imgs = generator(
                CU.to_device(
                    latent_codes[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :], device
                )
            )[0].cpu()

            for j in range(imgs.shape[0]):
                torchvision.utils.save_image(
                    tensor_transform_reverse(imgs[j : j + 1]),
                    f"{OUT_DIR}/{i * BATCH_SIZE + j}.png",
                    nrow=1,
                    padding=0,
                    normalize=True,
                    range=(0, 1),
                )

    # Clean up
    del generator
    CU.empty_cache()


# * From: https://github.com/microsoft/StyleSwin.git (MIT license)
def tensor_transform_reverse(image):
    assert image.dim() == 4
    moco_input = torch.zeros(image.size()).type_as(image)
    moco_input[:, 0, :, :] = image[:, 0, :, :] * 0.229 + 0.485
    moco_input[:, 1, :, :] = image[:, 1, :, :] * 0.224 + 0.456
    moco_input[:, 2, :, :] = image[:, 2, :, :] * 0.225 + 0.406
    return moco_input


def _load_generator():
    G = Generator(
        256,
        LATENT_CODE_LENGTH,
        8,
        channel_multiplier=G_CHANNEL_MULTIPLIER,
        lr_mlp=LR_MLP,
        enable_full_resolution=ENABLE_FULL_RESOLUTION,
        use_checkpoint=False,
    )

    # Load pretrained state
    ckpt = torch.load(str(PRETRAIN_DIR), map_location="cpu")
    G.load_state_dict(ckpt["g_ema" if USE_G_EMA else "g"])
    del ckpt

    # Extract mapping network (Z -> W)
    M = G.style

    # Replace mapping network
    G.style = torch.nn.Identity()

    return (
        CU.to_device(M, device).eval(),
        CU.to_device(G, device).eval(),
    )


# * For use by ASAD
def get_generator():
    _, G = _load_generator()
    return lambda z: G(z)[0].contiguous()


if __name__ == "__main__":
    main()
