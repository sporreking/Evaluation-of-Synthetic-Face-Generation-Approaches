import sys
from pathlib import Path
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from latent_code_receiver import receive_latent_codes

# * Add generator project root to python path
PROJECT_ROOT_DIR = Path().absolute()
GEN_ROOT_DIR = Path().absolute() / "environment" / "stylegan2ada"
sys.path.append(str(PROJECT_ROOT_DIR))
sys.path.append(str(GEN_ROOT_DIR))


# * Imports from base project root
import src.util.CudaUtil as CU

# * Imports from generator project root
import dnnlib
import legacy

# * Read system input

LAUNCH_MODE_Z = "z"
LAUNCH_MODE_W = "w"
LAUNCH_MODE_PROJECT = "p"

LAUNCH_MODE_LIST = [LAUNCH_MODE_Z, LAUNCH_MODE_W, LAUNCH_MODE_PROJECT]

parser = ArgumentParser()
parser.add_argument("--mode", type=str, choices=LAUNCH_MODE_LIST, default=LAUNCH_MODE_W)
parser.add_argument("--truncation_psi", type=float, default=0.7)
parser.add_argument("--truncation_cutoff", type=float, default=8)
parser.add_argument(
    "--noise_mode", type=str, choices=["const", "random", "none"], default="const"
)

ARGS = parser.parse_args(sys.argv[1:] if __name__ == "__main__" else [])
LAUNCH_MODE = ARGS.mode


# * Constants

# Directories
PRETRAIN_DIR = GEN_ROOT_DIR / "pretrain"
OUT_DIR = GEN_ROOT_DIR / "out"
OUT_W_FILE = OUT_DIR / "w.npy"

# Generation
LATENT_CODE_LENGTH = 512
TRUNCATION_PSI = ARGS.truncation_psi
TRUNCATION_CUTOFF = ARGS.truncation_cutoff
NOISE_MODE = ARGS.noise_mode
W_PLUS_DIM = 14
LATENT_CODES_PER_PROJECTION = 100

# Networking
NETWORK_PKL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq140k-paper256-ada-bcr.pkl"

# Load GPU if possible
device = CU.get_default_device()


def main():

    # Receive latent codes
    lc_data = receive_latent_codes(LATENT_CODE_LENGTH)
    latent_codes = torch.tensor(lc_data).reshape(
        (int(len(lc_data) / LATENT_CODE_LENGTH), LATENT_CODE_LENGTH)
    )

    # Create cache directory
    PRETRAIN_DIR.mkdir(exist_ok=True)

    # Create output directory
    OUT_DIR.mkdir(exist_ok=True)

    # Perform projection if applicable
    if LAUNCH_MODE == LAUNCH_MODE_PROJECT:
        G = CU.to_device(_load_generator(), device)

        print("Projecting latent codes from Z to W...")
        n_lc = latent_codes.shape[0]
        w = np.zeros((n_lc, LATENT_CODE_LENGTH))
        with torch.no_grad():
            for i in range(int(np.ceil(n_lc / LATENT_CODES_PER_PROJECTION))):
                w[
                    i
                    * LATENT_CODES_PER_PROJECTION : (i + 1)
                    * LATENT_CODES_PER_PROJECTION,
                    :,
                ] = (
                    G.mapping(
                        CU.to_device(
                            latent_codes[
                                i
                                * LATENT_CODES_PER_PROJECTION : (i + 1)
                                * LATENT_CODES_PER_PROJECTION,
                                :,
                            ],
                            device,
                        ),
                        None,
                        truncation_psi=TRUNCATION_PSI,
                        truncation_cutoff=TRUNCATION_CUTOFF,
                    )[:, 0, :]
                    .cpu()
                    .numpy()
                )
        print(f"Successfully projected {n_lc} latent codes!")

        print(f"Saving to file '{OUT_W_FILE}'...")
        np.save(OUT_W_FILE, w)
        print("Done!")

        return

    # Inform CLI about current launch state
    print(f"Assuming latent codes to be from the {LAUNCH_MODE.upper()}-domain.")

    # Load pretrained network
    G = get_generator()

    # Generate images
    with torch.no_grad():
        for i in tqdm(range(latent_codes.shape[0]), desc="Generating images"):
            img = G(CU.to_device(latent_codes[i : i + 1, :], device))
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            Image.fromarray(img[0].cpu().numpy(), "RGB").save(f"{OUT_DIR}/{i}.png")

    print(
        f"Successfully generated {latent_codes.shape[0]} images from requested latent codes!"
    )


def _load_generator():
    with dnnlib.util.open_url(NETWORK_PKL, cache_dir=str(PRETRAIN_DIR)) as f:
        return legacy.load_network_pkl(f)["G_ema"].eval()


# * For use by ASAD
def get_generator():
    G = CU.to_device(_load_generator(), device)
    return (
        (
            lambda w: G.synthesis(
                w.repeat(1, W_PLUS_DIM, 1),
                noise_mode=NOISE_MODE,
                force_fp32=True,
            )
        )
        if LAUNCH_MODE == LAUNCH_MODE_W
        else (
            lambda z: G(
                z,
                None,
                truncation_psi=TRUNCATION_PSI,
                truncation_cutoff=TRUNCATION_CUTOFF,
                noise_mode=NOISE_MODE,
            )
        )
    )


if __name__ == "__main__":
    main()
