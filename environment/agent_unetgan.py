# from https://github.com/ajbrock/BigGAN-PyTorch (MIT license) - some modifications
""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).
"""

# from https://github.com/boschresearch/unetgan (AGPL-3.0 license)
"""
    A u-net based discriminator for generative adversarial networks 2020
    Schonfeld, Edgar and Schiele, Bernt and Khoreva, Anna
"""
# Modifications to some functions.


import os
from tkinter import Y
import argparse
import sys
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import time as time
from latent_code_receiver import receive_latent_codes

# Constants
BATCH_SIZE = 2  # * Lower this if memory problems

# Setup path
old_path = Path().absolute()
new_path = old_path / "environment" / "unetgan"
sys.path.append(str(new_path))
os.chdir(new_path)

# Local from unetgan
import utils


def load_weights(
    G,
    config: dict,
    name_suffix: str = None,
    G_ema=None,
    strict: bool = True,
    load_optim: bool = True,
) -> None:
    """
    Load pretrained weight to the generator.

    Args:
        G (BigGAN.Generator): Generator from BigGAN.
        config (dict): Dictionary with the configurations.
        name_suffix (str, optional): Suffix of the name. Defaults to None.
        G_ema (BigGAN.Generator, optional): EMA generator from BigGAN.
            Defaults to None.
        strict (bool, optional): Parameter for pytorch model.load_state_dict.
            Defaults to True.
        load_optim (bool, optional): Load G_optim if true, else load G from U-NetGAN.
            Defaults to True.
    """
    root = config["resume_from"]
    epoch_id = config["epoch_id"]
    if name_suffix:
        print("Loading %s weights from %s..." % (name_suffix, root))
    else:
        print("Loading weights from %s..." % root)
    if G is not None:
        G.load_state_dict(
            torch.load(
                "%s/%s.pth"
                % (root, utils.join_strings("_", ["G", epoch_id, name_suffix]))
            ),
            strict=strict,
        )
        if load_optim:
            s = torch.load(
                "%s/%s.pth"
                % (
                    root,
                    utils.join_strings("_", ["G_optim", epoch_id, name_suffix]),
                )
            )
            G.optim.load_state_dict(
                torch.load(
                    "%s/%s.pth"
                    % (
                        root,
                        utils.join_strings("_", ["G_optim", epoch_id, name_suffix]),
                    )
                )
            )

    if G_ema is not None:
        G_ema.load_state_dict(
            torch.load(
                "%s/%s.pth"
                % (
                    root,
                    utils.join_strings("_", ["G_ema", epoch_id, name_suffix]),
                )
            ),
            strict=strict,
        )


def save_images(
    G_z: torch.Tensor, config: dict, true_batch_size: int, start_id: int
) -> None:
    """
    Save images to the disk.

    Args:
        G_z (torch.Tensor): Tensor containing `true_batch_size` images.
        config (dict): Dictionary with the configurations.
        true_batch_size (int): Number of images in one batch.
        start_id (int): Id tag for the first image.
    """
    # Create base image name
    image_filename = "%s/" % (config["sample_root"])

    # Loop through the images, saving them to disk
    for i in range(true_batch_size):
        im = G_z[i, :, :, :]
        im_name = image_filename + str(start_id) + ".png"
        torchvision.utils.save_image(im.float().cpu(), im_name, normalize=True)
        start_id += 1


def load_generator(config: dict):
    """
    Create the generator and load its weights using the function `load_weights`.

    Args:
        config (dict): Dictionary with the configurations.

    Returns:
        BigGAN.Generator: The generator.
    """
    # GPU
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    # TODO: how to handle seed?
    # Seed RNG
    utils.seed_rng(config["seed"])

    # Import the model
    model_name = "BigGAN"  # ! Code rewrite only supports BigGAN
    model = __import__(model_name)

    # Create generator and load it to the GPU
    G = model.Generator(**config).to(device)

    # If using EMA, prepare it
    if config["ema"]:
        G_ema = model.Generator(**{**config, "skip_init": True, "no_optim": True}).to(
            device
        )
        ema = utils.ema(G, G_ema, config["ema_decay"], config["ema_start"])
    else:
        G_ema, ema = None, None

    # If loading from a pre-trained model, load weights
    try:
        load_weights(G, config, G_ema=G_ema if config["ema"] else None)
    except:
        load_weights(G, config, G_ema=None)
        G_ema.load_state_dict(G.state_dict())

    # Switch to eval mode
    G.eval()
    if config["ema"]:
        G_ema.eval()

    return G_ema if config["ema"] and config["use_ema"] else G


def sample_generator(G, config: dict, z: torch.Tensor) -> None:
    """
    Sample images from the generator given the latent codes `z`.

    Args:
        G (BigGAN.Generator): The generator.
        config (dict): Dictionary with the configurations.
        z (torch.Tensor): Tensor containing the latent codes.
    """
    start_time = time.time()
    torch.cuda.empty_cache()

    # Setup batches
    n = z.size()[0]
    z_batches = torch.split(z, config["batch_size"])
    n_batches = len(z_batches)

    start_id = 0
    with torch.no_grad():
        for z_batch in tqdm(z_batches, desc="Generating images"):
            curr_batch_size = z_batch.size()[0]

            # Sample the generator
            # Define unconditional labels (zero in the unconditional case)
            G_z = G(
                z_batch,
                G.shared(torch.zeros(size=(curr_batch_size,)).float().to("cuda")),
            )

            # Save images to disk
            save_images(G_z, config, curr_batch_size, start_id)

            start_id += curr_batch_size
    print(
        f"Image generation completed in {round((time.time() - start_time)/60, 2)} minutes!"
    )


def setup_config(config: dict) -> dict:
    """
    Setup configuration for the image generation, many settings are
    based on the load_pretrained_ffhq.sh file from the U-NetGAN github.

    Don't change these settings if you dont know what you are doing.
    Args:
        config (dict): Dictionary with the configurations.

    Returns:
        dict: Dictionary with the updated configurations.
    """
    # From load_pretrained_ffhq.sh, dont change these
    config["epoch_id"] = "ep_82"
    config["unconditional"] = True
    config["resume_from"] = "pretrained_model"
    config["seed"] = 1337  # ? Not sure what this does for U-NetGAN
    config["G_ch"] = 64
    config["D_ch"] = 64
    config["G_eval_mode"] = True
    config["G_init"] = "ortho"
    config["D_init"] = "ortho"
    config["G_ortho"] = 0.0
    config["dataset"] = "FFHQ"
    config["resume"] = True
    config["id"] = "ffhq_unet_bce_noatt_cutmix_consist"
    config["hier"] = True
    config["dim_z"] = 128
    config["G_attn"] = "0"
    config["D_attn"] = "0"
    config["gpus"] = "0"
    config["use_ema"] = True
    config["ema"] = True
    config["ema_start"] = 21000
    """
    config["test_every"] = 10000
    config["save_every"] = 10000
    config["num_best_copies"] = 2
    config["num_save_copies"] = 1
    config["sample_every"] = 1
    config["num_G_accumulations"] = 1
    config["num_d_accumulations"] = 1
    config["sample_every"] = 1
    """
    # Further config options from train.py, dont change these
    config["resolution"] = utils.imsize_dict[config["dataset"]]
    config["n_classes"] = utils.nclass_dict[config["dataset"]]
    config["G_activation"] = utils.activation_dict[config["G_nl"]]
    config["skip_init"] = True
    return config


def setup_directories(config: dict) -> dict:
    """
    Create directories and update configs accordingly.

    Args:
        config (dict): Dictionary with the configurations.

    Returns:
        dict: Dictionary with the updated configurations.
    """
    # Create experiment directories
    if not os.path.isdir(config["base_root"]):
        os.makedirs(config["base_root"])
    if not os.path.isdir(os.path.join(config["base_root"], "sample")):
        os.makedirs(os.path.join(config["base_root"], "sample"))
    if not os.path.isdir(os.path.join(config["base_root"], "metadata")):
        os.makedirs(os.path.join(config["base_root"], "metadata"))

    # Update config roots
    for key in ["metadata", "sample"]:
        config["%s_root" % key] = "%s/%s" % (config["base_root"], key)
    return config


def main(config: dict, z: torch.Tensor):
    """
    Samples images from the U-Net Generator(BigGAN) given latent codes.

    Args:
        config (dict): Dictionary with the configurations.
        z (torch.Tensor): Tensor containing the latent codes.
    """
    # Setup arguments in config
    config = setup_config(config)

    # Create directories update roots
    config = setup_directories(config)

    # Load the generator
    G = load_generator(config)

    # Sample the generator
    sample_generator(G, config, z)


def float_list_to_tensor(config: dict, latent_codes: list) -> torch.Tensor:
    """
    Transform list to tensor by unflattening the latent codes
    the correct shape.

    Args:
        config (dict): _description_
        latent_codes (list): _description_

    Returns:
        torch.Tensor: _description_
    """
    z = torch.tensor(latent_codes, dtype=torch.float)
    dim_z = config["dim_z"]
    num_latent_codes = int(z.shape[0] / dim_z)
    return torch.reshape(z, (num_latent_codes, dim_z)).to("cuda")


def get_generator():
    """
    Returns the generator.

    Note that originally, BigGAN uses G(z,y), but in our unconditional case,
    it is transformed to return the generator in the classic G(z) format.

    Returns:
        BigGan: The generator.
    """
    # Setup their argparser
    parser = utils.prepare_parser()

    # Setup arguments in config
    config = vars(parser.parse_known_args()[0])
    config = setup_config(config)

    # Load the generator
    G = load_generator(config)

    # Change back path
    os.chdir(old_path)
    return lambda z: G(z, G.shared(torch.zeros(size=(z.size()[0],)).float().to("cuda")))


if __name__ == "__main__":
    # Setup their argparser
    parser = utils.prepare_parser()

    # Add argument
    parser.add_argument(
        "--sample_root",
        type=str,
        default="sample",
        help="Default location to store samples (default: %(default)s)",
    )

    config = vars(parser.parse_args())

    z = float_list_to_tensor(config, receive_latent_codes(config["dim_z"]))
    # print(receive_latent_codes(2))
    # exit()
    # Get latent codes from EnvironmentManager

    # Check that BATCH_SIZE is appropriate.
    num_latent_codes = z.shape[0]
    if BATCH_SIZE < num_latent_codes:
        config["batch_size"] = BATCH_SIZE
    else:
        raise ValueError("Batch size must be lower than number of samples")

    # Arguments from load_pretrained_ffhq.sh, relative paths from unetgan folder.
    # Base root directory where sampling directories are stored.
    config["base_root"] = "./out"  #! modify this

    main(config, z)
