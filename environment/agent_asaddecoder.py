from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Callable
import torch
import math
from tqdm import tqdm

# Setup path
new_path = Path().absolute()
sys.path.append(str(new_path))

import src.util.CudaUtil as CU

from src.controller.ASADControllerModels import (
    get_dec_arch,
    get_cls_arch,
)

from src.util.ModelUtil import AuxModelInfo, save_aux, load_aux_best
from src.generator.Generator import Generator


def _get_config() -> dict:
    # Setup parser
    parser = ArgumentParser()
    parser.add_argument("--generator_name", type=str)
    parser.add_argument("--decoder_name", type=str)
    parser.add_argument("--classifier_name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--iter_per_epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    return vars(parser.parse_args())


def main():
    """
    Train decoder specified by the `decoder_name` parameter using generator
    specified by the `generator_name` parameter.

    Requires classifier named `classifier_name`
    associated with the decoder.

    ! Needs to be extended when new generators are added.
    * Now supports the following generators: unetgan
    """
    config = _get_config()

    # Dynamic import of generator
    # * Add new generators here
    if config["generator_name"] == "unetgan":
        from environment.agent_unetgan import get_generator
        from src.generator.UNetGANGenerator import UNetGANGenerator as Gen

    # Get generator
    G = get_generator()
    _setup_decoder(config, Gen(), G)


def _setup_decoder(
    config: dict, gen: Generator, G: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    # Setup GPU
    device = CU.get_default_device()

    # Init decoder model
    name = config["decoder_name"]
    model = CU.to_device(get_dec_arch(gen), device)

    # Init and load pretrained classifier model
    cls_name = config["classifier_name"]
    cls_model = get_cls_arch()
    cls_model.load_state_dict(load_aux_best(cls_name).state)
    cls_model = CU.to_device(cls_model, device)
    cls_model.eval()

    # Fit decoder
    _fit_decoder(config, model, cls_model, device, name, gen, G)


def _decoder_loss(
    control_vectors: torch.Tensor,
    codes: torch.Tensor,
    cls_model: Callable[[torch.Tensor], torch.Tensor],
    G: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    # To avoid taking log of zero
    eps = 0.000001

    # According to ASAD paper
    sum_dim = (1, 2, 3)
    G_minus_n = G(codes - control_vectors)
    c1 = torch.mean(torch.log(cls_model(G_minus_n) + eps))
    mse1 = torch.mean(torch.sum((G_minus_n - G(codes)) ** 2, sum_dim))

    del G_minus_n
    torch.cuda.empty_cache()

    C = c1 + torch.mean(torch.log(1 - cls_model(G(codes + control_vectors)) + eps))
    MSE = mse1 + torch.mean(
        torch.sum((G(codes + control_vectors) - G(codes)) ** 2, sum_dim)
    )
    return C + MSE


def _fit_decoder(
    config: dict,
    model: Callable[[torch.Tensor], torch.Tensor],
    cls_model: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    name: str,
    gen: Generator,
    G: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    torch.cuda.empty_cache()

    # Losses
    tr_losses = []
    val_losses = []

    # Create optimizers
    opt = torch.optim.AdamW(model.parameters())

    # Setup batches
    n_batches = math.floor(config["iter_per_epoch"] / config["batch_size"])
    tr_percent = 0.8
    tr_batches = math.floor(n_batches * tr_percent)

    # Start training loop
    for epoch in range(config["epochs"]):
        avg_loss = 0
        for batch_nr in tqdm(range(n_batches)):
            # Create seed latent codes
            codes = gen.random_latent_code(config["batch_size"]).astype("float32")

            # Send data to GPU
            codes = CU.to_device(torch.from_numpy(codes), device)

            if batch_nr + 1 >= tr_batches:
                # Validate decoder
                val_loss = _validate_decoder(config, model, cls_model, codes, G)

                # Record losses
                avg_loss += val_loss
                val_losses.append((val_loss, batch_nr))
            else:
                # Train decoder
                tr_loss = _train_decoder(config, model, cls_model, codes, opt, G)

                # Record losses
                tr_losses.append((tr_loss, batch_nr))

        # Log losses & scores (last batch)
        avg_loss = avg_loss / (n_batches - tr_batches)
        print(
            "Epoch [{}/{}], train loss: {:.4f}, val loss: {:.4f}".format(
                epoch + 1, config["epochs"], tr_loss, avg_loss
            )
        )
        # Save result
        save_aux(
            name,
            AuxModelInfo(
                model.state_dict(),
                epoch + 1,
                batch_nr + 1,
                tr_batches,
                tr_loss,
                avg_loss,
            ),
        )
    pass


def _train_decoder(
    config: dict,
    model: Callable[[torch.Tensor], torch.Tensor],
    cls_model: Callable[[torch.Tensor], torch.Tensor],
    codes: torch.Tensor,
    opt,
    G: Callable[[torch.Tensor], torch.Tensor],
):
    # Clear model gradients
    opt.zero_grad()
    torch.cuda.empty_cache()

    # Get predictions
    control_vectors = model(codes)

    # Calc loss
    loss = _decoder_loss(control_vectors, codes, cls_model, G)

    # Update weights
    loss.backward()
    opt.step()
    return loss.item()


def _validate_decoder(
    config: dict,
    model: Callable[[torch.Tensor], torch.Tensor],
    cls_model: Callable[[torch.Tensor], torch.Tensor],
    codes: torch.Tensor,
    G: Callable[[torch.Tensor], torch.Tensor],
):
    # Clear model gradients
    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        # Get predictions
        control_vectors = model(codes)

        # Calc loss
        loss = _decoder_loss(control_vectors, codes, cls_model, G)
    model.train()
    return loss.item()


if __name__ == "__main__":
    main()
