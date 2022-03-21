import random

from src.dataset.Dataset import Dataset
from src.dataset.TorchImageDataset import TorchImageDataset
import src.util.ModelUtil as ModelUtil
from src.util.CudaUtil import to_device, get_default_device, empty_cache

from tqdm import tqdm
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.models.resnet as resnet

#! Data constants
DATA_SEED = 69
DATA_TRAINING_SPLIT = 0.8

#! Training Constants
BATCH_SIZE = 25  # Changing batch-size requires retraining from scratch
NUM_EPOCHS = 100
NUM_RAD_TRAIN_BATCHES_PER_VALIDATION = 10
NUM_BATCHES_PER_VALIDATION = 500  # The last batches are for radius training
NUM_VALIDATIONS_PER_LR = 150

#! Model Constants
PARAM_RADIUS = "radius"
PARAM_CENTER = "center"
AUX_MODEL_NAME = "evaluation_embedding"
MODEL_PARAM_NU = 0.01

#! Network
class DeepSVDDNet(torch.nn.Module):
    """
    Represents a Deep SVDD neural network.
    """

    OUTPUT_DIM = 1000

    def __init__(self):
        """
        Constructs a new Deep SVDD network. The network consist of a ResNet50
        network and a radius parameter.
        """
        super().__init__()

        # Main network
        self.net = resnet.resnet50()
        self.net.fc = torch.nn.Linear(512 * 4, 1000, bias=False)

        # Radius parameter
        self.register_parameter(
            name=PARAM_RADIUS,
            param=torch.nn.Parameter(torch.rand(1), requires_grad=True),
        )

        # Center parameter
        self.register_parameter(
            name=PARAM_CENTER,
            param=torch.nn.Parameter(torch.zeros(self.OUTPUT_DIM), requires_grad=False),
        )

    def forward(self, x):
        return self.net.forward(x)


def soft_boundary_loss_func(
    rep: torch.Tensor,
    radius: torch.nn.parameter.Parameter,
    center: torch.nn.parameter.Parameter,
    device: torch.device,
    nu: float = 0.01,
) -> torch.Tensor:
    """
    A soft boundary loss function, for usage with output from
    a Deep SVDD network.

    Args:
        rep (torch.Tensor): The network output to evaluate.
        radius (torch.nn.parameter.Parameter): The radius parameter
            of the Deep SVDD network.
        center (torch.nn.parameter.Parameter): The center parameter
            of the Deep SVDD network.
        device (torch.device): The device to compute on.
        nu (float, optional): A parameter in the formula. Defaults to 0.01.

    Returns:
        torch.Tensor: A one-dimensional tensor with the loss score.
    """
    n = rep.shape[0]

    return (n * radius**2) + 1 / nu * torch.sum(
        torch.max(
            to_device(torch.zeros(n), device),
            torch.sum((rep - center) ** 2, 1) - radius**2,
        ),
        0,
    )


#! Training


def _train_validate(
    epoch: int,
    batch: int,
    net: DeepSVDDNet,
    loader_train: torch.utils.data.DataLoader,
    loader_valid: torch.utils.data.DataLoader,
    train_loss: float,
    device: torch.device,
) -> None:
    """Compute validation score and save to disk."""

    # Set to evaluation mode
    net.eval()

    # Counters
    valid_loss = 0
    num_validation_batches = 0

    # Start evaluation of validation data
    with torch.no_grad():
        for data in tqdm(loader_valid, position=2, desc="Validation", leave=False):
            # Send batch to device
            data = to_device(data, device)

            # Increment number processed of batches
            num_validation_batches += 1

            # Add loss
            valid_loss += soft_boundary_loss_func(
                net(data),
                net.get_parameter(PARAM_RADIUS),
                net.get_parameter(PARAM_CENTER),
                device,
                MODEL_PARAM_NU,
            ).item()

    # Compute loss average
    valid_loss /= num_validation_batches

    # Set model to training mode
    net.train()

    # Inform CLI
    tqdm.write(
        f"(e: {epoch}, b: {batch}) | Validation loss: {valid_loss:.6f}"
        + f" | Radius: {net.get_parameter(PARAM_RADIUS).item():.6f}"
    )

    # Save results to disk
    ModelUtil.save_aux(
        AUX_MODEL_NAME,
        ModelUtil.AuxModelInfo(
            net.state_dict(),
            epoch,
            batch,
            len(loader_train),
            train_loss,
            valid_loss,
        ),
    )


# Used for keeping track of the full number of processed batches
_batch_iterator = 0


def _train_epoch(
    epoch: int,
    start_batch: int,
    net: DeepSVDDNet,
    loader_train: torch.utils.data.DataLoader,
    loader_valid: torch.utils.data.DataLoader,
    net_optimizer: torch.optim.AdamW,
    rad_optimizer: torch.optim.LBFGS,
    scheduler: torch.optim.lr_scheduler.StepLR,
    device: torch.device,
) -> None:
    """Train for a single epoch."""
    global _batch_iterator

    # Iterate through batches
    for i, data in tqdm(
        enumerate(loader_train),
        total=len(loader_train),
        position=1,
        desc="Batch",
        leave=False,
    ):
        # Increment batch counter
        _batch_iterator += 1

        # Fast forward to skip pre-trained batches
        if (
            start_batch > 0
            and _batch_iterator <= (epoch - 1) * len(loader_train) + start_batch
        ):
            continue

        # Send batch to GPU
        data = to_device(data, device)

        # Pick optimizer based on current batch
        train_net = (
            _batch_iterator % NUM_BATCHES_PER_VALIDATION
            < NUM_BATCHES_PER_VALIDATION - NUM_RAD_TRAIN_BATCHES_PER_VALIDATION
        )
        optimizer = net_optimizer if train_net else rad_optimizer

        def closure():
            """Calculates loss and gradients."""

            # Reset parameter gradients
            optimizer.zero_grad()

            # Forward + backward
            outputs = net(data)
            loss = soft_boundary_loss_func(
                outputs,
                net.get_parameter(PARAM_RADIUS),
                net.get_parameter(PARAM_CENTER),
                device,
                MODEL_PARAM_NU,
            )
            loss.backward()

            return loss

        # Optimize
        loss = None
        if train_net:
            # Train network (skip radius)
            loss = closure()
            optimizer.step()
        else:
            # Save old radius
            prev_rad = net.get_parameter(PARAM_RADIUS).item()

            # Optimize
            optimizer.step(closure)
            loss = closure()

            # Reset optimizer if loss is NaN
            if loss.item() != loss.item():
                optimizer.reset_state()
                net.get_parameter(PARAM_RADIUS).data = to_device(
                    torch.tensor([prev_rad]), device
                )

        # Validate and save to disk if applicable
        if (
            _batch_iterator - 1
        ) % NUM_BATCHES_PER_VALIDATION == NUM_BATCHES_PER_VALIDATION - 1:
            # Validate
            _train_validate(
                epoch, i + 1, net, loader_train, loader_valid, loss.item(), device
            )

            # Update schedule
            scheduler.step()


def train(dataset: Dataset, start_state: ModelUtil.AuxModelInfo = None) -> None:
    """
    Trains the evaluation embedding model and saves it to disk for
    different points along the training. The model is saved as an
    auxiliary model using `util.ModelUtil`, under the name given
    by the `metric.EvaluationEmbedding.AUX_MODEL_NAME` constant.

    Args:
        dataset (Dataset): The dataset to train on.
        start_state (AuxModelInfo): The state to start from, or
            `None` if the training should start from scratch.
            Defaults to None.
    """
    global _batch_iterator

    # Load device
    device = get_default_device()

    # Split into training / validation
    uri_list = dataset.get_image_paths()
    random.seed(DATA_SEED)
    random.shuffle(uri_list)

    n = len(uri_list)
    n_train = int(np.floor(n * DATA_TRAINING_SPLIT))

    uri_train = uri_list[:n_train]
    uri_valid = uri_list[n_train:]

    # Create composite transform
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Create datasets
    ds_train = TorchImageDataset(uri_train, transform)
    ds_valid = TorchImageDataset(uri_valid, transform)

    # Create dataset loaders
    loader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    loader_valid = torch.utils.data.DataLoader(
        ds_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Create network
    net = to_device(
        DeepSVDDNet(),
        device,
    )

    # Load start state if applicable
    if start_state is not None:
        net.load_state_dict(start_state.state)

    # Define optimizers
    net_optimizer = torch.optim.AdamW(
        net.net.parameters(),
        weight_decay=0.01,
        lr=0.0001,
    )

    rad_optimizer = torch.optim.LBFGS(
        [net.get_parameter(PARAM_RADIUS)],
        line_search_fn="strong_wolfe",
    )

    # Add function for resetting optimizer
    rad_optimizer_start_state = rad_optimizer.state_dict()
    rad_optimizer.reset_state = lambda: rad_optimizer.load_state_dict(
        rad_optimizer_start_state
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        net_optimizer, step_size=NUM_VALIDATIONS_PER_LR, gamma=0.1
    )

    # Set up batch iterator according to start_state
    _batch_iterator = (
        (start_state.epoch - 1) * len(loader_train) if start_state is not None else 0
    )

    # Derive center parameter if applicable
    if start_state is None:

        # Perform initial forward pass
        tqdm.write("Determining center parameter...")
        projs = np.zeros((n_train, DeepSVDDNet.OUTPUT_DIM))
        for i, data in tqdm(
            enumerate(loader_train),
            total=len(loader_train),
            desc="Initial forward pass",
        ):
            projs[(i * BATCH_SIZE) : ((i + 1) * BATCH_SIZE), :] = (
                net(to_device(data, device)).cpu().detach().numpy()
            )

        # Derive center parameter
        center = np.mean(projs, axis=0)

        # Send to GPU
        net.get_parameter(PARAM_CENTER).data = to_device(torch.tensor(center), device)

        # Inform CLI
        tqdm.write(
            f"Done! Center (mean, std) = ({np.mean(center):.6f}, {np.std(center):.6f})"
        )

        # Clear from stack
        del projs, center

    # Start training
    for epoch in tqdm(range(NUM_EPOCHS), position=0, desc="Epoch"):

        # Fast forward to skip pre-trained epochs
        if start_state is not None and epoch < start_state.epoch - 1:
            continue

        # Train epoch
        _train_epoch(
            epoch + 1,
            start_state.batch if start_state is not None else -1,
            net,
            loader_train,
            loader_valid,
            net_optimizer,
            rad_optimizer,
            scheduler,
            device,
        )

        # For next epoch, ignore start_state
        start_state = None

    # Clean up
    empty_cache()


#! Fetching


def get() -> DeepSVDDNet:
    """
    Fetches the evaluation embedding model if it has been pre-trained.

    For training the model, use `train(Dataset)` instead.

    Returns:
        DeepSVDDNet: The evaluation embedding model, or `None` if it
            has not yet been trained.
    """

    # Try to load model
    model_info = ModelUtil.load_aux_best(AUX_MODEL_NAME)
    if model_info is not None:

        # Create network
        model = DeepSVDDNet()

        # Load parameters
        model.load_state_dict(model_info.state)

        # Set to evaluation mode
        model.eval()

        # Return model
        return model

    # Not yet trained!
    return None
