from __future__ import annotations

from typing import Union

from src.dataset.Dataset import Dataset
import src.util.ModelUtil as ModelUtil
from src.util.CudaUtil import to_device, get_default_device, empty_cache

from tqdm import tqdm
import numpy as np

import torch
import torchvision
import torchvision.transforms as T

#! Data constants
DATA_SEED = 69
DATA_TRAINING_SPLIT = 0.8

#! Training Constants
BATCH_SIZE = 28  # Changing batch-size requires retraining from scratch
NUM_EPOCHS = 1000
NUM_RAD_TRAIN_BATCHES_PER_VALIDATION = 10
NUM_BATCHES_PER_VALIDATION = 2000 * 1  # The last batches are for radius training
NUM_VALIDATIONS_PER_LR = 20

#! Model Constants
PARAM_RADIUS = "radius"
PARAM_CENTER = "center"
AUX_MODEL_NAME = "evaluation_embedding"
MODEL_PARAM_NU = 0.01

#! Projection Constants
DS_PROJ_FILE_PREFIX = "ee_proj"
DS_PROJ_BATCH_SIZE = 25

#! Inception Constants
INCEPTION_OUTPUT_DIM = 2048
INCEPTION_PROJ_FILE_PREFIX = "inception_proj"

#! Network
class DeepSVDDNet(torch.nn.Module):
    """
    Represents a Deep SVDD neural network.
    """

    def __init__(
        self,
        num_hidden_layers: int = 2,
        num_nodes_per_hidden_layer: int = 128,
        num_output_nodes: int = 32,
    ):
        """
        Constructs a new Deep SVDD network. The network consist of several hidden layers
        combined with a final output layer, all according to the specified constructor
        arguments. After each hidden layer, a ReLU activation function is used.

        Note that there are no biases in the layers.

        Args:
            num_hidden_layers (int, optional): The number of hidden layers to have.
                Defaults to 2.
            num_nodes_per_hidden_layer (int, optional): The number of nodes to use
                for each hidden layer. Defaults to 512.
            num_output_nodes (int, optional): The number of output nodes, i.e., the
                dimensions of the entire network output. Defaults to 128.

        Raises:
            ValueError: If any of the network parameters are invalid.
        """
        super().__init__()

        # Sanity check
        if num_hidden_layers < 1:
            raise ValueError("Must have at least one hidden layer!")

        if num_nodes_per_hidden_layer < 1:
            raise ValueError("Must have at least one node per hidden layer!")

        if num_output_nodes < 1:
            raise ValueError("Must have at least one output node!")

        # Store network configuration
        self._num_hidden_layers = num_hidden_layers
        self._num_nodes_per_hidden_layer = num_nodes_per_hidden_layer
        self._num_output_nodes = num_output_nodes

        # DeepSVDDNetwork
        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                INCEPTION_OUTPUT_DIM, num_nodes_per_hidden_layer, bias=False
            ),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(),
            *(
                torch.nn.Sequential(
                    torch.nn.Linear(
                        num_nodes_per_hidden_layer,
                        num_nodes_per_hidden_layer,
                        bias=False,
                    ),
                    torch.nn.ReLU(inplace=True),
                    # torch.nn.Dropout(),
                )
                for _ in range(num_hidden_layers - 1)
            ),
            torch.nn.Linear(num_nodes_per_hidden_layer, num_output_nodes, bias=False),
        )

        # Radius parameter
        self.register_parameter(
            name=PARAM_RADIUS,
            param=torch.nn.Parameter(torch.tensor(0.5), requires_grad=True),
        )

        # Center parameter
        self.register_parameter(
            name=PARAM_CENTER,
            param=torch.nn.Parameter(
                torch.zeros(num_output_nodes), requires_grad=False
            ),
        )

    @property
    def num_hidden_layers(self):
        return self._num_hidden_layers

    @property
    def num_nodes_per_hidden_layer(self):
        return self._num_nodes_per_hidden_layer

    @property
    def num_output_nodes(self):
        return self._num_output_nodes

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


#! Inception


class InceptionDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self._data[idx, :]).float()


def get_inception_model() -> torchvision.models.inception.Inception3:
    """
    Creates a new inception v3 model instance and returns it.
    The top-layer is removed.

    Returns:
        torchvision.models.inception.Inception3: The new inception v3 instance.
    """

    m = torchvision.models.inception.inception_v3(
        pretrained=True, aux_logits=False, transform_input=False
    )
    m.fc = torch.nn.Flatten()

    return m.eval()


def get_inception_image_transform():
    """
    Returns a composite transform for converting images to
    inception v3 compatible input.

    Returns:
        Transform: A composite transform for inception v3 pre-processing.
    """
    return T.Compose(
        [
            T.Resize(299),
            T.CenterCrop(299),
            T.ToTensor(),
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def _inception_proj_file_name(dataset: Dataset) -> str:
    return f"{INCEPTION_PROJ_FILE_PREFIX}_{dataset.get_name(dataset.get_resolution())}"


def get_inception_dataset(
    dataset: Dataset,
    data_split: float = 1.0,
) -> Union[InceptionDataset, tuple[InceptionDataset, InceptionDataset]]:

    # Load inceptified data
    data = ModelUtil.get_file_jar().get_file(
        f"{_inception_proj_file_name(dataset)}.npy",
        np.load,
    )

    # Sanity check
    if data is None:
        raise RuntimeError(
            f"Dataset '{dataset.get_name(dataset.get_resolution())}' "
            + "has not been inceptified!"
        )

    # Check if data should be split
    if data_split >= 1.0:
        return InceptionDataset(data)

    # Split into training / validation datasets
    np.random.seed(DATA_SEED)
    np.random.shuffle(data)
    n_train = int(np.floor(data.shape[0] * data_split))

    ds_train = InceptionDataset(data[:n_train, :])
    ds_valid = InceptionDataset(data[n_train:, :])

    # Return inception datasets
    return ds_train, ds_valid


def is_inceptified(dataset: Dataset):
    return (
        ModelUtil.get_file_jar().get_file(
            f"{_inception_proj_file_name(dataset)}.npy",
            np.load,
        )
        is not None
    )


def inceptify(dataset: Dataset, device: torch.cuda.device) -> None:

    tqdm.write(
        f"Inceptifying dataset '{dataset.get_name(dataset.get_resolution())}'..."
    )

    # Create torch image dataset
    ds_im = dataset.to_torch_dataset(get_inception_image_transform(), use_labels=False)
    loader = torch.utils.data.DataLoader(
        ds_im, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Get inception model
    inception = to_device(get_inception_model(), device)

    # Inceptify data
    projections = np.zeros((len(dataset), INCEPTION_OUTPUT_DIM))
    for i, data in tqdm(
        enumerate(loader),
        total=len(loader),
        desc="Transforming with inception model",
    ):
        projections[(i * BATCH_SIZE) : ((i + 1) * BATCH_SIZE), :] = (
            inception(to_device(data, device)).cpu().detach().numpy()
        )

    # Save inceptified data
    ModelUtil.get_file_jar().store_file(
        _inception_proj_file_name(dataset),
        lambda p: np.save(p, projections),
    )

    # Inform CLI
    tqdm.write(f"Done! Inceptified data saved to disk.")


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
            # Send batch to GPU
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

        def closure(retain_graph: bool = False):
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
            loss.backward(retain_graph=retain_graph)

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
            optimizer.step(lambda: closure(True))
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
                epoch,
                i + 1,
                net,
                loader_train,
                loader_valid,
                loss.item(),
                device,
            )

            # Update schedule
            scheduler.step()


def _calc_center(net: DeepSVDDNet, loader, device, num_training_samples: int) -> None:
    # Perform initial forward pass
    tqdm.write("Determining center parameter...")
    projections = np.zeros((num_training_samples, net.num_output_nodes))
    for i, data in tqdm(
        enumerate(loader),
        total=len(loader),
        desc="Initial forward pass",
    ):
        projections[(i * BATCH_SIZE) : ((i + 1) * BATCH_SIZE), :] = (
            net(to_device(data, device)).cpu().detach().numpy()
        )
        break

    # Derive center parameter
    center = (
        np.ones(net.num_output_nodes) * 10
    )  # projections[0, :]  # np.mean(projections, axis=0)

    # Send to GPU
    net.get_parameter(PARAM_CENTER).data = to_device(torch.tensor(center), device)

    # Inform CLI
    tqdm.write(
        f"Done! Center (mean, std) = ({np.mean(center):.6f}, {np.std(center):.6f})"
    )


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

    # Inceptify
    if not is_inceptified(dataset):
        inceptify(dataset, device)

    # Create datasets
    ds_train, ds_valid = get_inception_dataset(dataset, DATA_TRAINING_SPLIT)

    # Create dataset loaders
    loader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
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
        # lr=0.0001,
    )

    rad_optimizer = torch.optim.LBFGS(
        [net.get_parameter(PARAM_RADIUS)], line_search_fn="strong_wolfe", max_iter=100
    )
    # rad_optimizer = net_optimizer

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
        _calc_center(net, loader_train, device, len(ds_train))

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


#! Projecting real samples


def _ds_proj_file_name(dataset: Dataset) -> str:
    return f"{DS_PROJ_FILE_PREFIX}_{dataset.get_name(dataset.get_resolution())}"


def project(dataset: Dataset) -> np.ndarray:
    """
    Projects the images in the given dataset through the evaluation embedding
    and saves the result to disk. Make sure that the evaluation embedding has
    been trained before calling this function.

    The projections may also be fetched with `get_projections(dataset)`.

    Args:
        dataset (Dataset): The dataset to project

    Raises:
        ValueError: If the evaluation embedding has not yet been trained.

    Returns:
        np.ndarray: The projections where each row corresponds to a sample.
    """

    # Sanity check
    if ModelUtil.load_aux_best(AUX_MODEL_NAME) is None:
        raise ValueError("Cannot project before training!")

    # Load device
    device = get_default_device()

    # Send evaluation embedding to device
    ee = to_device(get(), device)

    # Load dataset images
    ds = get_inception_dataset(dataset)

    # Create dataset loader
    loader = torch.utils.data.DataLoader(
        ds, batch_size=DS_PROJ_BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Project images
    projections = np.zeros((len(dataset), ee.num_output_nodes))
    for i, images in tqdm(
        enumerate(loader), total=len(loader), desc="Projecting dataset"
    ):
        projections[i * DS_PROJ_BATCH_SIZE : ((i + 1) * DS_PROJ_BATCH_SIZE), :] = (
            ee(to_device(images, device)).cpu().detach().numpy()
        )

    # Save projections
    ModelUtil.get_file_jar().store_file(
        _ds_proj_file_name(dataset),
        lambda p: np.save(p, projections),
    )

    return projections


def get_projections(dataset: Dataset) -> np.ndarray:
    """
    Returns the projections of the specified dataset through the
    evaluation embedding if they have been projected with
    `project(dataset)` before. If the projection has not yet been
    performed, `None` will be returned instead.

    Args:
        dataset (Dataset): The dataset whose projections should be fetched.

    Returns:
        np.ndarray: The projections where each row corresponds to a sample.
            Note that if the projections have not yet been performed (with
            `project(dataset)`) the function will return `None`.
    """
    return ModelUtil.get_file_jar().get_file(
        f"{_ds_proj_file_name(dataset)}.npy",
        np.load,
    )


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
