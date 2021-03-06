from collections import OrderedDict
from pathlib import Path
import re
from typing import Tuple, List
import inspect
from src.util.FileJar import FileJar

from matplotlib import pyplot as plt
import torch
import numpy as np


TRAIN_LOSS_STATE_KEY = "train_loss"
VALID_LOSS_STATE_KEY = "valid_loss"
EPOCH_STATE_KEY = "epoch"
BATCH_STATE_KEY = "batch"
NUM_BATCHES_PER_EPOCH_KEY = "num_batches_per_epoch"

SAVE_FILE_EXT = "pt"

_file_jar = FileJar(Path("auxiliary"), create_root_dir=True)


class AuxModelInfo:
    """
    Used to pass auxiliary model data.
    """

    def __init__(
        self,
        state: OrderedDict,
        epoch: int,
        batch: int,
        num_batches_per_epoch: int,
        train_loss: float,
        valid_loss: float,
    ):
        """
        Constructs a new AuxModelInfo instace for carying model metadata.

        Args:
            state (OrderedDict[str, torch.Tensor]): The state_dict of the model to save.
            epoch (int): The epoch with which the specified `state` is associated.
            batch (int): The batch with which the specified `state` is associated.
            num_batches_per_epoch (int): Number of batches per epoch.
            train_loss (float): The training loss score of the specified `state`.
            valid_loss (float): The validation loss score of the specified `state`.
        """
        self._state = state
        self._epoch = epoch
        self._batch = batch
        self._num_batches_per_epoch = num_batches_per_epoch
        self._train_loss = train_loss
        self._valid_loss = valid_loss

    @property
    def state(self) -> OrderedDict:
        """
        Get the state_dict of the auxiliary model.
        """
        return self._state

    @property
    def epoch(self) -> int:
        """
        Get the epoch with which the `state` is associated.
        """
        return self._epoch

    @property
    def batch(self) -> int:
        """
        Get the batch with which the `state` is associated.
        """
        return self._batch

    @property
    def num_batches_per_epoch(self) -> int:
        """
        Get the number of batches per epoch.
        """
        return self._num_batches_per_epoch

    @property
    def train_loss(self) -> float:
        """
        Get the training loss score.
        """
        return self._train_loss

    @property
    def valid_loss(self) -> float:
        """
        Get the validation loss score.
        """
        return self._valid_loss


def _full_name(name: str, epoch: int, batch: int) -> str:
    return f"{name}_e{epoch}_b{batch}.{SAVE_FILE_EXT}"


def _is_epoch_batch_format(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_\.]+_e[0-9]+_b[0-9]+\.pt$", name))


def _epoch_batch_from_full_name(full_name: str) -> Tuple[int, int]:
    e, b = full_name.split("_")[-2:]
    return (int(e[1:]), int(b.split(".")[0][1:]))


def _best_name(name: str) -> str:
    return f"{name}_best.{SAVE_FILE_EXT}"


def _aux_save_func(info: AuxModelInfo):
    return (
        (lambda p: torch.save(info.state, p))
        if "_use_new_zipfile_serialization"
        not in inspect.signature(torch.save).parameters
        else (lambda p: torch.save(info.state, p, _use_new_zipfile_serialization=False))
    )


def _load_aux_with_full_name(
    full_name: str,
) -> AuxModelInfo:
    state = _file_jar.get_file(full_name, lambda p: torch.load(p, map_location="cpu"))

    # Check if load failed
    if state is None:
        return None

    epoch = state[EPOCH_STATE_KEY].item()
    batch = state[BATCH_STATE_KEY].item()
    num_batches_per_epoch = state[NUM_BATCHES_PER_EPOCH_KEY].item()
    train_loss = state[TRAIN_LOSS_STATE_KEY].item()
    valid_loss = state[VALID_LOSS_STATE_KEY].item()

    del state[EPOCH_STATE_KEY]
    del state[BATCH_STATE_KEY]
    del state[NUM_BATCHES_PER_EPOCH_KEY]
    del state[TRAIN_LOSS_STATE_KEY]
    del state[VALID_LOSS_STATE_KEY]

    return AuxModelInfo(
        state, epoch, batch, num_batches_per_epoch, train_loss, valid_loss
    )


def delete_aux(name: str) -> None:
    """
    Deletes all saved instances of the specified auxiliary model from the disk.

    Args:
        name (str): The name of the auxiliary model to delete.
    """
    for p in _file_jar.iterdir():
        if (
            _is_epoch_batch_format(p.name) and "_".join(p.name.split("_")[:-2]) == name
        ) or p.name == _best_name(name):
            p.unlink(missing_ok=True)


def save_aux(
    name: str,
    info: AuxModelInfo,
    save_best: bool = True,
) -> None:
    """
    Associates the specified `name` with an auxiliary model `state`,
    and saves it to disk for the specified `epoch` and `batch`. The
    provided training and validation loss scores are also saved.

    Args:
        name (str): The name of the model to save.
        info (AudModelInfo): Information about the model to save.
        save_best (bool, optional): If `True`, the model will be saved as the "best"
            model if its validation loss score is lower than that of the previous
            best one. Defaults to True.
    """

    # Add loss scores to output
    info.state[EPOCH_STATE_KEY] = torch.tensor(info.epoch)
    info.state[BATCH_STATE_KEY] = torch.tensor(info.batch)
    info.state[NUM_BATCHES_PER_EPOCH_KEY] = torch.tensor(info.num_batches_per_epoch)
    info.state[TRAIN_LOSS_STATE_KEY] = torch.tensor(info.train_loss)
    info.state[VALID_LOSS_STATE_KEY] = torch.tensor(info.valid_loss)

    # Save model
    _file_jar.store_file(
        _full_name(name, info.epoch, info.batch),
        _aux_save_func(info),
    )

    # Save model as best if applicable
    if save_best:
        best = load_aux_best(name)
        if best is None or info.valid_loss < best.valid_loss:
            _file_jar.store_file(
                _best_name(name),
                _aux_save_func(info),
            )


def load_aux(name: str, epoch: int, batch: int) -> AuxModelInfo:
    """
    Loads the auxiliary model associated with the specified `name`,
    `epoch`, and `batch`. Metadata such as loss scores and epoch / batch
    info is also loaded.

    Args:
        name (str): The name of the model to load.
        epoch (int): The epoch to load from.
        batch (int): The batch to load from.

    Returns:
        AuxModelInfo: Information about the loaded model.
            If no such model exists, `None` is returned instead.
    """
    return _load_aux_with_full_name(_full_name(name, epoch, batch))


def load_aux_best(
    name: str,
) -> AuxModelInfo:
    """
    Loads the best auxiliary model associated with the specified `name`.
    Metadata such as loss scores and epoch / batch info is also loaded.

    Args:
        name (str): The name of the model to load.

    Returns:
        AuxModelInfo: Information about the loaded model.
            If no best model exists, `None` is returned instead.
    """
    return _load_aux_with_full_name(_best_name(name))


def get_available_aux_epoch_batch_pairs(name: str) -> List[Tuple[int, int]]:
    """
    Returns tuples of available epoch and batch versions of the auxiliary
    model associated with the specified name.

    Returns:
        list[tuple[int, int]]: Tuples of (epoch, batch) format.
    """
    return [
        _epoch_batch_from_full_name(path.name)
        for path in _file_jar.iterdir()
        if _is_epoch_batch_format(path.name) and path.name.startswith(name)
    ]


def plot_aux_loss(name: str, title: str = None):
    """
    Plots the training and validation loss for the auxiliary model
    associated with the specified name.

    Args:
        name (str): The name of the model to plot for.
        title (str, optional): A title to use for the plot, or `None`
            if no title should be used. Note that the suffix `" Training"`
            will be appended to the specified title. Defaults to None.

    Raises:
        ValueError: If there exists no model with the specified name.
    """

    # Get available versions
    epoch_batch_pairs = get_available_aux_epoch_batch_pairs(name)

    # Sanity check
    if not epoch_batch_pairs:
        raise ValueError(f"No auxiliary model with name: '{name}'")

    # Load data
    epochs = []
    train_loss_scores = []
    valid_loss_scores = []
    for e, b in epoch_batch_pairs:
        info = load_aux(name, e, b)

        # Derive the epoch points
        epochs.append((info.epoch - 1) + (info.batch) / info.num_batches_per_epoch)

        # Extract the loss scores and sort according to epoch order
        train_loss_scores.append(info.train_loss)
        valid_loss_scores.append(info.valid_loss)

    # Sort / convert data
    epochs = np.array(epochs)
    order = epochs.argsort()
    epochs = epochs[order]
    train_loss_scores = np.array(train_loss_scores)[order]
    valid_loss_scores = np.array(valid_loss_scores)[order]

    # Plot scores
    plt.plot(epochs, train_loss_scores, label="Training Loss Score")
    plt.plot(epochs, valid_loss_scores, label="Validation Loss Score")
    if title is not None:
        plt.title(f"{title} Training")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.show()


def get_file_jar() -> FileJar:
    """
    Returns the file jar used by the model util.

    Returns:
        FileJar: The file jar used by the model util.
    """
    return _file_jar
