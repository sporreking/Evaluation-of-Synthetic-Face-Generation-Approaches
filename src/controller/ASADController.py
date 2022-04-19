from src.controller.Controller import Controller
from src.dataset.Dataset import Dataset
from src.dataset.TorchImageDataset import TorchImageDataset
from src.generator.Generator import Generator
import numpy as np
import src.util.CudaUtil as CU
from typing import Any
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from src.util.AuxUtil import AuxModelInfo, save_aux, load_aux_best
from copy import deepcopy
from src.environment.EnvironmentManager import EnvironmentManager as em
from src.controller.ASADControllerModels import get_dec_arch, get_cls_arch

CLASS_PREFIX = "CLS"
DECODER_PREFIX = "DEC"
DECODER_AGENT_NAME = "asaddecoder"
ASAD_NAME = "ASAD"

# Setup modes
# TODO implement continue
SETUP_MODE_TRAIN_ALL = "train_all"
SETUP_MODE_TRAIN_MISSING = "train_missing"
SETUP_MODES = [SETUP_MODE_TRAIN_ALL, SETUP_MODE_TRAIN_MISSING]


class ASADController(Controller):
    """
    Subclass to Controller, implements adaptive
    semantic attribute decoupling.
    """

    def __init__(self, gen: Generator, ds: Dataset, attrs: list[str]):
        """
        Constructor to the ASADController

        Args:
            gen (Generator): The generator associated with the controller.
            ds (Dataset): Dataset for training the classifiers.
            attrs (list[str]): Attributes from the dataset `ds` to be used.
        """
        super().__init__(ASAD_NAME, gen)

        # Check attribute names
        if any("_" in a for a in attrs):
            raise ValueError("Attribute names may not contain underscores!")

        self._attrs = attrs
        self._ds = ds

        # For saving decoders
        self._decoders = {}

    def is_ready(self) -> bool:
        # Get model names
        cls_names, dec_names = self._get_model_names()

        # All model names
        all_model_names = cls_names + dec_names

        # Check if all models are ready.
        return not None in [load_aux_best(n) for n in all_model_names]

    def is_setup_dependent_on_generator(self) -> bool:
        return True

    def setup(self, mode: str) -> None:
        # Check if valid mode
        if mode not in SETUP_MODES:
            raise ValueError(f"Invalid setup mode: {mode}, use any of: {SETUP_MODES}")

        # Retrain all models
        if mode == SETUP_MODE_TRAIN_ALL:
            # Classifiers
            for attr in self._attrs:
                self._setup_classifier(attr)

            # Decoders
            for attr in self._attrs:
                self._setup_decoder(attr)

        # Train only missing models
        elif mode == SETUP_MODE_TRAIN_MISSING:
            # Reality check
            if self.is_ready():
                return

            # All model names
            cls_names, dec_names = self._get_model_names()

            # Train missing classifiers
            missing_cls = self._missing_models(cls_names)
            for cl in missing_cls:
                attr = self._get_attr_from_model_name(cl)
                print(f"Training {attr} classifier!")
                self._setup_classifier(attr)

            # Train missing decoders
            missing_dec = self._missing_models(dec_names)
            for dec in missing_dec:
                attr = self._get_attr_from_model_name(dec)
                print(f"Training {attr} decoder!")
                self._setup_decoder(attr)

    def parse_native_input(self, input: dict[str, np.ndarray]) -> np.ndarray:
        """
        Parse input from standardized representation to native representation.

        Standardized representation refers to the float range of [-1,1].

        For ASAD, the representation coincides with the float range of the native representation.
        However, each attribute manipulation must be done in a sequential step, thus, every dict
        in the output only contain one attribute and its values.

        For example:

        Let input be defined as:
        `input_example = {"gender" : np.ndarray[-1, 1, -0.5], "age": np.ndarray[1,0,-1]}`
        then input for image 1 should then be parsed as an image with gender = -1
        and age = 1. Thus, `input example` defines parameters for 3 images (length of the array).

        Note that a value of 0 means that no changes for that parameter will be done for that image.

        Args:
            input (dict[str, np.ndarray]): Standardized input representation.
                Where key = parameter name, and the value = array of values
                (in float range of [-1,1]) of that parameter. Each value in the
                array represents one generated image. For example, image 1 should
                be based on value indexed 1 in the array for each parameter in input.keys().

        Returns:
            np.ndarray[dict[str, Any]]: 2D array containing dictionary with
                parameter names and their native values. Axis 0 corresponds to the number
                of images to generate/manipulate, while axis 1 denotes different sequential
                manipulation steps that should be applied. Each entry in the array
                corresponds to a single manipulation specified by a dictionary, with parameters
                on native format.
        """

        attrs = list(input.keys())
        vals = list(input.values())

        nr_images = len(vals[0])
        nr_attrs = len(attrs)

        parsed_input = np.empty((nr_images, nr_attrs), object)
        for i in range(nr_images):
            for j in range(nr_attrs):
                parsed_input[i, j] = {attrs[j]: vals[j][i]}

        return parsed_input

    def generate_native(
        self, latent_codes: np.ndarray, native_input: np.ndarray
    ) -> list[str]:

        # Get device
        device = CU.get_default_device()

        # CPU device
        cpu_device = torch.device("cpu")

        # Load decoder architecture
        model = get_dec_arch(self._gen)

        # Load all Decoders first
        for j in range(native_input.shape[1]):
            attr = list(native_input[0, j].keys())[0]
            if attr not in self._decoders:
                name = "_".join(
                    (self._name, DECODER_PREFIX, self._gen.get_name(), attr)
                )
                dec = load_aux_best(name)
                model_cp = deepcopy(model)
                model_cp.load_state_dict(dec.state)
                self._decoders[name] = model_cp

        # Clear cache
        CU.empty_cache()

        # Update latent codes according to the native_input
        old_name = None
        with torch.no_grad():
            for j in range(native_input.shape[1]):
                for i, z in enumerate(latent_codes):
                    d = native_input[i, j]
                    val = list(d.values())[0]
                    # Only modify if value is non-zero.
                    if val != 0:
                        attr = list(d.keys())[0]
                        name = "_".join(
                            (self._name, DECODER_PREFIX, self._gen.get_name(), attr)
                        )

                        # Only transfer decoder to GPU when necessary.
                        if not old_name == name:
                            if old_name is not None:
                                # Set old decoder back to CPU
                                CU.to_device(self._decoders[old_name], cpu_device)

                            # Load decoder into GPU
                            model = CU.to_device(self._decoders[name], device)
                            model.eval()

                        # Update latent codes
                        latent_codes[i] = (
                            z
                            + val
                            * model(CU.to_device(torch.from_numpy(z), device))
                            .cpu()
                            .numpy()
                        )

                        old_name = name

        # Generate images based on updated latent codes
        return self._gen.generate(latent_codes)

    def _get_model_names(self) -> tuple[list[str], list[str]]:
        # Construct classifier model names
        cls_names = [
            "_".join(
                (
                    self._name,
                    CLASS_PREFIX,
                    self._ds.get_name(self._ds.get_resolution()),
                    attr,
                )
            )
            for attr in self._attrs
        ]

        # Construct decoder model names
        dec_names = [
            "_".join((self._name, DECODER_PREFIX, self._gen.get_name(), attr))
            for attr in self._attrs
        ]
        return cls_names, dec_names

    def _get_attr_from_model_name(self, name: str) -> str:
        return name.split("_")[-1]

    def _missing_models(self, names: list[str]) -> list[str]:
        return [n for n in names if load_aux_best(n) is None]

    def _setup_classifier(self, attr: str, batch_size: int = 64, epochs: int = 40):
        # Get labels
        df = self._ds.get_processed_labels()

        print("----- Load images -----")
        # Load data parameters
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        train_ds = TorchImageDataset(
            self._ds.get_image_paths(),
            T.Compose([T.ToTensor(), T.Normalize(*stats)]),
            attr,
            df,
        )

        # Create a dataloader
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)

        # Setup GPU
        device = CU.get_default_device()

        # Setup model architecture
        model = CU.to_device(get_cls_arch(), device)

        # Train and validate the classifier
        name = "_".join(
            (
                self._name,
                CLASS_PREFIX,
                self._ds.get_name(self._ds.get_resolution()),
                attr,
            )
        )
        self._fit_classifier(model, train_dl, device, epochs, name)

    def _fit_classifier(self, model, train_dl, device, epochs, name) -> None:
        torch.cuda.empty_cache()

        # Losses
        tr_losses = []
        val_losses = []

        # Create optimizers
        opt = torch.optim.AdamW(model.parameters())

        # Setup batches
        n_batches = len(train_dl)
        tr_percent = 0.8
        tr_batches = math.floor(n_batches * tr_percent)

        # Start training loop
        for epoch in range(epochs):
            val_avg = 0
            for batch_nr, (images, labels) in tqdm(
                enumerate(train_dl), total=n_batches
            ):
                # Send data to GPU
                images = CU.to_device(images, device)
                labels = CU.to_device(labels, device)

                if batch_nr + 1 >= tr_batches:
                    # Validate classifier
                    val_loss = self._validate_classifier(model, images, labels)

                    # Record losses
                    val_avg += val_loss
                    val_losses.append((val_loss, batch_nr))
                else:
                    # Train classifier
                    tr_loss = self._train_classifier(model, images, labels, opt)

                    # Record losses
                    tr_losses.append((tr_loss, batch_nr))

            val_avg = val_avg / (n_batches - tr_batches)
            # Log losses & scores (last batch)
            print(
                "Epoch [{}/{}], train loss: {:.4f}, val loss: {:.4f}".format(
                    epoch + 1, epochs, tr_loss, val_avg
                )
            )

            # TODO numofbatch -> training>_batch
            # Save result-
            save_aux(
                name,
                AuxModelInfo(
                    model.state_dict(),
                    epoch + 1,
                    batch_nr + 1,
                    tr_batches,
                    tr_loss,
                    val_avg,
                ),
            )

    def _train_classifier(self, model, images, labels, opt) -> float:
        # Clear model gradients
        opt.zero_grad()

        # Get predictions
        preds = model(images)

        # Reshape labels to fit preds
        labels = torch.reshape(labels, preds.shape)

        # Calc loss
        loss = F.binary_cross_entropy(preds, labels)

        # Update weights
        loss.backward()
        opt.step()
        return loss.item()

    def _validate_classifier(self, model, images, labels) -> float:
        model.eval()

        with torch.no_grad():
            # Make predictions
            preds = model(images)

            # Reshape labels to fit preds
            labels = torch.reshape(labels, preds.shape)

            # Calc loss
            loss = F.binary_cross_entropy(preds, labels)

        model.train()
        return loss.item()

    def _setup_decoder(
        self,
        attr: str,
        epochs: int = 15,
        iter_per_epoch: int = 2000,
        batch_size: int = 3,
    ) -> None:

        # Define parameters
        cls_name = "_".join(
            (
                self._name,
                CLASS_PREFIX,
                self._ds.get_name(self._ds.get_resolution()),
                attr,
            )
        )
        gen_name = self._gen.get_name()
        dec_name = "_".join(
            (
                self._name,
                DECODER_PREFIX,
                gen_name,
                attr,
            )
        )

        # Train decoder using generator env.
        em.run(
            DECODER_AGENT_NAME,
            gen_name,
            epochs=epochs,
            iter_per_epoch=iter_per_epoch,
            batch_size=batch_size,
            generator_name=gen_name,
            decoder_name=dec_name,
            classifier_name=cls_name,
        )
