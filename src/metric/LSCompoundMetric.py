from __future__ import annotations

import src.metric.CompoundMetric as CompoundMetric
from src.metric.CompoundMetric import CompoundMetric
from src.metric.SampleMetricManager import SampleMetricManager
from src.core.Setupable import SetupMode
from src.metric.CompoundMetricManager import CompoundMetricManager
from typing import Any
from src.controller.InterFaceGANController import (
    SETUP_POPULATION_NAME,
    SETUP_LABELS_NAME,
    SETUP_SVC_PREFIX,
)
from src.controller.SupportVectorClassifier import (
    get_svc_names,
    get_missing_model_names,
    setup_population,
    setup_auxillary,
    setup_labels,
    train_svc,
    get_svc,
    is_population_ready,
    get_population_name,
    get_population_info,
    is_labels_ready,
    get_labels,
)
from src.controller.ASADController import ASADController
from src.controller.Controller import Controller
import numpy as np
from src.util.AuxUtil import load_aux_best

LS_NAME = "LinearSeparability"


class LSCompoundMetric(CompoundMetric):
    def __init__(
        self,
        cmm: CompoundMetricManager,
        smm: SampleMetricManager = None,
    ):
        """
        Constructor for LSCompoundMetric class, subclass of the CompoundMetric class.

        Args:
            cmm (CompoundMetricManager): Manager used by metrics. Population, controller and dataset is derived
                from this manager.
            smm (SampleMetricManager, optional): Not used for this metric. Defaults to None.
        """
        super(LSCompoundMetric, self).__init__(LS_NAME, cmm, smm)

        # Init storage structure for this metric and models used
        self._svcs = {}
        self._ls = None
        self._ls_per_attr = {}

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        # Get the controller
        controller: Controller = self._cmm.get_controller()

        # To get classifier models
        asad_controller = ASADController(controller._gen, controller._attrs)

        # QoL
        mn_cls = lambda attr, omit_name=True: asad_controller.get_model_name(
            attr, True, omit_name
        )
        gen_name = controller._gen.get_name()
        controller_name = controller.get_name()
        # TODO: Implement 'improve' functionality

        # Construct modes
        return {
            SETUP_POPULATION_NAME: SetupMode(
                lambda _: self._setup_pop(),
                lambda: is_population_ready(
                    get_population_name(gen_name, controller_name)
                ),
                lambda: get_population_info(
                    get_population_name(gen_name, controller_name)
                ),
            ),
            **{
                mn_cls(attr): SetupMode(
                    lambda _, batch_size, epochs, attr=attr: self._setup_auxillary_classifier(
                        attr, batch_size, epochs
                    ),
                    lambda attr=attr: load_aux_best(mn_cls(attr, False)) is not None,
                    lambda attr=attr: asad_controller._setup_info_func(
                        mn_cls(attr, False)
                    ),
                    [SETUP_POPULATION_NAME],
                    batch_size=64,
                    epochs=40,
                )
                for attr in controller._attrs
            },
            **{
                SETUP_LABELS_NAME(attr): SetupMode(
                    lambda _, batch_size, epochs, attr=attr: self._setup_training_labels(
                        attr, batch_size, epochs
                    ),
                    lambda attr=attr, pop_name=get_population_name(
                        gen_name, controller.get_name()
                    ): is_labels_ready(attr, pop_name),
                    required_modes=[mn_cls(attr)],
                    batch_size=64,
                    epochs=40,
                )
                for attr in controller._attrs
            },
            **{
                SETUP_LABELS_NAME(attr)
                + "_target": SetupMode(
                    lambda _, batch_size, epochs, attr=attr: setup_labels(
                        controller,
                        attr,
                        batch_size,
                        epochs,
                        self._population.get_name(),
                    ),
                    lambda attr=attr: is_labels_ready(
                        attr, self._population.get_name()
                    ),
                    required_modes=[mn_cls(attr)],
                    batch_size=64,
                    epochs=40,
                )
                for attr in controller._attrs
            },
            **{
                (SETUP_SVC_PREFIX + "_" + attr): SetupMode(
                    lambda _, attr=attr: self._train_svc(attr),
                    lambda attr=attr: get_svc(attr, gen_name, controller_name)
                    is not None,
                    required_modes=[
                        SETUP_LABELS_NAME(attr),
                        SETUP_LABELS_NAME(attr) + "_target",
                    ],
                )
                for attr in controller._attrs
            },
        }

    def _setup_pop(self) -> None:
        """
        Setup a population used for training SVCs.
        Setup already existing population or if necessary constructs a new one
        and populate it.

        Note that a population is bound to a generator AND a controller.
        """
        setup_population(self._cmm.get_controller())

    def _setup_auxillary_classifier(
        self, attr: str, batch_size: int = 64, epochs: int = 40
    ) -> None:
        """
        Train auxiliary classifier (borrowed from ASADController) if
        necessary.

        Trained classifiers are saved according to ASADController.
        """
        setup_auxillary(attr, self.get_dataset(), batch_size, epochs)

    def _setup_training_labels(
        self, attr: str, batch_size: int = 64, epochs: int = 40
    ) -> None:
        """
        Setup labels given a auxiliary classifier (Will train one if missing.)

        See documentation for `_setup_auxillary_classifier` and the ASADController
        class for more information about the classifier.
        """
        setup_labels(self._cmm.get_controller(), attr, batch_size, epochs)

    def _train_svc(self, attr: str) -> None:
        """
        Train a support vector classifier to predict attributes on a population.
        """
        svc = train_svc(self._cmm.get_controller(), attr)
        self._svcs[attr] = svc

    def calc(self, **parameters: Any) -> Any:
        def binary_conditional_entropy(X: np.ndarray, Y: np.ndarray):
            """
            Computes the (binary) conditional entropy H(Y|X).
            """
            assert len(X) == len(Y)
            binary = [0, 1]
            cond_entropy = 0
            n = len(X)
            for x in binary:
                # p(X=x)
                p_x = len(X[X == x]) / n
                for y in binary:
                    # p(X=x,Y=y)
                    ind_x = X == x
                    Y_subset = Y[ind_x]
                    p_x_y = len(Y_subset[Y_subset == y]) / n

                    # Check for zeros to avoid -inf
                    if p_x_y == 0 or p_x == 0:
                        continue

                    p_div = p_x_y / p_x

                    # Accumulate the result
                    eps = 0.0000001
                    cond_entropy -= p_x_y * np.log2(p_div + eps)
            return cond_entropy

        # Get the controller
        controller = self._cmm.get_controller()

        # Get the latent codes
        latent_codes = np.stack(self._population.get_data()["latent_code"].to_numpy())

        # QoL
        gen_name = controller._gen.get_name()
        controller_name = controller.get_name()

        # Calculate the conditional entropy for each attribute
        conditional_entropy = 0
        for attr in controller._attrs:
            pop_name = self._population.get_name()

            # Get the labels from the auxiliary classifier
            labels = get_labels(
                attr,
                pop_name,
            )
            if labels is None:
                # Predict the labels if they don't exist.
                print("Labels does not exist for the target population.")
                print("Predicting labels...")
                setup_labels(controller, attr, pop_name=pop_name, **parameters)
                labels = get_labels(
                    attr,
                    pop_name,
                )
                print("Labels done, continuing with SVC training.")

            # Get the svc
            if attr in self._svcs:
                svc = self._svcs[attr]
            else:
                svc = get_svc(attr, gen_name, controller_name)

            # Get the SVC predictions
            preds = svc.predict(latent_codes).reshape(labels.shape)

            # Calc the conditional entropy
            ls_attr = binary_conditional_entropy(preds, labels)

            # Accumulate the result
            conditional_entropy += ls_attr

            # Save result per attr
            self._ls_per_attr[attr] = np.exp(ls_attr)

        self._ls = np.exp(conditional_entropy)
        return self._ls

    def get(self, calc_if_missing: bool = False, **parameters: Any) -> Any:
        # Check if metric already calculated
        if self._ls is not None:
            return self._ls

        # Check if calculate when missing
        elif calc_if_missing:
            return self.calc(**parameters)
        else:
            return None

    def print_result(self) -> None:
        print(f"Linear separability: {self._ls}")
        for attr in self._ls_per_attr.keys():
            print(f"Linear separability for {attr}: {self._ls_per_attr[attr]}")

    def plot_result(self) -> None:
        pass
