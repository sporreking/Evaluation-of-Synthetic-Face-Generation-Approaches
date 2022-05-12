from src.metric.CompoundMetric import CompoundMetric
from src.metric.SampleMetricManager import SampleMetricManager
from src.core.Setupable import SetupMode
from src.metric.CompoundMetricManager import CompoundMetricManager
from typing import Any
from src.metric.ImprovedPrecisionRecall import (
    ImprovedPrecisionRecall,
    get_ref_file_name,
    BATCH_SIZE,
    BATCH_SIZE_NAME,
    DIST_CALC_BATCH_SIZE,
    DIST_CALC_BATCH_SIZE_NAME,
)

PRECISION_NAME = "Precision"

# Setup constants
SETUP_NAME = "DATASET_MANIFOLD"


class PrecisionCompoundMetric(CompoundMetric):
    def __init__(
        self,
        cmm: CompoundMetricManager,
        smm: SampleMetricManager = None,
    ):
        """
        Constructor for PrecisionCompoundMetric class, subclass of the CompoundMetric class.

        Args:
            cmm (CompoundMetricManager): Manager used by metrics. Population and dataset is derived
                from this manager.
            smm (SampleMetricManager, optional): Not used for this metric. Defaults to None.
        """
        super(PrecisionCompoundMetric, self).__init__(PRECISION_NAME, cmm, smm)

        # Init storage structure for this metric
        self._precision = None
        self._ipr = ImprovedPrecisionRecall(self.get_dataset())
        self._pop = self.get_population()

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        ds = self.get_dataset()
        return {
            get_ref_file_name(ds, omit_extension=True): SetupMode(
                True,
                lambda _, batch_size, dist_calc_batch_size: self._ipr.compute_manifold_ref(
                    batch_size, dist_calc_batch_size
                ),
                lambda: self._ipr.check_precalculated_manifold(),
                batch_size=BATCH_SIZE,
                dist_calc_batch_size=DIST_CALC_BATCH_SIZE,
            )
        }

    def calc(self, filter_bit: int = 1, **parameters: Any) -> Any:
        """
        Calculates the precision of this compound metric's population.

        Args:
            filter_bit (int, optional): Filter bit used to select a subset of the
                population. Filter bit is defined by the order in FilterRegistry. For example,
                the first filter corresponds to filter bit 1. Defaults to 1 (IdentityFilter).
            batch_size (int, optional): The batch size to use when projecting images
                through the VGG16-network on the GPU. Defaults to `BATCH_SIZE`.
            dist_calc_batch_size (int, optional): The batch size to use when calculating
                pair-wise distances on the CPU (the operation is very memory-intensive).
                Defaults to `DIST_CALC_BATCH_SIZE`.

        Returns:
            float: The precision of this compound metric's population.
        """
        # Check manifold
        assert self._ipr.check_precalculated_manifold()

        # Fetch parameters

        batch_size = (
            parameters[BATCH_SIZE_NAME] if BATCH_SIZE_NAME in parameters else BATCH_SIZE
        )

        dist_calc_batch_size = (
            parameters[DIST_CALC_BATCH_SIZE_NAME]
            if DIST_CALC_BATCH_SIZE_NAME in parameters
            else DIST_CALC_BATCH_SIZE
        )

        # Check parameters
        assert type(batch_size) == int
        assert batch_size > 0

        assert type(dist_calc_batch_size) == int
        assert dist_calc_batch_size > 0

        # Calculate precision
        pop_data = self._pop.get_filtered_data(filter_bit)
        self._precision = self._ipr.calc_precision(
            pop_data[self._pop.COLUMN_URI].values.tolist(),
            batch_size,
            dist_calc_batch_size,
        )
        return self._precision

    def get(self, calc_if_missing: bool = False, **parameters: Any) -> Any:
        # Check if metric already calculated
        if self._precision is not None:
            return self._precision

        # Check if calculate when missing
        elif calc_if_missing:
            return self.calc(**parameters)
        else:
            return None

    def print_result(self) -> None:
        print("Precision: ", self._precision)

    def plot_result(self) -> None:
        pass
