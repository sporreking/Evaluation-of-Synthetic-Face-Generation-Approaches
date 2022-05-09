from src.metric.CompoundMetric import CompoundMetric
from src.metric.SampleMetricManager import SampleMetricManager
from src.core.Setupable import SetupMode
from src.metric.CompoundMetricManager import CompoundMetricManager
from typing import Any
from cleanfid import fid
from src.population.Population import Population
from src.dataset.FFHQDataset import FFHQDataset

FID_NAME = "FID"
FID_CALC_MODES = ["clean", "legacy_tensorflow", "legacy_pytorch"]
FID_DEFAULT_CALC_MODE = "clean"


# TODO: Fix this:
#! Only works when equal to zero, gets pickle error otherwise
NUM_WORKERS = 0


class FIDCompoundMetric(CompoundMetric):
    def __init__(
        self,
        cmm: CompoundMetricManager,
        smm: SampleMetricManager = None,
    ):
        """
        Constructor for FIDCompoundMetric class, subclass of the CompoundMetric class.

        Args:
            cmm (CompoundMetricManager): Manager used by metrics. Population and dataset is derived
                from this manager.
            smm (SampleMetricManager, optional): Not used for this metric. Defaults to None.
        """
        super(FIDCompoundMetric, self).__init__(FID_NAME, cmm, smm)

        # Init storage structure for this metric
        self._fid = dict()

    def reg_setup_modes(self) -> dict[str, SetupMode]:
        ds = self.get_dataset()
        return {
            f"statistics_{fcm}_{ds.get_name(ds.get_resolution())}": SetupMode(
                lambda _, fcm=fcm: self._setup(fcm),
                lambda fcm=fcm: self._is_ready(fcm),
            )
            for fcm in FID_CALC_MODES
        }

    def _setup(self, calc_mode: str = FID_DEFAULT_CALC_MODE) -> None:
        """
        Setup the needed statistics to calculate the metric.

        Note that each dataset and `calc_mode` combination needs a calculated statistic.

        For more information regarding `calc_mode`, see the documentation for `calc()`.

        Args:
            calc_mode (str, optional): Calc mode determines FID implementation, different statistics
                needed for different implementations. See documentation on `calc()` for more information.
                Defaults to `FID_DEFAULT_CALC_MODE` ("clean").

        Raises:
            ValueError: Error when non-valid `calc_mode`, valid modes are defined by `FID_CALC_MODES`.
        """
        # Check calc_mode
        if calc_mode not in FID_CALC_MODES:
            raise ValueError(
                f"{calc_mode} not supported, supported modes: {FID_CALC_MODES}"
            )

        # Calculate custom statistics
        ds = self.get_dataset()
        fid.make_custom_stats(
            ds.get_name(ds.get_resolution()), str(ds.get_image_dir()), mode=calc_mode
        )

    def _is_ready(self, calc_mode=FID_DEFAULT_CALC_MODE) -> bool:
        """
        Checks if compound metric is ready for calculations.

        Args:
            calc_mode (str, optional): Calc mode determines FID implementation, different statistics
                needed for different implementations. See documentation on `calc()` for more information.
                Defaults to `FID_DEFAULT_CALC_MODE` ("clean").

        Raises:
            ValueError: Error when non-valid `calc_mode`, valid modes are defined by `FID_CALC_MODES`.

        Returns:
            bool: True if the compound metrics is ready for calculations.
        """
        # Check calc_mode
        if calc_mode not in FID_CALC_MODES:
            raise ValueError(
                f"{calc_mode} not supported, supported modes: {FID_CALC_MODES}"
            )

        ds = self.get_dataset()

        if type(ds).get_resolution_invariant_name() == "FFHQ" and (
            ds.get_resolution() == 256 or ds.get_resolution() == 1024
        ):
            # pre-computed statistic by clean-fid
            return True
        else:
            return fid.test_stats_exists(ds.get_name(), calc_mode)

    def calc(self, **parameters: Any) -> Any:
        """
        Calculates the FID given the dataset and the population.

        No setup is needed for FFHQ 256/1024, if using other custom datasets `calc()`
        requires the user to run `setup()` first.

        Args:
            calc_mode (str, optional): Either "clean", "legacy_tensorflow, or "legacy_pytorch".
                This decides how the FID score should be calculated, i.e., using clean-fid,
                regular tensorflow implementation, or pytorch implementation. Default is "clean" (clean-fid).
        Raises:
            ValueError: Error when non-valid `calc_mode`, valid modes are defined by `FID_CALC_MODES`.
            ValueError: Error when the name of the dataset in conjunction with the
                specified `calc_mode` don't have a pre-computed statistic.

        Returns:
            Any: The FID value.
        """
        # Fetch parameters
        calc_mode = self._check_calc_mode(parameters)

        # Get variables for use in FID
        ds = self.get_dataset()
        pop_path = str(
            Population.POPULATION_ROOT_DIR / self.get_population().get_name()
        )
        resolution = ds.get_resolution()
        fid_score = None

        if type(ds).get_resolution_invariant_name() == "FFHQ" and (
            ds.get_resolution() == 256 or ds.get_resolution() == 1024
        ):
            # Use pre-computed statistic by clean-fid
            fid_score = fid.compute_fid(
                pop_path,
                dataset_name=type(ds).get_resolution_invariant_name(),
                dataset_res=resolution,
                mode=calc_mode,
                dataset_split="trainval70k",
                num_workers=NUM_WORKERS,
            )
        else:
            # Use custom pre-computed statistic
            dataset_name = ds.get_name(resolution)

            # Check if statistic exists
            if fid.test_stats_exists(dataset_name, calc_mode):
                fid_score = fid.compute_fid(
                    pop_path,
                    dataset_name=dataset_name,
                    mode=calc_mode,
                    dataset_split="custom",
                    num_workers=NUM_WORKERS,
                )
            else:
                raise ValueError(
                    f"Statistic named '{dataset_name}' with `calc_mode` '{calc_mode}'"
                    " has no statistic. Double check `calc_mode` or run 'setup()'"
                )

        # Save result
        self._fid[calc_mode] = fid_score
        return fid_score

    def get(self, calc_if_missing: bool = False, **parameters: Any) -> Any:
        # Check parameters
        calc_mode = self._check_calc_mode(parameters)

        # Check if metric already calculated
        if calc_mode in self._fid.keys() and self._fid[calc_mode] is not None:
            return self._fid[calc_mode]

        # Check if calculate when missing
        elif calc_if_missing:
            return self.calc(**parameters)

        else:
            return None

    def print_result(self) -> None:
        for calc_mode, fid_score in self._fid.items():
            print(calc_mode + " FID: ", fid_score)

    def plot_result(self) -> None:
        pass

    def _check_calc_mode(self, parameters) -> str:
        # Fetch parameters
        if "calc_mode" in parameters.keys():
            calc_mode = parameters["calc_mode"]

            # Check calc_mode
            if calc_mode not in FID_CALC_MODES:
                raise ValueError(
                    f"{calc_mode} not supported, supported modes: {FID_CALC_MODES}"
                )
        else:
            calc_mode = FID_DEFAULT_CALC_MODE
        return calc_mode
