from __future__ import annotations
from typing import Any, Type, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
from src.dataset.Dataset import Dataset
import src.metric.SampleMetric as SampleMetric
from src.util.FileJar import FileJar
from pathlib import Path

if TYPE_CHECKING:
    from src.population.Population import Population


class SampleMetricManager:
    """
    Class used for calculating and storing metrics for a dedicated population.
    """

    # Pickle file name
    PICKLE_METRICS_NAME = "sample_metric.pkl"

    def __init__(
        self,
        sample_metrics: list[Type],
        population: Union[Population, None],
        dataset: Dataset,
    ):
        """
        Constructs a new SampleMetricManager for handling the specified metrics
        for the provided population.

        Saves and loads metrics from pickle file located at:

        'population/`population.get_name()`/`PICKLE_METRICS_NAME`'

        Args:
            sample_metrics (list[Type]): SampleMetrics that this manager should support.
            population (Population | None): The population on which the metrics should be applied.
                If `None`, no metric activity may be performed, but setups are still possible.
            dataset (Dataset): A dataset that may be used by the different
                SampleMetrics to derive their results.

        Raises:
            ValueError: If any of the items passed to `sample_metrics` is not
                a valid SampleMetric.
        """

        # Input check
        for c in sample_metrics:
            if not issubclass(c, SampleMetric.SampleMetric):
                raise ValueError(f"Not a SampleMetric: '{c.__name__}'")

        # Save members
        self._population = population
        self._dataset = dataset

        # Save sample metrics
        self._sample_metrics = {sm.get_name(): sm(self) for sm in sample_metrics}

        # Create storage
        self._metrics = pd.DataFrame(columns=self.get_metric_names())

        # If there is no population, metrics will not be saved to disk
        if population is None:
            return

        # Get already saved metrics from disk
        self._file_jar_path = (
            self._population.POPULATION_ROOT_DIR / self._population.get_name()
        )
        self._file_jar = FileJar(self._file_jar_path)
        file_jar_metrics = self._file_jar.get_file(
            self.PICKLE_METRICS_NAME, pd.read_pickle
        )

        # Copy metrics specified by `sample_metrics` from saved metrics from disk
        if file_jar_metrics is not None:
            for column in file_jar_metrics:
                if column in self._metrics:
                    self._metrics[column] = file_jar_metrics[column]

    def get_metric_instances(self) -> list[SampleMetric.SampleMetric]:
        """
        Returns a list of all sample metric instances. There should be exactly
        one instance for each metric type specified upon manager construction.

        Returns:
            list[SampleMetric]: A list of all sample metric instances.
        """
        return list(self._sample_metrics.values())

    def get_metric_indices(self) -> pd.Index:
        """
        Returns the indices where one or more sample metrics
        have been calculated.

        Returns:
            pd.Index: The indices.
        """
        return self._metrics.index

    def get_population(self) -> Union[Population, None]:
        """
        Returns the dedicated population of this manager. This value
        was assigned upon the manger's instantiation.

        Returns:
            Population: The population of this manager, or `None` if no
                population was supplied upon construction.
        """
        return self._population

    def get_dataset(self) -> Dataset:
        """
        Returns the dataset contained by this manager. This value
        was assigned upon the manager's instantiation, and was passed
        to each SampleMetric upon their creation since some of the
        metrics may require real data.

        Returns:
            Dataset: The dataset contained by this manager.
        """
        return self._dataset

    def get_metric_names(self) -> list[str]:
        """
        Returns the names of the metrics contained by this manager.

        Returns:
            list[str]: The names of all metrics contained by this manager.
        """
        return list(self._sample_metrics.keys())

    def _parse_input(
        self,
        metric_names: Union[list[str], str],
        ids: Union[list[int], int],
        check_calc: bool = True,
    ) -> tuple[list[str], pd.Index]:
        """Converts input to a standardized form and performs sanity checks."""

        # Parse sample metric name input
        if metric_names is None:
            metric_names = self.get_metric_names()
        elif type(metric_names) == str:
            metric_names = [metric_names]

        # Parse sample indices
        if ids is None:
            ids = self._population.get_data().index
        if type(ids) == int:
            ids = [ids]

        # Check specified metric names
        for metric_name in metric_names:
            if metric_name not in self.get_metric_names():
                raise ValueError(f"Unknown metric: '{metric_names}'")

        # Check population specified IDs
        for id in ids:
            if id not in self._population.get_data().index:
                raise IndexError(
                    f"There is no sample in population '{self._population.get_name()}' "
                    + f"with ID '{id}'"
                )

        # Check self._metrics for specified IDs
        if check_calc:
            for id in ids:
                if id not in self._metrics.index:  # Check if any metrics exist
                    raise IndexError(
                        f"There are no metrics for sample with ID '{id}'! "
                        + "Make sure that the requested metrics have been "
                        + "calculated for the specified IDs."
                    )
                else:  # Check if requested metrics exist
                    for m in metric_names:
                        if not self._is_calculated(id, m):
                            raise IndexError(
                                f"There is no metric '{m}' for sample with ID '{id}'! "
                                + "Make sure that the requested metrics have been "
                                + "calculated for the specified IDs."
                            )

        return metric_names, ids

    def _is_calculated(self, id: int, metric_name: str):
        """Check whether a metric is calculated for a sample."""
        if id in self._metrics.index:
            val1 = self._metrics.loc[id, metric_name]
            val2 = self._metrics.loc[id, metric_name]
            return (
                np.array_equal(val1, val2)
                if isinstance(val1, np.ndarray)
                else val1 == val2
            )

        return False

    def get(
        self,
        metric_names: Union[list[str], str] = None,
        ids: Union[list[int], int] = None,
        calc_if_missing: bool = False,
        skip_if_missing: bool = False,
        **parameters: Any,
    ) -> pd.DataFrame:
        """
        Retrieves the requested metrics for the samples that correspond
        to the specified population IDs.

        Note that if `calc_if_missing=True`, a call will be made to `calc()`.
        More information about possible exceptions etc. may be be found in the
        docstring of that function.

        Args:
            metric_names (Union[list[str], str], optional): The metrics to retrieve.
                The manager must have been initialized with SampleMetrics
                which the specified metric names match. If `None`, all metrics will be
                retrieved for the specified IDs. Defaults to None.
            ids (Union[list[int], int], optional): The IDs of the population samples
                to retrieve the specified metrics for. If `None`, the metrics will be
                retrieved for all samples in the population. Defaults to None.
            calc_if_missing (bool, optional): If `True`, the specified metrics will
                be calculated automatically for the specified IDs if they do not exist.
                This is done through the `calc()` function.
                Note that this should not be True if `skip_if_missing` is True.
                Defaults to False.
            skip_if_missing (bool, optional): If `True`, the specified metrics will
                be skipped for the specified IDs if they do not exist.
                Note that this should not be True if `calc_if_missing` is True.
                Defaults to False.
            **parameters (Any): Arbitrary parameters required by the metrics.
                These are only used if `calc_if_missing=True`, and are all forwarded
                to each requested metric's SampleMetric upon calculation.
                Note that if different metrics require parameters with the same name,
                they may not have different values unless calls to `calc()` are
                performed for the metrics separately.

        Returns:
            pd.DataFrame: The requested metrics for the specified IDs.

        Raises:
            IndexError: If any of the specified sample IDs are not found within
                the manager's population.
            IndexError: If the requested metrics are not yet calculated for the
                specified sample IDs, and if `calc_if_missing=False` (default).
            ValueError: If any of the specified metrics are unknown. Note that they should
                be added upon manager instantiation.
            ValueError: If the result from a metric calculation is not a numpy array.
                This can only happen if `calc_if_missing=True`.
            ValueError: If the result from a metric does not contain precisely one
                result for each requested sample. This can only happen if `calc_if_missing=True`.
            RuntimeError: If the manager has no population, i.e., `None` was supplied upon construction.
            ValueError: If `calc_if_missing` and `skip_if_missing` is True.
        """

        if self._population is None:
            raise RuntimeError("The manager has no population!")

        if calc_if_missing and skip_if_missing:
            raise ValueError("Contradicting arguments, cannot skip AND calc missing.")

        # Adjust input
        metric_names, ids = self._parse_input(
            metric_names, ids, check_calc=not (calc_if_missing or skip_if_missing)
        )

        # Calculate if missing
        if calc_if_missing or skip_if_missing:

            # Check each metric individually
            for metric_name in metric_names:

                # Find samples with current metric missing
                calc_ids = []
                for id in ids:
                    if not self._is_calculated(id, metric_name):
                        calc_ids.append(id)

                # Calculate for missing metric
                if calc_ids:
                    if not skip_if_missing:
                        self.calc(metric_name, calc_ids, **parameters)
                    else:
                        ids = [id for id in ids if id not in calc_ids]

        # Fetch metrics
        return self._metrics.loc[ids, metric_names]

    def calc(
        self,
        metric_names: Union[list[str], str] = None,
        ids: Union[list[int], int] = None,
        **parameters: Any,
    ) -> None:
        """
        Calculates the requested metrics for the samples that correspond
        to the specified population IDs.

        Args:
            metric_names (Union[list[str], str], optional): The metrics to calculate.
                The manager must have been initialized with SampleMetrics
                which the specified metric names match. If `None`, all metrics will be
                calculated for the specified IDs. Defaults to None.
            ids (Union[list[int], int], optional): The IDs of the population samples
                to calculate the specified metrics for. If `None`, the metrics will be
                calculated for all samples in the population. Defaults to None.
            **parameters (Any): Arbitrary parameters required by the metrics.
                All of these parameters are forwarded to each requested metric's
                SampleMetric. Note that if different metrics require parameters
                with the same name, they may not have different values unless calls to
                `calc()` are performed for the metrics separately.

        Raises:
            IndexError: If any of the specified sample IDs are not found within
                the manager's population.
            ValueError: If any of the specified metrics are unknown. Note that they should
                be added upon manager instantiation.
            ValueError: If the result from a metric calculation is not a numpy array.
            ValueError: If the result from a metric does not contain precisely one
                result for each requested sample.
            RuntimeError: If the manager has no population, i.e., `None` was supplied upon construction.
        """

        if self._population is None:
            raise RuntimeError("The manager has no population!")

        # Adjust input
        metric_names, ids = self._parse_input(metric_names, ids, check_calc=False)

        # Fetch sample metrics
        sms: list[SampleMetric.SampleMetric] = [
            self._sample_metrics[metric_name] for metric_name in metric_names
        ]

        # Calculate metrics
        for sm in sms:
            msg = f"# Computing {sm.get_name()} #"
            print(len(msg) * "#")
            print(msg)
            print(len(msg) * "#")

            # Derive metric for samples
            result: np.ndarray = sm.calc(ids, **parameters)

            # Check valid result type
            if not isinstance(result, np.ndarray):
                raise ValueError(
                    f"Calculation failed for metric '{sm.get_name()}'! Returned {result}."
                )

            # Check valid result dimensions
            # Not applicable when using filtering
            if result.shape[0] != len(ids) and "filter_bit" not in parameters:
                raise ValueError(
                    f"Calculation failed for metric '{sm.get_name()}'! First axis must "
                    + "contain exactly one result for each specified ID."
                )

            # Store results
            for i in range(len(result)):
                self._metrics.loc[
                    self._population.get_data(ids).index[i], sm.get_name()
                ] = result[i]

            ## Save results to disk
            # Get metrics from disk
            file_jar_metrics = self._file_jar.get_file(
                self.PICKLE_METRICS_NAME, pd.read_pickle
            )

            # Update relevent columns with the newly calculated metrics if it exists
            if file_jar_metrics is not None:
                for column in self._metrics:
                    file_jar_metrics[column] = self._metrics[column]

                # Save results to disk
                self._file_jar.store_file(
                    self.PICKLE_METRICS_NAME,
                    file_jar_metrics.to_pickle,
                )
            else:
                # Save results to disk
                self._file_jar.store_file(
                    self.PICKLE_METRICS_NAME, self._metrics.to_pickle
                )

    def clear_sample_metrics(self) -> None:
        """
        Clear metrics from internal storage.
        """
        # TODO: Only remove row independent metrics
        self._metrics = pd.DataFrame(columns=self.get_metric_names())

    @staticmethod
    def clear_local_sample_metrics(population_dir: Path) -> None:
        """
        Clear metrics from local storage, will clear all sample metrics
        associated with the specified population directory.

        Args:
            population_dir (path): Name of the population to be cleared.
        """
        # TODO: Only remove row independent metrics
        file_path = population_dir / SampleMetricManager.PICKLE_METRICS_NAME
        if file_path.is_file():
            file_path.unlink()
