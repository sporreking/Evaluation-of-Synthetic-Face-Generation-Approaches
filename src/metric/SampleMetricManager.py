from __future__ import annotations
from typing import Any, Type, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
from src.dataset.Dataset import Dataset
import src.metric.SampleMetric as SampleMetric

if TYPE_CHECKING:
    from src.population.Population import Population


class SampleMetricManager:
    """
    Class used for calculating and storing metrics for a dedicated population.
    """

    def __init__(
        self,
        sample_metrics: list[Type],
        population: Population,
        dataset: Dataset,
    ):
        """
        Constructs a new SampleMetricManager for handling the specified metrics
        for the provided population.

        Args:
            sample_metrics (list[Type]): SampleMetrics that this manager should support.
            population (Population): The population on which the metrics should be applied.
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

    def get_metric_instances(self) -> list[SampleMetric.SampleMetric]:
        """
        Returns a list of all sample metric instances. There should be exactly
        one instance for each metric type specified upon manager construction.

        Returns:
            list[SampleMetric]: A list of all sample metric instances.
        """
        return list(self._sample_metrics.values())

    def get_population(self) -> Population:
        """
        Returns the dedicated population of this manager. This value
        was assigned upon the manger's instantiation.

        Returns:
            Population: The population of this manager.
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
        return (
            id in self._metrics.index
            and self._metrics.loc[id, metric_name] == self._metrics.loc[id, metric_name]
        )

    def get(
        self,
        metric_names: Union[list[str], str] = None,
        ids: Union[list[int], int] = None,
        calc_if_missing: bool = False,
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
                This is done through the `calc()` function. Defaults to False.
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

        """
        # Adjust input
        metric_names, ids = self._parse_input(
            metric_names, ids, check_calc=not calc_if_missing
        )

        # Calculate if missing
        if calc_if_missing:

            # Check each metric individually
            for metric_name in metric_names:

                # Find samples with current metric missing
                calc_ids = []
                for id in ids:
                    if not self._is_calculated(id, metric_name):
                        calc_ids.append(id)

                # Calculate for missing metric
                self.calc(metric_name, calc_ids, **parameters)

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
        """

        # Adjust input
        metric_names, ids = self._parse_input(metric_names, ids, check_calc=False)

        # Fetch sample metrics
        sms: list[SampleMetric.SampleMetric] = [
            self._sample_metrics[metric_name] for metric_name in metric_names
        ]

        # Calculate metrics
        for sm in sms:
            # Derive metric for samples
            result: np.ndarray = sm.calc(ids, **parameters)

            # Check valid result type
            if not isinstance(result, np.ndarray):
                raise ValueError(
                    f"Calculation failed for metric '{sm.get_name()}'! Returned {result}."
                )

            # Check valid result dimensions
            if result.shape[0] != len(ids):
                raise ValueError(
                    f"Calculation failed for metric '{sm.get_name()}'! First axis must "
                    + "contain exactly one result for each specified ID."
                )

            # Store results
            for i in range(len(result)):
                self._metrics.loc[
                    self._population.get_data(ids).index[i], sm.get_name()
                ] = result[i]
