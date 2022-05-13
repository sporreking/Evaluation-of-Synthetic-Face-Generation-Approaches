from __future__ import annotations

import src.metric.CompoundMetric as CompoundMetric
from src.metric.SampleMetricManager import SampleMetricManager
from src.population.Population import Population
from src.dataset.Dataset import Dataset
from src.util.FileJar import FileJar
from src.controller.Controller import Controller
from typing import Union, Any, Type
import pandas as pd
import numpy as np


class CompoundMetricManager:

    # Pickle file name
    PICKLE_METRICS_SUFFIX = "metrics_filter_bit_"
    PICKLE_METRICS_FILE_NAME = (
        lambda cls, filter_bit: f"{cls.PICKLE_METRICS_SUFFIX}{filter_bit}.pkl"
    )

    def __init__(
        self,
        compound_metrics: list[Type],
        population: Population,
        dataset: Dataset,
        smm: SampleMetricManager,
        controller: Controller,
        filter_bit: int,
    ):
        """
        Constructs a new CompoundMetricManager for handling the specified metrics
        for the provided population.

        Saves and loads metrics from pickle file located at:

        'population/`population.get_name()`/`PICKLE_METRICS_FILE_NAME(filter_bit)`'

        Args:
            compound_metrics (list[Type]): list of the
                the type of metrics that this manager should support. Compound metrics must be setup and ready.
            dataset (Dataset): A dataset that may be used by the different
                metrics to derive their results.
            population (Population): The population on which the metrics should be applied.
            smm (SampleMetricManager): Used for calculating per-sample metrics.
            controller (Controller): The controller associated with the population.
            filter_bit (int): Filter bit used to select a subset of the
                population. Filter bit is defined by the order in FilterRegistry. For example,
                the first filter (IdentityFilter) corresponds to filter bit 1.

        Raises:
            ValueError: If any of the items passed to `metrics` is not
                a valid CompoundMetric.
        """
        # Input check
        for c in compound_metrics:
            if not issubclass(c, CompoundMetric.CompoundMetric):
                raise ValueError(f"Not a CompoundMetric: '{c.__name__}'")

        self._dataset = dataset
        self._population = population
        self._smm = smm
        self._controller = controller
        self._filter_bit = filter_bit

        # Construct metrics
        compound_metrics_con = [metric(self, smm) for metric in compound_metrics]

        # Save compound metrics
        self._compound_metrics = dict(
            (metric.get_name(), metric) for metric in compound_metrics_con
        )

        # Create storage, init with NaN to keep track of calculated metrics
        self._metrics = pd.DataFrame(columns=self.get_metric_names())
        self._metrics.loc[0, :] = np.nan

        # Get already saved metrics from disk
        self._file_jar_path = (
            Population.POPULATION_ROOT_DIR / self._population.get_name()
        )
        self._file_jar = FileJar(self._file_jar_path)
        file_jar_metrics = self._file_jar.get_file(
            self.PICKLE_METRICS_FILE_NAME(self._filter_bit), pd.read_pickle
        )

        # Copy metrics specified by `compound_metrics` from saved metrics from disk
        if file_jar_metrics is not None:
            for column in file_jar_metrics:
                if column in self._metrics:
                    self._metrics[column] = file_jar_metrics[column]

    def get_metric_instances(self) -> list[CompoundMetric.CompoundMetric]:
        """
        Returns a list of all compound metric instances. There should be exactly
        one instance for each metric type specified upon manager construction.

        Returns:
            list[CompoundMetric]: A list of all compound metric instances.
        """
        return list(self._compound_metrics.values())

    def get_population(self) -> Population:
        """
        Returns the dedicated population of this manager. This value
        was assigned upon the manger's instantiation.

        Returns:
            Population: The population of this manager.
        """
        return self._population

    def get_controller(self) -> Controller:
        """
        Returns the controller associated with this manager.

        Returns:
            Controller: The controller of this manager.
        """
        return self._controller

    def get_dataset(self) -> Dataset:
        """
        Returns the dataset contained by this manager. This value
        was assigned upon the manager's instantiation.

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
        return list(self._compound_metrics.keys())

    def get(
        self,
        metric_names: Union[list[str], str] = None,
        calc_if_missing: bool = False,
        **parameters: Any,
    ) -> Any:
        """
        Retrieves the requested metrics.

        Note that if `calc_if_missing=True`, a call will be made to `calc()`.
        More information about possible exceptions, parameters etc. may be be found in the
        docstring of that function.

        Args:
            metric_names (Union[list[str], str], optional): The metrics to retrieve.
                The manager must have been initialized with CompoundMetrics
                which the specified metric names match. If `None`, all metrics will be
                retrieved. Defaults to None.
            calc_if_missing (bool, optional): If `True`, the specified metrics will
                be calculated automatically if they do not exist.
                This is done through the `calc()` function. Defaults to False.
            **parameters (Any): Arbitrary parameters required by the metrics.
                These are only used if `calc_if_missing=True`, and are all forwarded
                to each requested metric's CompoundMetrics upon calculation.
                Note that if different metrics require parameters with the same name,
                they may not have different values unless calls to `calc()` are
                performed for the metrics separately.

        Returns:
            Any: The requested metrics. If several metrics are requested, result will be
                returned as a pd.DataFrame, if only one metric is requested, the returned
                value will have the type associated with that metric (e.g. float, int)

        Raises:
            ValueError: If any of the specified metrics are unknown. Note that they should
                be added upon manager instantiation.

        """
        metric_names = self._parse_metric_names(metric_names)

        # Calculate if missing
        if calc_if_missing:

            # Check each metric individually
            for metric_name in metric_names:

                # Check if value is NaN
                if np.isnan(self._metrics.loc[0, metric_name]):
                    # Calculate for missing metric
                    self.calc(metric_name, **parameters)

        # Fetch metrics
        if len(metric_names) > 1:
            return self._metrics.loc[0, metric_names]
        else:
            return self._metrics.loc[0, metric_names].item()

    def calc(
        self,
        metric_names: Union[list[str], str] = None,
        **parameters: Any,
    ) -> None:
        """
        Calculates the requested metrics.

        Args:
            metric_names (Union[list[str], str], optional): The metrics to calculate.
                The manager must have been initialized with CompoundMetrics
                which the specified metric names match. If `None`, all metrics will be
                calculated. Defaults to None.
            **parameters (Any): Arbitrary parameters required by the metrics.
                All of these parameters are forwarded to each requested metric's
                CompoundMetric. Note that if different metrics require parameters
                with the same name, they may not have different values unless calls to
                `calc()` are performed for the metrics separately.

        Raises:
            ValueError: If any of the specified metrics are unknown. Note that they should
                be added upon manager instantiation.
        """
        metric_names = self._parse_metric_names(metric_names)

        # Fetch compound metrics
        cms = [self._compound_metrics[metric_name] for metric_name in metric_names]

        # Calculate metrics
        for cm in cms:
            msg = f"# Computing {cm.get_name()} #"
            print(len(msg) * "#")
            print(msg)
            print(len(msg) * "#")
            # Derive metric for samples
            result = cm.calc(filter_bit=self._filter_bit, **parameters)

            # Store result
            self._metrics.loc[0, cm.get_name()] = result

        ## Save results to disk
        # Get metrics from disk
        file_jar_metrics = self._file_jar.get_file(
            self.PICKLE_METRICS_FILE_NAME(self._filter_bit), pd.read_pickle
        )

        # Update relevent columns with the newly calculated metrics if it exists
        if file_jar_metrics is not None:
            for column in self._metrics:
                file_jar_metrics[column] = self._metrics[column]

            # Save results to disk
            self._file_jar.store_file(
                self.PICKLE_METRICS_FILE_NAME(self._filter_bit),
                file_jar_metrics.to_pickle,
            )
        else:
            # Save results to disk
            self._file_jar.store_file(
                self.PICKLE_METRICS_FILE_NAME(self._filter_bit), self._metrics.to_pickle
            )

    def clear_compound_metrics(self) -> None:
        """
        Clear metrics from internal storage.
        """
        self._metrics.loc[0, :] = np.nan

    def clear_local_compound_metrics(population_name: str) -> None:
        """
        Clear metrics from local storage, will clear all compound metrics in
        population directory.

        Args:
            population_name (str): Name of the population to be cleared.
        """
        dir_path = Population.POPULATION_ROOT_DIR / population_name

        # Remove local metrics
        [
            file.unlink()
            for file in dir_path.glob(
                f"{CompoundMetricManager.PICKLE_METRICS_SUFFIX}*.pkl"
            )
            if file.is_file()
        ]

    def _parse_metric_names(self, metric_names: Union[list, str]) -> list[str]:
        # Parse sample metric name input
        if metric_names is None:
            metric_names = self.get_metric_names()
        elif type(metric_names) == str:
            metric_names = [metric_names]

        # Check specified metric names
        for metric_name in metric_names:
            if metric_name not in self.get_metric_names():
                raise ValueError(f"Unknown metric: '{metric_names}'")
        return metric_names
