from typing import Type, Any
from src.filter.Filter import Filter
from src.controller.Controller import Controller
from src.metric.CompoundMetricManager import CompoundMetricManager
from src.metric.SampleMetricManager import SampleMetricManager
from src.metric.SampleMetric import SampleMetric
from src.metric.CompoundMetric import CompoundMetric
from src.population.Population import Population
from src.filter.FilterRegistry import FilterRegistry
import numpy as np

COMPOUND_MODEL_FACTORY_PREFIX_NAME = "CMF"


class CompoundModelFactory:
    """
    This class represents a compound model factory. Each instance is responsible for producing
    compound models that are based on a fixed controller and generator.

    A compound model factory consists mainly of a controller, generator, and filters.
    Furthermore, a compound model is defined as the controller, generator and a filter, such that
    the number of filters determines how many compound models a factory represents.

    Each factory shares a population between all of its compound models, named after the controller and the generator.
    However, each compound model calculates its own metrics using their own metric manager on aforementioned
    population , by applying them to the separate filterings.
    """

    def __init__(
        self,
        controller: Controller,
        filters: list[Type[Filter]],
        sample_metrics: list[Type[SampleMetric]],
        compound_metrics: list[Type[CompoundMetric]],
    ):
        """
        Constructs a CompoundModelFactory with the specified arguments.

        Args:
            controller (Controller): Controller of the CompoundModelFactory. Generator and Dataset will be inferred from
                the controller.
            filters (list[Type[Filter]]): Filters of the CompoundModelFactory, each filter in combination with the
                controller is defined as a compound model.
            sample_metrics (list[Type[SampleMetric]]): All sample metrics used to evaluate the model.
            compound_metrics (list[Type[CompoundMetric]]): All compound metrics used to evaluate the model.
        """
        # Define key compound model components
        self._controller = controller
        self._generator = controller.get_generator()
        self._filters = filters

        # Define data objects
        self._dataset = self._generator.get_dataset()
        self._compound_model_name_prefix = "_".join(
            [
                controller.get_name(),
                self._generator.get_name(),
            ]
        )
        self._name = "_".join(
            [COMPOUND_MODEL_FACTORY_PREFIX_NAME, self._compound_model_name_prefix]
        )
        self._population = Population(self._name)

        # Define metric related objects, one manager for each compound model
        self._smms = {
            filter.get_name(): SampleMetricManager(
                sample_metrics, self._population, self._dataset
            )
            for filter in self._filters
        }
        self._cmms = {
            filter.get_name(): CompoundMetricManager(
                compound_metrics,
                self._population,
                self._dataset,
                self._smms[filter.get_name()],
                controller,
                filter.get_bit(FilterRegistry),
            )
            for filter in self._filters
        }

    def generate_custom_population(
        self,
        latent_codes: np.ndarray,
        input: dict[str, np.ndarray],
        pop_name: str,
        append=True,
    ):
        """
        Generates samples based on specified latent codes and input (which specify
        attribute manipulations).

        Samples will be saved under the specified population name.

        An example of the input:

        Let input be defined as:
        `input_example = {"gender" : np.ndarray[-1, 1, -0.5], "age": np.ndarray[1,0,-1]}`
        then input for image 1 should then be parsed as an image with gender = -1
        and age = 1. Thus, `input_example` defines parameters for 3 images (length of the array).

        Note that a value of 0 means that no changes for that parameter will be done for that image.

        Note that the custom population is not meant to be evaluated. However, it can be done
        by constructing a metric manager for the population, this is not intended and may produce
        strange results.

        Note that this function clears local and internal compound metrics stored by the compound metric manager for this
        population.

        Args:
            latent_codes (np.ndarray): Latent codes that will be generated. Should be at least
                2-dimensional. First axis should represent the number of samples, the other ones should
                represent the dimensions of a latent code.
            input (dict[str, np.ndarray]): Standardized input representation.
                Where key = parameter name, and the value = array of values
                (in float range of [-1,1]) of that parameter. Each value in the
                array represents one generated image. For example, image 1 should
                be based on value indexed 1 in the array for each parameter in input.keys().
                Standardized representation refers to the float range of [-1,1].
            pop_name (str): Name of the population.
            append (bool, optional): True if generated samples should be appended to the population.
                False if the new samples should replace the population instead. Defaults to True.

        Raises:
            AssertionError: When the number of latent codes are below 1.
            AssertionError: When shape of the `latent_codes` is not valid.
            AssertionError: When a attribute is not supported by the controller.
        """
        # Check input
        assert len(latent_codes.shape) > 1
        num_samples = latent_codes.shape[0]
        assert num_samples > 0
        for attr in input.keys():
            assert attr in self._controller.get_attributes()

        pop = Population(pop_name)

        # Parse input
        parsed_input = self._controller.parse_native_input(input)

        # Generate images
        uris, manipulated_latent_codes = self._controller.generate_native(
            latent_codes, parsed_input
        )

        # Add images to population
        pop.add_all(
            manipulated_latent_codes,
            latent_codes,
            uris,
            [1] * len(uris),
            append=append,
            **input,
        )

        # Clear compound metrics
        self._clear_compound_metrics(pop_name)

    def generate_population(
        self, num_samples: int, num_attrs=-1, replace=False
    ) -> None:
        """
        Generates a random population with specified number of samples, or appends them to a
        already existing population.

        The number of attributes manipulated for each sample is specified by number of attributes.
        (Input sampling to be used by the controller is defined by the controller function `sample_random_input()`.)

        An example how the generated population interacts with pre-existing population:

        Let P be a population with 5 number of samples with id 1,2,3,4,5.
        'generate_population(num_samples = 15, replace=False)' will result in a population with the prexisting samples
        1-5 and newly created samples 6-15. If replace would be set to true, id 1-5 would be replaced by new samples.

        Another example:

        Let P be a population with 5 number of samples with id 1,2,3,4,5.
        'generate_population(num_samples = 3, replace=True)' will result in id 1-3 being replaced by new samples and
        4-5 staying as they were.

        In the last example, if the wanted result would be a new population with just 3 samples, then one would
        have to manually delete the old population folder first.

        Note that this function clears local and internal compound metrics stored by the compound metric manager for this
        population.

        Args:
            num_samples (int): Aspiring number of samples to generate. The actual number of samples to generate
                depends on `replace`. When `replace` is True, `num_samples` samples will be generated, these samples
                will replace any pre-existing samples (with that ID) in the population associated this factory.
                When `replace` is False, generated samples will be appended to the existing population (even if its empty)
                such that the population grows to up to `num_samples` samples.
            num_attrs (int, optional): Number of attributes to manipulate for each sample. Should not exceed the number of
                attributes of a controller. If set to -1, each sample will get random number of manipulated attributes.
                Defaults to -1.
            replace (bool, optional): True if samples in the population should be replaced, if false samples are appended to
                the population instead. Defaults to False.

        Raises:
            AssertionError: When `num_samples` or `num_attrs` are not valid values.
        """

        # Check input
        assert num_samples > 0
        assert num_attrs >= -1
        assert num_attrs <= len(self._controller.get_attributes())

        pop_samples = self._population.num_samples()

        # Check early stopping criteria
        if (not replace) and pop_samples >= num_samples:
            print(
                f"The population already consists of {pop_samples} ({num_samples} was the goal) samples."
            )
            return

        # Set number of samples to generate
        samples_to_gen = num_samples if replace else num_samples - pop_samples

        # Set sample mode, see sample_random_input() docs for more info
        attr_sample_mode = "random" if num_attrs == -1 else num_attrs
        random_input = self._controller.sample_random_input(
            samples_to_gen, attr_sample_mode
        )

        # Parse input
        parsed_input = self._controller.parse_native_input(random_input)

        # Get latent codes
        latent_codes = self._generator.random_latent_code(samples_to_gen)

        # Generate images
        uris, manipulated_latent_codes = self._controller.generate_native(
            latent_codes, parsed_input
        )

        # Add images to population
        self._population.add_all(
            manipulated_latent_codes,
            latent_codes,
            uris,
            [1] * len(uris),
            append=(not replace),
            **random_input,
        )

        # Clear compound metrics
        self._clear_compound_metrics(self._population.get_name())

    def _clear_compound_metrics(self, population_name: str) -> None:
        # Clear compound metrics from manager
        for filter in self._filters:
            cmm = self._cmms[filter.get_name()]
            if population_name == cmm.get_population().get_name():
                cmm.clear_compound_metrics()

        # Clear locally stored compound metrics from population
        CompoundMetricManager.clear_local_compound_metrics(population_name)

    def evaluate_compound_models(
        self,
        filters: Type[Filter] | list[Type[Filter]] = None,
        compound_metrics: list[str] | str = None,
        recalculate_metrics: bool = False,
        **parameters: Any,
    ) -> None:
        """
        Evaluates a selection or all of the compound models on a selection of compound metrics.
        Result are saved to the managers storage.

        Args:
            filters (Type[Filter] | list[Type[Filter]], optional): Selection of filters to use, must be subset
                of the filters which was specified in the construction, see `get_filters()`.
                `filters` can either be a filter, a list of filters or `None` which will result in the use of all available filters.
                Defaults to None.
            compound_metrics (list[str] | str, optional): List of compound metric names to calculate, must be subset
                of the compound metrics which was specified in the construction, see `get_compound_metric_names()`.
                `compound_metrics` can either be a compound metric name, a list of compound metric names or `None` which
                will result in the use of all available compound metrics. Defaults to None.
            recalculate_metrics (bool, optional): True if metrics should be recalculated even if they already exist.
                False means that metrics already calculated will be skipped. Defaults to False.
            **parameters(Any): Parameters passed down to the calculation functions `calc()` and `get()`
                of the compound metric manager. These parameters will then be passed down to the each
                underlying metric of that compound metric manager. See their documentation for more details.

        Raises:
            AssertionError: When arguments `filters` or `compound_metrics` are not valid values.
        """
        # Format input
        filters = self._format_filters_input(filters)
        compound_metrics = self._format_compound_metrics_input(compound_metrics)

        # Check input
        self._check_filters(filters)
        self._check_compound_metrics(compound_metrics)

        # Evaluate each compound model
        for filter in filters:
            self._evaluate_compound_model(
                filter, compound_metrics, recalculate_metrics, **parameters
            )

    def _evaluate_compound_model(
        self,
        filter: Type[Filter],
        compound_metrics: list[str],
        recalculate_metrics: bool = False,
        **parameters: Any,
    ) -> None:

        msg = f"# Evaluating metrics for compound model: {self._compound_model_name_prefix}_{filter.get_name()} #"
        print(len(msg) * "#")
        print(msg)
        print(len(msg) * "#")

        # Apply filter
        # TODO: Implement a smart check if filter already applied
        self._population.apply_filter(filter, self._smms[filter.get_name()])

        # Calc all
        if recalculate_metrics:
            self._cmms[filter.get_name()].calc(compound_metrics, **parameters)

        # Calc missing
        else:
            msg = f"# Trying to fetch metrics #"
            print(len(msg) * "#")
            print(msg)
            print(len(msg) * "#")
            self._cmms[filter.get_name()].get(
                compound_metrics,
                calc_if_missing=True,
                **parameters,
            )

        msg = "# Metric Evaluation complete #"
        print(len(msg) * "#")
        print(msg)
        print(len(msg) * "#")

    def get_filters(self) -> list[Type[Filter]]:
        """
        Returns the filters.

        Returns:
            list[Type[Filter]]: The filters.
        """
        return self._filters

    def get_compound_metric_names(self) -> list[str]:
        """
        Returns a list of all compound metric names.

        Returns:
            list[str]: Name of the compound metrics.
        """
        return self._cmms[self._filters[0].get_name()].get_metric_names()

    def get_name(self) -> str:
        """
        Returns the name of the compond metric factory.

        Returns:
            str: The name.
        """
        return self._name

    def get_compound_model_names(self) -> list[str]:
        """
        Returns a list of all compound model names.

        Returns:
            list[str]: Name of the compound models.
        """
        return [
            f"{self._compound_model_name_prefix}_{filter.get_name()}"
            for filter in self._filters
        ]

    def _check_filters(self, filters: list[Type[Filter]]) -> None:
        for filter in filters:
            assert filter in self._filters

    def _check_compound_metrics(self, compound_metrics: list[str]) -> None:
        for compound_metric in compound_metrics:
            assert compound_metric in self.get_compound_metric_names()

    def _format_filters_input(
        self, filters: Type[Filter] | list[Type[Filter]] = None
    ) -> list[Type[Filter]]:
        if type(filters) is list:
            return filters
        elif filters is None:
            return self._filters
        else:
            return [filters]

    def _format_compound_metrics_input(
        self, compound_metrics: str | list[str] = None
    ) -> list[Type[Filter]]:
        if type(compound_metrics) is list:
            return compound_metrics
        elif compound_metrics is None:
            return self.get_compound_metric_names()
        else:
            return [compound_metrics]
