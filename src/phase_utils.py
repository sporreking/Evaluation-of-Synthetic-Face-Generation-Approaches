from src.dataset.DatasetRegistry import DatasetRegistry
from src.dataset.Dataset import Dataset

from src.controller.ControllerRegistry import ControllerRegistry
from src.controller.Controller import Controller

from src.generator.GeneratorRegistry import GeneratorRegistry
from src.generator.Generator import Generator

from src.metric.SampleMetricRegistry import SampleMetricRegistry
from src.metric.SampleMetric import SampleMetric

from src.filter.FilterRegistry import FilterRegistry
from src.filter.Filter import Filter

from src.metric.CompoundMetricRegistry import CompoundMetricRegistry
from src.metric.CompoundMetric import CompoundMetric

import src.util.PromptUtil as PU


def get_prompt_save_file_name(name: str) -> str:
    """
    Derives a file name to use for storing default prompt values.

    Args:
        name (str): The unique identifier of the file-name.

    Returns:
        str: The file name.
    """
    return f"{name}.sav"


def confirm_lists(
    message: str, *lists: tuple[str, list[str]], confirm_phrase: str = "Confirm?"
) -> bool:
    """
    Displays a number of lists, each with their own header, and asks the user
    whether they would like to confirm what has been displayed.

    Args:
        message (str): The main message to present, headering all of the lists.
        *lists (tuple[str, list[str]]): Lists to present, coupled with proceding headers.
        confirm_phrase (str, optional): Message for the yes/no prompt. Default is "Confirm?".
    Returns:
        bool: `True` if the user confirms the lists, otherwise `False`.
    """
    PU.push_indent(1)
    PU.print_with_border(message, "=", "||")

    for header, li in lists:
        PU.print_list(
            header,
            li,
            header_border_symbol="-",
            header_border_side_symbol="|",
        )

    accept_selection = PU.prompt_yes_no("Confirm?")

    PU.pop_indent()

    return accept_selection


def select_dataset() -> Dataset:
    """
    Asks the user to pick a dataset from the dataset registry.

    Returns:
        Dataset: The picked dataset.
    """
    ds_name = PU.prompt_options("Select a dataset to use:", DatasetRegistry.get_names())
    ds_type = DatasetRegistry.get_resource(ds_name)
    ds_resolution = PU.prompt_options(
        f"Select a valid resolution for dataset '{ds_type.get_resolution_invariant_name()}':",
        DatasetRegistry.get_available_resolutions(ds_name),
    )

    return ds_type(ds_resolution)


def select_attributes(dataset: Dataset) -> list[str]:
    """
    Asks the user to select attributes from the given `dataset`.

    Returns:
        list[str]: The selected attributes.
    """
    return PU.prompt_multi_options(
        "What attributes should be used?",
        dataset.get_labels().columns,
        default_file=get_prompt_save_file_name(
            f"attributes_{dataset.get_name(dataset.get_resolution())}"
        ),
    )


def select_controllers() -> tuple[list[str], list[type[Controller]]]:
    """
    Asks the user to select controllers from the controller registry.

    Returns:
        tuple[list[str], list[type[Controller]]]: Two lists.
            The first one contains names of the chosen controllers,
            while the second one contains the actual controller types.
    """
    controller_names = PU.prompt_multi_options(
        "What controllers should be used?",
        ControllerRegistry.get_names(),
        default_file=get_prompt_save_file_name("controllers"),
    )
    return controller_names, [
        ControllerRegistry.get_resource(name) for name in controller_names
    ]


def select_generators() -> tuple[list[str], list[type[Generator]]]:
    """
    Asks the user to select generators from the generator registry.

    Returns:
        tuple[list[str], list[type[Generator]]]: Two lists.
            The first one contains names of the chosen generators,
            while the second one contains the actual generator types.
    """
    generator_names = PU.prompt_multi_options(
        "What generators should be used?",
        GeneratorRegistry.get_names(),
        default_file=get_prompt_save_file_name("generators"),
    )
    return generator_names, [
        GeneratorRegistry.get_resource(name) for name in generator_names
    ]


def select_sample_metrics() -> tuple[list[str], list[type[SampleMetric]]]:
    """
    Asks the user to select sample metrics from the sample metric registry.

    Returns:
        tuple[list[str], list[type[SampleMetric]]]: Two lists.
            The first one contains names of the chosen sample metrics,
            while the second one contains the actual sample metric types.
    """
    sample_metric_names = PU.prompt_multi_options(
        "What sample metrics should be used?",
        SampleMetricRegistry.get_names(),
        default_file=get_prompt_save_file_name("sample_metrics"),
    )
    return sample_metric_names, [
        SampleMetricRegistry.get_resource(name) for name in sample_metric_names
    ]


def select_filters() -> tuple[list[str], list[type[Filter]]]:
    """
    Asks the user to select filters from the filter registry.

    Returns:
        tuple[list[str], list[type[Filter]]]: Two lists.
            The first one contains names of the chosen filters,
            while the second one contains the actual filter types.
    """
    filter_names = PU.prompt_multi_options(
        "What filters should be used?",
        FilterRegistry.get_names(),
        default_file=get_prompt_save_file_name("filters"),
    )
    return filter_names, [FilterRegistry.get_resource(name) for name in filter_names]


def select_compound_metrics() -> tuple[list[str], list[type[CompoundMetric]]]:
    """
    Asks the user to select compound metrics from the compound metric registry.

    Returns:
        tuple[list[str], list[type[CompoundMetric]]]: Two lists.
            The first one contains names of the chosen compound metrics,
            while the second one contains the actual compound metric types.
    """
    compound_metric_names = PU.prompt_multi_options(
        "What compound metrics should be used?",
        CompoundMetricRegistry.get_names(),
        default_file=get_prompt_save_file_name("compound_metrics"),
    )
    return compound_metric_names, [
        CompoundMetricRegistry.get_resource(name) for name in compound_metric_names
    ]
