from src.dataset.Dataset import Dataset

from src.controller.ControllerRegistry import ControllerRegistry
from src.generator.GeneratorRegistry import GeneratorRegistry

from src.metric.SampleMetricRegistry import SampleMetricRegistry
from src.metric.SampleMetricManager import SampleMetricManager

from src.metric.CompoundMetricRegistry import CompoundMetricRegistry
from src.metric.CompoundMetricManager import CompoundMetricManager
from src.metric.CompoundMetric import CompoundMetric

from src.filter.FilterRegistry import FilterRegistry
from src.filter.Filter import Filter

from src.compound_model.CompoundModelFactory import CompoundModelFactory

import src.util.PromptUtil as PU

from src.phase_utils import (
    confirm_lists,
    select_dataset,
    select_attributes,
    select_filters,
    select_compound_metrics,
)


def evaluation_phase() -> None:
    """
    Evaluate generated populations.
    """

    # * Select dataset / attributes
    dataset = select_dataset()
    attributes = select_attributes(dataset)

    # * Create compound model factories
    generators = [g(dataset) for g in GeneratorRegistry.get_resources()]
    controllers = [
        ControllerRegistry.get_resource(c_name)(g, attributes)
        for c_name in ControllerRegistry.get_names()
        for g in generators
        if g.get_name() in ControllerRegistry.get_compatible_generator_names(c_name)
    ]
    cmfs = [CompoundModelFactory(c, [], [], []) for c in controllers]

    # * Find available populations
    invalid_cmfs = list(
        filter(lambda cmf: cmf.get_population().num_samples() <= 0, cmfs)
    )

    if len(invalid_cmfs) > 0:
        if PU.prompt_yes_no("Some populations are not yet generated. Show them?"):
            PU.print_list(
                "The following populations are not yet generated:",
                [cmf.get_population().get_name() for cmf in invalid_cmfs],
            )
            PU.input_continue()

        # Remove invalid factories
        cmfs = [cmf for cmf in cmfs if cmf not in invalid_cmfs]

    # * Population selection
    cmfs: list[CompoundModelFactory] = [
        cmfs[i]
        for i in PU.prompt_multi_options(
            "What populations would you like to evaluate?",
            PU.tablify(
                [
                    [cmf.get_population().get_name() for cmf in cmfs],
                    [f"Size: {cmf.get_population().num_samples()}" for cmf in cmfs],
                ]
            ),
            allow_empty=True,
            return_index=True,
        )
    ]

    if len(cmfs) <= 0:
        print("No population selected. Aborting.")
        return

    # * Select filters
    filters = _select_filters(dataset)
    if len(filters) <= 0:
        print("No filter selected. Aborting.")
        return
    refilter = PU.prompt_yes_no("Refilter populations if they are already filtered?")

    # * Select compound metrics
    compound_metric_types = _select_compound_metrics(dataset, cmfs)
    if len(compound_metric_types) <= 0:
        print("No compound metrics selected. Aborting.")
        return
    recalculate = PU.prompt_yes_no("Recalculate metrics if they already exist?")

    # * Reconstruct factories with selections
    cmfs = [
        CompoundModelFactory(
            cmf.get_controller(),
            filters,
            SampleMetricRegistry.get_resources(),
            compound_metric_types[cmf.get_name()],
        )
        for cmf in cmfs
    ]

    # * Confirmation
    if not confirm_lists(
        "Use the following filters per controller-generator pair for "
        + "constituting compound models?",
        *[
            (
                cmf.get_controller().get_name()
                + " | "
                + cmf.get_controller().get_generator().get_name(),
                [f"filter: {f.get_name()}" for f in filters],
            )
            for cmf in cmfs
        ],
    ) or not confirm_lists(
        "Calculate the following metrics per controller-generator pair? "
        + "(all filters for each)",
        *[
            (
                cmf.get_controller().get_name()
                + " | "
                + cmf.get_controller().get_generator().get_name(),
                cmf.get_compound_metric_names(),
            )
            for cmf in cmfs
        ],
    ):
        print("Aborting.")
        return

    # * Perform filtering
    filter_status = _perform_filtering(filters, cmfs, refilter)

    # * Perform evaluation
    metric_status = _perform_evaluation(filters, filter_status, cmfs, recalculate)

    # * Status
    _display_status(cmfs, filter_status, metric_status)

def _display_status(
    cmfs: list[CompoundModelFactory],
    filter_status: dict[str, dict[str, bool]],
    metric_status: dict[str, dict[str, bool]],
) -> None:
    PU.push_indent(1)
    PU.print_with_border("Filtering and Evaluation Status", "=", "||")
    for cmf in cmfs:
        for f in cmf.get_filters():
            PU.push_indent(1)
            PU.print_list(
                " | ".join(
                    (
                        f"Compound Model: ("
                        + ", ".join(
                            (
                                cmf.get_controller().get_name(),
                                cmf.get_controller().get_generator().get_name(),
                                f.get_name(),
                            )
                        )
                        + ")",
                        (
                            "FILTER SUCCESS | THE FOLLOWING METRICS WERE CALCULATED"
                            if filter_status[cmf.get_name()][f.get_name()]
                            else "FILTER FAILURE | NO METRICS WERE CALCULATED"
                        ),
                    )
                ),
                PU.tablify(
                    [
                        cmf.get_compound_metric_names(),
                        [
                            "SUCCESS"
                            if metric_status[cmf.get_name()][name]
                            else "FAILURE"
                            for name in cmf.get_compound_metric_names()
                        ],
                    ]
                ),
                header_border_symbol="-",
                header_border_side_symbol="|",
            )
            PU.pop_indent()
    PU.pop_indent()


def _perform_evaluation(
    filters: list[type[Filter]],
    filter_status: dict[str, dict[str, bool]],
    cmfs: list[CompoundModelFactory],
    recalculate: bool,
) -> dict[str, dict[str, bool]]:
    metric_status = {
        cmf.get_name(): {m: True for m in cmf.get_compound_metric_names()}
        for cmf in cmfs
    }
    PU.push_indent(1)
    PU.print_with_border("Evaluating Populations", "#", "###")
    for cmf in cmfs:
        PU.push_indent(1)
        PU.print_with_border(
            f"Performing evaluation of population: '{cmf.get_population().get_name()}'",
            "=",
            "||",
        )
        for metric_name in cmf.get_compound_metric_names():

            # Get failed filters
            failed_filters = [
                f for f in filters if not filter_status[cmf.get_name()][f.get_name()]
            ]

            # Evaluate (calculate metrics)
            try:
                PU.print_with_border(
                    f"Calculating compound metric '{metric_name}'.", "-", "|"
                )
                if len(failed_filters) == len(filters):  # Skip metric
                    PU.push_indent(3)
                    PU.print_with_indent("There were no successful filters!")
                    PU.print_with_indent("Skipping metric.")
                    PU.pop_indent()
                    continue
                elif len(failed_filters) > 0:  # Continue with successful filters
                    PU.print_list(
                        "Skipping failed filters:",
                        [f.get_name() for f in failed_filters],
                    )
                PU.print_list(
                    "Calculating for the following filters:",
                    [f.get_name() for f in filters if f not in failed_filters],
                )
                cmf.evaluate_compound_models(
                    [f for f in filters if f not in failed_filters],
                    metric_name,
                    recalculate,
                )
            except Exception as error:
                metric_status[cmf.get_name()][metric_name] = False
                PU.push_indent(3)
                PU.print_with_border(
                    f"Failed to calculate metric '{metric_name}' for population "
                    + f"'{cmf.get_population().get_name()}'",
                    "!",
                )
                PU.print_with_indent(repr(error))
                PU.pop_indent()
        PU.pop_indent()
    PU.pop_indent()

    return metric_status


def _perform_filtering(
    filters: list[type[Filter]], cmfs: list[CompoundModelFactory], refilter: bool
) -> dict[str, dict[str, bool]]:
    filter_status = {
        cmf.get_name(): {f.get_name(): True for f in filters} for cmf in cmfs
    }
    PU.push_indent(1)
    PU.print_with_border("Filtering Populations", "#", "###")
    for cmf in cmfs:
        PU.push_indent(1)
        PU.print_with_border(
            f"Performing filtering of population: '{cmf.get_population().get_name()}'",
            "=",
            "||",
        )
        for f in filters:
            if not refilter and any(
                cmf.get_population().get_filtering_indices(f.get_bit())
            ):  # Filter already applied
                PU.print_with_border(
                    f"Filter '{f.get_name()}' has already been applied.", "-", "|"
                )
                continue

            # Apply filter
            try:
                PU.print_with_border(f"Applying filter '{f.get_name()}'.", "-", "|")
                cmf.apply_filters(f)
            except Exception as error:
                filter_status[cmf.get_name()][f.get_name()] = False
                PU.push_indent(3)
                PU.print_with_border(
                    f"Failed to apply filter '{f.get_name()}' to population "
                    + f"'{cmf.get_population().get_name()}'",
                    "!",
                )
                PU.print_with_indent(repr(error))
                PU.pop_indent()
        PU.pop_indent()
    PU.pop_indent()

    return filter_status


def _select_compound_metrics(
    dataset: Dataset, cmfs: list[CompoundModelFactory]
) -> dict[str, list[type[CompoundMetric]]]:
    _, compound_metric_types = select_compound_metrics()

    smm = SampleMetricManager(SampleMetricRegistry.get_resources(), None, dataset)

    invalid_metrics = {}

    # * Check metric status
    for cmf in cmfs:

        # Create CMM for current CMF's controller
        cmm = CompoundMetricManager(
            compound_metric_types, None, dataset, smm, cmf.get_controller(), -1
        )

        # Find invalid metrics
        invalid_metrics[cmf.get_name()] = list(
            filter(lambda m: not m.is_ready(), cmm.get_metric_instances())
        )

    # * Inform about metric status
    if any(len(li) > 0 for li in invalid_metrics.values()) > 0:
        PU.push_indent(3)
        PU.print_with_border("Some metrics were not ready.", symbol="*")
        PU.print_with_indent(
            "Make sure that they have been set up prior to evaluation."
        )
        for k in invalid_metrics.keys():
            if len(invalid_metrics[k]) <= 0:
                continue
            PU.print_list(
                header=f"{k}:",
                items=PU.tablify(
                    [
                        [m.get_name() for m in invalid_metrics[k]],
                        "SETUP NOT COMPLETE",
                    ]
                ),
            )

        if not PU.prompt_yes_no("Skip these and continue?"):
            PU.pop_indent()
            return {}

        # Add valid metric types
        PU.pop_indent()

    return {
        cmf.get_name(): [
            cmt
            for cmt in compound_metric_types
            if cmt not in [type(im) for im in invalid_metrics[cmf.get_name()]]
        ]
        for cmf in cmfs
    }


def _select_filters(dataset: Dataset) -> list[type[Filter]]:
    _, filters = select_filters()

    smm = SampleMetricManager(SampleMetricRegistry.get_resources(), None, dataset)

    # * Check filter status
    invalid_filters = list(
        filter(lambda f: not f(_setup_only=True, smm=smm).is_ready(), filters)
    )

    if len(invalid_filters) > 0:
        PU.push_indent(3)
        PU.print_with_border("Some filters were not ready.", symbol="*")
        PU.print_with_indent("Make sure that they have been set up prior to filtering.")
        PU.print_list(
            items=PU.tablify(
                [
                    [f.get_name() for f in filters],
                    "SETUP NOT COMPLETE",
                ]
            )
        )
        if not PU.prompt_yes_no("Skip these and continue?"):
            PU.pop_indent()
            return []

        # Remove invalid filters
        filters = [f for f in filters if f not in invalid_filters]
        PU.pop_indent()

    return filters
