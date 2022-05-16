from src.controller.ControllerRegistry import ControllerRegistry
from src.metric.SampleMetricManager import SampleMetricManager
from src.metric.CompoundMetricManager import CompoundMetricManager

from src.core.Setupable import Setupable

import src.util.PromptUtil as PU

from typing import Any

from src.phase_utils import (
    confirm_lists,
    select_dataset,
    select_attributes,
    select_controllers,
    select_generators,
    select_sample_metrics,
    select_filters,
    select_compound_metrics,
)


def setup_phase() -> None:
    """
    Allows the user to perform setups for their desired controllers, generators, filters, and metrics.
    """

    # * Initial setupable selection
    dataset = select_dataset()
    attributes = select_attributes(dataset)
    controller_names, controller_types = select_controllers()
    generator_names, generator_types = select_generators()
    sample_metric_names, sample_metrics_types = select_sample_metrics()
    filter_names, filter_types = select_filters()
    compound_metric_names, compound_metrics_types = select_compound_metrics()

    # * Final presentation before config
    if not confirm_lists(
        "Initial Selection",
        ("Dataset", [dataset.get_name(dataset.get_resolution())]),
        ("Attributes", attributes),
        ("Controllers", controller_names),
        ("Generators", generator_names),
        ("Sample Metrics", sample_metric_names),
        ("Filters", filter_names),
        ("Compound Metrics", compound_metric_names),
    ):
        print("Aborting.")
        return

    # * Create setupable instances
    generators = [g(dataset) for g in generator_types]
    controllers = [
        c(g, attributes)
        for i, c in enumerate(controller_types)
        for g in generators
        if g.get_name()
        in ControllerRegistry.get_compatible_generator_names(controller_names[i])
    ]
    sample_metric_manager = SampleMetricManager(sample_metrics_types, None, dataset)
    filters = [f(_setup_only=True, smm=sample_metric_manager) for f in filter_types]
    compound_metric_managers = [
        CompoundMetricManager(
            compound_metrics_types, None, dataset, sample_metric_manager, c, -1
        )
        for c in controllers
    ]

    # * Configure controller setup
    (
        selected_controllers,
        selected_controller_names,
        controller_setup_config,
    ) = _select_setup_modes(
        "Controller",
        controllers,
        [c.get_name() + "_" + c.get_generator().get_name() for c in controllers],
    )

    # * Configure sample metric setup
    (
        selected_sample_metrics,
        selected_sample_metric_names,
        sample_metric_setup_config,
    ) = _select_setup_modes(
        "Sample Metric",
        sample_metric_manager.get_metric_instances(),
        [m.get_name() for m in sample_metric_manager.get_metric_instances()],
    )

    # * Configure filter setup
    (
        selected_filters,
        selected_filter_names,
        filter_setup_config,
    ) = _select_setup_modes(
        "Filter",
        filters,
        [f.get_name() for f in filter_types],
    )

    # * Configure compound metric setup
    (
        selected_compound_metrics,
        selected_compound_metric_names,
        compound_metric_setup_config,
    ) = _select_setup_modes(
        "Compound Metric",
        [m for cmm in compound_metric_managers for m in cmm.get_metric_instances()],
        [
            m.get_name()
            + "_"
            + cmm.get_controller().get_name()
            + "_"
            + cmm.get_controller().get_generator().get_name()
            for cmm in compound_metric_managers
            for m in cmm.get_metric_instances()
        ],
    )

    # * Final confirmation before launch
    if not confirm_lists(
        "Run setups for the following targets?",
        ("Controllers", selected_controller_names),
        ("Sample Metrics", selected_sample_metric_names),
        ("Filters", selected_filter_names),
        ("Compound Metrics", selected_compound_metric_names),
    ):
        print("Aborting.")
        return

    # * Perform controller setups
    status_controllers: dict[str, dict[str, bool]]
    if len(selected_controllers) > 0:
        PU.push_indent(1)
        PU.print_with_border("Performing Setup for Controllers")
        status_controllers = _run_setup_modes(
            selected_controllers, selected_controller_names, controller_setup_config
        )
        PU.pop_indent()

    # * Perform sample metric setups
    status_sample_metrics: dict[str, dict[str, bool]]
    if len(selected_sample_metrics) > 0:
        PU.push_indent(1)
        PU.print_with_border("Performing Setup for Sample Metrics")
        status_sample_metrics = _run_setup_modes(
            selected_sample_metrics,
            selected_sample_metric_names,
            sample_metric_setup_config,
        )
        PU.pop_indent()

    # * Perform filter setups
    status_filters: dict[str, dict[str, bool]]
    if len(selected_filters) > 0:
        PU.push_indent(1)
        PU.print_with_border("Performing Setup for Filters")
        status_filters = _run_setup_modes(
            selected_filters, selected_filter_names, filter_setup_config
        )
        PU.pop_indent()

    # * Perform compound metric setups
    status_compound_metrics: dict[str, dict[str, bool]]
    if len(selected_compound_metrics) > 0:
        PU.push_indent(1)
        PU.print_with_border("Performing Setup for Compound Metrics")
        status_compound_metrics = _run_setup_modes(
            selected_compound_metrics,
            selected_compound_metric_names,
            compound_metric_setup_config,
        )
        PU.pop_indent()

    # * Print status
    PU.push_indent(1)

    PU.push_indent(1)
    PU.print_with_border("Finished", "=", "||")
    PU.pop_indent()

    if len(selected_controllers) > 0:
        _display_status("Controller", status_controllers)

    if len(selected_sample_metrics) > 0:
        _display_status("Sample Metric", status_sample_metrics)

    if len(selected_filters) > 0:
        _display_status("Filter", status_filters)

    if len(selected_compound_metrics) > 0:
        _display_status("Compound Metric", status_compound_metrics)

    PU.pop_indent()


def _display_status(
    type_name: str, status_setupables: dict[str, dict[str, bool]]
) -> None:
    PU.push_indent(1)
    PU.print_with_border(f"{type_name} Status")
    for setupable_name, status_modes in status_setupables.items():
        if len(status_modes) <= 0:
            PU.print_with_indent("No setup modes were performed.")
            continue

        PU.print_list(
            setupable_name,
            PU.tablify(
                [
                    list(status_modes.keys()),
                    [
                        "SUCCESS" if success else "FAILURE"
                        for success in status_modes.values()
                    ],
                ]
            ),
            bullet_symbol="  *",
            header_border_symbol="-",
            header_border_side_symbol="|",
        )
    PU.pop_indent()


def _run_with_prerequisite_modes(
    setupable: Setupable, config: dict[str, tuple[bool, dict[str, Any]]], mode: str
) -> None:
    for r in setupable.get_required_modes(mode):
        _run_with_prerequisite_modes(setupable, config, r)

    run, parameters = config[mode]
    if run:
        setupable.setup(mode, parameters, skip_if_completed=True)
    elif not setupable.is_ready(mode):
        raise RuntimeError(
            f"Required mode: '{mode}' is not ready, yet it was not queried for execution."
        )


def _run_setup_modes(
    setupables: list[Setupable],
    setupable_names: list[str],
    setup_config: list[dict[str, tuple[bool, dict[str, Any]]]],
) -> dict[str, dict[str, bool]]:

    # Contains information about what modes were successful
    status = {setupable_name: {} for setupable_name in setupable_names}

    # Run all selected modes
    for setupable, name, config in zip(setupables, setupable_names, setup_config):

        # Run all modes
        for mode, (run, _) in config.items():
            if not run:
                continue
            try:
                _run_with_prerequisite_modes(setupable, config, mode)
                status[name][mode] = True
            except Exception as error:
                status[name][mode] = False
                PU.push_indent(3)
                PU.print_with_border(f"Failed to setup {name} | {mode}", "!")
                PU.print_with_indent(repr(error))
                PU.pop_indent()

    return status


def _select_setup_modes(
    type_name: str, setupables: list[Setupable], setupable_names: list[str]
) -> tuple[list[Setupable], list[str], list[dict[str, tuple[bool, dict[str, Any]]]]]:

    assert len(setupables) == len(setupable_names)

    PU.push_indent(1)

    # Compile table of setupable names and 'ready' status
    setupable_names_ready = PU.tablify(
        [
            setupable_names,
            [
                "SETUP COMPLETE" if setupable.is_ready() else "SETUP NOT COMPLETE"
                for setupable in setupables
            ],
        ]
    )

    # Select what to setup
    setupable_selection = PU.prompt_multi_options(
        f"What {type_name.lower()}(s) would you like to setup?",
        setupable_names_ready,
        default_indices=[i for i, s in enumerate(setupables) if not s.is_ready()],
        allow_empty=True,
        return_index=True,
    )

    # TODO: Warn if completed setupables were selected (same for modes?)

    # Selection context
    setupable_config = [  # [setupable{mode: (run, params), ...}, ...]
        {
            m: [not setupables[i].is_ready(m), setupables[i].get_setup_parameters(m)]
            for m in setupables[i].get_setup_modes()
        }
        for i in setupable_selection
    ]
    setupable_config_modified = [False for _ in setupable_config]

    # Early exit if there was no selection
    if len(setupable_selection) <= 0:
        PU.pop_indent()
        return [], [], setupable_config

    # Select what to configure
    config_index = -1
    while config_index != len(setupable_selection):
        config_index = PU.prompt_options(
            f"Would you like to customize the setup of any {type_name.lower()}(s)?",
            PU.tablify(
                [
                    [setupable_names_ready[i] for i in setupable_selection],
                    [
                        "RUN CUSTOMIZED SETUP"
                        if modified
                        else "RUN ALL UNCOMPLETED MODES WITH DEFAULT PARAMETERS"
                        for modified in setupable_config_modified
                    ],
                ]
            )
            + ["confirm"],
            default_index=len(setupable_selection),
            return_index=True,
        )

        # Configure setupable if applicable
        if config_index < len(setupable_selection):
            setupable = setupables[setupable_selection[config_index]]
            name = setupable_names[setupable_selection[config_index]]
            config = setupable_config[config_index]

            # Print indent
            PU.push_indent(1)
            PU.print_with_border(f"Customization of {type_name} Setup: {name}")

            # Pick setup modes
            valid_selection = False
            while not valid_selection:
                mode_selection = PU.prompt_multi_options(
                    f"What setup modes would you like to run? | {name}",
                    PU.tablify(
                        [
                            setupable.get_setup_modes(),  # Mode name
                            [  # Ready?
                                (
                                    "SETUP COMPLETE"
                                    if setupable.is_ready(m)
                                    else "SETUP NOT COMPLETE"
                                )
                                for m in setupable.get_setup_modes()
                            ],
                            [  # Requirements
                                (
                                    "Info: " + setupable.get_setup_info(m)
                                    if setupable.is_ready(m)
                                    else "UNSATISFIED PREREQUISITES: "
                                    + (
                                        "None"
                                        if all(
                                            setupable.is_ready(r)
                                            for r in setupable.get_required_modes(m)
                                        )
                                        else ", ".join(
                                            [
                                                r
                                                for r in setupable.get_required_modes(m)
                                                if not setupable.is_ready(r)
                                            ]
                                        )
                                    )
                                )
                                for m in setupable.get_setup_modes()
                            ],
                        ]
                    ),
                    default_indices=[
                        i
                        for i, m in enumerate(setupable.get_setup_modes())
                        if config[m][0]
                    ],
                    allow_empty=True,
                    return_index=True,
                )

                # Confirm selection
                valid_selection = True
                for i, m in enumerate(setupable.get_setup_modes()):
                    if i in mode_selection:
                        # Check if selection is valid w.r.t. prerequisites
                        if any(
                            r
                            not in [
                                sm
                                for j, sm in enumerate(setupable.get_setup_modes())
                                if j in mode_selection
                            ]
                            and not setupable.is_ready(r)
                            for r in setupable.get_required_modes(m)
                        ):
                            valid_selection = False
                            PU.print_with_border(
                                f"Prerequisites of '{m}' are not met", "!"
                            )
                            PU.input_continue()

                            continue
                        # Selection valid -> add mode
                        config[m][0] = True
                        setupable_config_modified[
                            config_index
                        ] = True  # Signal modification
                    else:
                        # Remove mode (always allowed)
                        config[m][0] = False
                        setupable_config_modified[
                            config_index
                        ] = True  # Signal modification

            # Get selected setup modes
            setup_mode_selection = list(
                enumerate(m for m in setupable.get_setup_modes() if config[m][0])
            )

            # Check if no modes were selected
            if len(setup_mode_selection) <= 0:
                PU.pop_indent()
                continue

            # Configure parameters for selected modes
            mode_config_index = -1
            while True:

                # Pick mode to configure
                mode_config_index = PU.prompt_options(
                    f"Would you like to configure any parameters? | {name}",
                    PU.tablify(
                        [
                            [m for _, m in setup_mode_selection],
                            [
                                ", ".join(
                                    [f"{p} = {v}" for p, v in config[m][1].items()]
                                )
                                for _, m in setup_mode_selection
                            ],
                        ]
                    )
                    + ["confirm"],
                    default_index=len(setup_mode_selection),
                    return_index=True,
                )

                # Check if done
                if mode_config_index == len(setup_mode_selection):
                    break

                # Print indent
                mode = setup_mode_selection[mode_config_index][1]
                PU.push_indent(1)
                PU.print_with_border(
                    f"Configuration of Parameters for Setup Mode: {name} | {mode}"
                )

                # Pick parameter to configure
                param_index = -1
                while True:
                    # Pick parameter
                    param_index = PU.prompt_options(
                        f"What parameter would you like to configure? | {name} | {mode}",
                        [f"{p} = {v}" for p, v in config[mode][1].items()]
                        + ["confirm"],
                        default_index=len(config[mode][1]),
                        return_index=True,
                    )

                    # Check if done
                    if param_index == len(config[mode][1]):
                        break

                    # Update parameter
                    param_name = list(config[mode][1].keys())[param_index]
                    param_type = type(config[mode][1][param_name])
                    config[mode][1][param_name] = PU.input_type(
                        f"New value for '{param_name}'", param_type
                    )

                PU.pop_indent()  # Parameter config

            PU.pop_indent()  # Setupable config

    PU.pop_indent()  # Done

    return (
        [setupables[i] for i in setupable_selection],
        [setupable_names[i] for i in setupable_selection],
        setupable_config,
    )
