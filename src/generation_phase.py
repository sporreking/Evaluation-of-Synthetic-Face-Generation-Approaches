from src.compound_model.CompoundModelFactory import CompoundModelFactory
from src.controller.ControllerRegistry import ControllerRegistry

import src.util.PromptUtil as PU

from src.phase_utils import (
    confirm_lists,
    select_dataset,
    select_attributes,
    select_controllers,
    select_generators,
)

DEFAULT_POPULATION_SIZE = 70000


def generation_phase() -> None:
    """
    Generate populations.
    """

    # * Initial selection
    dataset = select_dataset()
    attributes = select_attributes(dataset)
    controller_names, controller_types = select_controllers()
    generator_names, generator_types = select_generators()

    # * Final presentation before config
    if not confirm_lists(
        "Initial Selection",
        ("Dataset", [dataset.get_name(dataset.get_resolution())]),
        ("Attributes", attributes),
        ("Controllers", controller_names),
        ("Generators", generator_names),
    ):
        print("Aborting.")
        return

    # * Create controllers
    generators = [g(dataset) for g in generator_types]
    controllers = [
        c(g, attributes)
        for i, c in enumerate(controller_types)
        for g in generators
        if g.get_name()
        in ControllerRegistry.get_compatible_generator_names(controller_names[i])
    ]

    # * Check controller setup status
    invalid_controllers = list(filter(lambda c: not c.is_ready(), controllers))
    if len(invalid_controllers) > 0:
        PU.push_indent(3)
        PU.print_with_border("Some controllers were not ready.", symbol="*")
        PU.print_with_indent(
            "Make sure that they have been set up prior to generation."
        )
        PU.print_list(
            items=PU.tablify(
                [
                    [
                        c.get_name() + "_" + c.get_generator().get_name()
                        for c in invalid_controllers
                    ],
                    "SETUP NOT COMPLETE",
                ]
            )
        )
        if not PU.prompt_yes_no("Skip these and continue?"):
            PU.pop_indent()
            print("Aborting.")
            return

        # Remove invalid controllers
        controllers = [c for c in controllers if c not in invalid_controllers]
        PU.pop_indent()

    # * Create compound model factories
    cmfs = [CompoundModelFactory(c, [], [], []) for c in controllers]

    # * Selection
    cmfs: list[CompoundModelFactory] = [
        cmfs[i]
        for i in PU.prompt_multi_options(
            "What controller-generator-pairs would you like to generate for?",
            [cmf.get_name() for cmf in cmfs],
            allow_empty=True,
            return_index=True,
        )
    ]

    if len(cmfs) <= 0:
        print("No items selected. Aborting.")
        return

    # * Configuration
    cmf_names = [cmf.get_name() for cmf in cmfs]
    config = {name: [DEFAULT_POPULATION_SIZE, False] for name in cmf_names}
    config_index = -1
    while config_index < len(cmfs) + 1:
        config_index = PU.prompt_options(
            "Would you like to configure the generation process of any population?",
            PU.tablify(
                [
                    [name for name in cmf_names],
                    [
                        f"Current Size: {cmf.get_population().num_samples()}"
                        for cmf in cmfs
                    ],
                    [f"Target Size: {config[name][0]}" for name in cmf_names],
                    [
                        (
                            "DROP OLD SAMPLES AND REGENERATE TO TARGET SIZE"
                            if config[name][1]
                            else (
                                "CONTINUE TO TARGET SIZE"
                                if cmfs[i].get_population().num_samples()
                                < config[name][0]
                                else "DO NOTHING (CURRENT SIZE >= TARGET SIZE)"
                            )
                        )
                        for i, name in enumerate(cmf_names)
                    ],
                ]
            )
            + ["set for all", "confirm"],
            default_index=len(cmfs) + 1,
            return_index=True,
        )

        # Set values
        if config_index < len(cmfs):  # Set for single
            config[cmf_names[config_index]] = [
                PU.input_int("New number of samples/images", 1),
                PU.prompt_yes_no("Replace old population and regenerate?"),
            ]
        elif config_index == len(cmfs):  # Set for all
            cfg = [
                PU.input_int("New number of samples/images", 1),
                PU.prompt_yes_no("Replace old population and regenerate?"),
            ]
            for name in cmf_names:
                config[name] = cfg.copy()

    # * Final confirmation
    if not PU.prompt_yes_no(f"Start generation of {len(cmfs)} population(s)?"):
        print("Aborting.")
        return

    # * Generate populations
    status = {name: False for name in cmf_names}
    PU.push_indent(1)
    for cmf_name, cmf in zip(cmf_names, cmfs):
        try:
            n, replace = config[cmf_name]
            PU.print_with_border(
                f"Creating population of {n} images for {cmf_name}", "-", "|"
            )
            cmf.generate_population(n, replace=replace)
            status[cmf_name] = True
        except Exception as error:
            status[cmf_name] = False
            PU.push_indent(3)
            PU.print_with_border(f"Failed to generate population '{cmf_name}'", "!")
            PU.print_with_indent(repr(error))
            PU.pop_indent()
    PU.pop_indent()

    # * Display status
    PU.push_indent(1)
    PU.print_list(
        "Population Status",
        PU.tablify(
            [cmf_names, ["SUCCESS" if s else "FAILURE" for s in status.values()]]
        ),
        bullet_symbol="  *",
        header_border_symbol="=",
        header_border_side_symbol="||",
    )
    PU.pop_indent()
