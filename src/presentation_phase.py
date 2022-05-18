from src.controller.ControllerRegistry import ControllerRegistry
from src.generator.GeneratorRegistry import GeneratorRegistry

from src.metric.SampleMetricRegistry import SampleMetricRegistry

from src.metric.CompoundMetricRegistry import CompoundMetricRegistry
from src.metric.CompoundMetricManager import CompoundMetricManager

from src.filter.FilterRegistry import FilterRegistry

from src.compound_model.CompoundModelFactory import (
    CompoundModelFactory,
    CompoundModelFactoryContext,
)

import src.util.PromptUtil as PU

from src.phase_utils import (
    select_dataset,
    select_attributes,
)

from typing import Any

from pathlib import Path

import pandas as pd
import numpy as np

LATEX_ROOT = Path(".latex")


def presentation_phase():
    """
    Present evaluation results.
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
    cmfs = [
        CompoundModelFactory(
            c,
            FilterRegistry.get_resources(),
            SampleMetricRegistry.get_resources(),
            CompoundMetricRegistry.get_resources(),
        )
        for c in controllers
    ]
    cmfs = list(filter(lambda cmf: cmf.get_population().num_samples() > 0, cmfs))

    CompoundModelFactoryContext(cmfs)

    # * Select presentation mode
    if (
        PU.prompt_options(
            "How would you like to present the results?",
            ["Show per compound model.", "Show per metric."],
            return_index=True,
        )
        == 0
    ):
        _show_per_model(cmfs)
    else:
        _show_per_metric(cmfs)


def _valid_metric(val: Any) -> bool:
    return not (
        pd.isnull(val) if type(val) == int or type(val) == float else pd.isnull(val)
    )


def _show_per_metric(cmfs: list[CompoundModelFactory]):

    # * Select metrics
    metric_names = PU.prompt_multi_options(
        "What metrics would you like to see?",
        [mn for mn in cmfs[0].get_compound_metric_names()],
    )

    # * Construct compound models [(name, cmf, f, cmm), ...]
    compound_models = [
        (
            "("
            + ", ".join(
                (
                    cmf.get_controller().get_name(),
                    cmf.get_controller().get_generator().get_name(),
                    f.get_name(),
                )
            )
            + ")",
            cmf,
            f,
            cmf.get_compound_metric_managers()[i],
        )
        for cmf in cmfs
        for i, f in enumerate(cmf.get_filters())
        if any(
            _valid_metric(cmf.get_compound_metric_managers()[i].get(n))
            for n in metric_names
        )
    ]

    # * Present results
    for metric_name in metric_names:
        PU.print_list(
            f"Metric: {metric_name}",
            PU.tablify(
                [
                    [name for name, _, _, _ in compound_models],
                    [
                        str(_get_metric_rep(cmm, metric_name))
                        for _, _, _, cmm in compound_models
                    ],
                ]
            ),
            header_border_symbol="-",
            header_border_side_symbol="|",
        )

    # * Export latex
    for metric_name in metric_names:
        _save_latex(
            metric_name,
            metric_name,
            pd.DataFrame(
                [
                    [cmm.get(metric_name) for cmm in cmf.get_compound_metric_managers()]
                    for cmf in cmfs
                ],
                columns=[f.get_name() for f in cmfs[0].get_filters()],
                index=[
                    ", ".join(
                        (
                            cmf.get_controller().get_name(),
                            cmf.get_controller().get_generator().get_name(),
                        )
                    )
                    for cmf in cmfs
                ],
            ),
        )

    # * Plot results
    if not PU.prompt_yes_no("Plot results?"):
        return

    for metric_name in metric_names:
        for _, _, _, cmm in compound_models:
            cm = cmm.get_metric_instances()[cmm.get_metric_names().index(metric_name)]
            if not cm.has_plot() or not _valid_metric(cmm.get(metric_name)):
                continue
            cm.plot_result()


def _show_per_model(cmfs: list[CompoundModelFactory]):

    # * Construct compound models [(name, cmf, f, cmm), ...]
    compound_models = [
        (
            "("
            + ", ".join(
                (
                    cmf.get_controller().get_name(),
                    cmf.get_controller().get_generator().get_name(),
                    f.get_name(),
                )
            )
            + ")",
            cmf,
            f,
            cmf.get_compound_metric_managers()[i],
        )
        for cmf in cmfs
        for i, f in enumerate(cmf.get_filters())
    ]

    # * Select compound models
    compound_model_selection = PU.prompt_multi_options(
        "What compound models would you like to see?",
        PU.tablify(
            [
                [name for name, _, _, _ in compound_models],
                [
                    "Available Metrics: "
                    + (lambda x: x if len(x) > 0 else "None")(
                        ", ".join(
                            list(
                                filter(
                                    lambda mn: _valid_metric(cmm.get(mn)),
                                    cmf.get_compound_metric_names(),
                                )
                            )
                        )
                    )
                    for _, cmf, _, cmm in compound_models
                ],
            ],
        ),
        default_indices=[
            i
            for i, (_, cmf, _, cmm) in enumerate(compound_models)
            if any(_valid_metric(cmm.get(n)) for n in cmf.get_compound_metric_names())
        ],
        return_index=True,
    )
    compound_models = [
        cm for i, cm in enumerate(compound_models) if i in compound_model_selection
    ]

    # * Present results
    for compound_model_name, cmf, f, cmm in compound_models:
        if len(cmf.get_population().get_filtering_indices(f.get_bit())) <= 0:
            continue
        PU.print_list(
            f"Compound Model: {compound_model_name}",
            [
                f"{name} = {_get_metric_rep(cmm, name)}"
                for name in cmf.get_compound_metric_names()
            ],
            header_border_symbol="-",
            header_border_side_symbol="|",
        )

    # * Export latex
    for cmf in cmfs:
        _save_latex(
            cmf.get_name(),
            ", ".join(
                (
                    cmf.get_controller().get_name(),
                    cmf.get_controller().get_generator().get_name(),
                )
            ),
            pd.DataFrame(
                [
                    cmm.get(cmf.get_compound_metric_names())
                    for cmm in cmf.get_compound_metric_managers()
                ],
                index=[f.get_name() for f in cmf.get_filters()],
            ).transpose(),
        )

    # * Plot results
    if not PU.prompt_yes_no("Plot results?"):
        return

    for compound_model_name, cmf, f, cmm in compound_models:
        if len(cmf.get_population().get_filtering_indices(f.get_bit())) <= 0:
            continue

        for cm in cmm.get_metric_instances():
            if not cm.has_plot() or not _valid_metric(cmm.get(cm.get_name())):
                continue
            cm.plot_result()


def _rep_val(val: Any) -> str:
    if type(val) not in (float, int, str, bool) and not isinstance(val, np.number):
        return "-"

    if val != val:
        return "-"

    if type(val) == float or isinstance(val, np.floating):
        return f"{val:.4f}"

    return str(val)


def _get_metric_rep(cmm: CompoundMetricManager, metric_name: str) -> str:
    val = cmm.get(metric_name)
    return _rep_val(val)


def _save_latex(file_name: str, label: str, df: pd.DataFrame) -> None:
    if not LATEX_ROOT.exists():
        LATEX_ROOT.mkdir()

    df = df.applymap(_rep_val)

    with open(LATEX_ROOT / f"{file_name}.tex", "w") as f:
        f.writelines(
            df.style.to_latex(column_format="|l|" + "r|" * df.shape[1])
            .replace(
                " &", "\multicolumn{1}{|c|}{\\textbf{" + label + "}} &", 1
            )  # Insert label
            .replace("\n", "\n\hline\n", 2)  # Top line
            .replace("\\end", "\\hline\n\\end")  # Bottom line
        )
