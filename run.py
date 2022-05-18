from argparse import ArgumentParser
from typing import Callable
import time
import sys

from src.init_phase import init_phase, init_done
from src.setup_phase import setup_phase
from src.generation_phase import generation_phase
from src.evaluation_phase import evaluation_phase

import src.util.PromptUtil as PU

#####################
# Utility functions #
#####################


def _require_phase(
    target_phase_name: str, required_phase_name: str, check_func: Callable[[], bool]
) -> None:
    """Checks whether a required phase has been completed for a specified target phase."""
    if not check_func():
        print(
            f"Could not run phase '{target_phase_name}' since the previous phase "
            + f"'{required_phase_name}' has not yet been completed. Exiting."
        )
        exit(1)


def _prompt_satisfied(phase_name: str, check_func: Callable[[], bool]) -> None:
    """If the specified phase is already satisfied, the user will be asked if they want to re-run."""
    if check_func():
        if PU.prompt_yes_no(
            f"Phase '{phase_name}' is already satisfied. Continue anyway?"
        ):
            print(f"Re-running phase '{phase_name}'.")
        else:
            print(f"Skipping phase '{phase_name}'. Exiting.")
            exit(0)


#################################################
# Phase wrappers for specific argument handling #
#################################################


def _init_wrapper(name: str, argv: list = None) -> None:
    # Ask user if phase should be re-run (if applicable)
    _prompt_satisfied(name, init_done)

    # Execute init phase
    init_phase()


def _setup_wrapper(name: str, argv: list = None) -> None:
    # Require init phase
    _require_phase(name, "init", init_done)

    # Execute setup phase
    setup_phase()


def _generation_wrapper(name: str, argv: list = None) -> None:
    # Require init phase
    _require_phase(name, "init", init_done)

    #! Setup phase check depends on selection (i.e., it cannot be checked here)

    # Execute generation phase
    generation_phase()


def _evaluation_wrapper(name: str, argv: list = None) -> None:
    # Require init phase
    _require_phase(name, "init", init_done)

    # Execute evaluation phase
    evaluation_phase()


##########################
# Core argument handling #
##########################

# Phase names
phases = {
    "init": _init_wrapper,
    "setup": _setup_wrapper,
    "generate": _generation_wrapper,
    "evaluate": _evaluation_wrapper,
}

# Parser core arguments
parser = ArgumentParser()
parser.add_argument("phase", choices=phases.keys(), help="the phase to run")
core_args = parser.parse_args(sys.argv[1:2])

# Evoke appropriate wrapper
t = time.time()
phases[core_args.phase](core_args.phase, sys.argv[2:])
print(f"Total elapsed time: {time.time() - t:.1f} seconds.")
