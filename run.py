from argparse import ArgumentParser
import time
import sys

from src.init_phase import init_phase

#################################################
# Phase wrappers for specific argument handling #
#################################################


def _init_wrapper(name, argv=None):
    # Execute init phase
    init_phase()


##########################
# Core argument handling #
##########################

# Process names
phases = {
    "init": _init_wrapper,
}

# Parser core arguments
parser = ArgumentParser()
parser.add_argument("phase", choices=phases.keys(), help="the phase to run")
core_args = parser.parse_args(sys.argv[1:2])

# Evoke appropriate wrapper
t = time.time()
phases[core_args.phase](core_args.phase, sys.argv[2:])
print(f"Total elapsed time: {time.time() - t:.1f} seconds.")
