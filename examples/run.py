r"""
Run file
"""

import time
import shutil
import os
import argparse
import importlib.util

from gtf import constants
from gtf import initial_conditions
from gtf import state
from gtf import time_evolution


def load_config(config_path):
    r"""
    Load a configuration module from a Python file.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


parser = argparse.ArgumentParser(description="Run the SIDM gravothermal fluid solver.")

parser.add_argument("config", help="Path to the configuration file.")

args = parser.parse_args()

config = load_config(args.config)


r_first = constants.FLOATDTYPE(config.R_FIRST)

constants.R_OUTER = constants.FLOATDTYPE(config.R_OUTER)
constants.N_SHELL = config.N_SHELL

constants.SIGMA_OVER_M = constants.FLOATDTYPE(config.SIGMA_OVER_M)
constants.BETA = constants.FLOATDTYPE(config.BETA)
constants.ALPHA = constants.FLOATDTYPE(config.ALPHA)

constants.F_TOL = constants.FLOATDTYPE(config.F_TOL)
constants.F_ACCEPT = constants.FLOATDTYPE(config.F_ACCEPT)
constants.X_TOL = constants.FLOATDTYPE(config.X_TOL)
constants.ITER_MAX = config.ITER_MAX

constants.DT_TOL = constants.FLOATDTYPE(config.DT_TOL)

constants.RHO_STOP = constants.FLOATDTYPE(config.RHO_STOP)
constants.OUTPUT_FACTOR = constants.FLOATDTYPE(config.OUTPUT_FACTOR)
constants.RUN_ID = config.RUN_ID

# output directory
out_direc = f"data/run_{constants.RUN_ID:d}/"

if os.path.exists(out_direc):
    shutil.rmtree(out_direc)

os.makedirs(out_direc)

constants.OUT_DIREC = out_direc

# set up NFW halo at t=0
ln_r_edge, u = initial_conditions.set_up_initial_conditions(r_first)

# create initial state of the system with all relevant state variables
state_init = state.state_from_unknowns(ln_r_edge, u)

# evolve the halo until central density is greater than constants.RHO_STOP

start_time = time.perf_counter()

time_evolution.evolve(state_init)

end_time = time.perf_counter()

execution_time = end_time - start_time

print(execution_time)
