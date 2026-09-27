r"""
Run file
"""

import time
import shutil
import os
import argparse
import importlib.util

from gtf import settings
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


r_first = settings.FLOATDTYPE(config.R_FIRST)

settings.R_OUTER = settings.FLOATDTYPE(config.R_OUTER)
settings.N_SHELL = config.N_SHELL

settings.SIGMA_OVER_M = settings.FLOATDTYPE(config.SIGMA_OVER_M)
settings.BETA = settings.FLOATDTYPE(config.BETA)
settings.ALPHA = settings.FLOATDTYPE(config.ALPHA)

settings.F_TOL = settings.FLOATDTYPE(config.F_TOL)
settings.F_ACCEPT = settings.FLOATDTYPE(config.F_ACCEPT)
settings.X_TOL = settings.FLOATDTYPE(config.X_TOL)
settings.ITER_MAX = config.ITER_MAX

settings.DT_TOL = settings.FLOATDTYPE(config.DT_TOL)

settings.RHO_STOP = settings.FLOATDTYPE(config.RHO_STOP)
settings.OUTPUT_FACTOR = settings.FLOATDTYPE(config.OUTPUT_FACTOR)
settings.RUN_ID = config.RUN_ID

# output directory
out_direc = f"data/run_{settings.RUN_ID:d}/"

if os.path.exists(out_direc):
    shutil.rmtree(out_direc)

os.makedirs(out_direc)

settings.OUT_DIREC = out_direc

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
