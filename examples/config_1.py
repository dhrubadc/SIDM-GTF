r"""
Config file for a run.
Pass as argument to run.py
"""

# first non-zero radius at t=0
R_FIRST = 0.001

# outermost radius (innermost is set to zer)
R_OUTER = 200.0

# number of shells
N_SHELL = 200

# SIDM cross section
SIGMA_OVER_M = 0.3

# LMFP conductivity constant
BETA = 0.8

# transition parameter
ALPHA = 1.0

# convergence of Newton iterations
F_TOL = 1.0e-12

# stagnated but acceptable
F_ACCEPT = 1.0e-9

# determines stagnation
X_TOL = 1.0e-12

# maximum allowed iterations
ITER_MAX = 50

# timestep controller
DT_TOL = 0.01

# total time limiter
RHO_STOP = 1.0e6

# controls output frequency
OUTPUT_FACTOR = 1.02

# run id matched config file suffix
RUN_ID = 1
