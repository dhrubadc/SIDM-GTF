r"""
Define float precision and runtime constants.
"""

import numpy as np

FLOATDTYPE = np.float64
"""Float data type precision, defaults to np.float64.
"""

A = FLOATDTYPE(4.0) / np.sqrt(FLOATDTYPE(np.pi))
r"""
Relevant for SMFP conductivity calculation, determined from kinetic theory. 
"""

B = FLOATDTYPE(25.0) * np.sqrt(FLOATDTYPE(np.pi)) / FLOATDTYPE(32.0)
r"""
Relevant for SMFP conductivity calculation, determined from kinetic theory. 
"""

SIGMA_OVER_M = None
r"""
SIDM cross section per unit mass. 
Set by user at runtime.
"""

BETA = None
r"""
Coefficient for the LMFP conductivity.
Set by user at runtime.
"""

ALPHA = None
r"""
Determines transition between the LMFP and SMFP regimes.
Set by user at runtime.
"""

DM = None
r"""
Array of Lagrangian masses cells of shape :obj:`N_SHELL`.
Set by user at runtime.
"""

M_EDGE = None
r"""
Enclosed mass array of shape :obj:`N_SHELL` + 1.
Set by user at runtime.
"""

R_OUTER = None
r"""
Outermost grid edge (Innermost edge is zero).
Set by user at runtime.
"""

N_SHELL = None
r"""
Number of Lagrangian cells.
Set by user at runtime.
"""

F_TOL = None
r"""
Residual tolerance for convergence of Newton iterations.
Set by user at runtime.
"""

ITER_MAX = None
r"""
Maximum number of allowed Newton iterations.
Set by user at runtime.
"""

X_TOL = None
r"""
Tolerance for determining stagnation of Newton iterations.
Set by user at runtime.
"""

F_ACCEPT = None
r"""
Residual tolerance for accepting stagnated Newton iterations.
Set by user at runtime.
"""

DT_TOL = None
r"""
Controls the time-stepping of the solver.
Set by user at runtime.
"""

RHO_STOP = None
r"""
Central density for stoping the time evolution. 
Must be greater than the central density at t=0
for time evolution to proceed.
"""

OUT_DIREC = None
r"""
Output directory for storing the data files.
"""

OUTPUT_FACTOR = None
r"""
Controls frequency of outputs
"""
