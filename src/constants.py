r"""
Define runtime constants and solver parameters
in dimensionless units.
"""

import numpy as np


FLOATDTYPE = np.float64
"""Float data type precision (default: float64).
"""

A = FLOATDTYPE(4.0) / np.sqrt(FLOATDTYPE(np.pi))
r"""Relevant for SMFP conductivity calculation, determined from kinetic theory. 
"""

B = FLOATDTYPE(25.0) * np.sqrt(FLOATDTYPE(np.pi)) / FLOATDTYPE(32.0)
r"""Relevant for SMFP conductivity calculation, determined from kinetic theory. 
"""

SIGMA_OVER_M = None
r"""SIDM cross section per unit mass. 
Set by user at runtime.
"""

BETA = None
r"""Coefficient for the LMFP conductivity.
Set by user at runtime.
"""

ALPHA = None
r"""Determines transition between the LMPF and SMFP regimes.
Set by user at runtime.
"""

DM = None
r"""Array of Lagrangian shell masses of shape N_SHELL.
Set by user at runtime.
"""

M_EDGE = None
r"""Enclosed mass array of shape N_SHELL + 1.
Set by user at runtime.
"""

R_OUTER = None
r"""Outermost grid edge (Innermost edge is zero).
Set by user at runtime.
"""

N_SHELL = None
r"""Number of Lagrangian shells.
Set by user at runtime.
"""

F_TOL = None
r"""Tolerance for Newton iteration convergence.
Set by user at runtime.
"""

ITER_MAX = None
r"""
Maximum number of allowed Newton iterations.
Set by user at runtime.
"""

X_TOL = None
r"""Tolerance for determing stagnation of Newton iterations.
Set by user at runtime.
"""

F_ACCEPT = None
r"""Tolerance for accepting stagnated Newton iterations.
Set by user at runtime.
"""

DT_TOL = None
r"""Controls the time-stepping of the solver.
Set by user at runtime.
"""
