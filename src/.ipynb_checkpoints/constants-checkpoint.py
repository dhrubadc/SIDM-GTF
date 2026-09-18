"""
Module to define the float precision and
quantities that remain constant throughout a run
in dimensionless units (Nishikawa 2020).
"""

import numpy as np

# =============================================================
# float precision
# =============================================================

FLOATDTYPE = np.float64

# ============================================================
# Short Mean Free Path Conductivity constants (Nishikawa 2020)
# ============================================================

A = FLOATDTYPE(4.0) / np.sqrt(FLOATDTYPE(np.pi))
B = FLOATDTYPE(25.0) * np.sqrt(FLOATDTYPE(np.pi)) / FLOATDTYPE(32.0)

# ===============================================================
# cross section for scattering.
# ===============================================================

SIGMA_OVER_M = None

# ===============================================================
# Long mean free path conductivity constant
# ===============================================================
BETA = None

# ===============================================================
# Conductivity transition parameter
# ===============================================================
ALPHA = None

# ===============================================================
# lagrangian shell and enclosed mass
# ===============================================================
DM = None
M_EDGE = None

# ===============================================================
# grid parameters (r_inner is always set to zero)
# ===============================================================
R_OUTER = None
N_SHELL = None

# ==============================================================
# Newton iteration convergence related parameters
# ===============================================================
F_TOL = None
ITER_MAX = None

# ===============================================================
# timestep controller
# ===============================================================
DT_TOL = None
