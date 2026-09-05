"""
Physical constants.
"""

import numpy as np

FLOAT_DTYPE = np.float64


# ============================================================
# Conductivity constants
# ============================================================

a = FLOAT_DTYPE(4.0) / np.sqrt(FLOAT_DTYPE(np.pi))

b = FLOAT_DTYPE(25.0) * np.sqrt(FLOAT_DTYPE(np.pi)) / FLOAT_DTYPE(32.0)
