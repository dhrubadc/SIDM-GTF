"""
Module to define constants and float precision.
"""

import numpy as np

FLOATDTYPE = np.float64

# ============================================================
# Short Mean Free Path Conductivity constants (Nishikawa 2020)
# ============================================================

a = FLOATDTYPE(4.0) / np.sqrt(FLOATDTYPE(np.pi))

b = FLOATDTYPE(25.0) * np.sqrt(FLOATDTYPE(np.pi)) / FLOATDTYPE(32.0)
