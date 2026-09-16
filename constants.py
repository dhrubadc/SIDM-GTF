"""Physical constants."""

import numpy as np

FLOATDTYPE = np.float64

# ============================================================
# Conductivity constants
# ============================================================

a = FLOATDTYPE(4.0) / np.sqrt(FLOATDTYPE(np.pi))

b = FLOATDTYPE(25.0) * np.sqrt(FLOATDTYPE(np.pi)) / FLOATDTYPE(32.0)
