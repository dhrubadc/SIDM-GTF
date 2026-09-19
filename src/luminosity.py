"""
Module to calculate conductive luminosity
in dimensionless units (Nishikawa 2020).
"""

import numpy as np
from . import constants


def luminosity(r_edge, u, r, kappa):
    """
    Return luminosity at shell edges.
    The luminosities at r=0 and r=r_outer are fixed to 0.
    """
    n_shell = len(u)
    l = np.zeros(n_shell + 1, dtype=constants.FLOATDTYPE)

    kappa_face = constants.FLOATDTYPE(0.5) * (kappa[:-1] + kappa[1:])
    du = u[1:] - u[:-1]
    dr = r[1:] - r[:-1]

    l[1:-1] = (
        -constants.FLOATDTYPE(2.0 / 3.0) * r_edge[1:-1] ** 2 * kappa_face * du / dr
    )

    return l
