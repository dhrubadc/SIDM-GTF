"""
Conductive luminosity using linear cell variables and gradients.
"""

import numpy as np
import constants
import grid


def luminosity(r_edge, u, kappa):
    """
    Return luminosity at shell edges.

    For interior edge j,

        L[j] = -(2/3) r_edge[j]**2 kappa_face du/dr,

    with kappa_face being the arithmetic mean of kappa
    at adjacent cell centers.

    The luminosities at r=0 and r=r_max at fixed to 0.
    """
    N = len(u)
    if len(r_edge) != N + 1 or len(kappa) != N:
        raise ValueError("Inconsistent r_edge, u, and kappa lengths.")

    r = grid.cell_centers(r_edge)
    L = np.zeros(N + 1, dtype=constants.FLOAT_DTYPE)

    kappa_face = constants.FLOAT_DTYPE(0.5) * (kappa[:-1] + kappa[1:])
    du = u[1:] - u[:-1]
    dr = r[1:] - r[:-1]

    L[1:-1] = (
        -constants.FLOAT_DTYPE(2.0 / 3.0) * r_edge[1:-1] ** 2 * kappa_face * du / dr
    )

    return L


def luminosity_divergence(L, dm):
    """
    Return dL/dm = (L[i+1]-L[i])/dm[i].
    """
    return (L[1:] - L[:-1]) / dm


def total_kinetic_energy(u, dm):
    """
    Return U=sum(dm*u)
    """
    return np.sum(dm * u)


def total_gravitational_energy(r_edge, dm, M_edge):
    """
    Return sum(dm M(r)/r)
    where M(r) is the arithmatic mean of enclosed mass
    at the two adjacent edges
    """
    r = grid.cell_centers(r_edge)
    M_center = constants.FLOAT_DTYPE(0.5) * (M_edge[:-1] + M_edge[1:])
    return -np.sum(dm * M_center / r)


def total_energy(r_edge, u, dm, M_edge):
    """
    Return total energy E=U+W.
    """
    U = total_kinetic_energy(u, dm)
    W = total_gravitational_energy(r_edge, dm, M_edge)
    return U + W
