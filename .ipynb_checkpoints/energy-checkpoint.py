"""Conductive luminosity using linear cell variables and gradients."""

import numpy as np

from constants import FLOAT_DTYPE
import gravity
from grid import cell_centers


def luminosity(r_edge, u, kappa):
    """
    Return luminosity at shell edges.

    For interior edge j,

        L[j] = -(2/3) r_edge[j]**2 kappa_face du/dr,

    with arithmetic cell-center radii and arithmetic kappa_face.  The inner
    and outer luminosities are fixed to zero.
    """
    r_edge = np.asarray(r_edge, dtype=FLOAT_DTYPE)
    u = np.asarray(u, dtype=FLOAT_DTYPE)
    kappa = np.asarray(kappa, dtype=FLOAT_DTYPE)
    N = len(u)
    if len(r_edge) != N + 1 or len(kappa) != N:
        raise ValueError("Inconsistent r_edge, u, and kappa lengths.")

    r = cell_centers(r_edge)
    L = np.zeros(N + 1, dtype=FLOAT_DTYPE)

    kappa_face = FLOAT_DTYPE(0.5) * (kappa[:-1] + kappa[1:])
    du = u[1:] - u[:-1]
    dr = r[1:] - r[:-1]

    L[1:N] = (
        -(FLOAT_DTYPE(2.0) / FLOAT_DTYPE(3.0))
        * r_edge[1:N]**2
        * kappa_face
        * du
        / dr
    )

    return L


def luminosity_divergence(L, dm):
    """Return dL/dm = (L[i+1]-L[i])/dm[i]."""
    L = np.asarray(L, dtype=FLOAT_DTYPE)
    dm = np.asarray(dm, dtype=FLOAT_DTYPE)
    return (L[1:] - L[:-1]) / dm


def total_internal_energy(u, dm):
    """Return U=sum(dm*u) in float64."""
    u = np.asarray(u, dtype=FLOAT_DTYPE)
    dm = np.asarray(dm, dtype=FLOAT_DTYPE)
    if u.shape != dm.shape:
        raise ValueError("u and dm must have the same shape.")
    return FLOAT_DTYPE(np.sum(dm * u, dtype=FLOAT_DTYPE))


def total_energy(r_edge, u, dm, M_edge, G=1.0):
    """
    Return the conserved total energy E=U+W in float64.

    W is the exact piecewise-uniform-shell gravitational energy whose
    gradient is used by the variational hydrostatic residual.
    """
    r_edge = np.asarray(r_edge, dtype=FLOAT_DTYPE)
    u = np.asarray(u, dtype=FLOAT_DTYPE)
    dm = np.asarray(dm, dtype=FLOAT_DTYPE)
    M_edge = np.asarray(M_edge, dtype=FLOAT_DTYPE)
    G = FLOAT_DTYPE(G)

    U = total_internal_energy(u, dm)
    W = gravity.gravitational_energy(
        r_edge,
        dm,
        M_edge,
        G=G,
    )
    return FLOAT_DTYPE(U + W)
