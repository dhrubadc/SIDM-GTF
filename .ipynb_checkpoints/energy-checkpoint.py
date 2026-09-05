"""Conductive luminosity using linear cell variables and gradients."""

import numpy as np

from constants import FLOAT_DTYPE
from grid import cell_centers


def luminosity(r_edge, u, kappa):
    """
    Return luminosity at shell edges.

    For interior edge j,

        L[j] = -(2/3) r_edge[j]**2 kappa_edge du/dr,

    with kappa_edge being the arithmetic mean of kappa 
    at adjacent cell centers.
    
    The luminosities at r=0 and r=r_max at fixed to 0. 
    """
    
    
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
    """Return dL/dm = (L[i+1]-L[i])/dm[i].
    """
    return (L[1:] - L[:-1]) / dm

def total_kinetic_energy(u, dm):
    """Return U=sum(dm*u)
    """
    return np.sum(dm * u)

def total_gravitational_energy(r, dm, M_edge):
    """Return sum(dm M(r)/r)
    where M(r) is the arithmatic mean of enclosed mass
    at the two adjacent edges
    """
    M_center = 0.5 * (M_edge[:-1] + M_edge[1:])
    return np.sum(dm * M_center / r)
    
def total_energy(r_edge, u, dm, M_edge, G=1.0):
    """
    Return total energy E=U+W.
    """
    U = total_kinetic_energy(u, dm)
    W = total_gravitational_energy(r, dm, M_edge)
    return U + W