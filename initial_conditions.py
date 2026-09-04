"""Halo initial conditions in linear variables."""

import numpy as np
from scipy.optimize import brentq
from constants import FLOAT_DTYPE
from grid import shell_volume


def truncated_nfw_dimensionless(r, r_t, n):
    """Return rho (in units of rho_s) 
    as a function of r (in units of r_s) 
    for a truncated NFW profile.
    """
    return np.exp(-(r / r_t)**n) / (r * (FLOAT_DTYPE(1.0) + r)**2)

def nfw_dimensionless(r):
    """Return rho (in units of rho_s)
    as a function of r (in units of r_s)
    for an NFW profile.
    """
    return FLOAT_DTYPE(1.0) / (r * (FLOAT_DTYPE(1.0) + r)**2)

def shell_mass(rho, volume):
    """Return the Lagrangian mass of each shell
    """
    return rho * volume

def enclosed_mass(dm):
    """Return enclosed mass at all shell edges
    """
    M_edge = np.zeros(len(dm) + 1)
    M_edge[1:] = np.cumsum(dm)
    return M_edge

def hydrostatic_u_from_rho(r_edge, rho, M_edge, u_outer):
    """
    Integrate inwards the same logarithmic hydrostatic residual
    as in residual.py to get the initial u profile. 
    The outermost u is specified.  
    """
    N = len(rho)
    if len(r_edge) != N + 1 or len(M_edge) != N + 1:
        raise ValueError("Inconsistent initial-condition array lengths.")
    if u_outer <= FLOAT_DTYPE(0.0):
        raise ValueError("u_outer must be positive.")

    r = cell_centers(r_edge)
    ln_r = np.log(r)
    ln_rho = np.log(rho)
    u = np.empty(N)
    u[-1] = u_outer

    for i in range(N - 2, -1, -1):
        delta_ln_r = ln_r[i + 1] - ln_r[i]
        delta_ln_rho = ln_rho[i + 1] - ln_rho[i]
        
        gravity_coefficient = (M_edge[i + 1] * FLOAT_DTYPE(3.0/2.0)
                               / r_edge[i + 1]
                              )
        
        
        u_right = u[i + 1]
        ln_u_right = np.log(u_right)

        def shell_balance(ln_u_left):
            u_left = np.exp(ln_u_left)
            u_edge = FLOAT_DTYPE(0.5) * (u_left + u_right)
            return (
                (
                    delta_ln_rho
                    + ln_u_right
                    - ln_u_left
                )
                / delta_ln_r
                + gravity_coefficient / u_edge
            )

        lower = ln_u_right - FLOAT_DTYPE(10.0)
        upper = ln_u_right + FLOAT_DTYPE(10.0)
        
        while shell_balance(lower) < FLOAT_DTYPE(0.0):
            lower -= FLOAT_DTYPE(10.0)
        
        while shell_balance(upper) > FLOAT_DTYPE(0.0):
            upper += FLOAT_DTYPE(10.0)

        ln_u_left = brentq(
            shell_balance,
            lower,
            upper,
            xtol=FLOAT_DTYPE(1.0e-13),
            rtol=FLOAT_DTYPE(1.0e-13),
        )
        
        u[i] = np.exp(ln_u_left)

    if np.any(~np.isfinite(u)) or np.any(u <= FLOAT_DTYPE(0.0)):
        raise RuntimeError("Hydrostatic initialization produced invalid u.")

    return u
