"""
Module to set up initial conditions for an SIDM halo.
Current support for standard and truncated NFW profiles
in dimensionless units (Nishikawa 2020).
"""

import numpy as np
from scipy.optimize import brentq
import constants
import grid


def truncated_nfw_dimensionless(r, r_t, n):
    """
    Return rho (in units of rho_s)
    as a function of r (in units of r_s)
    for a truncated NFW profile.
    """
    return np.exp(-((r / r_t) ** n)) / (r * (constants.FLOATDTYPE(1.0) + r) ** 2)


def shell_mass(rho, volume):
    """
    Return the Lagrangian mass of each shell.
    """
    return rho * volume


def enclosed_mass(dm):
    """
    Return enclosed mass at all shell edges.
    """
    m_edge = np.zeros(len(dm) + 1, dtype=constants.FLOATDTYPE)
    m_edge[1:] = np.cumsum(dm)

    return m_edge


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
def hydrostatic_u_from_rho(r_edge, ln_r, ln_rho, m_edge, u_outer):
    """
    Integrate inwards the same hydrostatic residual
    as in residual.py to get the initial u profile.
    The outermost u is specified.
    """
    n_shell = len(ln_r)
    u = np.empty(n_shell, dtype=constants.FLOATDTYPE)
    u[-1] = u_outer

    for i in range(n_shell - 2, -1, -1):
        dln_r = ln_r[i + 1] - ln_r[i]
        dln_rho = ln_rho[i + 1] - ln_rho[i]

        g_coefficient = m_edge[i + 1] * constants.FLOATDTYPE(3.0 / 2.0) / r_edge[i + 1]

        u_right = u[i + 1]
        ln_u_right = np.log(u_right)

        def shell_balance(
            ln_u_left,
            dln_r=dln_r,
            dln_rho=dln_rho,
            g_coefficient=g_coefficient,
            ln_u_right=ln_u_right,
            u_right=u_right,
        ):
            u_left = np.exp(ln_u_left)
            u_face = constants.FLOATDTYPE(0.5) * (u_left + u_right)

            return (dln_rho + ln_u_right - ln_u_left) / dln_r + g_coefficient / u_face

        lower = ln_u_right - constants.FLOATDTYPE(10.0)
        upper = ln_u_right + constants.FLOATDTYPE(10.0)

        while shell_balance(lower) < constants.FLOATDTYPE(0.0):
            lower -= constants.FLOATDTYPE(10.0)

        while shell_balance(upper) > constants.FLOATDTYPE(0.0):
            upper += constants.FLOATDTYPE(10.0)

        ln_u_left = brentq(
            shell_balance,
            lower,
            upper,
            xtol=constants.FLOATDTYPE(1.0e-13),
            rtol=constants.FLOATDTYPE(1.0e-13),
        )

        u[i] = np.exp(ln_u_left)

    if np.any(~np.isfinite(u)) or np.any(u <= constants.FLOATDTYPE(0.0)):
        raise RuntimeError("Hydrostatic initialization produced invalid u.")

    return u


def set_up_initial_conditions(r_first, r_outer, n_shell, r_t=np.inf, n=1.0):
    """
    Set up initial condtions.
    Default is standard NFW without any truncation.
    """
    ln_r_edge = grid.make_radial_grid(r_first, r_outer, n_shell)
    ln_r = grid.log_cell_centers(ln_r_edge)
    ln_v = grid.log_shell_volumes(ln_r_edge)

    r = np.exp(ln_r)
    v = np.exp(ln_v)
    r_edge = np.exp(ln_r_edge)

    rho = truncated_nfw_dimensionless(r, r_t, n)
    ln_rho = np.log(rho)

    dm = shell_mass(rho, v)
    m_edge = enclosed_mass(dm)

    u = hydrostatic_u_from_rho(r_edge, ln_r, ln_rho, m_edge, u_outer=0.001)

    return ln_r_edge, u, dm, m_edge
