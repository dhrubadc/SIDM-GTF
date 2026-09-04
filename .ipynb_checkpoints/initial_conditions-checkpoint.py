"""Halo initial conditions in linear variables."""

import numpy as np

from constants import FLOAT_DTYPE
import gravity
from grid import shell_volume


def truncated_nfw_dimensionless(r, r_s, r_t, truncation_n, M_s):
    """Return rho/rho_s for a truncated NFW profile."""
    del M_s  # rho/rho_s is independent of the chosen mass scale.
    r = np.asarray(r, dtype=FLOAT_DTYPE)
    r_s = FLOAT_DTYPE(r_s)
    r_t = FLOAT_DTYPE(r_t)
    truncation_n = FLOAT_DTYPE(truncation_n)
    x = r / r_s
    return np.exp(-(r / r_t)**truncation_n) / (
        x * (FLOAT_DTYPE(1.0) + x)**2
    )


def nfw_dimensionless(r, r_s, M_s):
    """Return rho/rho_s for an NFW profile."""
    del M_s
    r = np.asarray(r, dtype=FLOAT_DTYPE)
    r_s = FLOAT_DTYPE(r_s)
    x = r / r_s
    return FLOAT_DTYPE(1.0) / (
        x * (FLOAT_DTYPE(1.0) + x)**2
    )


def shell_mass(rho, volume):
    """Return the fixed mass in each shell."""
    rho = np.asarray(rho, dtype=FLOAT_DTYPE)
    volume = np.asarray(volume, dtype=FLOAT_DTYPE)
    return rho * volume


def enclosed_mass(dm):
    """Return enclosed mass at all shell edges."""
    dm = np.asarray(dm, dtype=FLOAT_DTYPE)
    M_edge = np.zeros(len(dm) + 1, dtype=FLOAT_DTYPE)
    M_edge[1:] = np.cumsum(dm, dtype=FLOAT_DTYPE)
    return M_edge


def hydrostatic_u_from_rho(
    r_edge,
    rho,
    M_edge,
    u_outer,
    gamma=5.0 / 3.0,
):
    """
    Integrate the energy-compatible variational hydrostatic residual inward.

    The outermost u is specified.  The pressure jump at every interior edge
    is fixed by the exact piecewise-uniform-shell gravitational-energy
    gradient used by the coupled residual.
    """
    r_edge = np.asarray(r_edge, dtype=FLOAT_DTYPE)
    rho = np.asarray(rho, dtype=FLOAT_DTYPE)
    M_edge = np.asarray(M_edge, dtype=FLOAT_DTYPE)
    u_outer = FLOAT_DTYPE(u_outer)
    gamma = FLOAT_DTYPE(gamma)
    N = len(rho)
    if len(r_edge) != N + 1 or len(M_edge) != N + 1:
        raise ValueError("Inconsistent initial-condition array lengths.")
    if u_outer <= FLOAT_DTYPE(0.0):
        raise ValueError("u_outer must be positive.")

    V = shell_volume(r_edge)
    dm = rho * V
    _W, gravitational_gradient, _hdiag, _hoff = (
        gravity.gravitational_energy_gradient_hessian(
            r_edge, dm, M_edge
        )
    )

    pressure = np.empty(N, dtype=FLOAT_DTYPE)
    pressure[-1] = (
        (gamma - FLOAT_DTYPE(1.0)) * rho[-1] * u_outer
    )

    for i in range(N - 2, -1, -1):
        edge = i + 1
        pressure[i] = (
            pressure[i + 1]
            + gravitational_gradient[edge] / r_edge[edge]**3
        )

    u = pressure * V / ((gamma - FLOAT_DTYPE(1.0)) * dm)

    if np.any(~np.isfinite(u)) or np.any(u <= FLOAT_DTYPE(0.0)):
        raise RuntimeError("Hydrostatic initialization produced invalid u.")

    return u
