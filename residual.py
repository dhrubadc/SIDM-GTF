import numpy as np
from constants import FLOAT_DTYPE
from grid import log_cell_centers


def hydrostatic_terms_and_scale(ln_r_edge, u, ln_rho, M_edge):
    """
    Return the individual terms and scale of the
    logarithmic hydrostatic residual
    """
    r_edge = np.exp(ln_r_edge)
    ln_u = np.log(u)

    ln_r = log_cell_centers(ln_r_edge)
    delta_ln_r = ln_r[1:] - ln_r[:-1]

    delta_ln_rho = ln_rho[1:] - ln_rho[:-1]
    delta_ln_u = ln_u[1:] - ln_u[:-1]

    u_face = FLOAT_DTYPE(0.5) * (u[1:] + u[:-1])

    rho_slope = delta_ln_rho / delta_ln_r
    u_slope = delta_ln_u / delta_ln_r

    gravity_slope = FLOAT_DTYPE(3.0 / 2.0) * M_edge[1:-1] / (r_edge[1:-1] * u_face)

    hydro_scale = np.maximum(
        FLOAT_DTYPE(1.0),
        np.abs(rho_slope) + np.abs(u_slope) + np.abs(gravity_slope),
    )

    return rho_slope, u_slope, gravity_slope, hydro_scale


def hydrostatic_residual(ln_r_edge, u, ln_rho, M_edge):
    """
    Return the N-1 hydrostatic residuals,
    where N is the number of shells.

    F_H[i] = (Delta ln(rho) + Delta_ln(u)) / Delta ln(r)
             + (3.0/2.0) * M_edge[1:-1] / (r_edge[1:-1] * u_face).
    """
    rho_slope, u_slope, gravity_slope, hydro_scale = hydrostatic_terms_and_scale(
        ln_r_edge, u, ln_rho, M_edge
    )

    F_H = rho_slope + u_slope + gravity_slope
    scaled_F_H = F_H / hydro_scale

    return F_H, scaled_F_H


def energy_terms(u, u_old, V, V_old, dm, L, dt):
    """
    Return the individual terms and scale of the
    liner energy residual.
    """
    energy_term = u - u_old
    compression_term = FLOAT_DTYPE(2.0 / 3.0) * u * (FLOAT_DTYPE(1.0) - V_old / V)
    conduction_term = dt * energy.luminosity_divergence(L, dm)

    energy_scale = np.maximum(
        u_old,
        np.abs(energy_term) + np.abs(compression_term) + np.abs(conduction_term),
    )

    return energy_term, compression_term, conduction_term, energy_scale


def energy_residual(u, u_old, V, V_old, dm, L, dt):
    """
    Return the N energy residuals,
    where N is the number of shells.
    """
    energy_term, compression_term, conduction_term, energy_scale = energy_terms(
        u, u_old, V, V_old, dm, L, dt
    )

    F_E = energy_term + compression_term + conduction_term
    scaled_F_E = F_E / energy_scale

    return F_E, scaled_F_E
