"""Module to calculate coupled residual"""

import numpy as np
import constants
import grid
import state
import energy


def hydrostatic_terms_and_scale(r_edge, ln_r, u, ln_u, ln_rho, M_edge):
    """
    Return the individual terms and scale of the
    logarithmic hydrostatic residual.
    """
    delta_ln_r = ln_r[1:] - ln_r[:-1]
    delta_ln_rho = ln_rho[1:] - ln_rho[:-1]
    delta_ln_u = ln_u[1:] - ln_u[:-1]

    u_face = constants.FLOAT_DTYPE(0.5) * (u[1:] + u[:-1])

    rho_slope = delta_ln_rho / delta_ln_r
    u_slope = delta_ln_u / delta_ln_r

    gravity_slope = (
        constants.FLOAT_DTYPE(3.0 / 2.0) * M_edge[1:-1] / (r_edge[1:-1] * u_face)
    )

    hydro_scale = np.maximum(
        constants.FLOAT_DTYPE(1.0),
        np.abs(rho_slope) + np.abs(u_slope) + np.abs(gravity_slope),
    )

    return rho_slope, u_slope, gravity_slope, hydro_scale


def hydrostatic_residual(r_edge, ln_r, u, ln_u, ln_rho, M_edge):
    """
    Return the N-1 hydrostatic residuals,
    where N is the number of shells.

    F_H[i] = (Delta ln(rho) + Delta_ln(u)) / Delta ln(r)
             + (3.0/2.0) * M_edge[1:-1] / (r_edge[1:-1] * u_face).
    """
    rho_slope, u_slope, gravity_slope, hydro_scale = hydrostatic_terms_and_scale(
        r_edge, ln_r, u, ln_u, ln_rho, M_edge
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
    compression_term = (
        constants.FLOAT_DTYPE(2.0 / 3.0) * u * (constants.FLOAT_DTYPE(1.0) - V_old / V)
    )
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


def residual(
    x,
    ln_r_edge_old,
    u_old,
    dt,
    dm,
    M_edge,
    sigma_over_m,
    C,
    alpha,
    r_inner,
    r_outer,
    N_shell,
):
    """Construct combined hydro and energy residual."""
    ln_r_edge, u = unpack_unknowns(x, r_inner, r_outer, N_shell)

    state_this = state.state_from_unknowns(ln_r_edge, u, dm, sigma_over_m, C, alpha)

    r_edge = state_this["r_edge"]
    ln_r = state_this["ln_r"]
    V = state_this["V"]
    ln_rho = state_this["ln_rho"]
    ln_u = state_this["ln_u"]
    L = state_this["L"]

    F_H, scaled_F_H = hydrostatic_residual(r_edge, ln_r, u, ln_u, ln_rho, M_edge)

    r_edge_old = np.exp(ln_r_edge_old)
    V_old = grid.shell_volumes(r_edge_old)

    F_E, scaled_F_E = energy_residual(u, u_old, V, V_old, dm, L, dt)

    return np.concatenate((F_H, F_E)), np.concatenate((scaled_F_H, scaled_F_E))


def pack_unknowns(ln_r_edge, u):
    """
    Pack x = [ln_r_edge[1:-1], u].
    """
    return np.concatenate((ln_r_edge[1:-1], u))


def unpack_unknowns(x, r_inner, r_outer, N_shell):
    """
    Unpack x into ln_r_edge and u.
    """
    nr = N_shell - 1
    if len(x) != 2 * N_shell - 1:
        raise ValueError("Incorrect Newton-vector length.")

    ln_r_edge = np.empty(N_shell + 1, dtype=constants.FLOAT_DTYPE)

    ln_r_edge[0] = -np.inf if r_inner == 0.0 else np.log(r_inner)
    ln_r_edge[-1] = np.log(r_outer)
    ln_r_edge[1:-1] = x[:nr]

    u = x[nr:]

    return ln_r_edge, u
