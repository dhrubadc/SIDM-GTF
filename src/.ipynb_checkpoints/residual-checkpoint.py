"""
Module to calculate the coupled residual
for the current state of the system
in dimensionless units (Nishikawa 2020).
"""

import numpy as np
import constants


def hydrostatic_terms(state):
    """
    Return the individual terms of each
    hydrostatic residual.
    """
    # pylint: disable=unsubscriptable-object

    dln_r = state["ln_r"][1:] - state["ln_r"][:-1]
    dln_rho = state["ln_rho"][1:] - state["ln_rho"][:-1]
    dln_u = state["ln_u"][1:] - state["ln_u"][:-1]

    u_face = constants.FLOATDTYPE(0.5) * (state["u"][1:] + state["u"][:-1])

    rho_slope = dln_rho / dln_r
    u_slope = dln_u / dln_r

    g_term = (
        constants.FLOATDTYPE(3.0 / 2.0)
        * constants.M_EDGE[1:-1]
        / (state["r_edge"][1:-1] * u_face)
    )

    return rho_slope, u_slope, g_term


def hydrostatic_residual(state):
    """
    Return the n_shell-1 raw hydrostatic residuals.
    """
    rho_slope, u_slope, g_term = hydrostatic_terms(state)

    fhydro = rho_slope + u_slope + g_term
    return fhydro


def energy_terms(state, v_old, u_old, dt):
    """
    Return the individual terms of each
    energy residual.
    """
    energy_term = state["u"] - u_old
    compression_term = (
        constants.FLOATDTYPE(2.0 / 3.0)
        * state["u"]
        * (constants.FLOATDTYPE(1.0) - v_old / state["v"])
    )
    conduction_term = dt * (state["l"][1:] - state["l"][:-1]) / constants.DM

    return energy_term, compression_term, conduction_term


def energy_residual(state, v_old, u_old, dt):
    """
    Return the n_shell energy residuals.
    """
    energy_term, compression_term, conduction_term = energy_terms(
        state, v_old, u_old, dt
    )

    fenergy = energy_term + compression_term + conduction_term
    return fenergy


def residual(state, v_old, u_old, dt):
    """
    Construct coupled hydro and energy residual
    from the current state of the system.
    """
    fhydro = hydrostatic_residual(state)
    fenergy = energy_residual(state, v_old, u_old, dt)

    return np.concatenate((fhydro, fenergy))
