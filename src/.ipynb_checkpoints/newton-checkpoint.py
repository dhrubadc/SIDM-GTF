"""
Module to implement Newton iterations
and helper functions for packing
and upacking of primary unknowns.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
import constants
import residual
import jacobian
import state


def pack_unknowns(ln_r_edge, u):
    """
    Pack x = [ln_r_edge[1:-1], u].
    """
    return np.concatenate((ln_r_edge[1:-1], u))


def unpack_unknowns(x):
    """
    Unpack x into ln_r_edge and u.
    """
    ln_r_edge = np.empty(constants.N_SHELL + 1, dtype=constants.FLOATDTYPE)

    ln_r_edge[0] = -np.inf
    ln_r_edge[-1] = np.log(constants.R_OUTER)
    ln_r_edge[1:-1] = x[: constants.N_SHELL - 1]

    u = x[constants.N_SHELL - 1 :]

    return ln_r_edge, u


def iterate(state_prev, dt):
    # pylint: disable=too-many-locals
    """Implement Newton iterations."""

    x_trial = pack_unknowns(state_prev["ln_r_edge"], state_prev["u"])
    state_trial = state_prev

    v_prev = state_prev["v"]
    u_prev = state_prev["u"]

    f_trial = residual.residual(state_trial, v_prev, u_prev, dt)
    j_trial = jacobian.analytic_jacobian(state_trial, v_prev, dt)

    if np.max(np.abs(f_trial)) < constants.F_TOL:
        print("state is same as previous time", np.max(f_trial))
        return state_trial

    for i in range(0, constants.ITER_MAX):

        dx = spsolve(j_trial.tocsr(), -f_trial)

        x_now = x_trial + dx

        ln_r_edge_now, u_now = unpack_unknowns(x_now)

        state_now = state.state_from_unknowns(ln_r_edge_now, u_now)

        f_now = residual.residual(state_now, v_prev, u_prev, dt)
        j_now = jacobian.analytic_jacobian(state_now, v_prev, dt)

        print(i, np.max(np.abs(f_now)))

        if np.max(np.abs(f_now)) < constants.F_TOL:
            print("newton has converged")
            return state_now

        f_trial = f_now
        j_trial = j_now
        x_trial = x_now

    print("newton has not converged")
    return None
