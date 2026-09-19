"""
Module to implement Newton iterations
and helper functions for packing
and upacking of primary unknowns.
"""

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from . import constants
from . import residual
from . import jacobian
from . import state


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


def increment_newton_variables(x_trial, dx):
    """
    Implement line search to ensure
    shells are always ordered and u>0.
    """
    alpha = 1.0

    while True:

        x_now = x_trial + alpha * dx

        ln_r_edge_now, u_now = unpack_unknowns(x_now)

        valid_r = np.all(ln_r_edge_now[1:] > ln_r_edge_now[:-1])
        valid_u = np.all(u_now > 0.0)

        if valid_r and valid_u:
            break

        alpha *= 0.5

    return x_now, ln_r_edge_now, u_now, alpha


def iterate(state_prev, dt):
    # pylint: disable=too-many-locals
    """
    Implement Newton iterations.
    """

    x_trial = pack_unknowns(state_prev["ln_r_edge"], state_prev["u"])

    v_prev = state_prev["v"]
    u_prev = state_prev["u"]

    f_trial = residual.residual(state_prev, v_prev, u_prev, dt)
    j_trial = jacobian.analytic_jacobian(state_prev, v_prev, dt)

    if np.max(np.abs(f_trial)) < constants.F_TOL:

        print(
            "state is not updated and same as previous time",
            "\n max(F) = ",
            np.max(np.abs(f_trial)),
        )

        return state_prev

    for i in range(1, constants.ITER_MAX):

        dx = spsolve(csr_array(j_trial), -f_trial)

        x_now, ln_r_edge_now, u_now, alpha = increment_newton_variables(x_trial, dx)

        dx_r = np.max(
            np.abs(x_now[: constants.N_SHELL - 1] - x_trial[: constants.N_SHELL - 1])
        )

        dx_u = np.max(
            np.abs(x_now[constants.N_SHELL - 1 :] - x_trial[constants.N_SHELL - 1 :])
            / x_now[constants.N_SHELL - 1 :]
        )

        dx_max = max(dx_r, dx_u)

        state_now = state.state_from_unknowns(ln_r_edge_now, u_now)
        f_now = residual.residual(state_now, v_prev, u_prev, dt)
        j_now = jacobian.analytic_jacobian(state_now, v_prev, dt)

        print(
            "Newton iteration",
            i,
            "\n max(F) = ",
            np.max(np.abs(f_now)),
            "\n max(dx) = ",
            dx_max,
            "\n alpha = ",
            alpha,
        )

        if np.max(np.abs(f_now)) < constants.F_TOL:

            print("Newton iterations have converged in ", i, "iterations.")

            return state_now

        if dx_max < constants.X_TOL:

            print("Newton has stagnated after ", i, "iterations.")

            if np.max(np.abs(f_now)) < constants.F_ACCEPT:

                print("Stagnated but acceptable")
                return state_now

            print("Stagnated and not acceptable")
            return None

        f_trial = f_now
        j_trial = j_now
        x_trial = x_now

    print("Increase number of iterations")
    return None
