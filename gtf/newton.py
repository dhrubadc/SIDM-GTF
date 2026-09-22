r"""
Implement Newton iterations and related helper functions.
"""

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from . import constants
from . import residual
from . import jacobian
from . import state


def pack_unknowns(ln_r_edge, u):
    r"""
    Pack unknown variables x = [ln_r_edge[1:-1], u].

    :param ln_r_edge: log of cell edges
    :type ln_r_edge: np.ndarray

    :param u: specific energy at cell midpoints
    :type u: np.ndarray

    :return: x = [ln_r_edge[1:-1], u]
    :rtype: np.ndarray
    """
    return np.concatenate((ln_r_edge[1:-1], u))


def unpack_unknowns(x):
    r"""
    Unpack unknown variables x into ln_r_edge and u.
    x = [ln_r_edge[1:-1], u].

    :param x: Array of unknown variables
    :type x: np.ndarray

    :return: log of cell edges and specific energy at cell midpoints
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    ln_r_edge = np.empty(constants.N_SHELL + 1, dtype=constants.FLOATDTYPE)

    ln_r_edge[0] = -np.inf
    ln_r_edge[-1] = np.log(constants.R_OUTER)
    ln_r_edge[1:-1] = x[: constants.N_SHELL - 1]

    u = x[constants.N_SHELL - 1 :]

    return ln_r_edge, u


def increment_newton_variables(x_trial, dx):
    r"""
    Implement line search to ensure
    cells are always ordered and u>0.

    .. math ::

       x_{\rm new} = x_{\rm trial} + \alpha\ {\rm d}x

    :math:`\alpha` is initialized with 1.0
    and continually halved until a physically valid :math:`x_{\rm new}`
    is found (ordered cells and u>0)

    :param x_trial: trial solutions from previous iteration
    :type x_trial: np.ndarray

    :param dx: proposed increment in the trial solutions from solving J * dx = -F
    :type dx: np.ndarray

    :return: physically valid new trial solutions,
             corresponding full physical state,
             and the accepted :math:`\alpha`.
    :rtype: tuple[np.ndarray, dict, :obj:`gtf.constants.FLOATDTYPE`]

    Here J is the output of :obj:`gtf.jacobian.analytic_jacobian` and
    F is the output of :obj:`gtf.residual.residual`.
    """
    alpha = 1.0

    while True:

        x_new = x_trial + alpha * dx

        ln_r_edge_new, u_new = unpack_unknowns(x_new)

        valid_r = np.all(ln_r_edge_new[1:] > ln_r_edge_new[:-1])
        valid_u = np.all(u_new > 0.0)

        if valid_r and valid_u:
            break

        alpha *= 0.5

    state_new = state.state_from_unknowns(ln_r_edge_new, u_new)

    return x_new, state_new, alpha


def iterate(state_old, dt):
    # pylint: disable=too-many-locals
    r"""
    Implement Newton iterations.

    Iteratively solve  J * dx = -F, starting from a trial solution
    corresponding to the state of the system at time t-dt.

    Here J is the output of :obj:`gtf.jacobian.analytic_jacobian` and
    F is the output of :obj:`gtf.residual.residual`.

    A converged solution for time t is found
    if :math:`{\rm MAX}(|F|)` < :obj:`gtf.constants.F_TOL`

    A stagnated but acceptable solution for time t is found
    if between last two iterations
    :math:`{\rm MAX}(|\Delta\ \ln r_{\rm edge}|, |\Delta\ u|/u)`
    < :obj:`gtf.constants.X_TOL` but :math:`{\rm MAX}(|F|)`
    < :obj:`gtf.constants.F_ACCEPT`

    Maximum number of iterations tried is :obj:`gtf.constants.ITER_MAX`.

    :param state_old: physical state of the system at time t-dt
    :type state_old: dict

    :param dt: timestep
    :type dt: :obj:`gtf.constants.FLOATDTYPE`

    :return: physical state of the system at time t,
             final :math:`{\rm MAX}(|F|)`, total number of iterations
    :rtype: tuple[dict, :obj:`gtf.constants.FLOATDTYPE`, int]
            or None if a suitably converged state is not found
    """
    x_trial = pack_unknowns(state_old["ln_r_edge"], state_old["u"])

    v_old = state_old["v"]
    u_old = state_old["u"]

    f_trial = residual.residual(state_old, v_old, u_old, dt)
    j_trial = jacobian.analytic_jacobian(state_old, v_old, dt)

    f_max = np.max(np.abs(f_trial))

    if f_max < constants.F_TOL:

        print(
            "state is not updated and same as previous time",
            "\n max(F) = ",
            np.max(np.abs(f_trial)),
        )

        return state_old, f_max, 0

    for i in range(1, constants.ITER_MAX):

        dx = spsolve(csr_array(j_trial), -f_trial)

        x_new, state_new, alpha = increment_newton_variables(x_trial, dx)

        dx_r = np.max(
            np.abs(x_new[: constants.N_SHELL - 1] - x_trial[: constants.N_SHELL - 1])
        )

        dx_u = np.max(
            np.abs(x_new[constants.N_SHELL - 1 :] - x_trial[constants.N_SHELL - 1 :])
            / x_trial[constants.N_SHELL - 1 :]
        )

        dx_max = np.maximum(dx_r, dx_u)

        f_new = residual.residual(state_new, v_old, u_old, dt)
        j_new = jacobian.analytic_jacobian(state_new, v_old, dt)

        print(
            "Newton iteration",
            i,
            "\n max(F) = ",
            np.max(np.abs(f_new)),
            "\n max(dx) = ",
            dx_max,
            "\n alpha = ",
            alpha,
        )

        f_max = np.max(np.abs(f_new))

        if f_max < constants.F_TOL:

            print("Newton iterations have converged in ", i, "iterations.")
            return state_new, f_max, i

        if dx_max < constants.X_TOL:

            print("Newton has stagnated after ", i, "iterations.")

            if f_max < constants.F_ACCEPT:

                print("Stagnated but acceptable")
                return state_new, f_max, i

            print("Stagnated and not acceptable")
            return None

        f_trial = f_new
        j_trial = j_new
        x_trial = x_new

    print("Increase number of iterations")
    return None
