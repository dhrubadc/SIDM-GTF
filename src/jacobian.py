"""
Module to calculate the analytical jacobian
corresponding to the coupled residual
for the current state of the system
in dimensionless units (Nishikawa 2020).
"""

import numpy as np
import constants


def diff_quant(i, quant):
    """
    Difference of a cell quantity
    between consequtive cells.
    """
    return quant[i + 1] - quant[i]


def delta(m, n):
    """
    Delta Function.
    """
    if n == m:
        return constants.FLOATDTYPE(1.0)

    return constants.FLOATDTYPE(0.0)


def volume_ratio(m, state, v_old):
    """
    Ratio of volume at previous timestep
    and current volume.
    """
    return v_old[m] / state["v"][m]


def gravity_term(i, state, m_edge):
    """
    Self gravity dependent term of the hydro residual.
    """
    return (
        constants.FLOATDTYPE(3.0)
        * m_edge[i + 1]
        / (state["r_edge"][i + 1] * (state["u"][i] + state["u"][i + 1]))
    )


def kappa_derivative_common(i, state, alpha):
    """
    Common coefficient for the kappa derivatives.
    """
    return state["kappa_s"][i] ** (alpha) / (
        state["kappa_s"][i] ** alpha + state["kappa_l"][i] ** alpha
    )


def dln_r_dln_redge(i, j, state):
    """
    Derivative of ln_r
    with respect to ln_r_edge.
    """
    den = state["r_edge"][i] + state["r_edge"][i + 1]
    num = state["r_edge"][i] * delta(i, j) + state["r_edge"][i + 1] * delta(i + 1, j)
    return num / den


def dr_dln_redge(i, j, state):
    """
    Derivative of r
    with respect to ln_r_edge.
    """
    return constants.FLOATDTYPE(0.5) * (
        state["r_edge"][i] * delta(i, j) + state["r_edge"][i + 1] * delta(i + 1, j)
    )


def dln_v_dln_redge(i, j, state):
    """
    Derivative of ln_v
    with respect to ln_r_edge.
    """
    den = state["r_edge"][i + 1] ** 3.0 - state["r_edge"][i] ** 3.0

    num = constants.FLOATDTYPE(3.0) * (
        state["r_edge"][i + 1] ** 3.0 * delta(i + 1, j)
        - state["r_edge"][i] ** 3.0 * delta(i, j)
    )
    return num / den


def dln_u_du(i, k, state):
    """
    Detivative of ln_u
    with respect to u.
    """
    return delta(i, k) / state["u"][i]


def dfhydro_dln_redge(i, j, state, m_edge):
    """
    Derivative of hydro residual
    with respect to ln_r_edge.
    """
    dln_r = diff_quant(i, state["ln_r"])
    dln_rho = diff_quant(i, state["ln_rho"]) + diff_quant(i, state["ln_u"])
    g_term = gravity_term(i, state, m_edge)

    term_1 = (dln_v_dln_redge(i, j, state) - dln_v_dln_redge(i + 1, j, state)) / dln_r
    term_2 = (
        dln_rho
        * (dln_r_dln_redge(i + 1, j, state) - dln_r_dln_redge(i, j, state))
        / dln_r**2.0
    )
    term_3 = g_term * delta(i + 1, j)

    return term_1 - term_2 - term_3


def dfhydro_du(i, k, state, m_edge):
    """
    Derivative of hydro residual
    with respect to u.
    """
    dln_r = diff_quant(i, state["ln_r"])
    g_term = gravity_term(i, state, m_edge)

    term_1 = (dln_u_du(i + 1, k, state) - dln_u_du(i, k, state)) / dln_r
    term_2 = (
        g_term / (state["u"][i] + state["u"][i + 1]) * (delta(i, k) + delta(i + 1, k))
    )

    return term_1 - term_2


def dkappa_dln_redge(i, j, state, alpha):
    """
    Derivative of kappa
    with respect to ln_redge.
    """
    kappa_common = kappa_derivative_common(i, state, alpha)
    return -state["kappa"][i] * kappa_common * dln_v_dln_redge(i, j, state)


def dkappa_du(i, k, state, alpha):
    """
    Derivative of kappa
    with respect to u.
    """
    kappa_common = kappa_derivative_common(i, state, alpha)
    return (
        state["kappa"][i]
        / state["u"][i]
        * (constants.FLOATDTYPE(1.0 / 2.0) + kappa_common)
        * delta(i, k)
    )


def dl_dln_redge(i, j, state, alpha):
    """
    Derivative of l at the interior edges
    with respect to ln_r_edge.
    """
    dr = diff_quant(i, state["r"])
    du = diff_quant(i, state["u"])

    term_1 = constants.FLOATDTYPE(2.0) * delta(i + 1, j)
    term_2 = (
        constants.FLOATDTYPE(1.0)
        / dr
        * (dr_dln_redge(i + 1, j, state) - dr_dln_redge(i, j, state))
    )

    term_3 = dkappa_dln_redge(i, j, state, alpha)
    term_4 = dkappa_dln_redge(i + 1, j, state, alpha)

    term_5 = (
        constants.FLOATDTYPE(2.0)
        / constants.FLOATDTYPE(3.0)
        * state["r_edge"][i + 1] ** 2.0
        * du
        / dr
        * constants.FLOATDTYPE(0.5)
        * (term_3 + term_4)
    )

    return state["l"][1:-1][i] * (term_1 - term_2) - term_5


def dl_du(i, k, state, alpha):
    """
    Derivative of l at the interior edges
    with respect to u.
    """
    dr = diff_quant(i, state["r"])
    du = diff_quant(i, state["u"])

    kappa_face = constants.FLOATDTYPE(0.5) * (state["kappa"][i] + state["kappa"][i + 1])

    term_1 = kappa_face * (delta(i + 1, k) - delta(i, k))

    term_2 = dkappa_du(i, k, state, alpha)
    term_3 = dkappa_du(i + 1, k, state, alpha)

    term_4 = du * constants.FLOATDTYPE(0.5) * (term_2 + term_3)

    return (
        -constants.FLOATDTYPE(2.0)
        / constants.FLOATDTYPE(3.0)
        * state["r_edge"][i + 1] ** 2.0
        / dr
        * (term_1 + term_4)
    )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def dfenergy_dln_r_edge(m, j, state, v_old, dm, dt, alpha, n_shell):
    """
    Derivative of energy residual
    with respect to ln_r_edge.
    """
    z = volume_ratio(m, state, v_old)

    term_1 = (
        constants.FLOATDTYPE(2.0)
        / constants.FLOATDTYPE(3.0)
        * state["u"][m]
        * z
        * dln_v_dln_redge(m, j, state)
    )

    if m == 0:
        term_2 = dt / dm[m] * dl_dln_redge(m, j, state, alpha)
    elif m == n_shell - 1:
        term_2 = -dt / dm[m] * dl_dln_redge(m - 1, j, state, alpha)
    else:
        term_2 = (
            dt
            / dm[m]
            * (dl_dln_redge(m, j, state, alpha) - dl_dln_redge(m - 1, j, state, alpha))
        )

    return term_1 + term_2


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def dfenergy_du(m, k, state, v_old, dm, dt, alpha, n_shell):
    """
    Derivative of energy residual
    with respect to u.
    """
    z = volume_ratio(m, state, v_old)

    term_1 = (
        constants.FLOATDTYPE(1.0)
        + constants.FLOATDTYPE(2.0)
        / constants.FLOATDTYPE(3.0)
        * (constants.FLOATDTYPE(1.0) - z)
    ) * delta(m, k)

    if m == 0:
        term_2 = dt / dm[m] * dl_du(m, k, state, alpha)
    elif m == n_shell - 1:
        term_2 = -dt / dm[m] * dl_du(m - 1, k, state, alpha)
    else:
        term_2 = (
            dt / dm[m] * (dl_du(m, k, state, alpha) - dl_du(m - 1, k, state, alpha))
        )

    return term_1 + term_2


def jacobian_hydro_block(state, m_edge):
    """
    Hydro block of the Jacobian
    with fixed edges at ln_r_edge[0] and ln_r_edge[-1].
    df_dln_redge is tri-diagonal and df_du is bi-diagonal.
    """
    n_shell = len(m_edge) - 1
    df_dln_redge = np.full(
        (n_shell - 1, n_shell + 1), np.nan, dtype=constants.FLOATDTYPE
    )
    df_du = np.full((n_shell - 1, n_shell), np.nan, dtype=constants.FLOATDTYPE)

    for i in range(0, n_shell - 1):

        for j in range(0, n_shell + 1):
            df_dln_redge[i, j] = dfhydro_dln_redge(i, j, state, m_edge)

        for k in range(0, n_shell):
            df_du[i, k] = dfhydro_du(i, k, state, m_edge)

    jac_hydro = np.column_stack((df_dln_redge[:, 1:-1], df_du))
    return jac_hydro


def jacobian_energy_block(state, v_old, dm, dt, alpha):
    """
    Energy block of Jacobian
    with fixed edges at ln_r_edge[0] and ln_r_edge[-1].
    df_dln_redge is four-diagonal and df_du is tri-diagonal.
    """
    n_shell = len(dm)
    df_dln_redge = np.full((n_shell, n_shell + 1), np.nan, dtype=constants.FLOATDTYPE)
    df_du = np.full((n_shell, n_shell), np.nan, dtype=constants.FLOATDTYPE)

    for m in range(0, n_shell):

        for j in range(0, n_shell + 1):
            df_dln_redge[m, j] = dfenergy_dln_r_edge(
                m, j, state, v_old, dm, dt, alpha, n_shell
            )

        for k in range(0, n_shell):
            df_du[m, k] = dfenergy_du(m, k, state, v_old, dm, dt, alpha, n_shell)

    jac_energy = np.column_stack((df_dln_redge[:, 1:-1], df_du))
    return jac_energy


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def analytic_jacobian(state, v_old, m_edge, dm, dt, alpha):
    """
    Analytical jacobian for the coupled residual.
    """
    jac_hydro = jacobian_hydro_block(state, m_edge)
    jac_energy = jacobian_energy_block(state, v_old, dm, dt, alpha)

    jac = np.row_stack((jac_hydro, jac_energy))
    return jac
