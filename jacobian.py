"""Module to construct the jacobian
for the coupled residual.
"""

import numpy as np
from constants import FLOATDTYPE


def diff_quant(i, quant):
    """Difference of a quantity
    between consequtive cells.
    """
    return quant[i + 1] - quant[i]


def delta(m, n):
    """Delta Function."""
    if n == m:
        return FLOATDTYPE(1.0)

    return FLOATDTYPE(0.0)


def volume_ratio(m, state, v_old):
    """Ratio of volume at previous timestep
    and current volume.
    """
    return v_old[m] / state["v"][m]


def gravity_term(i, state, m_edge):
    """gravity term of the hydro residual."""
    return (
        FLOATDTYPE(3.0)
        * m_edge[i + 1]
        / (state["r_edge"][i + 1] * (state["u"][i] + state["u"][i + 1]))
    )


def kappa_derivative_common(i, state, alpha):
    """Common coefficient for the kappa derivatives."""
    return state["kappa_s"][i] ** (alpha) / (
        state["kappa_s"][i] ** alpha + state["kappa_l"][i] ** alpha
    )


def dln_r_dln_redge(i, j, state):
    """Derivative of ln_r
    with respect to ln_r_edge.
    """
    den = state["r_edge"][i] + state["r_edge"][i + 1]
    num = state["r_edge"][i] * delta(i, j) + state["r_edge"][i + 1] * delta(i + 1, j)
    return num / den


def dr_dln_redge(i, j, state):
    """Derivative of r
    with respect to ln_r_edge.
    """
    return FLOATDTYPE(0.5) * (
        state["r_edge"][i] * delta(i, j) + state["r_edge"][i + 1] * delta(i + 1, j)
    )


def dln_v_dln_redge(i, j, state):
    """Derivative of ln_v
    with respect to ln_r_edge.
    """
    den = state["r_edge"][i + 1] ** 3.0 - state["r_edge"][i] ** 3.0

    num = FLOATDTYPE(3.0) * (
        state["r_edge"][i + 1] ** 3.0 * delta(i + 1, j)
        - state["r_edge"][i] ** 3.0 * delta(i, j)
    )
    return num / den


def dln_u_du(i, k, state):
    """Detivative of ln_u
    with respect to u.
    """
    return delta(i, k) / state["u"][i]


def dfhydro_dln_redge(i, j, state, m_edge):
    """Derivative of hydro residual
    with respect to ln_r_edge.
    """
    d = diff_quant(i, state["ln_r"])
    a = diff_quant(i, state["ln_rho"]) + diff_quant(i, state["ln_u"])
    g = gravity_term(i, state, m_edge)

    term_1 = (dln_v_dln_redge(i, j, state) - dln_v_dln_redge(i + 1, j, state)) / d
    term_2 = (
        a * (dln_r_dln_redge(i + 1, j, state) - dln_r_dln_redge(i, j, state)) / d**2.0
    )
    term_3 = g * delta(i + 1, j)

    return term_1 - term_2 - term_3


def dfhydro_du(i, k, state, m_edge):
    """Derivative of hydro residual
    with respect to u.
    """
    d = diff_quant(i, state["ln_r"])
    g = gravity_term(i, state, m_edge)

    term_1 = (dln_u_du(i + 1, k, state) - dln_u_du(i, k, state)) / d
    term_2 = g / (state["u"][i] + state["u"][i + 1]) * (delta(i, k) + delta(i + 1, k))

    return term_1 - term_2


def dkappa_dln_redge(i, j, state, alpha):
    """Derivative of kappa
    with respect to ln_redge.
    """
    theta = kappa_derivative_common(i, state, alpha)
    return -state["kappa"][i] * theta * dln_v_dln_redge(i, j, state)


def dkappa_du(i, k, state, alpha):
    """Derivative of kappa
    with respect to u.
    """
    theta = kappa_derivative_common(i, state, alpha)
    return (
        state["kappa"][i]
        / state["u"][i]
        * (FLOATDTYPE(1.0 / 2.0) + theta)
        * delta(i, k)
    )


def dl_dln_redge(i, j, state, alpha):
    """Derivative of L at the interior edges
    with respect to ln_r_edge.
    """
    dr = diff_quant(i, state["r"])
    du = diff_quant(i, state["u"])

    term_1 = FLOATDTYPE(2.0) * delta(i + 1, j)
    term_2 = (
        FLOATDTYPE(1.0)
        / dr
        * (dr_dln_redge(i + 1, j, state) - dr_dln_redge(i, j, state))
    )

    term_3 = dkappa_dln_redge(i, j, state, alpha)
    term_4 = dkappa_dln_redge(i + 1, j, state, alpha)

    term_5 = (
        FLOATDTYPE(2.0)
        / FLOATDTYPE(3.0)
        * state["r_edge"][i + 1] ** 2.0
        * du
        / dr
        * FLOATDTYPE(0.5)
        * (term_3 + term_4)
    )

    return state["l"][1:-1][i] * (term_1 - term_2) - term_5


def dl_du(i, k, state, alpha):
    """Derivative of L at the interior edges
    with respect to u.
    """
    dr = diff_quant(i, state["r"])
    du = diff_quant(i, state["u"])

    kappa_face = FLOATDTYPE(0.5) * (state["kappa"][i] + state["kappa"][i + 1])

    term_1 = kappa_face * (delta(i + 1, k) - delta(i, k))

    term_2 = dkappa_du(i, k, state, alpha)
    term_3 = dkappa_du(i + 1, k, state, alpha)

    term_4 = du * FLOATDTYPE(0.5) * (term_2 + term_3)

    return (
        -FLOATDTYPE(2.0)
        / FLOATDTYPE(3.0)
        * state["r_edge"][i + 1] ** 2.0
        / dr
        * (term_1 + term_4)
    )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def dfenergy_dln_r_edge(m, j, state, v_old, alpha, dm, dt, n_shell):
    """Derivative of energy residual
    with respect to ln_r_edge.
    """
    z = volume_ratio(m, state, v_old)

    term_1 = (
        FLOATDTYPE(2.0)
        / FLOATDTYPE(3.0)
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
def dfenergy_du(m, k, state, v_old, alpha, dm, dt, n_shell):
    """Derivative of energy residual
    with respect to u.
    """
    z = volume_ratio(m, state, v_old)

    term_1 = (
        FLOATDTYPE(1.0) + FLOATDTYPE(2.0) / FLOATDTYPE(3.0) * (FLOATDTYPE(1.0) - z)
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


def jacobian_hydro_block(state, m_edge, n_shell):
    """Hydro block of the Jacobian
    with fixed edges at ln_r_edge[0] and ln_r_edge[-1].
    df_dln_redge is tri-diagonal and df_du is bi-diagonal.
    """
    df_dln_redge = np.full((n_shell - 1, n_shell + 1), np.nan, dtype=FLOATDTYPE)
    df_du = np.full((n_shell - 1, n_shell), np.nan, dtype=FLOATDTYPE)

    for i in range(0, n_shell - 1):

        for j in range(0, n_shell + 1):
            df_dln_redge[i, j] = dfhydro_dln_redge(i, j, state, m_edge)

        for k in range(0, n_shell):
            df_du[i, k] = dfhydro_du(i, k, state, m_edge)

    jac_hydro = np.column_stack((df_dln_redge[:, 1:-1], df_du))
    return jac_hydro


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def jacobian_energy_block(state, v_old, alpha, dm, dt, n_shell):
    """Energy block of Jacobian
    with fixed edges at ln_r_edge[0] and ln_r_edge[-1].
    df_dln_redge is four-diagonal and df_du is bi-diagonal.
    """
    df_dln_redge = np.full((n_shell, n_shell + 1), np.nan, dtype=FLOATDTYPE)
    df_du = np.full((n_shell, n_shell), np.nan, dtype=FLOATDTYPE)

    for m in range(0, n_shell):

        for j in range(0, n_shell + 1):
            df_dln_redge[m, j] = dfenergy_dln_r_edge(
                m, j, state, v_old, alpha, dm, dt, n_shell
            )

        for k in range(0, n_shell):
            df_du[m, k] = dfenergy_du(m, k, state, v_old, alpha, dm, dt, n_shell)

    jac_energy = np.column_stack((df_dln_redge[:, 1:-1], df_du))
    return jac_energy


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def jacobian(state, m_edge, v_old, alpha, dm, dt, n_shell):
    """Jacobian for the coupled residual."""
    jac_hydro = jacobian_hydro_block(state, m_edge, n_shell)
    jac_energy = jacobian_energy_block(state, v_old, alpha, dm, dt, n_shell)

    jac = np.row_stack((jac_hydro, jac_energy))
    return jac
