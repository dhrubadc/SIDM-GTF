"""
Module to calculate the analytical jacobian
corresponding to the coupled residual
for the current state of the system
in dimensionless units (Nishikawa 2020).
"""


import numpy as np
from . import constants


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


def gravity_term(i, state):
    """
    Self gravity dependent term of the hydro residual.
    """
    # pylint: disable=unsubscriptable-object

    return (
        constants.FLOATDTYPE(3.0)
        * constants.M_EDGE[i + 1]
        / (state["r_edge"][i + 1] * (state["u"][i] + state["u"][i + 1]))
    )


def kappa_derivative_common(i, state):
    """
    Common coefficient for the kappa derivatives.
    """
    return state["kappa_s"][i] ** (constants.ALPHA) / (
        state["kappa_s"][i] ** constants.ALPHA + state["kappa_l"][i] ** constants.ALPHA
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


def dfhydro_dln_redge(i, j, state):
    """
    Derivative of hydro residual
    with respect to ln_r_edge.
    """
    dln_r = diff_quant(i, state["ln_r"])
    dln_rho = diff_quant(i, state["ln_rho"]) + diff_quant(i, state["ln_u"])
    g_term = gravity_term(i, state)

    term_1 = (dln_v_dln_redge(i, j, state) - dln_v_dln_redge(i + 1, j, state)) / dln_r
    term_2 = (
        dln_rho
        * (dln_r_dln_redge(i + 1, j, state) - dln_r_dln_redge(i, j, state))
        / dln_r**2.0
    )
    term_3 = g_term * delta(i + 1, j)

    return term_1 - term_2 - term_3


def dfhydro_du(i, k, state):
    """
    Derivative of hydro residual
    with respect to u.
    """
    dln_r = diff_quant(i, state["ln_r"])
    g_term = gravity_term(i, state)

    term_1 = (dln_u_du(i + 1, k, state) - dln_u_du(i, k, state)) / dln_r
    term_2 = (
        g_term / (state["u"][i] + state["u"][i + 1]) * (delta(i, k) + delta(i + 1, k))
    )

    return term_1 - term_2


def dkappa_dln_redge(i, j, state):
    """
    Derivative of kappa
    with respect to ln_redge.
    """
    kappa_common = kappa_derivative_common(i, state)
    return -state["kappa"][i] * kappa_common * dln_v_dln_redge(i, j, state)


def dkappa_du(i, k, state):
    """
    Derivative of kappa
    with respect to u.
    """
    kappa_common = kappa_derivative_common(i, state)
    return (
        state["kappa"][i]
        / state["u"][i]
        * (constants.FLOATDTYPE(1.0 / 2.0) + kappa_common)
        * delta(i, k)
    )


def dl_dln_redge(i, j, state):
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

    term_3 = dkappa_dln_redge(i, j, state)
    term_4 = dkappa_dln_redge(i + 1, j, state)

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


def dl_du(i, k, state):
    """
    Derivative of l at the interior edges
    with respect to u.
    """
    dr = diff_quant(i, state["r"])
    du = diff_quant(i, state["u"])

    kappa_face = constants.FLOATDTYPE(0.5) * (state["kappa"][i] + state["kappa"][i + 1])

    term_1 = kappa_face * (delta(i + 1, k) - delta(i, k))

    term_2 = dkappa_du(i, k, state)
    term_3 = dkappa_du(i + 1, k, state)

    term_4 = du * constants.FLOATDTYPE(0.5) * (term_2 + term_3)

    return (
        -constants.FLOATDTYPE(2.0)
        / constants.FLOATDTYPE(3.0)
        * state["r_edge"][i + 1] ** 2.0
        / dr
        * (term_1 + term_4)
    )


def dfenergy_dln_r_edge(m, j, state, v_old, dt):
    """
    Derivative of energy residual
    with respect to ln_r_edge.
    """
    # pylint: disable=unsubscriptable-object

    z = volume_ratio(m, state, v_old)

    term_1 = (
        constants.FLOATDTYPE(2.0)
        / constants.FLOATDTYPE(3.0)
        * state["u"][m]
        * z
        * dln_v_dln_redge(m, j, state)
    )

    if m == 0:
        term_2 = dt / constants.DM[m] * dl_dln_redge(m, j, state)
    elif m == constants.N_SHELL - 1:
        term_2 = -dt / constants.DM[m] * dl_dln_redge(m - 1, j, state)
    else:
        term_2 = (
            dt
            / constants.DM[m]
            * (dl_dln_redge(m, j, state) - dl_dln_redge(m - 1, j, state))
        )

    return term_1 + term_2


def dfenergy_du(m, k, state, v_old, dt):
    """
    Derivative of energy residual
    with respect to u.
    """
    # pylint: disable=unsubscriptable-object

    z = volume_ratio(m, state, v_old)

    term_1 = (
        constants.FLOATDTYPE(1.0)
        + constants.FLOATDTYPE(2.0)
        / constants.FLOATDTYPE(3.0)
        * (constants.FLOATDTYPE(1.0) - z)
    ) * delta(m, k)

    if m == 0:
        term_2 = dt / constants.DM[m] * dl_du(m, k, state)
    elif m == constants.N_SHELL - 1:
        term_2 = -dt / constants.DM[m] * dl_du(m - 1, k, state)
    else:
        term_2 = dt / constants.DM[m] * (dl_du(m, k, state) - dl_du(m - 1, k, state))

    return term_1 + term_2


def jacobian_hydro_block(state):
    """
    Hydro block of the Jacobian
    with fixed edges at ln_r_edge[0] and ln_r_edge[-1].
    df_dln_redge is tri-diagonal and df_du is bi-diagonal.
    """

    df_dln_redge = np.zeros(
        (constants.N_SHELL - 1, constants.N_SHELL + 1), dtype=constants.FLOATDTYPE
    )
    df_du = np.zeros(
        (constants.N_SHELL - 1, constants.N_SHELL), dtype=constants.FLOATDTYPE
    )

    for i in range(0, constants.N_SHELL - 1):

        for j in (i, i + 1, i + 2):
            if j < constants.N_SHELL + 1:
                df_dln_redge[i, j] = dfhydro_dln_redge(i, j, state)

        for k in (i, i + 1):
            if k < constants.N_SHELL:
                df_du[i, k] = dfhydro_du(i, k, state)

    jac_hydro = np.hstack([df_dln_redge[:, 1:-1], df_du])
    return jac_hydro


def jacobian_energy_block(state, v_old, dt):
    """
    Energy block of Jacobian
    with fixed edges at ln_r_edge[0] and ln_r_edge[-1].
    df_dln_redge is four-diagonal and df_du is tri-diagonal.
    """

    df_dln_redge = np.zeros(
        (constants.N_SHELL, constants.N_SHELL + 1), dtype=constants.FLOATDTYPE
    )
    df_du = np.zeros((constants.N_SHELL, constants.N_SHELL), dtype=constants.FLOATDTYPE)

    for m in range(0, constants.N_SHELL):

        for j in (m - 1, m, m + 1, m + 2):
            if 0 <= j < constants.N_SHELL + 1:
                df_dln_redge[m, j] = dfenergy_dln_r_edge(m, j, state, v_old, dt)

        for k in (m - 1, m, m + 1):
            if 0 <= k < constants.N_SHELL:
                df_du[m, k] = dfenergy_du(m, k, state, v_old, dt)

    jac_energy = np.hstack([df_dln_redge[:, 1:-1], df_du])
    return jac_energy


def analytic_jacobian(state, v_old, dt):
    """
    Analytical jacobian for the coupled residual.
    """
    jac_hydro = jacobian_hydro_block(state)
    jac_energy = jacobian_energy_block(state, v_old, dt)

    jac = np.vstack([jac_hydro, jac_energy])
    return jac
