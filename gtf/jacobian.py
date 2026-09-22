r"""
Calculate the analytical jacobian matrix
corresponding to the fully implicit coupled residuals.
"""

import numpy as np
from . import constants


def diff_quant(i, quant):
    r"""
    Difference of a cell quantity
    between consecutive cells.

    :param i: index
    :type i: int

    :param quant: physical quantity defined at cell midpoints
    :type quant: np.ndarray
    """
    return quant[i + 1] - quant[i]


def delta(m, n):
    r"""
    Kronecker delta function.

    :param m: index
    :type m: int

    :param n: index
    :type n: int

    :return: 1 if m=n, 0 otherwise
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    if n == m:
        return constants.FLOATDTYPE(1.0)

    return constants.FLOATDTYPE(0.0)


def volume_ratio(m, state, v_old):
    r"""
    Ratio of cell volume at previous timestep
    and current cell volume.

    :param m: index
    :type m: int

    :param state: current physical state of the system
    :type state: dict

    :param v_old: cell volumes at time t-dt
    :type v_old: np.ndarray

    :return: volume ratio (old to current)
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    return v_old[m] / state["v"][m]


def gravity_term(i, state):
    r"""
    Self gravity dependent term of the hydrostatic residual.

    :param i: index
    :type i: int

    :param state: current physical state of the system
    :type state: dict

    :return: gravity term for hydrostatic residual
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    # pylint: disable=unsubscriptable-object

    return (
        constants.FLOATDTYPE(3.0)
        * constants.M_EDGE[i + 1]
        / (state["r_edge"][i + 1] * (state["u"][i] + state["u"][i + 1]))
    )


def kappa_derivative_common(i, state):
    r"""
    Common coefficient for kappa derivatives.

    :param i: index
    :type i: int

    :param state: current physical state of the system
    :type state: dict

    :return: common term for kappa derivatives
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    return state["kappa_s"][i] ** (constants.ALPHA) / (
        state["kappa_s"][i] ** constants.ALPHA + state["kappa_l"][i] ** constants.ALPHA
    )


def dln_r_dln_redge(i, j, state):
    r"""
    Calculate derivative of :math:`\ln\ r`
    with respect to :math:`\ln\ r_{\rm edge}`.

    :param i: index
    :type i: int

    :param j: index
    :type j: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of :math:`\ln\ r` with respect to :math:`\ln\ r_{\rm edge}`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    den = state["r_edge"][i] + state["r_edge"][i + 1]
    num = state["r_edge"][i] * delta(i, j) + state["r_edge"][i + 1] * delta(i + 1, j)
    return num / den


def dr_dln_redge(i, j, state):
    r"""
    Calculate derivative of :math:`r`
    with respect to :math:`\ln\ r_{\rm edge}`.

    :param i: index
    :type i: int

    :param j: index
    :type j: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of :math:`r` with respect to :math:`\ln\ r_{\rm edge}`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    return constants.FLOATDTYPE(0.5) * (
        state["r_edge"][i] * delta(i, j) + state["r_edge"][i + 1] * delta(i + 1, j)
    )


def dln_v_dln_redge(i, j, state):
    r"""
    Calculate derivative of :math:`\ln\ v`
    with respect to :math:`\ln\ r_{\rm edge}`.

    :param i: index
    :type i: int

    :param j: index
    :type j: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of :math:`\ln\ v` with respect to :math:`\ln\ r_{\rm edge}`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    den = state["r_edge"][i + 1] ** 3.0 - state["r_edge"][i] ** 3.0

    num = constants.FLOATDTYPE(3.0) * (
        state["r_edge"][i + 1] ** 3.0 * delta(i + 1, j)
        - state["r_edge"][i] ** 3.0 * delta(i, j)
    )
    return num / den


def dln_u_du(i, k, state):
    r"""
    Derivative of :math:`\ln\ u`
    with respect to :math:`u`.

    :param i: index
    :type i: int

    :param k: index
    :type k: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of :math:`\ln\ u` with respect to :math:`u`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    return delta(i, k) / state["u"][i]


def dfhydro_dln_redge(i, j, state):
    r"""
    Calculate derivative of the hydrostatic residual
    with respect to :math:`\ln\ r_{\rm edge}`.

    :param i: index
    :type i: int

    :param j: index
    :type j: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of hydrostatic residual with respect to :math:`\ln\ r_{\rm edge}`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
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
    r"""
    Derivative of the hydrostatic residual
    with respect to :math:`u`.

    :param i: index
    :type i: int

    :param k: index
    :type k: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of hydrostatic residual with respect to :math:`u`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    dln_r = diff_quant(i, state["ln_r"])
    g_term = gravity_term(i, state)

    term_1 = (dln_u_du(i + 1, k, state) - dln_u_du(i, k, state)) / dln_r
    term_2 = (
        g_term / (state["u"][i] + state["u"][i + 1]) * (delta(i, k) + delta(i + 1, k))
    )

    return term_1 - term_2


def dkappa_dln_redge(i, j, state):
    r"""
    Calculate derivative of :math:`\kappa`
    with respect to :math:`\ln\ r_{\rm edge}`.

    :param i: index
    :type i: int

    :param j: index
    :type j: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of :math:`\kappa` with respect to :math:`\ln\ r_{\rm edge}`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    kappa_common = kappa_derivative_common(i, state)
    return -state["kappa"][i] * kappa_common * dln_v_dln_redge(i, j, state)


def dkappa_du(i, k, state):
    r"""
    Calculate derivative of :math:`\kappa`
    with respect to :math:`u`.

    :param i: index
    :type i: int

    :param k: index
    :type k: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of :math:`\kappa` with respect to :math:`u`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
    """
    kappa_common = kappa_derivative_common(i, state)
    return (
        state["kappa"][i]
        / state["u"][i]
        * (constants.FLOATDTYPE(1.0 / 2.0) + kappa_common)
        * delta(i, k)
    )


def dl_dln_redge(i, j, state):
    r"""
    Calculate derivative of :math:`l`
    with respect to :math:`\ln\ r_{\rm edge}`.

    :param i: index
    :type i: int

    :param j: index
    :type j: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of :math:`l` with respect to :math:`\ln\ r_{\rm edge}`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
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
    r"""
    Calculate derivative of :math:`l`
    with respect to :math:`u`

    :param i:  index
    :type i: int

    :param k: index
    :type k: int

    :param state: current physical state of the system
    :type state: dict

    :return: derivative of :math:`l` with respect to :math:`u`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
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
    r"""
    Calculate derivative of the energy residual
    with respect to :math:`\ln\ r_{\rm edge}`.

    :param m: index
    :type m: int

    :param j: index
    :type j: int

    :param state: current physical state of the system
    :type state: dict

    :param v_old: cell volumes at time t-dt
    :type v_old: np.ndarray

    :param dt: timestep
    :type dt: :obj:`gtf.constants.FLOATDTYPE`

    :return: derivative of energy residual with respect to :math:`\ln\ r_{\rm edge}`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
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
    r"""
    Calculate derivative of the energy residual
    with :math:`u`.

    :param m: index
    :type m: int

    :param k: index
    :type k: int

    :param state: current physical state of the system
    :type state: dict

    :param v_old: cell volumes at time t-dt
    :type v_old: np.ndarray

    :param dt: timestep
    :type dt: :obj:`gtf.constants.FLOATDTYPE`

    :return: derivative of energy residual with respect to :math:`u`
    :rtype: :obj:`gtf.constants.FLOATDTYPE`
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
    r"""
    Hydrostatic block of the jacobian matrix.
    Fixed edges at ln_r_edge[0] and ln_r_edge[-1].
    df_dln_redge is tri-diagonal and df_du is bi-diagonal.

    :param state: current physical state of the system
    :type state: dict

    :return: hydrostatic block of the analytical jacobian matrix
    :rtype: np.ndarray
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
    r"""
    Energy block of the jacobian matrix.
    Fixed edges at ln_r_edge[0] and ln_r_edge[-1].
    df_dln_redge is four-diagonal and df_du is tri-diagonal.

    :param state: current physical state of the system
    :type state: dict

    :param v_old: cell volumes at time t-dt
    :type v_old: np.ndarray 
    
    :param dt: timestep
    :type dt: :obj:`gtf.constants.FLOATDTYPE`

    :return: energy block of the analytical jacobian matrix
    :rtype: np.ndarray
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
    r"""
    Analytical jacobian matrix corresponding to the coupled residuals.

    :param state: current physical state of the system
    :type state: dict

    :param v_old: cell volumes at time t-dt
    :type v_old: np.ndarray

    :param dt: timestep
    :type dt: :obj:`gtf.constants.FLOATDTYPE`

    :return: analytical jacobian matrix
    :rtype: np.ndarray
    """
    jac_hydro = jacobian_hydro_block(state)
    jac_energy = jacobian_energy_block(state, v_old, dt)

    jac = np.vstack([jac_hydro, jac_energy])
    return jac
