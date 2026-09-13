import numpy as np
from constants import FLOAT_DTYPE


def delta(m, n):
    """Delta Function"""
    if n == m:
        return FLOAT_DTYPE(1.0)
    else:
        return FLOAT_DTYE(0.0)


def dln_r_dln_redge(i, j, r_edge):
    """Derivative of log of cell mid points
    with respect to log of cell edges
    as a function of r_edge.
    """
    den = r_edge[i] + r_edge[i + 1]
    num = r_edge[j] * (delta(j, i) + delta(j, i + 1))
    return num / den


def dln_V_dln_redge(i, j, r_edge):
    """Derivative of log of cell volume
    with respect to log of cell edges
    as a function of r_edge
    """
    den = r_edge[i + 1] ** 3.0 - r_edge[i] ** 3.0
    num = 3.0 * r_edge[j] ** 3.0 * (delta(j, i + 1) - delta(j, i))
    return num / den


def dln_u_du(i, k, u):
    """Detivative of log u with respect to u"""
    return delta_func(i, k) / u[i]


def dFHydro_dln_redge(i, j, ln_r_edge, u, state, M_edge):
    """Derivative of hydro residual
    with respect to ln_redge
    """
    ln_r = state["ln_r"]
    ln_rho = state["ln_rho"]
    ln_u = state["ln_u"]
    r_edge = state["r_edge"]

    D = ln_r[i + 1] - ln_r[i]
    A = (ln_rho[i + 1] - ln_rho[i]) + (ln_u[i + 1] - ln_u[i])
    g = 3.0 * M_edge[i + 1] / (r_edge[i + 1] * (u[i] + u[i + 1]))

    term_1 = (dln_V_dln_redge(i, j, r_edge) - dln_V_dln_redge(i + 1, j, r_edge)) / D
    term_2 = (
        A * (dln_r_dln_redge(i + 1, j, r_edge) - dln_r_dln_redge(i, j, r_edge)) / D**2.0
    )
    term_3 = g * delta(i + 1, j)

    return term_1 - term_2 - term_3


def dFhydro_du(i, k, ln_r_edge, u, state, M_edge):
    """Derivative of Hydro residual
    with respect to u
    """
    ln_r = state["ln_r"]
    r_edge = state["r_edge"]

    D = ln_r[i + 1] - ln_r[i]
    g = 3.0 * M_edge[i + 1] / (r_edge[i + 1] * (u[i] + u[i + 1]))

    term_1 = (dln_u_du(i + 1, k, u) - dln_u_du(i, k, u)) / D
    term_2 = g / (u[i] + u[i + 1]) * (delta(i, k) + delta(i + 1, k))

    return term_1 - term_2


def jacobian_hydro_block(ln_r_edge, u, state, M_edge, N_shell):
    """Hydro block of the Jacobian"""
    dF_dln_redge = np.full((N_shell - 1, N_shell - 1), np.nan, dtype=FLOAT_DTYPE)
    dF_du = np.full((N_shell - 1, N_shell), np.nan, dtype=FLOAT_DTYPE)

    for i in range(0, N_shell - 2):

        for j in range(0, N_shell - 2):
            dF_dln_redge[i, j] = dFHydro_dln_redge(i, j, ln_r_edge, u, state, M_edge)

        for k in range(0, N_shell - 1):
            dF_du[i, k] = dFhydro_du(i, k, ln_r_edge, u, state, M_edge)

    jac_hydro = np.column_stack((dF_dln_redge, dF_du))
    return dF_dln_redge, dF_du, jac_hydro
