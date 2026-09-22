r"""
Calculate the fully implicit coupled hydrostatic and energy residuals.
"""

import numpy as np
from . import constants


def hydrostatic_terms(state):
    r"""
    Calculate individual terms of each
    hydrostatic residual.

    :param state: current physical state of the system
    :type state: dict

    :return: individual terms of each hydrostatic residual
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
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
    r"""
    Calculate the raw hydrostatic residuals defined at each interior edge.

    .. math ::

       F_{\rm H} &= \frac{{\rm d} \ln \rho}{{\rm d} \ln r} +
       \frac{{\rm d} \ln u}{{\rm d} \ln r}
       + \frac{3}{2} \frac{m_{\rm edge}[1:-1]}{r_{\rm edge}[1:-1]\ u_{\rm face}}

       u_{\rm face} &= \frac{1}{2} (u[:-1] + u[1:])

    Here :math:`m_{\rm edge}` is :obj:`gtf.constants.M_EDGE`.

    :param state: current physical state of the system
    :type state: dict

    :return: raw hydrostatic residuals
    :rtype: np.ndarray
    """
    rho_slope, u_slope, g_term = hydrostatic_terms(state)

    fhydro = rho_slope + u_slope + g_term

    return fhydro


def energy_terms(state, v_old, u_old, dt):
    r"""
    Calculate individual terms of each
    energy residual.

    :param state: current physical state of the system
    :type state: dict

    :param v_old: cell volumes at time t-dt
    :type v_old: np.ndarray

    :param u_old: cell specific energies at time t-dt
    :type u_old: np.ndarray

    :param dt: timestep
    :type dt: :obj:`gtf.constants.FLOATDTYPE`

    :return: individual terms of each energy residual
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
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
    r"""
    Calculate the raw energy residuals defined at each cell midpoint.

    .. math ::

       F_{\rm E} = u - u_{\rm old}
       + {\rm d}t\ \frac{{\rm d} l}{{\rm d} m} +
       \frac{2}{3}\ u\ (1 - \frac{v_{\rm old}}{v})

    Here dm is :obj:`gtf.constants.DM`.

    :param state: current physical state of the system
    :type state: dict

    :param v_old: cell volumes at time t-dt
    :type v_old: np.ndarray

    :param u_old: cell specific energies at time t-dt
    :type u_old: np.ndarray

    :param dt: timestep
    :type dt: :obj:`gtf.constants.FLOATDTYPE`

    :return: raw energy residuals
    :rtype: np.ndarray
    """
    energy_term, compression_term, conduction_term = energy_terms(
        state, v_old, u_old, dt
    )

    fenergy = energy_term + compression_term + conduction_term

    return fenergy


def residual(state, v_old, u_old, dt):
    r"""
    Combine the hydrostatic and energy residuals.

    :param state: current physical state of the system
    :type state: dict

    :param v_old: cell volumes at time t-dt
    :type v_old: np.ndarray

    :param u_old: cell specific energies at time t-dt
    :type u_old: np.ndarray

    :param dt: timestep
    :type dt: :obj:`gtf.constants.FLOATDTYPE`

    :return: combined raw residual
    :rtype: np.ndarray
    """
    fhydro = hydrostatic_residual(state)
    fenergy = energy_residual(state, v_old, u_old, dt)

    return np.concatenate((fhydro, fenergy))
