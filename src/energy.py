"""
Calculate kinetic, potential, and total
energy.
"""

import numpy as np
from . import constants


def total_kinetic_energy(u):
    r"""
    Calculate total kinetic energy.

    .. math::

       K = \Sigma ({\rm d} m\ u)

    Here :math:`{\rm d} m` is :obj:`src.constants.DM`.

    :param u: specific energy at cell midpoints
    :type u: 1D array of shape :obj:`src.constants.N_SHELL`

    :return: total kinetic energy
    :rtype: float
    """
    return np.sum(constants.DM * u)


def total_gravitational_energy(r):
    r"""
    Calculate total potential energy.

    .. math::

       W &= \Sigma \frac{{\rm d} m\ M_{\rm center}}{r}

       M_{\rm center} &= 0.5\ (M_{\rm edge}[0:-1] + M_{\rm edge}[1:])

    Here :math:`{\rm d} m` is :obj:`src.constants.DM` and
    :math:`M_{\rm edge}` is :obj:`src.constants.M_EDGE`.

    :param r: cell midpoints
    :type r: 1D array of shape :obj:`src.constants.N_SHELL`

    :return: total potential energy
    :rtype: float
    """
    # pylint: disable=unsubscriptable-object

    m_center = constants.FLOATDTYPE(0.5) * (
        constants.M_EDGE[:-1] + constants.M_EDGE[1:]
    )
    return -np.sum(constants.DM * m_center / r)


def total_energy(r, u):
    r"""
    Calculate total energy.

    .. math::

       E = K + W

    :param r: cell midpoints
    :type r: 1D array of shape :obj:`src.constants.N_SHELL`

    :param u: specific energy at cell midpoints
    :type u: 1D array of shape :obj:`src.constants.N_SHELL`

    :return: total energy
    :rtype: float
    """
    ke = total_kinetic_energy(u)
    pe = total_gravitational_energy(r)
    return ke + pe
