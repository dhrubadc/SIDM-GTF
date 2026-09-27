"""
Calculate kinetic, potential, and total
energy.
"""

import numpy as np
from . import settings


def total_kinetic_energy(u):
    r"""
    Calculate total kinetic energy.

    .. math::

       K = \Sigma ({\rm d} m\ u)

    Here :math:`{\rm d} m` is :obj:`gtf.settings.DM`.

    :param u: specific energy at cell midpoints
    :type u: np.ndarray

    :return: total kinetic energy
    :rtype: :obj:`gtf.settings.FLOATDTYPE`
    """
    return np.sum(settings.DM * u)


def total_gravitational_energy(r):
    r"""
    Calculate total potential energy.

    .. math::

       W &= \Sigma \frac{{\rm d} m\ M_{\rm center}}{r}

       M_{\rm center} &= 0.5\ (M_{\rm edge}[0:-1] + M_{\rm edge}[1:])

    Here :math:`{\rm d} m` is :obj:`gtf.settings.DM` and
    :math:`M_{\rm edge}` is :obj:`gtf.settings.M_EDGE`.

    :param r: cell midpoints
    :type r: np.ndarray

    :return: total potential energy
    :rtype: :obj:`gtf.settings.FLOATDTYPE`
    """
    # pylint: disable=unsubscriptable-object

    m_center = settings.FLOATDTYPE(0.5) * (settings.M_EDGE[:-1] + settings.M_EDGE[1:])
    return -np.sum(settings.DM * m_center / r)


def total_energy(r, u):
    r"""
    Calculate total energy.

    .. math::

       E = K + W

    :param r: cell midpoints
    :type r: np.ndarray

    :param u: specific energy at cell midpoints
    :type u: np.ndarray

    :return: total energy
    :rtype: :obj:`gtf.settings.FLOATDTYPE`
    """
    ke = total_kinetic_energy(u)
    pe = total_gravitational_energy(r)
    return ke + pe
