"""
Calculate kinetic, potential, and total
energy in dimensionless units.
"""

import numpy as np
from . import constants


def total_kinetic_energy(u):
    r"""
    Calculate total kinetic energy.

    :math:`K = \Sigma ({\rm DM}\ u)`

    :param u: specific energy
    :type u: float or array-like

    :return: total kinetic energy
    :rtype: same as u
    """
    return np.sum(constants.DM * u)


def total_gravitational_energy(r):
    r"""
    Calculate total potential energy.

    :math:`K = \Sigma \frac{{\rm DM}\ {\rm M}_{{\rm CENTER}}}{r}`

    M_CENTER = 0.5 * (M_EDGE[0:-1] + M_EDGE[1:])

    :param r: cell midpoint
    :type r: float or array-like

    :return: total potential energy
    :rtype: same as r
    """
    # pylint: disable=unsubscriptable-object

    m_center = constants.FLOATDTYPE(0.5) * (
        constants.M_EDGE[:-1] + constants.M_EDGE[1:]
    )
    return -np.sum(constants.DM * m_center / r)


def total_energy(r, u):
    r"""
    Calculate total energy.

    :math:`E = K + W`

    :param r: cell midpoint
    :type r: float or array-like

    :param u: specific energy
    :type u: float or array-like

    :return: total energy
    :rtype: same as r and u
    """
    ke = total_kinetic_energy(u)
    pe = total_gravitational_energy(r)
    return ke + pe
