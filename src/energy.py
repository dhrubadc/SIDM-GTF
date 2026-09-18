"""
Module to calculate kinetic, potential, and total
energy of the system in dimensionless units (Nishikawa 2020).
"""

import numpy as np
import constants


def total_kinetic_energy(u):
    """
    Return total kinetic energy.
    """
    return np.sum(constants.DM * u)


def total_gravitational_energy(r):
    """
    Return total potential energy.
    """
    # pylint: disable=unsubscriptable-object

    m_center = constants.FLOATDTYPE(0.5) * (
        constants.M_EDGE[:-1] + constants.M_EDGE[1:]
    )
    return -np.sum(constants.DM * m_center / r)


def total_energy(r, u):
    """
    Return total energy.
    """
    ke = total_kinetic_energy(u)
    pe = total_gravitational_energy(r)
    return ke + pe
