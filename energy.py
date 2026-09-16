"""
Module to calculate kinetic, potential, and total
energy of the system in dimensionless units (Nishikawa 2020).
"""

import numpy as np
import constants


def total_kinetic_energy(u, dm):
    """
    Return total kinetic energy.
    """
    return np.sum(dm * u)


def total_gravitational_energy(r, dm, m_edge):
    """
    Return total potential energy.
    """
    m_center = constants.FLOATDTYPE(0.5) * (m_edge[:-1] + m_edge[1:])
    return -np.sum(dm * m_center / r)


def total_energy(r, u, dm, m_edge):
    """
    Return total energy.
    """
    ke = total_kinetic_energy(u, dm)
    pe = total_gravitational_energy(r, dm, m_edge)
    return ke + pe
