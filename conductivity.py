"""
Module to calculate SIDM thermal conductivities
in dimensionless units (Nishikawa 2020).
"""

import numpy as np
import constants


def smfp_conductivity(u, sigma_over_m):
    """
    Short mean free path conductivity.
    """
    v = np.sqrt((constants.FLOATDTYPE(2.0) / constants.FLOATDTYPE(3.0)) * u)
    return constants.FLOATDTYPE(1.5) * (constants.b / constants.a) * v / sigma_over_m**2


def lmfp_conductivity(rho, u, beta):
    """
    Long mean free path conductivity.
    """
    v = np.sqrt((constants.FLOATDTYPE(2.0) / constants.FLOATDTYPE(3.0)) * u)
    return constants.FLOATDTYPE(1.5) * beta * rho * v**3


def total_conductivity(kappa_s, kappa_l, alpha):
    """
    Total conductivity.
    """
    num = kappa_s * kappa_l
    den = (kappa_s ** (alpha) + kappa_l ** (alpha)) ** (1.0 / alpha)
    return num / den
