"""
Module to calculate SIDM thermal conductivities
in dimensionless units (Nishikawa 2020).
"""

import numpy as np
import constants


def smfp_conductivity(u):
    """
    Short mean free path conductivity.
    """
    v = np.sqrt((constants.FLOATDTYPE(2.0) / constants.FLOATDTYPE(3.0)) * u)
    return (
        constants.FLOATDTYPE(1.5)
        * (constants.B / constants.A)
        * v
        / constants.SIGMA_OVER_M**2
    )


def lmfp_conductivity(rho, u):
    """
    Long mean free path conductivity.
    """
    v = np.sqrt((constants.FLOATDTYPE(2.0) / constants.FLOATDTYPE(3.0)) * u)
    return constants.FLOATDTYPE(1.5) * constants.BETA * rho * v**3


def total_conductivity(kappa_s, kappa_l):
    """
    Total conductivity.
    """
    num = kappa_s * kappa_l
    den = (kappa_s ** (constants.ALPHA) + kappa_l ** (constants.ALPHA)) ** (
        1.0 / constants.ALPHA
    )
    return num / den
