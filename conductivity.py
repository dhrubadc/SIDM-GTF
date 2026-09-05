"""
Module to calculate SIDM thermal conductivity.
"""

import numpy as np
import constants


def smfp_conductivity(u, sigma_over_m):
    """
    Short mean free path conductivity
    """
    v = np.sqrt((constants.FLOAT_DTYPE(2.0) / constants.FLOAT_DTYPE(3.0)) * u)
    return (
        constants.FLOAT_DTYPE(1.5) * (constants.b / constants.a) * v / sigma_over_m**2
    )


def lmfp_conductivity(rho, u, C):
    """
    Long mean free path conductivity
    """
    v = np.sqrt((constants.FLOAT_DTYPE(2.0) / constants.FLOAT_DTYPE(3.0)) * u)
    return constants.FLOAT_DTYPE(1.5) * C * rho * v**3


def total_conductivity(rho, u, sigma_over_m, C, alpha):
    """
    Total conductivity
    """
    k_smfp = smfp_conductivity(u, sigma_over_m)
    k_lmfp = lmfp_conductivity(rho, u, C)

    num = k_smfp * k_lmfp
    den = (k_smfp ** (alpha) + k_lmfp ** (alpha)) ** (1.0 / alpha)
    return num / den
