r"""
Calculate SIDM thermal conductivities
in dimensionless units.
"""

import numpy as np
from . import constants


def smfp_conductivity(u):
    r"""
    Calculate the short mean free path conductivity.

    :math:`\kappa_s = \frac{3}{2} \frac{B}{A} (\frac{2}{3} u)^{1/2} \frac{1}{\sigma_{m}^2}`

    :param u: specific energy
    :type u: float or array-like

    :return: SMFP conductivity
    :rtype: same as u
    """
    v = np.sqrt((constants.FLOATDTYPE(2.0) / constants.FLOATDTYPE(3.0)) * u)
    return (
        constants.FLOATDTYPE(1.5)
        * (constants.B / constants.A)
        * v
        / constants.SIGMA_OVER_M**2
    )


def lmfp_conductivity(rho, u):
    r"""
    Calculate the long mean free path conductivity.

    :math:`\kappa_l = \frac{3}{2} \beta \rho (\frac{2}{3} u)^{3/2}`

    :param rho: density
    :type rho: float or array-like

    :param u: specific energy
    :type u: float or array-like

    :return: LMFP conductivity
    :rtype: same as rho and u
    """
    v = np.sqrt((constants.FLOATDTYPE(2.0) / constants.FLOATDTYPE(3.0)) * u)
    return constants.FLOATDTYPE(1.5) * constants.BETA * rho * v**3


def total_conductivity(kappa_s, kappa_l):
    r"""
    Calculate total conductivity.

    :math:`\kappa = \frac{\kappa_s \kappa_l}{(\kappa_s^{\alpha} + \kappa_l^{\alpha})^{1/\alpha}}`

    :param kappa_s: smfp conductivity
    :type kappa_s: float or array-like

    :param kappa_l: lmfp conductivity
    :type kappa_l: float or array-like

    :return: total conductivity
    :rtype: same as kappa_s and kappa_l
    """
    num = kappa_s * kappa_l
    den = (kappa_s ** (constants.ALPHA) + kappa_l ** (constants.ALPHA)) ** (
        1.0 / constants.ALPHA
    )
    return num / den
