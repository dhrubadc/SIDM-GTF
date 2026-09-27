r"""
Calculate SIDM thermal conductivities.
"""

import numpy as np
from . import settings


def smfp_conductivity(u):
    r"""
    Calculate the short mean free path conductivity.

    .. math::

       \kappa_s = \frac{3}{2}
       \frac{b}{a}
       (\frac{2}{3} u)^{1/2}
       \frac{1}{\sigma_{m}^2}

    Here :math:`b` is :obj:`gtf.global.B`,
    :math:`a` is :obj:`gtf.global.A`,
    and :math:`\sigma_{m}` is :obj:`gtf.global.SIGMA_OVER_M`.

    :param u: specific energy at cell midpoints
    :type u: np.ndarray

    :return: SMFP conductivity at cell midpoints
    :rtype: np.ndarray
    """
    v = np.sqrt((settings.FLOATDTYPE(2.0) / settings.FLOATDTYPE(3.0)) * u)
    return (
        settings.FLOATDTYPE(1.5)
        * (settings.B / settings.A)
        * v
        / settings.SIGMA_OVER_M**2
    )


def lmfp_conductivity(rho, u):
    r"""
    Calculate the long mean free path conductivity.

    .. math::

      \kappa_l = \frac{3}{2}
      \beta
      \rho
      (\frac{2}{3} u)^{3/2}

    Here :math:`\beta` is :obj:`gtf.settings.BETA`.

    :param rho: density at cell midpoints
    :type rho: np.ndarray

    :param u: specific energy at cell midpoints
    :type u: np.ndarray

    :return: LMFP conductivity at cell midpoints
    :rtype: np.ndarray
    """
    v = np.sqrt((settings.FLOATDTYPE(2.0) / settings.FLOATDTYPE(3.0)) * u)
    return settings.FLOATDTYPE(1.5) * settings.BETA * rho * v**3


def total_conductivity(kappa_s, kappa_l):
    r"""
    Calculate total conductivity.

    .. math::

      \kappa = \frac{\kappa_s \kappa_l}
      {(\kappa_s^{\alpha} + \kappa_l^{\alpha})^{1/\alpha}}

    Here :math:`\alpha` is :obj:`gtf.settings.ALPHA`.

    :param kappa_s: SMFP conductivity at cell midpoints
    :type kappa_s: np.ndarray

    :param kappa_l: LMFP conductivity at cell midpoints
    :type kappa_l: np.ndarray

    :return: total conductivity at cell midpoints
    :rtype: np.ndarray
    """
    num = kappa_s * kappa_l
    den = (kappa_s ** (settings.ALPHA) + kappa_l ** (settings.ALPHA)) ** (
        1.0 / settings.ALPHA
    )
    return num / den
