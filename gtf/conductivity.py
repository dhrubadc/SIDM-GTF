r"""
Calculate SIDM thermal conductivities.
"""

import numpy as np
from . import constants


def smfp_conductivity(u):
    r"""
    Calculate the short mean free path conductivity.

    .. math::

       \kappa_s = \frac{3}{2}
       \frac{b}{a}
       (\frac{2}{3} u)^{1/2}
       \frac{1}{\sigma_{m}^2}

    Here :math:`b` is :obj:`src.constants.B`,
    :math:`a` is :obj:`src.constants.A`,
    and :math:`\sigma_{m}` is :obj:`src.constants.SIGMA_OVER_M`.

    :param u: specific energy at cell midpoints
    :type u: 1D array of shape :obj:`src.constants.N_SHELL`

    :return: SMFP conductivity at cell midpoints
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL`
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

    .. math::

      \kappa_l = \frac{3}{2}
      \beta
      \rho
      (\frac{2}{3} u)^{3/2}

    Here :math:`\beta` is :obj:`src.constants.BETA`.

    :param rho: density at cell midpoints
    :type rho: 1D array of shape :obj:`src.constants.N_SHELL`

    :param u: specific energy at cell midpoints
    :type u: 1D array of shape :obj:`src.constants.N_SHELL`

    :return: LMFP conductivity at cell midpoints
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL`
    """
    v = np.sqrt((constants.FLOATDTYPE(2.0) / constants.FLOATDTYPE(3.0)) * u)
    return constants.FLOATDTYPE(1.5) * constants.BETA * rho * v**3


def total_conductivity(kappa_s, kappa_l):
    r"""
    Calculate total conductivity.

    .. math::

      \kappa = \frac{\kappa_s \kappa_l}
      {(\kappa_s^{\alpha} + \kappa_l^{\alpha})^{1/\alpha}}

    Here :math:`\alpha` is :obj:`src.constants.ALPHA`.

    :param kappa_s: SMFP conductivity at cell midpoints
    :type kappa_s: 1D array of shape :obj:`src.constants.N_SHELL`

    :param kappa_l: LMFP conductivity at cell midpoints
    :type kappa_l: 1D array of shape :obj:`src.constants.N_SHELL`

    :return: total conductivity at cell midpoints
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL`
    """
    num = kappa_s * kappa_l
    den = (kappa_s ** (constants.ALPHA) + kappa_l ** (constants.ALPHA)) ** (
        1.0 / constants.ALPHA
    )
    return num / den
