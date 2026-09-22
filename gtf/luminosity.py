r"""
Calculate conductive luminosity.
"""

import numpy as np
from . import constants


def luminosity(r_edge, r, u, kappa):
    r"""
    Calculate luminosity at shell edges.

    .. math::

       l[1:-1] &=
       -\ \frac{2}{3}\,r_{\rm edge}[1:-1]^2\,
       \kappa_{\rm face}\,
       \frac{\Delta u}{\Delta r}

       \kappa_{\rm face} &= \frac{1}{2} (\kappa[:-1] + \kappa[1:])

    l[0] at r=0 and l[-1] at :math:`r=r_{\rm outer}` are fixed to 0.

    Here :math:`r_{\rm outer}` is :obj:`gtf.constants.R_OUTER`.

    :param r_edge: cell edges
    :type r_edge: np.ndarray

    :param r: cell midpoints
    :type r: np.ndarray

    :param u: specific energy at cell midpoints
    :type u: np.ndarray

    :param kappa: conductivity at cell midpoints
    :type kappa: np.ndarray

    :return: luminosity at cell edges
    :rtype: np.ndarray
    """
    l = np.zeros(constants.N_SHELL + 1, dtype=constants.FLOATDTYPE)

    kappa_face = constants.FLOATDTYPE(0.5) * (kappa[:-1] + kappa[1:])
    du = u[1:] - u[:-1]
    dr = r[1:] - r[:-1]

    l[1:-1] = (
        -constants.FLOATDTYPE(2.0 / 3.0) * r_edge[1:-1] ** 2 * kappa_face * du / dr
    )

    return l
