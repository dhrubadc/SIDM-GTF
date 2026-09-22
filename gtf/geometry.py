"""
Calculate geometric quantities.
"""

import numpy as np
from . import constants


def log_cell_centers(ln_r_edge):
    r"""
    Calculate log of cell midpoints from logarithmic edges.

    .. math::

       \ln r =
       \ln\left(
       e^{\ln r_{\rm edge}[1:]}
       + e^{\ln r_{\rm edge}[:-1]}
       \right) - \ln 2

    :param ln_r_edge: log of cell edges
    :type ln_r_edge: 1D array of shape :obj:`src.constants.N_SHELL` + 1

    :return: log of cell midpoints
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL`
    """
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]

    return np.logaddexp(ln_right, ln_left) - np.log(constants.FLOATDTYPE(2.0))


def log_cell_volumes(ln_r_edge):
    r"""
    Calculate log of cell volumes from logarithmic edges.

    .. math::

       \ln v =
       \ln\left(
       e^{\ln r^3_{\rm edge}[1:]}
       - e^{\ln r^3_{\rm edge}[:-1]}
       \right) - \ln 3

    :param ln_r_edge: log of cell edges
    :type ln_r_edge: 1D array of shape :obj:`src.constants.N_SHELL` + 1

    :return: log of cell volumes
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL`
    """
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]
    delta = constants.FLOATDTYPE(3.0) * (ln_left - ln_right)

    return (
        constants.FLOATDTYPE(3.0) * ln_right
        + np.log(-np.expm1(delta))
        - np.log(constants.FLOATDTYPE(3.0))
    )
