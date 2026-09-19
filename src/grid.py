"""
Calculate grid related geometric quantities
in dimensionless units.
"""

import numpy as np
from . import constants


def make_radial_grid(r_first):
    """
    Build the initial radial grid in log-space.
    ln_r_edge[0] = -np.inf and
    ln_r_edge[1:] = linspace(ln_r_first, ln_r_outer, n_shell).

    :param r_first: first non-zero cell edge
    :type r_first: float 

    :return: initial radial grid
    :rtype: array-like
    """
    ln_r_edge = np.empty(constants.N_SHELL + 1, dtype=constants.FLOATDTYPE)

    ln_r_edge[0] = -np.inf

    ln_r_edge[1:] = np.linspace(
        np.log(r_first), np.log(constants.R_OUTER), constants.N_SHELL
    )

    return ln_r_edge


def log_cell_centers(ln_r_edge):
    """
    Calculate log of cell midpoints from logarithmic edges.

    :param ln_r_edge: log of cell edges 
    :type ln_r_edge: float or array-like

    :return: log of cell midpoints
    :rtype: float or array-like
    """
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]

    return np.logaddexp(ln_right, ln_left) - np.log(constants.FLOATDTYPE(2.0))


def log_cell_volumes(ln_r_edge):
    """
    Calculate log of shell volumes from logarithmic edges.

    :param ln_r_edge: log of cell edges 
    :type ln_r_edge: float or array-like

    :return: log of cell volumes
    :rtype: float or array-like
    """
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]
    delta = constants.FLOATDTYPE(3.0) * (ln_left - ln_right)

    return (
        constants.FLOATDTYPE(3.0) * ln_right
        + np.log(-np.expm1(delta))
        - np.log(constants.FLOATDTYPE(3.0))
    )
