"""
Module to construct a fixed radial grid
in dimensionless units (Nishikawa 2020).
"""

import numpy as np
import constants


def make_radial_grid(r_first):
    """
    Return a radial grid in log-space
    with ln_r_edge[0] = -np.inf and
    ln_r_edge[1:] = linspace(ln_r_first, ln_r_outer, n_shell).
    """
    ln_r_edge = np.empty(constants.N_SHELL + 1, dtype=constants.FLOATDTYPE)

    ln_r_edge[0] = -np.inf

    ln_r_edge[1:] = np.linspace(
        np.log(r_first), np.log(constants.R_OUTER), constants.N_SHELL
    )

    return ln_r_edge


def log_cell_centers(ln_r_edge):
    """
    Return log of cell midpoints from logarithmic edges.
    """
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]

    return np.logaddexp(ln_right, ln_left) - np.log(constants.FLOATDTYPE(2.0))


def log_shell_volumes(ln_r_edge):
    """
    Return log of shell volumes from logarithmic edges.
    """
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]
    delta = constants.FLOATDTYPE(3.0) * (ln_left - ln_right)

    return (
        constants.FLOATDTYPE(3.0) * ln_right
        + np.log(-np.expm1(delta))
        - np.log(constants.FLOATDTYPE(3.0))
    )
