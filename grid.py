"""
Module to construct a fixed radial grid
"""

import numpy as np
import constants


def make_radial_grid(r_first, r_max, N_shell):
    """
    Return a logarithmically-spaced radial grid
    centered at the origin.

    r_edge[0] = 0
    r_edge[1:] = geomspace(r_first, r_max, N_shell).
    """
    r_edge = np.empty(N_shell + 1, dtype=constants.FLOAT_DTYPE)
    r_edge[0] = constants.FLOAT_DTYPE(0.0)
    r_edge[1:] = np.geomspace(r_first, r_max, N_shell)

    return r_edge


def cell_centers(r_edge):
    """
    Return cell midpoints from linear edges,
    r = (r_right+r_left) / 2.
    """
    return constants.FLOAT_DTYPE(0.5) * (r_edge[:-1] + r_edge[1:])


def shell_volumes(r_edge):
    """
    Return shell volumes V=(r_right**3-r_left**3)/3
    from linear edges without subtracting cubes.
    The factorized form is more accurate when
    a shell becomes narrow.
    """
    r_left = r_edge[:-1]
    r_right = r_edge[1:]

    delta = r_right - r_left

    if np.any(delta <= constants.FLOAT_DTYPE(0.0)):
        raise ValueError("Shell edges must be strictly increasing.")

    return (
        delta * (r_right**2 + r_right * r_left + r_left**2) / constants.FLOAT_DTYPE(3.0)
    )


def log_cell_centers(ln_r_edge):
    """
    Return natural log of
    cell midpoints from logarithmic edges.
    """
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]

    return np.logaddexp(ln_right, ln_left) - np.log(constants.FLOAT_DTYPE(2.0))


def log_shell_volumes(ln_r_edge):
    """
    Return natural log of
    shell volumes from logarithmic edges.
    """
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]
    delta = constants.FLOAT_DTYPE(3.0) * (ln_left - ln_right)

    if np.any(delta >= constants.FLOAT_DTYPE(0.0)):
        raise ValueError("Shell edges must be strictly increasing.")

    return (
        constants.FLOAT_DTYPE(3.0) * ln_right
        + np.log(-np.expm1(delta))
        - np.log(constants.FLOAT_DTYPE(3.0))
    )
