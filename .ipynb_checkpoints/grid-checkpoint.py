"""Module to construct a fixed radial grid
"""
import numpy as np
from constants import FLOAT_DTYPE


def make_radial_grid(r_first, r_max, N_shell):
    """
    Return a logarithmically-spaced radial grid 
    centered at the origin.

    r_edge[0] = 0
    r_edge[1:] = geomspace(r_first, r_max, N_shell).
    """

    r_edge = np.empty(Nshell + 1, dtype=FLOAT_DTYPE)
    r_edge[0] = FLOAT_DTYPE(0.0)
    r_edge[1:] = np.geomspace(r_first, r_max, N_shell)

    return r_edge

def cell_centers(r_edge):
    """Return cell-centered radii from linear edges, 
    r = (r_right+r_left) / 2.
    """

    return FLOAT_DTYPE(0.5) * (r_edge[:-1] + r_edge[1:])

def shell_volume(r_edge):
    """
    Return V=(r_right**3-r_left**3)/3 from linear edges
    without subtracting cubes. The factorized form is 
    more accurate when a shell becomes narrow.
    """

    r_left = r_edge[:-1]
    r_right = r_edge[1:]

    delta = (r_right - r_left)

    if np.any(delta <= FLOAT_DTYPE(0.0)):
        raise ValueError("Shell edges must be strictly increasing.")

    return (
        delta
        * (
            r_right**2
            + r_right * r_left
            + r_left**2
        )
        / FLOAT_DTYPE(3.0)
    )

def log_cell_centers(ln_r_edge):
    """Return natural log of 
    cell-centered radii from logarithmic edges.
    """
    
    return (
        np.logaddexp(ln_r_edge[:-1], ln_r_edge[1:])
        - np.log(FLOAT_DTYPE(2.0))
    )

def log_shell_volume(ln_r_edge):
    """
    Return ln[(r_right**3-r_left**3)/3] from logarithmic edges.

    ``ln_r_edge[0]`` may be ``-inf`` for the origin.  ``expm1`` avoids
    subtracting nearly equal cubes when a Lagrangian shell becomes thin.
    """
    
    ln_left = ln_r_edge[:-1]
    ln_right = ln_r_edge[1:]
    delta = FLOAT_DTYPE(3.0) * (ln_left - ln_right)

    if np.any(delta >= FLOAT_DTYPE(0.0)):
        raise ValueError("Shell edges must be strictly increasing.")

    return (
        FLOAT_DTYPE(3.0) * ln_right
        + np.log(-np.expm1(delta))
        - np.log(FLOAT_DTYPE(3.0))
    )


