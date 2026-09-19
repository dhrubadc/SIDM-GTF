"""
Module to construct current state of the system
from the primary unknowns ln_r_edge and u
in dimensionless units (Nishikawa 2020).
"""

import numpy as np
from . import conductivity
from . import luminosity
from . import grid
from . import constants


def state_from_unknowns(ln_r_edge, u):
    """
    Construct all state variables required by
    residual and jacobian evaluation from ln_r_edge and u.
    """
    r_edge = np.exp(ln_r_edge)

    ln_r = grid.log_cell_centers(ln_r_edge)
    r = np.exp(ln_r)

    ln_v = grid.log_shell_volumes(ln_r_edge)
    v = np.exp(ln_v)

    ln_rho = np.log(constants.DM) - ln_v
    rho = np.exp(ln_rho)

    ln_u = np.log(u)

    kappa_s = conductivity.smfp_conductivity(u)
    kappa_l = conductivity.lmfp_conductivity(rho, u)
    kappa = conductivity.total_conductivity(kappa_s, kappa_l)

    l = luminosity.luminosity(r_edge, u, r, kappa)

    return {
        "r_edge": r_edge,
        "ln_r_edge": ln_r_edge,
        "u": u,
        "ln_u": ln_u,
        "r": r,
        "ln_r": ln_r,
        "ln_rho": ln_rho,
        "v": v,
        "l": l,
        "kappa_s": kappa_s,
        "kappa_l": kappa_l,
        "kappa": kappa,
    }
