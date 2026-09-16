"""Module to construct current state of the system
from ln_r_edge and u.
"""

import numpy as np
import conductivity
import energy
import grid


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
def state_from_unknowns(ln_r_edge, u, dm, sigma_over_m, beta, alpha):
    """
    Construct all state variables required by
    residual and jacobian from the primary unknowns.
    """
    r_edge = np.exp(ln_r_edge)

    ln_r = grid.log_cell_centers(ln_r_edge)
    r = np.exp(ln_r)

    ln_v = grid.log_shell_volumes(ln_r_edge)
    v = np.exp(ln_v)

    ln_rho = np.log(dm) - ln_v
    rho = np.exp(ln_rho)

    ln_u = np.log(u)

    kappa_s = conductivity.smfp_conductivity(u, sigma_over_m)
    kappa_l = conductivity.lmfp_conductivity(rho, u, beta)
    kappa = conductivity.total_conductivity(kappa_s, kappa_l, alpha)

    l = energy.luminosity(r_edge, u, kappa)

    return {
        "u": u,
        "r_edge": r_edge,
        "r": r,
        "ln_r": ln_r,
        "ln_rho": ln_rho,
        "ln_u": ln_u,
        "v": v,
        "l": l,
        "kappa_s": kappa_s,
        "kappa_l": kappa_l,
        "kappa": kappa,
    }
