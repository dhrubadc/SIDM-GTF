"""
Module to construct current state from the system from ln_r_edge and u.
"""

import conductivity
import energy
import grid


def state_from_unknowns(ln_r_edge, u, dm, sigma_over_m, C, alpha):
    """
    Construct state variables required by residual
    from the primary unknowns.
    """
    r_edge = np.exp(ln_r_edge)

    ln_r = grid.log_cell_centers(ln_r_edge)

    V = grid.shell_volumes(r_edge)
    ln_V = grid.log_shell_volumes(ln_r_edge)

    ln_rho = np.log(dm) - ln_V
    rho = np.exp(ln_rho)

    ln_u = np.log(u)

    kappa = conductivity.conductivity(rho, u, sigma_over_m, C, alpha)
    L = energy.luminosity(r_edge, u, kappa)

    return {
        "r_edge": r_edge,
        "ln_r": ln_r,
        "V": V,
        "ln_rho": ln_rho,
        "ln_u": ln_u,
        "L": L,
    }
