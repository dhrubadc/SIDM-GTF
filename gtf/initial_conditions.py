r"""
Set up initial conditions for a symmerically symmetric, isotropic SIDM halo.
Current support for standard and truncated NFW profiles.
"""

import numpy as np
from scipy.optimize import brentq
from . import constants
from . import geometry


def make_initial_grid(r_first):
    r"""
    Build initial radial grid in log-space.
    ln_r_edge[0] = -np.inf and
    ln_r_edge[1:] = linspace(ln_r_first, ln_r_outer, n_shell).

    Here n_shell is :obj:`src.constants.N_SHELL` and
    ln_r_outer is log of :obj:`src.constants.R_OUTER`.

    :param r_first: first non-zero cell edge
    :type r_first: float

    :return: initial radial grid
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL` + 1
    """
    ln_r_edge = np.empty(constants.N_SHELL + 1, dtype=constants.FLOATDTYPE)

    ln_r_edge[0] = -np.inf

    ln_r_edge[1:] = np.linspace(
        np.log(r_first), np.log(constants.R_OUTER), constants.N_SHELL
    )

    return ln_r_edge


def truncated_nfw(r, r_t, n):
    r"""
    Initial NFW profile with truncation option.

    :param r: cell midpoints
    :type r: 1D array of shape :obj:`src.constants.N_SHELL`

    :param r_t: truncation radius
    :type r_t: float

    :param n: truncation exponent
    :type n: float

    :return: initial density at cell midpoints
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL`
    """
    return np.exp(-((r / r_t) ** n)) / (r * (constants.FLOATDTYPE(1.0) + r) ** 2)


def cell_mass(rho, v):
    r"""
    Calculate the Lagrangian mass of each cell.

    :param rho: density at cell midpoints
    :type rho: 1D array of shape :obj:`src.constants.N_SHELL`

    :param v: cell volumes
    :type v: 1D array of shape :obj:`src.constants.N_SHELL`

    :return: return cell masses
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL`

    """
    return rho * v


def enclosed_mass(dm):
    r"""
    Calculate cumulative Lagrangian masses at all cell edges.

    :param dm: Lagrangian cell masses
    :type dm: 1D array of shape :obj:`src.constants.N_SHELL`

    :return: cumulative cell masses
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL` + 1
    """
    m_edge = np.zeros(len(dm) + 1, dtype=constants.FLOATDTYPE)
    m_edge[1:] = np.cumsum(dm)

    return m_edge


def hydrostatic_u_from_rho(r_edge, ln_r, ln_rho, u_outer):
    r"""
    Calculate initial specific energy from density.
    Integrate inwards the same hydrostatic :obj:`src.residual.hydrostatic_residual`
    to get the initial specific energy profile.
    The specific energy at the outermost cell is specified.

    :param r_edge: cell edges
    :type r_edge: 1D array of shape :obj:`src.constants.N_SHELL` + 1

    :param ln_r: log of cell midpoints
    :type ln_r: 1D array of shape :obj:`src.constants.N_SHELL`

    :param ln_rho: log of density at cell midpoints
    :type ln_rho: 1D array of shape :obj:`src.constants.N_SHELL`

    :param u_outer: specific energy of the last cell (strictly positive)
    :type u_outer: float

    :return: specific energy at cell midpoints
    :rtype: 1D array of shape :obj:`src.constants.N_SHELL`
    """

    u = np.empty(constants.N_SHELL, dtype=constants.FLOATDTYPE)
    u[-1] = u_outer

    for i in range(constants.N_SHELL - 2, -1, -1):
        dln_r = ln_r[i + 1] - ln_r[i]
        dln_rho = ln_rho[i + 1] - ln_rho[i]

        g_coefficient = (
            constants.M_EDGE[i + 1] * constants.FLOATDTYPE(3.0 / 2.0) / r_edge[i + 1]
        )

        u_right = u[i + 1]
        ln_u_right = np.log(u_right)

        # pylint: disable=too-many-arguments,too-many-positional-arguments
        def shell_balance(
            ln_u_left,
            dln_r=dln_r,
            dln_rho=dln_rho,
            g_coefficient=g_coefficient,
            ln_u_right=ln_u_right,
            u_right=u_right,
        ):
            u_left = np.exp(ln_u_left)
            u_face = constants.FLOATDTYPE(0.5) * (u_left + u_right)

            return (dln_rho + ln_u_right - ln_u_left) / dln_r + g_coefficient / u_face

        lower = ln_u_right - constants.FLOATDTYPE(10.0)
        upper = ln_u_right + constants.FLOATDTYPE(10.0)

        while shell_balance(lower) < constants.FLOATDTYPE(0.0):
            lower -= constants.FLOATDTYPE(10.0)

        while shell_balance(upper) > constants.FLOATDTYPE(0.0):
            upper += constants.FLOATDTYPE(10.0)

        ln_u_left = brentq(
            shell_balance,
            lower,
            upper,
            xtol=constants.FLOATDTYPE(1.0e-13),
            rtol=constants.FLOATDTYPE(1.0e-13),
        )

        u[i] = np.exp(ln_u_left)

    if np.any(~np.isfinite(u)) or np.any(u <= constants.FLOATDTYPE(0.0)):
        raise RuntimeError("Hydrostatic initialization produced invalid u.")

    return u


def set_up_initial_conditions(r_first, r_t=np.inf, n=1.0):
    r"""
    Set up initial condtions.
    Default is standard NFW without any truncation.

    :param r_first: first non-zero cell edge
    :type r_first: float

    :param r_t: truncation radius, defaults to infinity
    :type r_t: float

    :param n: truncation exponent, defaults to 1.0
    :type n: float

    :return: log of initial cell edges and specific energy at cell midpoints
    :rtpye: Tuple[:obj:`src.constants.N_SHELL` + 1, :obj:`src.constants.N_SHELL`]

    Also sets :obj:`src.constants.DM` and :obj:`src.constants.M_EDGE`.
    """
    ln_r_edge = make_initial_grid(r_first)
    ln_r = geometry.log_cell_centers(ln_r_edge)
    ln_v = geometry.log_cell_volumes(ln_r_edge)

    r = np.exp(ln_r)
    v = np.exp(ln_v)
    r_edge = np.exp(ln_r_edge)

    rho = truncated_nfw(r, r_t, n)
    ln_rho = np.log(rho)

    constants.DM = cell_mass(rho, v)
    constants.M_EDGE = enclosed_mass(constants.DM)

    u = hydrostatic_u_from_rho(r_edge, ln_r, ln_rho, u_outer=0.001)

    return ln_r_edge, u
