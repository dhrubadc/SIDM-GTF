r"""
Implement time evolution starting from an
initial state.
"""

import numpy as np
from . import constants
from . import newton


def evolve(state_init, rho_stop, dt_init=0.001):
    r"""Evolve the initial state of the system
    forward in time.

    Time stepping is controlled by the condtion
    :math:`{\rm MAX}(\epsilon_{r}, \epsilon_{u})`
    < :obj:`gtf.constants.DT_TOL`.

    Here :math:`\epsilon_{r} = \frac{|\Delta r_{\rm edge}[1:-1]|}{r_{\rm edge}[1:-1]}`
    and :math:`\epsilon_{u}= \frac{|\Delta u|}{u}`
    are the relative changes in r_edge and u
    between the current and previous times.

    :param state_init: physical state of the system at t=0
    :type state_init: dict

    :param rho_stop: maximum central density for stoping the evolution 
    :type rho_stop: :obj:`gtf.constants.FLOATDTYPE`

    :param dt_init: initial trial dt, defaults to 0.001
    :type dt_init: :obj:`gtf.constants.FLOATDTYPE`

    rho_stop must be greater than the central density at t=0 
    for time evolution to proceed.
    """
    t = 0

    state_old = state_init
    dt_trial = dt_init

    while np.exp(state_old["ln_rho"]).max() <= rho_stop:

        dt_step = dt_trial

        state_new = newton.iterate(state_old, dt_step)

        if state_new is None:
            raise RuntimeError("Newton iterations failed")

        eps_r = np.max(
            np.abs(state_new["r_edge"][1:-1] - state_old["r_edge"][1:-1])
            / state_old["r_edge"][1:-1]
        )
        eps_u = np.max(np.abs(state_new["u"] - state_old["u"]) / state_old["u"])

        eps = max(eps_r, eps_u)

        if eps > constants.DT_TOL:

            # reject this timestep
            dt_trial = dt_step * constants.FLOATDTYPE(0.99) * constants.DT_TOL / eps

            print("reject:", "dt =", dt_step, "eps =", eps, "new dt =", dt_trial)

            continue

        # Accept the timestep.
        t += dt_step

        dt_trial = dt_step * constants.FLOATDTYPE(0.99) * constants.DT_TOL / eps

        print(
            "accept:",
            "t=",
            t,
            "dt =",
            dt_step,
            "eps =",
            eps,
            "new dt =",
            dt_trial,
            "rho_0=",
            np.exp(state_new["ln_rho"]).max(),
        )

        state_old = state_new
