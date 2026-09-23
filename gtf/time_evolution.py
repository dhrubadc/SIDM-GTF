r"""
Implement time evolution starting from an
initial state.
"""

import numpy as np
from . import constants
from . import newton
from . import output


def evolve(state_init, dt_init=0.001):
    r"""Evolve the initial state of the system
    forward in time.

    Evolution stops when central density reaches
    :obj:`gtf.constants.RHO_STOP`

    Initial central density must be less than
    :obj:`gtf.constants.RHO_STOP` for evolution to proceed.

    Time stepping is controlled by the condtion
    :math:`{\rm MAX}(\epsilon_{r}, \epsilon_{u})`
    < :obj:`gtf.constants.DT_TOL`.

    Here :math:`\epsilon_{r} = \frac{|\Delta r_{\rm edge}[1:-1]|}{r_{\rm edge}[1:-1]}`
    and :math:`\epsilon_{u}= \frac{|\Delta u|}{u}`
    are the relative changes in r_edge and u
    between the current and previous times.

    :param state_init: physical state of the system at t=0
    :type state_init: dict

    :param dt_init: initial trial dt, defaults to 0.001
    :type dt_init: :obj:`gtf.constants.FLOATDTYPE`
    """

    # pylint: disable=too-many-locals

    if np.exp(state_init["ln_rho"]).max() > constants.RHO_STOP:
        raise RuntimeError("Increase RHO_STOP")

    history = []

    t = 0
    state_old = state_init
    dt_trial = dt_init

    rho_c_last_output = np.exp(state_init["ln_rho"]).max()

    snapshot_index = 1

    while np.exp(state_old["ln_rho"]).max() <= constants.RHO_STOP:

        dt_step = dt_trial

        state_new, f_max, n_iter = newton.iterate(state_old, dt_step)

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
            dt_trial = dt_step * constants.FLOATDTYPE(0.9) * constants.DT_TOL / eps

            print("reject:", "dt =", dt_step, "eps =", eps, "new dt =", dt_trial)

            continue

        # Accept the timestep.
        t += dt_step

        if (
            np.exp(state_new["ln_rho"]).max()
            >= rho_c_last_output * constants.OUTPUT_FACTOR
            or np.exp(state_new["ln_rho"]).max()
            <= rho_c_last_output / constants.OUTPUT_FACTOR
        ):

            write_data = np.concatenate(
                ([t], state_new["r_edge"][1:-1], state_new["u"])
            )

            output.write_snapshot(write_data, snapshot_index)

            rho_c_last_output = np.exp(state_new["ln_rho"]).max()

            snapshot_index += 1

        history.append(
            [t, dt_step, eps, f_max, n_iter, np.exp(state_new["ln_rho"]).max()]
        )

        dt_trial = dt_step * constants.FLOATDTYPE(0.9) * constants.DT_TOL / eps

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

    output.write_stats(np.asarray(history))
