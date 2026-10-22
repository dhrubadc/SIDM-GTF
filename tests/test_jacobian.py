"""Structural and finite-difference checks for the analytic Jacobian."""

import numpy as np

from gtf import jacobian, newton, residual, settings, state


def test_jacobian_shape_and_finite_entries(sample_state):
    j = jacobian.analytic_jacobian(sample_state, sample_state["v"], 0.01)
    assert j.shape == (2 * settings.N_SHELL - 1, 2 * settings.N_SHELL - 1)
    assert np.all(np.isfinite(j))


def test_analytic_jacobian_matches_central_finite_difference(sample_state):
    """Check every column at a smooth, positive test state."""
    ln_edges = sample_state["ln_r_edge"]
    u = sample_state["u"]
    x = newton.pack_unknowns(ln_edges, u)
    v_old = sample_state["v"].copy()
    dt = 1.0e-3
    analytic = jacobian.analytic_jacobian(sample_state, v_old, dt)
    numeric = np.empty_like(analytic)
    step = 2.0e-6

    def evaluate(x_trial):
        ln_edge_trial, u_trial = newton.unpack_unknowns(x_trial)
        trial_state = state.state_from_unknowns(ln_edge_trial, u_trial)
        return residual.residual(trial_state, v_old, u, dt)

    for j in range(x.size):
        dx = np.zeros_like(x)
        dx[j] = step * max(1.0, abs(x[j]))
        numeric[:, j] = (evaluate(x + dx) - evaluate(x - dx)) / (2.0 * dx[j])

    # This is intentionally a numerical verification test; failures may reveal
    # a derivative error or an overly sensitive finite-difference step.
    np.testing.assert_allclose(analytic, numeric, rtol=3e-4, atol=3e-6)
