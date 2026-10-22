"""Tests for Newton unknown packing and physical line search."""

import numpy as np

from gtf import newton, settings


def test_pack_unpack_round_trip(sample_unknowns):
    ln_edges, u = sample_unknowns
    x = newton.pack_unknowns(ln_edges, u)
    ln_edges_out, u_out = newton.unpack_unknowns(x)
    np.testing.assert_allclose(ln_edges_out[1:-1], ln_edges[1:-1])
    assert np.isneginf(ln_edges_out[0])
    assert ln_edges_out[-1] == np.log(settings.R_OUTER)
    np.testing.assert_allclose(u_out, u)


def test_increment_newton_variables_returns_ordered_edges_and_positive_u(sample_state):
    x = newton.pack_unknowns(sample_state["ln_r_edge"], sample_state["u"])
    dx = np.zeros_like(x)
    # Make the final energy negative for a full step; backtracking should recover.
    dx[-1] = -2.0 * x[-1]
    x_new, state_new, alpha = newton.increment_newton_variables(x, dx)
    assert 0.0 < alpha <= 1.0
    assert np.all(np.diff(state_new["r_edge"]) > 0.0)
    assert np.all(state_new["u"] > 0.0)
    np.testing.assert_allclose(x_new, newton.pack_unknowns(state_new["ln_r_edge"], state_new["u"]))
