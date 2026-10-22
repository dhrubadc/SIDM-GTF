"""Tests for coupled hydrostatic and energy residuals."""

import numpy as np

from gtf import residual, settings


def test_residual_shapes_and_concatenation(sample_state):
    n = settings.N_SHELL
    f_h = residual.hydrostatic_residual(sample_state)
    f_e = residual.energy_residual(sample_state, sample_state["v"], sample_state["u"], 0.01)
    f = residual.residual(sample_state, sample_state["v"], sample_state["u"], 0.01)
    assert f_h.shape == (n - 1,)
    assert f_e.shape == (n,)
    assert f.shape == (2 * n - 1,)
    np.testing.assert_allclose(f, np.concatenate((f_h, f_e)))


def test_energy_residual_is_zero_for_unchanged_state(sample_state):
    f = residual.energy_residual(sample_state, sample_state["v"], sample_state["u"], 0.0)
    np.testing.assert_allclose(f, 0.0, atol=1e-14)


def test_energy_residual_terms_are_finite(sample_state):
    terms = residual.energy_terms(sample_state, sample_state["v"] * 1.01,
                                  sample_state["u"] * 0.99, 0.005)
    assert len(terms) == 3
    for term in terms:
        assert term.shape == (settings.N_SHELL,)
        assert np.all(np.isfinite(term))
