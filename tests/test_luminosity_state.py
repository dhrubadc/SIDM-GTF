"""Tests for conductive luminosity and full-state construction."""

import numpy as np

from gtf import luminosity, settings


def test_constant_energy_has_zero_luminosity(sample_state):
    r_edge = sample_state["r_edge"]
    r = sample_state["r"]
    u = np.full(settings.N_SHELL, 0.5)
    kappa = np.full(settings.N_SHELL, 1.2)
    lum = luminosity.luminosity(r_edge, r, u, kappa)
    np.testing.assert_allclose(lum, 0.0, atol=0.0)


def test_luminosity_boundary_values_are_zero(sample_state):
    lum = sample_state["l"]
    assert lum.shape == (settings.N_SHELL + 1,)
    assert lum[0] == 0.0
    assert lum[-1] == 0.0


def test_state_fields_have_expected_shapes_and_positive_values(sample_state):
    n = settings.N_SHELL
    for key in ("u", "ln_u", "r", "ln_r", "ln_rho", "v", "kappa_s", "kappa_l", "kappa"):
        assert sample_state[key].shape == (n,)
    for key in ("r_edge", "ln_r_edge", "l"):
        assert sample_state[key].shape == (n + 1,)
    for key in ("u", "r", "v", "kappa_s", "kappa_l", "kappa"):
        assert np.all(np.isfinite(sample_state[key]))
        assert np.all(sample_state[key] > 0.0)
