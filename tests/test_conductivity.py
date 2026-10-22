"""Tests for SIDM conductivity prescriptions."""

import numpy as np

from gtf import conductivity, settings


def test_smfp_conductivity_matches_formula(configured_settings):
    u = np.array([0.2, 1.3, 4.0])
    speed = np.sqrt(2.0 * u / 3.0)
    expected = 1.5 * (settings.B / settings.A) * speed / settings.SIGMA_OVER_M**2
    np.testing.assert_allclose(conductivity.smfp_conductivity(u), expected)


def test_lmfp_conductivity_matches_formula(configured_settings):
    rho = np.array([0.1, 1.0, 3.0])
    u = np.array([0.2, 1.3, 4.0])
    expected = 1.5 * settings.BETA * rho * np.sqrt(2.0 * u / 3.0)**3
    np.testing.assert_allclose(conductivity.lmfp_conductivity(rho, u), expected)


def test_total_conductivity_is_positive_and_below_each_input(configured_settings):
    k_s = np.array([0.1, 1.0, 10.0])
    k_l = np.array([10.0, 1.0, 0.1])
    total = conductivity.total_conductivity(k_s, k_l)
    assert np.all(np.isfinite(total))
    assert np.all(total > 0.0)
    assert np.all(total <= np.minimum(k_s, k_l) * (1.0 + 1e-14))
