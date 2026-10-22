"""Tests for logarithmic cell geometry."""

import numpy as np

from gtf import geometry


def test_log_cell_centers_matches_arithmetic_midpoints(configured_settings):
    edges = np.array([-np.inf, np.log(0.2), np.log(0.7), np.log(2.0)])
    actual = np.exp(geometry.log_cell_centers(edges))
    expected = 0.5 * (np.exp(edges[:-1]) + np.exp(edges[1:]))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)


def test_log_cell_volumes_matches_spherical_shell_volumes(configured_settings):
    edges = np.array([-np.inf, np.log(0.2), np.log(0.7), np.log(2.0)])
    actual = np.exp(geometry.log_cell_volumes(edges))
    r = np.exp(edges)
    expected = (r[1:]**3 - r[:-1]**3) / 3.0
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)
    assert np.all(actual > 0.0)


def test_log_cell_volumes_remains_finite_for_thin_shells(configured_settings):
    edges = np.log(np.array([1.0, 1.0 + 1e-10, 1.0 + 2e-10]))
    actual = geometry.log_cell_volumes(edges)
    assert np.all(np.isfinite(actual))
