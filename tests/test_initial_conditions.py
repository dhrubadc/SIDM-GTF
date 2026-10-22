"""Tests for initial grid, mass bookkeeping, and hydrostatic initialization."""

import numpy as np

from gtf import geometry, initial_conditions, residual, settings


def test_initial_grid_has_origin_and_fixed_outer_edge(configured_settings):
    edges = initial_conditions.make_initial_grid(0.1)
    assert edges.shape == (settings.N_SHELL + 1,)
    assert np.isneginf(edges[0])
    assert np.all(np.diff(edges[1:]) > 0.0)
    assert np.exp(edges[-1]) == settings.R_OUTER


def test_truncated_nfw_reduces_density_for_finite_truncation(configured_settings):
    r = np.array([0.2, 0.5, 1.0])
    plain = initial_conditions.truncated_nfw(r, np.inf, 1.0)
    truncated = initial_conditions.truncated_nfw(r, 0.7, 1.0)
    assert np.all(truncated < plain)
    np.testing.assert_allclose(plain, 1.0 / (r * (1.0 + r)**2))


def test_cell_mass_and_enclosed_mass(configured_settings):
    rho = np.array([2.0, 3.0, 4.0])
    volume = np.array([0.1, 0.2, 0.3])
    dm = initial_conditions.cell_mass(rho, volume)
    np.testing.assert_allclose(dm, [0.2, 0.6, 1.2])
    np.testing.assert_allclose(initial_conditions.enclosed_mass(dm), [0.0, 0.2, 0.8, 2.0])


def test_set_up_initial_conditions_returns_finite_positive_energy(configured_settings):
    ln_edges, u = initial_conditions.set_up_initial_conditions(0.1)
    assert ln_edges.shape == (settings.N_SHELL + 1,)
    assert u.shape == (settings.N_SHELL,)
    assert np.all(np.isfinite(ln_edges[1:]))
    assert np.all(np.isfinite(u)) and np.all(u > 0.0)
    assert settings.DM.shape == (settings.N_SHELL,)
    assert settings.M_EDGE.shape == (settings.N_SHELL + 1,)
    assert np.all(settings.DM > 0.0)
    assert settings.M_EDGE[0] == 0.0
    assert np.all(np.diff(settings.M_EDGE) > 0.0)

    # The initializer integrates the same hydrostatic stencil used by residual.py.
    r_edge = np.exp(ln_edges)
    ln_r = geometry.log_cell_centers(ln_edges)
    rho = settings.DM / np.exp(geometry.log_cell_volumes(ln_edges))
    state = {
        "r_edge": r_edge,
        "ln_r": ln_r,
        "ln_rho": np.log(rho),
        "u": u,
        "ln_u": np.log(u),
    }
    hydro = residual.hydrostatic_residual(state)
    np.testing.assert_allclose(hydro, 0.0, atol=2e-10, rtol=0.0)
