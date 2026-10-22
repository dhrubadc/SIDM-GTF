"""Integration tests exercising the coupled solver with real Newton steps.

These tests intentionally use a small shell count and short time intervals so
that they remain suitable for the regular test suite. They exercise the actual
initial-condition, state, residual, Jacobian, Newton, and energy code paths.
"""

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest

from gtf import energy, initial_conditions, newton, settings, state, time_evolution


def _make_initial_state(monkeypatch, r_first=None):
    """Create a fresh, hydrostatic initial halo for an integration test."""
    # set_up_initial_conditions writes these module-level arrays; monkeypatch
    # ensures the test does not leak them into other tests.
    monkeypatch.setattr(settings, "DM", None)
    monkeypatch.setattr(settings, "M_EDGE", None)

    if r_first is None:
        r_first = settings.R_FIRST
    ln_r_edge, u = initial_conditions.set_up_initial_conditions(r_first)
    return state.state_from_unknowns(ln_r_edge, u)


def _advance(state_old, dt, steps=1):
    """Advance using real Newton solves, failing clearly if any solve fails."""
    current = state_old
    for step in range(steps):
        # Newton emits per-iteration diagnostics; keep the test output concise.
        with redirect_stdout(StringIO()):
            current_new, f_max, n_iter = newton.iterate(current, dt)
        assert current_new is not None, (
            f"Newton failed at step {step + 1}/{steps}: "
            f"max residual={f_max}, iterations={n_iter}, dt={dt}"
        )
        assert np.isfinite(f_max)
        current = current_new
    return current


def _relative_state_difference(a, b, reference):
    """Maximum relative difference in free radii and specific energies."""
    radius_error = np.max(
        np.abs(a["r_edge"][1:-1] - b["r_edge"][1:-1])
        / reference["r_edge"][1:-1]
    )
    energy_error = np.max(np.abs(a["u"] - b["u"]) / reference["u"])
    return max(radius_error, energy_error)


def _assert_physical_state(current):
    """Check basic physical invariants of a computed state."""
    for key in ("r_edge", "r", "u", "rho", "v", "kappa"):
        if key in current:
            assert np.all(np.isfinite(current[key])), f"non-finite values in {key}"
    assert np.all(current["r_edge"][1:] > current["r_edge"][:-1])
    assert np.all(current["u"] > 0.0)
    assert np.all(current["r"] > 0.0)
    assert np.all(np.exp(current["ln_rho"]) > 0.0)


def test_short_multistep_evolution_stays_physical(configured_settings, monkeypatch):
    """Several coupled steps should retain finite, positive physical fields."""
    initial = _make_initial_state(monkeypatch)
    final = _advance(initial, dt=1.0e-4, steps=3)

    _assert_physical_state(final)
    assert final["u"].shape == initial["u"].shape
    # The Lagrangian cell masses are fixed during evolution.
    np.testing.assert_allclose(settings.DM, np.diff(settings.M_EDGE), rtol=1e-14)


def test_total_energy_drift_is_small_over_complete_evolve_run(
    configured_settings, monkeypatch, tmp_path
):
    """Energy drift stays bounded until evolve() reaches its density stop."""
    # This compact test configuration makes the central density increase, so
    # evolve() reaches RHO_STOP after a few accepted steps rather than running
    # indefinitely on a halo whose core is initially expanding.
    monkeypatch.setattr(settings, "BETA", 100.0)
    monkeypatch.setattr(settings, "DT_TOL", 1.0e-3)
    monkeypatch.setattr(settings, "OUT_DIREC", str(tmp_path) + "/")
    monkeypatch.setattr(settings, "OUTPUT_FACTOR", 1.00000001)

    initial = _make_initial_state(monkeypatch)
    rho_initial = np.max(np.exp(initial["ln_rho"]))
    monkeypatch.setattr(settings, "RHO_STOP", rho_initial + 5.0e-3)
    energy_initial = energy.total_energy(initial["r"], initial["u"])

    with redirect_stdout(StringIO()):
        time_evolution.evolve(initial, dt_init=1.0e-3)

    history = np.atleast_2d(np.loadtxt(tmp_path / "stats.txt"))
    assert history.shape[0] >= 1
    assert np.all(np.isfinite(history))
    assert history[-1, -1] > settings.RHO_STOP

    snapshots = sorted(tmp_path.glob("snapshot_*.txt"))
    assert snapshots, "evolve() did not write a snapshot"
    snapshot = np.loadtxt(snapshots[-1])
    n = settings.N_SHELL
    assert snapshot.shape == (2 * n,)

    # Snapshot layout: [time, free linear radial edges, specific energies].
    r_edge = np.concatenate(([0.0], snapshot[1:n], [settings.R_OUTER]))
    ln_r_edge = np.concatenate(([-np.inf], np.log(r_edge[1:])))
    final = state.state_from_unknowns(ln_r_edge, snapshot[n:])
    energy_final = energy.total_energy(final["r"], final["u"])
    relative_drift = abs(energy_final - energy_initial) / abs(energy_initial)

    assert np.isfinite(relative_drift)
    # Regression bound for this short, tightly time-stepped integration run.
    assert relative_drift < 1.0e-4


def test_solution_converges_when_timestep_is_refined(
    configured_settings, monkeypatch
):
    """Two half steps should be closer to a refined solution than one full step."""
    initial = _make_initial_state(monkeypatch)

    coarse = _advance(initial, dt=1.0e-3, steps=1)
    medium = _advance(initial, dt=5.0e-4, steps=2)
    refined = _advance(initial, dt=2.5e-4, steps=4)

    coarse_error = _relative_state_difference(coarse, refined, initial)
    medium_error = _relative_state_difference(medium, refined, initial)

    assert np.isfinite(coarse_error) and np.isfinite(medium_error)
    assert medium_error < coarse_error


def test_high_density_initial_state_completes_implicit_step(
    configured_settings, monkeypatch
):
    """A centrally concentrated halo should still admit a small Newton step."""
    initial = _make_initial_state(monkeypatch, r_first=1.0e-3)
    rho_initial = np.exp(initial["ln_rho"])
    assert np.max(rho_initial) > 1.0e3

    with redirect_stdout(StringIO()):
        final, f_max, n_iter = newton.iterate(initial, 1.0e-7)

    assert final is not None, "Newton failed for the high-density test halo"
    assert np.isfinite(f_max)
    assert f_max < settings.F_ACCEPT
    assert n_iter < settings.ITER_MAX
    _assert_physical_state(final)
    assert np.all(np.isfinite(np.exp(final["ln_rho"])))
