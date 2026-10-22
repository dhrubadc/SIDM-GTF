"""Focused tests of evolve() control flow using a mocked Newton solver."""

import numpy as np
import pytest

from gtf import settings, time_evolution


def _state(rho, radius_scale=1.0, u_scale=1.0):
    return {
        "ln_rho": np.log(np.array([rho, rho * 0.8])),
        "r_edge": np.array([0.0, 0.5 * radius_scale, 1.0 * radius_scale]),
        "u": np.array([1.0 * u_scale, 0.8 * u_scale]),
    }


def test_evolve_rejects_initial_density_above_stop(configured_settings):
    initial = _state(settings.RHO_STOP * 1.1)
    with pytest.raises(RuntimeError, match="Increase RHO_STOP"):
        time_evolution.evolve(initial)


def test_evolve_raises_when_newton_fails(configured_settings, monkeypatch):
    monkeypatch.setattr(settings, "RHO_STOP", 10.0)
    monkeypatch.setattr(time_evolution.newton, "iterate", lambda state, dt: (None, np.inf, 0))
    with pytest.raises(RuntimeError, match="Newton iterations failed"):
        time_evolution.evolve(_state(1.0))


def test_evolve_accepts_step_and_writes_stats(configured_settings, monkeypatch):
    monkeypatch.setattr(settings, "RHO_STOP", 2.0)
    monkeypatch.setattr(settings, "DT_TOL", 0.1)
    monkeypatch.setattr(settings, "OUTPUT_FACTOR", 100.0)  # suppress snapshots in this test
    initial = _state(1.0)
    updated = _state(3.0, radius_scale=1.001, u_scale=1.001)
    monkeypatch.setattr(time_evolution.newton, "iterate", lambda state, dt: (updated, 1e-8, 2))
    written = {}
    monkeypatch.setattr(time_evolution.output, "write_stats", lambda history: written.setdefault("history", history))
    monkeypatch.setattr(time_evolution.output, "write_snapshot", lambda *args: pytest.fail("unexpected snapshot"))

    time_evolution.evolve(initial, dt_init=0.01)
    assert "history" in written
    assert written["history"].shape == (1, 6)
    assert written["history"][0, 0] == pytest.approx(0.01)


def test_evolve_rejects_then_accepts_smaller_step(configured_settings, monkeypatch):
    monkeypatch.setattr(settings, "RHO_STOP", 2.0)
    monkeypatch.setattr(settings, "DT_TOL", 0.1)
    monkeypatch.setattr(settings, "OUTPUT_FACTOR", 100.0)
    initial = _state(1.0)
    # First call has a large relative change and is rejected. Second accepted
    # call crosses RHO_STOP, terminating the loop on the next condition check.
    calls = []
    rejected = _state(1.0, radius_scale=1.2, u_scale=1.2)
    accepted = _state(3.0, radius_scale=1.001, u_scale=1.001)

    def fake_iterate(old, dt):
        calls.append(dt)
        if len(calls) == 1:
            return rejected, 1e-5, 2
        return accepted, 1e-8, 2

    monkeypatch.setattr(time_evolution.newton, "iterate", fake_iterate)
    monkeypatch.setattr(time_evolution.output, "write_snapshot", lambda *args: None)
    monkeypatch.setattr(time_evolution.output, "write_stats", lambda history: None)
    time_evolution.evolve(initial, dt_init=0.01)

    assert len(calls) == 2
    assert calls[1] < calls[0]


def test_evolve_raises_for_nonfinite_error_estimate(configured_settings, monkeypatch):
    monkeypatch.setattr(settings, "RHO_STOP", 10.0)
    initial = _state(1.0)
    bad = _state(1.0, radius_scale=np.nan)
    monkeypatch.setattr(time_evolution.newton, "iterate", lambda state, dt: (bad, 1e-8, 1))
    with pytest.raises(RuntimeError, match="Non-finite timestep error estimate"):
        time_evolution.evolve(initial)
