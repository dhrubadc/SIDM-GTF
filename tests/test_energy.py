"""Tests for total kinetic, gravitational, and total energy diagnostics."""

import numpy as np

from gtf import energy, settings


def test_energy_components_match_hand_calculation(configured_settings, monkeypatch):
    dm = np.array([1.0, 2.0, 3.0])
    m_edge = np.array([0.0, 1.0, 3.0, 6.0])
    u = np.array([2.0, 3.0, 4.0])
    r = np.array([0.5, 1.0, 2.0])
    monkeypatch.setattr(settings, "DM", dm)
    monkeypatch.setattr(settings, "M_EDGE", m_edge)

    kinetic = np.sum(dm * u)
    m_center = 0.5 * (m_edge[:-1] + m_edge[1:])
    grav = -np.sum(dm * m_center / r)
    assert energy.total_kinetic_energy(u) == kinetic
    np.testing.assert_allclose(energy.total_gravitational_energy(r), grav)
    np.testing.assert_allclose(energy.total_energy(r, u), kinetic + grav)


def test_gravitational_energy_is_negative(configured_settings, monkeypatch):
    monkeypatch.setattr(settings, "DM", np.array([0.5, 0.5]))
    monkeypatch.setattr(settings, "M_EDGE", np.array([0.0, 0.5, 1.0]))
    assert energy.total_gravitational_energy(np.array([0.25, 0.75])) < 0.0
