"""Shared fixtures for the SIDM-GTF unit tests."""

import numpy as np
import pytest

from gtf import settings
from gtf import state as state_module


@pytest.fixture
def configured_settings(monkeypatch):
    """Install a small, valid parameter set and restore it after each test."""
    values = {
        "FLOATDTYPE": np.float64,
        "SIGMA_OVER_M": 1.0,
        "BETA": 1.0,
        "ALPHA": 2.0,
        "N_SHELL": 4,
        "R_OUTER": 2.0,
        "R_FIRST": 0.1,
        "F_TOL": 1.0e-10,
        "ITER_MAX": 30,
        "X_TOL": 1.0e-10,
        "F_ACCEPT": 1.0e-7,
        "DT_TOL": 1.0e-2,
        "RHO_STOP": 1.0e6,
        "OUT_DIREC": "",
        "OUTPUT_FACTOR": 2.0,
        "R_T": np.inf,
        "N": 1.0,
    }
    for name, value in values.items():
        monkeypatch.setattr(settings, name, value)
    return settings


@pytest.fixture
def sample_unknowns(configured_settings):
    """Return ordered log-edges and positive specific energies."""
    ln_edges = np.empty(configured_settings.N_SHELL + 1)
    ln_edges[0] = -np.inf
    ln_edges[1:] = np.log(np.geomspace(0.1, configured_settings.R_OUTER,
                                      configured_settings.N_SHELL))
    u = np.array([0.8, 0.65, 0.45, 0.3], dtype=float)
    return ln_edges, u


@pytest.fixture
def sample_state(configured_settings, sample_unknowns):
    """Build a physical state and the global mass arrays it depends on."""
    ln_edges, u = sample_unknowns
    # Positive, nonuniform cell masses keep all density and conduction terms defined.
    settings.DM = np.array([0.15, 0.25, 0.35, 0.45], dtype=float)
    settings.M_EDGE = np.concatenate(([0.0], np.cumsum(settings.DM)))
    return state_module.state_from_unknowns(ln_edges, u)
