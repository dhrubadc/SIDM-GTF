"""Tests for snapshot and statistics file writing."""

import numpy as np

from gtf import output, settings


def test_snapshot_values_are_written_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "OUT_DIREC", str(tmp_path) + "/")
    values = np.array([0.25, 0.1, 0.2, 0.3, 1.1, 1.2])
    output.write_snapshot(values, 3)
    loaded = np.loadtxt(tmp_path / "snapshot_00003.txt")
    np.testing.assert_allclose(loaded, values)
    assert loaded.ndim == 1


def test_stats_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "OUT_DIREC", str(tmp_path) + "/")
    history = np.array([[0.1, 0.1, 0.01, 1e-8, 3, 1.2],
                        [0.2, 0.1, 0.02, 2e-8, 4, 1.3]])
    output.write_stats(history)
    loaded = np.loadtxt(tmp_path / "stats.txt")
    np.testing.assert_allclose(loaded, history)
