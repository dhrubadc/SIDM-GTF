r"""Output a state and overall stats"""

import numpy as np
from . import settings


def write_snapshot(write_data, snapshot_index):
    r"""
    Write data for a snapshot.

    :param write_data: array containing time and primary unknowns
    :type write_data: np.ndarray

    :snapshot_index: snapshot number
    :type snapshot_index: int
    """
    with open(
        settings.OUT_DIREC + f"snapshot_{snapshot_index:05d}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        np.savetxt(f, write_data)


def write_stats(history):
    r"""
    Write overall stats.

    :param history: array of important global diagnostics
    :type history: np.ndarray
    """
    with open(settings.OUT_DIREC + "stats.txt", "w", encoding="utf-8") as f:
        np.savetxt(f, history)
