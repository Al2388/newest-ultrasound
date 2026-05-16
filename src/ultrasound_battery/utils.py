"""Small cross-cutting helpers used by the acquisition services."""

from __future__ import annotations

import time


def session_timestamp(t: float | None = None) -> str:
    """
    Return a filesystem-safe, lexically-sortable, human-readable timestamp string.

    Format: ``YYYY-MM-DD_HH-MM-SS`` (local time, second resolution).

    Used in C-scan / A-scan / gauging session folder names and the files inside
    them, so an operator can identify when an acquisition was taken from the
    folder name alone — e.g. ``cscan_scan_2026-05-13_18-36-08/`` rather than
    ``cscan_scan_1778693768/``.

    Why ``-`` instead of ``:`` between hours/minutes/seconds: Windows forbids
    ``:`` in filenames. Why local time: this is single-lab research data; the
    operator reading the folder name benefits more from local clock matching
    than from UTC strictness. (Absolute timestamps in HDF5/JSON metadata are
    still recorded with full precision via ``time.time()``.)

    Parameters
    ----------
    t : float, optional
        Unix epoch seconds. Defaults to the current wall-clock time.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(t))
