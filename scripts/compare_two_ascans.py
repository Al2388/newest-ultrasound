"""
Overlay two A-scan cycling sessions on shared, glitch-robust axes.

Plots ToF / amplitude / energy for two HDF5 sessions on the same axes so their
*shapes* can be compared directly. Y-limits are set from the 1st-99th
percentile of the combined data, so a handful of dropout/glitch samples don't
stretch the view and hide the real signal.

Usage
-----
  python compare_two_ascans.py SESSION_A.h5 SESSION_B.h5 [--smooth 60] [--out out.png]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(h5_path: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        return {
            "t_h": (f["timestamps"][:] - f["timestamps"][0]) / 3600.0,
            "tof_us": f["tof_us"][:],
            "amplitude": f["amplitude"][:],
            "energy": f["energy"][:],
        }


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float64, copy=True)
    k = np.ones(win) / win
    y = np.convolve(x.astype(np.float64), k, mode="same")
    h = win // 2
    y[:h] = np.nan
    y[-h:] = np.nan
    return y


def _robust_ylim(arrays: list[np.ndarray], pad: float = 0.08) -> tuple[float, float]:
    """Y-limits from the combined 1st-99th percentile, ignoring glitch outliers."""
    cat = np.concatenate(arrays)
    lo, hi = np.percentile(cat, 1), np.percentile(cat, 99)
    span = hi - lo or 1.0
    return lo - pad * span, hi + pad * span


def compare(a_path: str, b_path: str, out_path: str | None, smooth_s: float) -> str:
    a, b = _load(a_path), _load(b_path)
    a_lbl, b_lbl = Path(a_path).stem, Path(b_path).stem

    fields = [("tof_us", "ToF (us)"), ("amplitude", "Amplitude (V)"), ("energy", "Energy")]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=140, sharex=True)
    fig.suptitle("A-scan session comparison (shared axes, 1-99 pct y-limits)", fontsize=11)

    for ax, (key, name) in zip(axes, fields):
        for d, lbl, col in ((a, a_lbl, "tab:blue"), (b, b_lbl, "tab:orange")):
            dt = float(np.median(np.diff(d["t_h"]))) * 3600.0
            win = max(1, int(round(smooth_s / max(dt, 1e-6))))
            y = _rolling_mean(d[key], win) if win > 1 else d[key]
            ax.plot(d["t_h"], y, color=col, linewidth=1.3, label=lbl)
        ax.set_ylim(*_robust_ylim([a[key], b[key]]))
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("elapsed time (hours)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is None:
        out_path = "reports/ascan_compare.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--smooth", type=float, default=60.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    print("saved:", compare(args.a, args.b, args.out, args.smooth))


if __name__ == "__main__":
    main()
