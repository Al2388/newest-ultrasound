"""
Overlay two cycling sessions across six channels on a shared time axis.

Extends compare_two_ascans.py: in addition to the three ultrasound channels
(ToF / amplitude / energy from the A-scan HDF5) it adds the cycler/logger
context channels - voltage, SOC and temperature - so the ultrasound features
can be read against the electrochemical state directly.

Voltage and temperature are interpolated onto the A-scan snapshot times from the
Maccor cycler export and the PicoLog temperature CSV. SOC is Coulomb-counted from
the cycler current over the whole session and normalised to the session's own
min/max throughput (0 % = fully discharged, 100 % = fully charged), because the
session-wide SOC is not stored (only per-segment SOC exists in segments/).

Usage
-----
  python compare_two_sessions_full.py EXP_DIR_A EXP_DIR_B [--smooth 60] [--out out.png]

  e.g. python compare_two_sessions_full.py data/experiments/18-5 data/experiments/21-5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the validated readers rather than re-parsing the cycler/temperature formats.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_charge_discharge import _find_one, read_ascan, read_cycler, read_temperature


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
    cat = np.concatenate([a[np.isfinite(a)] for a in arrays])
    lo, hi = np.percentile(cat, 1), np.percentile(cat, 99)
    span = hi - lo or 1.0
    return lo - pad * span, hi + pad * span


def _coulomb_soc(t_s: np.ndarray, current_a: np.ndarray, md: np.ndarray) -> np.ndarray:
    """Session-wide SOC (%) from Coulomb counting, normalised to its own min/max.

    The Maccor current column is unsigned magnitude; direction lives in the MD
    flag (C=charge, D=discharge, R/S=rest). We sign the current by MD, integrate
    (trapezoid) to cumulative charge throughput, then map the lowest point (end of
    full discharge) to 0 % and the highest (top of full charge) to 100 %.
    """
    sign = np.where(md == "C", 1.0, np.where(md == "D", -1.0, 0.0))
    order = np.argsort(t_s)
    t, i = t_s[order], (current_a * sign)[order]
    cap = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))]) / 3600.0
    span = cap.max() - cap.min()
    soc = 100.0 * (cap - cap.min()) / (span if span else 1.0)
    return t, soc


def _load(exp_dir: str) -> dict:
    exp = Path(exp_dir)
    asc = read_ascan(_find_one(str(exp / "ascan" / "*.h5")))
    cyc = read_cycler(_find_one(str(exp / "cycler" / "*.txt")))
    tmp = read_temperature(_find_one(str(exp / "temperature" / "*.csv")))

    a_t = asc["elapsed_s"]
    c_order = np.argsort(cyc["elapsed_s"])
    t_order = np.argsort(tmp["elapsed_s"])
    soc_t, soc = _coulomb_soc(cyc["elapsed_s"], cyc["current"], cyc["md"])

    return {
        "label": _find_one(str(exp / "ascan" / "*.h5")),
        "t_h": a_t / 3600.0,
        "tof_us": asc["tof_us"],
        "amplitude": asc["amplitude"],
        "energy": asc["energy"],
        "voltage": np.interp(a_t, cyc["elapsed_s"][c_order], cyc["voltage"][c_order]),
        "soc": np.interp(a_t, soc_t, soc),
        "temp_c": np.interp(a_t, tmp["elapsed_s"][t_order], tmp["temp_c"][t_order]),
    }


def compare(a_dir: str, b_dir: str, out_path: str | None, smooth_s: float) -> str:
    a, b = _load(a_dir), _load(b_dir)
    a_lbl, b_lbl = Path(a["label"]).stem, Path(b["label"]).stem

    fields = [
        ("tof_us", "ToF (us)", True),
        ("amplitude", "Amplitude (V)", True),
        ("energy", "Energy", True),
        ("voltage", "Voltage (V)", False),
        ("soc", "SOC (%)", False),
        ("temp_c", "Temp (C)", False),
    ]
    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 14), dpi=140, sharex=True)
    fig.suptitle("Two-session comparison: ultrasound + cycler context "
                 "(shared axes, 1-99 pct y-limits)", fontsize=11)

    for ax, (key, name, smooth) in zip(axes, fields):
        for d, lbl, col in ((a, a_lbl, "tab:blue"), (b, b_lbl, "tab:orange")):
            y = d[key]
            if smooth:
                dt = float(np.median(np.diff(d["t_h"]))) * 3600.0
                win = max(1, int(round(smooth_s / max(dt, 1e-6))))
                y = _rolling_mean(y, win)
            ax.plot(d["t_h"], y, color=col, linewidth=1.3, label=lbl)
        ax.set_ylim(*_robust_ylim([a[key], b[key]]))
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("elapsed time (hours)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if out_path is None:
        out_path = "reports/ascan_compare_full.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="experiment dir A (contains ascan/ cycler/ temperature/)")
    p.add_argument("b", help="experiment dir B")
    p.add_argument("--smooth", type=float, default=60.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    print("saved:", compare(args.a, args.b, args.out, args.smooth))


if __name__ == "__main__":
    main()
