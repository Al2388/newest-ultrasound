"""
Overlay two sessions across the full charge + full discharge only, time-aligned.

The raw sessions drift apart on the elapsed-time axis because the initial partial
discharge had a different duration in each run, shifting everything downstream.
Here we crop to just the full charge (Maccor steps 4+5) and full discharge
(step 7) - already extracted to data/experiments/<exp>/segments/*.csv - and
re-zero time to the start of charge, so both runs start together. Because the
charge and discharge durations match to ~1 %, they also end together; a short gap
marks the rest step that was cropped between them.

Six channels are shown on a shared time axis: ToF / amplitude / energy (ultrasound)
and voltage / SOC / temperature (cycler context). SOC here is the per-segment SOC
written by extract_charge_discharge.py (charge 0->100 %, discharge 100->0 %).

Usage
-----
  python compare_charge_discharge_aligned.py EXP_DIR_A EXP_DIR_B [--smooth 60] [--out out.png]
  e.g. python compare_charge_discharge_aligned.py data/experiments/18-5 data/experiments/21-5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_two_sessions_full import _robust_ylim, _rolling_mean


def _read_segment(csv_path: str) -> dict:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def _load_aligned(exp_dir: str) -> dict:
    """Concatenate full_charge then full_discharge on a charge-start-relative axis."""
    seg = Path(exp_dir) / "segments"
    chg = _read_segment(str(seg / "full_charge.csv"))
    dis = _read_segment(str(seg / "full_discharge.csv"))

    # Re-zero: charge starts at t=0; discharge follows after the charge duration,
    # with a NaN gap so the cropped rest step isn't drawn as a connecting line.
    t_chg = chg["elapsed_s"] - chg["elapsed_s"][0]
    chg_dur = t_chg[-1]
    t_dis = (dis["elapsed_s"] - dis["elapsed_s"][0]) + chg_dur

    keys = ["tof_us", "amplitude_v", "energy", "voltage_v", "soc_pct", "temp_c"]
    out = {"t_h": np.concatenate([t_chg, [np.nan], t_dis]) / 3600.0,
           "chg_end_h": chg_dur / 3600.0}
    for k in keys:
        out[k] = np.concatenate([chg[k], [np.nan], dis[k]])
    return out


def compare(a_dir: str, b_dir: str, out_path: str | None, smooth_s: float) -> str:
    a, b = _load_aligned(a_dir), _load_aligned(b_dir)
    a_lbl, b_lbl = Path(a_dir).name, Path(b_dir).name

    fields = [
        ("tof_us", "ToF (us)", True),
        ("amplitude_v", "Amplitude (V)", True),
        ("energy", "Energy", True),
        ("voltage_v", "Voltage (V)", False),
        ("soc_pct", "SOC (%)", False),
        ("temp_c", "Temp (C)", False),
    ]
    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 14), dpi=140, sharex=True)
    fig.suptitle("Charge + discharge only, aligned to charge start "
                 "(shared axes, 1-99 pct y-limits)", fontsize=11)

    for ax, (key, name, smooth) in zip(axes, fields):
        for d, lbl, col in ((a, a_lbl, "tab:blue"), (b, b_lbl, "tab:orange")):
            y = d[key]
            if smooth:
                dt = float(np.nanmedian(np.diff(d["t_h"]))) * 3600.0
                win = max(1, int(round(smooth_s / max(dt, 1e-6))))
                y = _rolling_mean(y, win)
            ax.plot(d["t_h"], y, color=col, linewidth=1.3, label=lbl)
        # mark the charge->discharge boundary (mean of the two runs' charge ends)
        ax.axvline((a["chg_end_h"] + b["chg_end_h"]) / 2.0, color="0.6",
                   linestyle="--", linewidth=0.9)
        ax.set_ylim(*_robust_ylim([d[key] for d in (a, b)]))
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time since charge start (hours)  -  dashed line = charge/discharge boundary")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if out_path is None:
        out_path = "reports/charge_discharge_aligned.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="experiment dir A (must contain segments/full_*.csv)")
    p.add_argument("b", help="experiment dir B")
    p.add_argument("--smooth", type=float, default=60.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    print("saved:", compare(args.a, args.b, args.out, args.smooth))


if __name__ == "__main__":
    main()
