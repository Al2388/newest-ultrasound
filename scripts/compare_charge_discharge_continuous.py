"""
Overlay two sessions over one continuous charge->discharge window, time-aligned.

Crops each session to the main cycle only - from the start of charge current
(Maccor step 4) to the end of discharge current (step 7) - and re-zeros time to
the charge start so both runs begin together. Unlike the segment-concatenation
view, the rest step *between* charge and discharge (step 6) is kept, so the
timeline is the continuous physical process with no artificial gap. The leading
partial discharge / initial rest and the trailing top-up are dropped because they
are the parts whose timing differed between runs.

Six channels on a shared axis: ToF / amplitude / energy (ultrasound) and
voltage / SOC / temperature (cycler). SOC is signed-current Coulomb counting over
the window, normalised so discharge-end = 0 % and charge-end = 100 %.

Usage
-----
  python compare_charge_discharge_continuous.py EXP_DIR_A EXP_DIR_B [--smooth 60] [--out out.png]
  e.g. python compare_charge_discharge_continuous.py data/experiments/18-5 data/experiments/21-5
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_two_sessions_full import _coulomb_soc, _robust_ylim, _rolling_mean
from extract_charge_discharge import _find_one, read_ascan, read_cycler, read_temperature

CHARGE_START_STEP = 4   # CC charge begins (main cycle start)
DISCHARGE_STEP = 7      # CC discharge (main cycle end)


def _load_window(exp_dir: str) -> dict:
    exp = Path(exp_dir)
    asc = read_ascan(_find_one(str(exp / "ascan" / "*.h5")))
    cyc = read_cycler(_find_one(str(exp / "cycler" / "*.txt")))
    tmp = read_temperature(_find_one(str(exp / "temperature" / "*.csv")))

    # Window: charge current starts (step 4) -> discharge current ends (step 7).
    t_start = cyc["elapsed_s"][cyc["step"] == CHARGE_START_STEP].min()
    t_end = cyc["elapsed_s"][cyc["step"] == DISCHARGE_STEP].max()
    t_disch = cyc["elapsed_s"][cyc["step"] == DISCHARGE_STEP].min()  # charge->discharge boundary

    a_t = asc["elapsed_s"]
    in_win = (a_t >= t_start) & (a_t <= t_end)
    a_t = a_t[in_win]

    c_order = np.argsort(cyc["elapsed_s"])
    t_order = np.argsort(tmp["elapsed_s"])
    soc_t, soc_full = _coulomb_soc(cyc["elapsed_s"], cyc["current"], cyc["md"])
    soc_win = np.interp(a_t, soc_t, soc_full)
    # renormalise SOC within the window: discharge-end = 0 %, charge-end = 100 %
    span = soc_win.max() - soc_win.min()
    soc_win = 100.0 * (soc_win - soc_win.min()) / (span if span else 1.0)

    rel = (a_t - t_start) / 3600.0
    return {
        "t_h": rel,
        "boundary_h": (t_disch - t_start) / 3600.0,
        "tof_us": asc["tof_us"][in_win],
        "amplitude_v": asc["amplitude"][in_win],
        "energy": asc["energy"][in_win],
        "voltage_v": np.interp(a_t, cyc["elapsed_s"][c_order], cyc["voltage"][c_order]),
        "soc_pct": soc_win,
        "temp_c": np.interp(a_t, tmp["elapsed_s"][t_order], tmp["temp_c"][t_order]),
    }


def compare(a_dir: str, b_dir: str, out_path: str | None, smooth_s: float) -> str:
    a, b = _load_window(a_dir), _load_window(b_dir)
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
    fig.suptitle("Continuous charge->discharge window, aligned to charge start "
                 "(shared axes, 1-99 pct y-limits)", fontsize=11)

    for ax, (key, name, smooth) in zip(axes, fields):
        for d, lbl, col in ((a, a_lbl, "tab:blue"), (b, b_lbl, "tab:orange")):
            y = d[key]
            if smooth:
                dt = float(np.median(np.diff(d["t_h"]))) * 3600.0
                win = max(1, int(round(smooth_s / max(dt, 1e-6))))
                y = _rolling_mean(y, win)
            ax.plot(d["t_h"], y, color=col, linewidth=1.3, label=lbl)
        ax.axvline((a["boundary_h"] + b["boundary_h"]) / 2.0, color="0.6",
                   linestyle="--", linewidth=0.9)
        ax.set_ylim(*_robust_ylim([d[key] for d in (a, b)]))
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time since charge start (hours)  -  dashed line = discharge start")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if out_path is None:
        out_path = "reports/charge_discharge_continuous.png"
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
