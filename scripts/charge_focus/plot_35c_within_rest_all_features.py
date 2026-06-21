"""35C: within-rest evolution of cell-ROI features at each SOC plateau.

5 rows x 4 cols figure (no reference subtraction):
  rows:  ΔV (mV),  ΔT (mC),  Δamp (mV),  ΔToF (ns),  Δenergy (a.u.)
  cols:  SOC 20% / 40% / 60% / 80%
  x:     rest time [min]
  y:     value relative to t = 0 of that plateau
  bands: ±2σ_ROI noise floor on the 3 acoustic rows
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"

UNITS = {
    "amplitude": ("mV",   1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns",   1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

REST_STEPS  = [5, 7, 9, 11]
SOC_LABELS  = [20, 40, 60, 80]
SOC_COLORS  = {20: "tab:blue", 40: "tab:green", 60: "tab:orange", 80: "tab:red"}


def _roi_mean(arr, mask):
    return np.nanmean(arr.reshape(arr.shape[0], -1)[:, mask.reshape(-1)], axis=1)


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    roi = d["roi_mask"]
    arrs = {"amplitude": d["amplitude"], "tof": d["tof"], "energy": d["energy"]}

    fig, axes = plt.subplots(5, 4, figsize=(17, 13), constrained_layout=True, sharex=True)

    for col, (step, soc) in enumerate(zip(REST_STEPS, SOC_LABELS)):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        idx = sel.index.values
        t = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        color = SOC_COLORS[soc]

        # --- Row 0: ΔV (mV) ---
        V_mV = sel["voltage_at_scan"].values * 1000.0
        dV = V_mV - V_mV[0]
        ax = axes[0, col]
        ax.plot(t, dV, "o-", color=color, lw=1.6, ms=5)
        ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
        ax.set_title(f"SOC ≈ {soc}% rest (step {step})",
                     fontsize=11, color=color, fontweight="bold")
        if col == 0:
            ax.set_ylabel("ΔV [mV]\n(rest start = 0)")
        ax.annotate(f"end {dV[-1]:+.1f} mV", xy=(0.97, 0.04), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85))

        # --- Row 1: ΔT (mC) ---
        T_C = sel["line_T_mean_c"].values
        dT_mC = (T_C - T_C[0]) * 1000.0
        ax = axes[1, col]
        ax.plot(t, dT_mC, "s-", color="tab:purple", lw=1.1, ms=4, alpha=0.85)
        ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
        ax.fill_between(t, -200, 200, color="tab:purple", alpha=0.05)
        if col == 0:
            ax.set_ylabel("ΔT [mC]\nTC08 per-scan")
        ax.annotate(f"end {dT_mC[-1]:+.0f} mC", xy=(0.97, 0.04), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85))

        # --- Rows 2-4: acoustic features (raw, no reference subtraction) ---
        for r, mod in enumerate(["amplitude", "tof", "energy"], start=2):
            unit, scale, sigma = UNITS[mod]
            roi_series = _roi_mean(arrs[mod][idx], roi) * scale
            d_roi = roi_series - roi_series[0]

            ax = axes[r, col]
            # 2σ_ROI noise floor band
            ax.fill_between(t, -2*sigma, 2*sigma, color="gray", alpha=0.18,
                            label=f"±2σ_ROI = ±{2*sigma:.2g} {unit}" if (col == 0 and r == 2) else None)
            ax.plot(t, d_roi, "o-", color=color, lw=1.6, ms=5)
            ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"Δ{mod} [{unit}]\n(σ_ROI = {sigma:.2g})")
            # end-of-rest annotation with z value
            z = d_roi[-1] / sigma
            ax.annotate(f"end {d_roi[-1]:+.2g} {unit}\n({z:+.1f}σ_ROI)",
                        xy=(0.97, 0.04), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85))

        axes[-1, col].set_xlabel("rest time [min]")

    fig.suptitle(
        "35 °C charge-focus: within-rest evolution of cell-ROI features\n"
        "rows: ΔV, ΔT, Δamplitude, ΔToF, Δenergy   |   columns: SOC plateaus 20–80 %\n"
        "raw cell-ROI mean (no reference subtraction);  grey band = ±2 σ_ROI noise floor",
        fontsize=12,
    )

    out = OUT_ROOT / "within_rest_evolution_all_features.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
