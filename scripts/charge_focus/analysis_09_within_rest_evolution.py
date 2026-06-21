"""Within-rest evolution per SOC plateau.

For each of the 5 rest plateaus, plot rest-time evolution of:
- voltage (direct electrochemistry: OCV relaxation)
- ROI-mean amplitude
- ROI-mean ToF
- ROI-mean energy

A signal that is still drifting after 100 min, relative to its own noise
floor, is direct evidence that the cell interior is not at equilibrium.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
LONGRUN = PROJ / "scripts" / "longrun_analysis"
sys.path.insert(0, str(LONGRUN))

import common  # noqa: E402
from common import NOISE_FLOOR, Stack, rest_segments  # noqa: E402

NEW_OUT = PROJ / "reports" / "longrun_cycling_22h_charge_focus"
common.OUT_ROOT = NEW_OUT
common.CACHE_PATH = NEW_OUT / "_cache" / "stack.npz"
common.META_PATH = NEW_OUT / "_cache" / "meta.csv"
OUT = NEW_OUT / "09_within_rest_evolution"
OUT.mkdir(parents=True, exist_ok=True)

UNITS_DISPLAY = {
    "amplitude": ("mV", 1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns", 1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

_orig_load = common.load


def filtered_load(rebuild: bool = False) -> Stack:
    s = _orig_load(rebuild=False)
    keep = s.meta.step_tag.isin({"charge", "rest"}).values
    idx = np.where(keep)[0]
    new_meta = s.meta.iloc[idx].copy().reset_index(drop=True)
    soc_min = float(new_meta.soc_pct.min())
    new_meta["soc_pct"] = new_meta["soc_pct"] - soc_min
    return Stack(
        amplitude=s.amplitude[idx], tof=s.tof[idx], energy=s.energy[idx],
        x_mm=s.x_mm, y_mm=s.y_mm, roi_mask=s.roi_mask, meta=new_meta,
    )


def main():
    stack = filtered_load()
    segs = rest_segments(stack.meta)
    print(f"{len(segs)} rest segments")

    fig, axes = plt.subplots(4, len(segs), figsize=(4 * len(segs), 11), constrained_layout=True, sharex=True)

    summary = ["# Within-rest evolution per SOC plateau\n",
               "Each rest plateau ~100 min. Y axis centered on first-scan value so 0 = baseline.\n",
               "Grey band = ±2 σ_ROI noise floor. Any excursion outside is statistically real change.\n\n",
               "| SOC | V drift [mV] | amp drift [mV] | amp σ_ROI×2 | ToF drift [ns] | ToF σ_ROI×2 | energy drift | energy σ_ROI×2 |\n",
               "|---:|---:|---:|---:|---:|---:|---:|---:|\n"]

    for col, seg_meta in enumerate(segs):
        soc = float(seg_meta.soc_pct.iloc[0])
        seg_idx = np.where(np.isin(stack.meta.run_idx.values, seg_meta.run_idx.values))[0]
        t = pd.to_datetime(seg_meta.time_utc)
        t_min = (t - t.iloc[0]).dt.total_seconds().values / 60.0

        # Row 0: voltage
        V_mv = seg_meta.voltage_at_scan.values * 1000.0
        dV = V_mv - V_mv[0]
        ax = axes[0, col]
        ax.plot(t_min, dV, "o-", color="tab:red", ms=5, lw=1.2)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"SOC ≈ {soc:.0f} %")
        if col == 0:
            ax.set_ylabel("ΔV [mV]\n(OCV relaxation)")
        ax.grid(alpha=0.3)
        v_drift = float(dV[-1])

        # Rows 1-3: amp, tof, energy
        drifts = {}
        for r, mod in enumerate(["amplitude", "tof", "energy"], start=1):
            unit, scale, sigma_native = UNITS_DISPLAY[mod]
            arr = getattr(stack, mod)[seg_idx]
            roi_mean = np.array([np.nanmean(a[stack.roi_mask]) for a in arr]) * scale
            d = roi_mean - roi_mean[0]
            sigma_disp = sigma_native if scale == 1e3 else sigma_native

            ax = axes[r, col]
            ax.plot(t_min, d, "o-", color="#222", ms=5, lw=1.2)
            ax.axhline(0, color="k", lw=0.4)
            ax.fill_between(t_min, -2 * sigma_disp, 2 * sigma_disp, color="gray", alpha=0.18,
                            label="±2 σ_ROI" if col == 0 and r == 1 else None)
            if col == 0:
                ax.set_ylabel(f"Δ{mod} [{unit}]\n(σ_ROI = {sigma_disp:.2g})")
            ax.grid(alpha=0.3)
            drift = float(d[-1])
            drifts[mod] = (drift, sigma_disp)

            # mark drift z value on plot
            z = drift / sigma_disp if sigma_disp else np.nan
            ax.text(0.97, 0.04 if drift > 0 else 0.94, f"Δend={drift:+.2f} {unit}\n({z:+.1f}σ_ROI)",
                    transform=ax.transAxes, ha="right",
                    va="bottom" if drift > 0 else "top",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        axes[3, col].set_xlabel("rest time [min]")

        summary.append(
            f"| {soc:.0f}% | {v_drift:+.1f} | {drifts['amplitude'][0]:+.2f} | {drifts['amplitude'][1]*2:.2g} | "
            f"{drifts['tof'][0]:+.2f} | {drifts['tof'][1]*2:.2g} | {drifts['energy'][0]:+.3f} | {drifts['energy'][1]*2:.3g} |\n"
        )

    fig.suptitle("Within-rest evolution per SOC plateau  —  is the cell at equilibrium?", fontsize=13)
    fig.savefig(OUT / "within_rest_evolution.png", dpi=130)
    plt.close(fig)

    summary.append("\n## Verdict\n")
    summary.append(
        "If any cell of the table shows |drift| > σ_ROI×2, that modality is **still changing** at end of 100 min rest.\n"
        "Voltage drift > 5 mV on a flat-OCV chemistry like LFP corresponds to multiple % SoC of equivalent un-relaxation.\n"
    )
    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
