"""Standalone Panel (b): ROI-mean ToF drift during 2 h rest, all 4 SOCs.

Shows:
  - 4 overlaid ΔToF traces (one per SOC)
  - ±σ_ROI noise band (gray)
  - 3σ_T thermal envelope (warm yellow) using the worst-case per-plateau σ_T
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

CACHE = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "fig_roi_mean_tof_rest.pdf"
OUT_PNG = OUT_DIR / "fig_roi_mean_tof_rest.png"

ROI_X_MM = (14.6, 64.5)
ROI_Y_MM = (16.1, 55.4)
REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]
SOC_COLOR = {
    20: "#3b75c6",
    40: "#3aa86c",
    60: "#e08e1c",
    80: "#c34141",
}

ALPHA_T_NS_PER_C = +73.5    # ns / degC, in-situ measured

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":      300,
    "pdf.fonttype":      42,
})


def build_roi(x_mm, y_mm):
    x_idx = np.where((x_mm >= ROI_X_MM[0]) & (x_mm <= ROI_X_MM[1]))[0]
    y_idx = np.where((y_mm >= ROI_Y_MM[0]) & (y_mm <= ROI_Y_MM[1]))[0]
    roi = np.zeros((y_mm.size, x_mm.size), dtype=bool)
    roi[np.ix_(y_idx, x_idx)] = True
    return roi


def main():
    d = np.load(CACHE / "stack.npz")
    meta = pd.read_csv(CACHE / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    tof_ns = d["tof"] * 1e3
    roi = build_roi(d["x_mm"], d["y_mm"])
    sigma_ROI = NOISE_FLOOR["tof_sigma_roi_ns"]    # 4.70 ns

    series = {}
    sigma_T_per_plateau_mC = {}
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        t = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        idx = sel.index.values
        y = np.array([float(np.nanmean(tof_ns[i][roi])) for i in idx])
        series[soc] = (t, y)
        sigma_T_per_plateau_mC[soc] = float(sel["line_T_mean_c"].std(ddof=1) * 1000)

    # Worst-case 3 sigma_T thermal envelope (use the largest per-plateau sigma_T)
    sigma_T_max_mC = max(sigma_T_per_plateau_mC.values())
    sigma_T_max_C = sigma_T_max_mC / 1000.0
    thermal_3sig_ns = 3 * sigma_T_max_C * ALPHA_T_NS_PER_C

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)

    # Thermal envelope (drawn first, behind everything)
    ax.fill_between([-3, 130], -thermal_3sig_ns, +thermal_3sig_ns,
                    color="#f1c060", alpha=0.30, zorder=0,
                    label=f"3σ_T thermal envelope = ±{thermal_3sig_ns:.2f} ns "
                          f"(worst-case σ_T = {sigma_T_max_mC:.0f} mC)")
    # Noise floor band
    ax.fill_between([-3, 130], -sigma_ROI, +sigma_ROI,
                    color="#9e9e9e", alpha=0.30, zorder=1,
                    label=f"±σ_ROI noise floor = ±{sigma_ROI:.2f} ns")

    ax.axhline(0, color="#444", lw=0.6, zorder=2)

    # The 4 SOC traces
    for soc in SOC_LABELS:
        t, y = series[soc]
        drift = y - y[0]
        ax.plot(t, drift, "o-", color=SOC_COLOR[soc], lw=2.0, ms=6,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=4, label=f"SOC {soc} %")
        ax.annotate(f"{drift[-1]:+.1f} ns",
                    xy=(t[-1], drift[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=10, color=SOC_COLOR[soc], fontweight="bold",
                    va="center")

    ax.set_xlim(-3, 128)
    ax.set_xlabel("rest time [min]")
    ax.set_ylabel("ΔToF since rest start  [ns]")
    ax.set_title("ROI-mean ToF drift during 2 h rest plateau, four SOCs overlaid\n"
                 "(35 °C cell; gray band = noise floor; yellow band = thermal envelope)",
                 fontweight="bold")
    ax.grid(alpha=0.20, linewidth=0.7)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92, ncol=1)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"\nworst-case sigma_T across plateaus = {sigma_T_max_mC:.1f} mC")
    print(f"3 sigma_T thermal envelope = +/- {thermal_3sig_ns:.2f} ns")
    print(f"sigma_ROI noise floor      = +/- {sigma_ROI:.2f} ns")
    print(f"\nper-plateau ROI-mean ToF endpoint drift:")
    for soc in SOC_LABELS:
        t, y = series[soc]
        drift_end = y[-1] - y[0]
        thermal_mult = drift_end / thermal_3sig_ns
        noise_n = drift_end / sigma_ROI
        print(f"  SOC {soc}%: dToF = {drift_end:+7.2f} ns  "
              f"(thermal x = {thermal_mult:+5.2f}, n_sigma = {noise_n:+5.2f})")


if __name__ == "__main__":
    main()
