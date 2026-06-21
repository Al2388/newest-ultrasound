"""ROI-mean ToF during rest, 4 SOC plateaus overlaid on a single axis.

Two-panel layout:
  (a) absolute ROI-mean ToF (ns) vs rest time -- four offset traces
  (b) ROI-mean ToF normalised to rest start (ΔToF = ToF(t) - ToF(0))
      so the four relaxation shapes can be compared directly
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
    20: "#3b75c6",   # blue
    40: "#3aa86c",   # green
    60: "#e08e1c",   # orange
    80: "#c34141",   # red
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
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
    sigma_ROI = NOISE_FLOOR["tof_sigma_roi_ns"]

    series = {}
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        t = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        idx = sel.index.values
        # ROI MEAN (the user asked for mean, not median)
        y = np.array([float(np.nanmean(tof_ns[i][roi])) for i in idx])
        series[soc] = (t, y)

    # ---- Figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)

    # Panel A: absolute ToF -- 4 traces
    for soc in SOC_LABELS:
        t, y = series[soc]
        axA.plot(t, y, "o-", color=SOC_COLOR[soc], lw=1.8, ms=5,
                 markeredgecolor="white", markeredgewidth=0.7,
                 label=f"SOC {soc} %")
    axA.set_xlabel("rest time [min]")
    axA.set_ylabel("ROI-mean ToF [ns]")
    axA.set_title("(a) Absolute ROI-mean ToF during 2 h rest", fontweight="bold")
    axA.set_xlim(-3, 115)
    axA.grid(alpha=0.20, linewidth=0.7)
    axA.legend(loc="best", fontsize=9.5, framealpha=0.92)

    # Panel B: normalised (ToF(t) - ToF(0))
    for soc in SOC_LABELS:
        t, y = series[soc]
        drift = y - y[0]
        axB.plot(t, drift, "o-", color=SOC_COLOR[soc], lw=1.8, ms=5,
                 markeredgecolor="white", markeredgewidth=0.7,
                 label=f"SOC {soc} %")
        axB.annotate(f"{drift[-1]:+.1f} ns",
                     xy=(t[-1], drift[-1]),
                     xytext=(4, 0), textcoords="offset points",
                     fontsize=8.5, color=SOC_COLOR[soc], fontweight="bold",
                     va="center")
    # noise floor band
    axB.axhline(0, color="#444", lw=0.5)
    axB.fill_between([-3, 130], -sigma_ROI, sigma_ROI, color="gray", alpha=0.18,
                     label=f"±σ_ROI = ±{sigma_ROI:.2f} ns")
    axB.set_xlabel("rest time [min]")
    axB.set_ylabel("ΔToF since rest start  [ns]")
    axB.set_title("(b) Drift relative to rest start", fontweight="bold")
    axB.set_xlim(-3, 125)
    axB.grid(alpha=0.20, linewidth=0.7)
    axB.legend(loc="best", fontsize=9.5, framealpha=0.92)

    fig.suptitle("ROI-mean ToF during the 2 h rest plateau at each SOC  (35 °C)",
                 fontsize=11.5, y=1.04)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")

    print("\nendpoint ROI-mean ToF and Δ_120min per SOC:")
    print(f"{'SOC':>4}  {'ToF(0) [ns]':>13}  {'ToF(end) [ns]':>14}  {'Δ_endpoint [ns]':>16}")
    for soc in SOC_LABELS:
        t, y = series[soc]
        print(f"{soc:>3}%  {y[0]:>13.1f}  {y[-1]:>14.1f}  {y[-1]-y[0]:>+16.1f}")


if __name__ == "__main__":
    main()
