"""ROI-mean AND ROI-median ToF drift during 2 h rest, 4 SOCs overlaid.

Two separate PDFs (one for mean, one for median), identical style.
Legend moved outside the plot area so it never blocks data.
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

ALPHA_T_NS_PER_C = +73.5

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


def make_plot(d, meta, roi, aggregator, agg_label, out_pdf, out_png):
    tof_ns = d["tof"] * 1e3
    sigma_ROI = NOISE_FLOOR["tof_sigma_roi_ns"]

    series = {}
    sigma_T_per_plateau = {}
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        t = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        idx = sel.index.values
        y = np.array([float(aggregator(tof_ns[i][roi])) for i in idx])
        series[soc] = (t, y)
        sigma_T_per_plateau[soc] = float(sel["line_T_mean_c"].std(ddof=1) * 1000)

    sigma_T_max_mC = max(sigma_T_per_plateau.values())
    thermal_3sig = 3 * (sigma_T_max_mC / 1000.0) * ALPHA_T_NS_PER_C

    # Figure: reserve right strip for legend
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)

    ax.fill_between([-3, 130], -thermal_3sig, +thermal_3sig,
                    color="#f1c060", alpha=0.30, zorder=0,
                    label=f"3σ_T thermal envelope (±{thermal_3sig:.2f} ns)")
    ax.fill_between([-3, 130], -sigma_ROI, +sigma_ROI,
                    color="#9e9e9e", alpha=0.30, zorder=1,
                    label=f"±σ_ROI noise floor (±{sigma_ROI:.2f} ns)")
    ax.axhline(0, color="#444", lw=0.6, zorder=2)

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

    ax.set_xlim(-3, 130)
    ax.set_xlabel("rest time [min]")
    ax.set_ylabel(f"Δ{agg_label} ToF since rest start  [ns]")
    ax.set_title(f"ROI-{agg_label} ToF drift during 2 h rest (35 °C), four SOCs overlaid",
                 fontweight="bold")
    ax.grid(alpha=0.20, linewidth=0.7)

    # Legend outside the axes on the right
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9.5, framealpha=0.95, frameon=True,
              borderaxespad=0)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[{agg_label}] worst-case sigma_T = {sigma_T_max_mC:.1f} mC, "
          f"3 sigma_T envelope = ±{thermal_3sig:.2f} ns")
    print(f"[{agg_label}] endpoint drift per SOC:")
    for soc in SOC_LABELS:
        t, y = series[soc]
        de = y[-1] - y[0]
        print(f"  SOC {soc}%:  dToF = {de:+7.2f} ns   "
              f"(thermal x = {de/thermal_3sig:+5.2f},  n_sigma = {de/sigma_ROI:+5.2f})")


def main():
    d = np.load(CACHE / "stack.npz")
    meta = pd.read_csv(CACHE / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    roi = build_roi(d["x_mm"], d["y_mm"])

    make_plot(d, meta, roi,
              aggregator=np.nanmean,
              agg_label="mean",
              out_pdf=OUT_DIR / "fig_roi_mean_tof_rest.pdf",
              out_png=OUT_DIR / "fig_roi_mean_tof_rest.png")

    make_plot(d, meta, roi,
              aggregator=np.nanmedian,
              agg_label="median",
              out_pdf=OUT_DIR / "fig_roi_median_tof_rest.pdf",
              out_png=OUT_DIR / "fig_roi_median_tof_rest.png")


if __name__ == "__main__":
    main()
