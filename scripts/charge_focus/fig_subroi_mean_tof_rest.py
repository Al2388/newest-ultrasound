"""Sub-ROI ToF drift during 2 h rest, split into tab-distal / mid / tab-proximal.

For each X-band (tab-distal, mid, tab-proximal) compute the band-mean ToF over
the canonical Y range, then plot ΔToF since rest start for all 4 SOC plateaus
(20/40/60/80 %) in a 1x3 panel figure. Noise floor (±σ_ROI) and worst-case
3 σ_T thermal envelope are drawn behind the traces, identical to the
whole-ROI version.

Hypothesis being checked: during a 2 h rest after charge step, Li redistributes
and density gradients relax; ToF should drift up (slower sound speed as Li
spreads more uniformly). Splitting by X-band tests whether this signal lives
in a specific region (e.g. tab-distal) while being averaged out at whole-ROI.
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

ROI_Y_MM = (16.1, 55.4)
SUBREGIONS = [
    ("tab-distal", 14.6, 30.0),
    ("mid",        30.0, 50.0),
    ("tab-prox",   50.0, 64.5),
]
REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]
SOC_COLOR = {20: "#3b75c6", 40: "#3aa86c", 60: "#e08e1c", 80: "#c34141"}

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


def band_mask(x_mm, y_mm, x_lo, x_hi):
    x_idx = np.where((x_mm >= x_lo) & (x_mm <= x_hi))[0]
    y_idx = np.where((y_mm >= ROI_Y_MM[0]) & (y_mm <= ROI_Y_MM[1]))[0]
    m = np.zeros((y_mm.size, x_mm.size), dtype=bool)
    m[np.ix_(y_idx, x_idx)] = True
    return m


def compute_series(tof_ns, meta, mask, aggregator):
    series = {}
    sigma_T_mC = {}
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        t = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        idx = sel.index.values
        y = np.array([float(aggregator(tof_ns[i][mask])) for i in idx])
        series[soc] = (t, y)
        sigma_T_mC[soc] = float(sel["line_T_mean_c"].std(ddof=1) * 1000)
    return series, sigma_T_mC


def main():
    d = np.load(CACHE / "stack.npz")
    meta = pd.read_csv(CACHE / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    tof_ns = d["tof"] * 1e3
    x_mm, y_mm = d["x_mm"], d["y_mm"]
    sigma_ROI = NOISE_FLOOR["tof_sigma_roi_ns"]

    band_series, band_sigma_T = {}, {}
    for name, x_lo, x_hi in SUBREGIONS:
        m = band_mask(x_mm, y_mm, x_lo, x_hi)
        s, sT = compute_series(tof_ns, meta, m, np.nanmean)
        band_series[name] = s
        band_sigma_T[name] = sT
        print(f"[{name}]  X in [{x_lo}, {x_hi}] mm   pixels = {int(m.sum())}")

    sigma_T_max_mC = max(max(v.values()) for v in band_sigma_T.values())
    thermal_3sig = 3 * (sigma_T_max_mC / 1000.0) * ALPHA_T_NS_PER_C

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), sharey=True,
                             constrained_layout=True)

    for ax, (name, x_lo, x_hi) in zip(axes, SUBREGIONS):
        ax.fill_between([-3, 130], -thermal_3sig, +thermal_3sig,
                        color="#f1c060", alpha=0.30, zorder=0,
                        label=f"3σ_T thermal (±{thermal_3sig:.1f} ns)")
        ax.fill_between([-3, 130], -sigma_ROI, +sigma_ROI,
                        color="#9e9e9e", alpha=0.30, zorder=1,
                        label=f"±σ_ROI noise (±{sigma_ROI:.2f} ns)")
        ax.axhline(0, color="#444", lw=0.6, zorder=2)

        series = band_series[name]
        for soc in SOC_LABELS:
            t, y = series[soc]
            drift = y - y[0]
            ax.plot(t, drift, "o-", color=SOC_COLOR[soc], lw=1.8, ms=5,
                    markeredgecolor="white", markeredgewidth=0.7,
                    zorder=4, label=f"SOC {soc} %")
            ax.annotate(f"{drift[-1]:+.1f}", xy=(t[-1], drift[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=9, color=SOC_COLOR[soc], fontweight="bold",
                        va="center")

        ax.set_xlim(-3, 130)
        ax.set_xlabel("rest time [min]")
        ax.set_title(f"{name}   X = {x_lo:.1f}–{x_hi:.1f} mm",
                     fontweight="bold")
        ax.grid(alpha=0.20, linewidth=0.7)

    axes[0].set_ylabel("Δmean ToF since rest start  [ns]")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=9, framealpha=0.95, frameon=True,
                    borderaxespad=0)

    fig.suptitle("Sub-ROI mean ToF drift during 2 h rest (35 °C), 4 SOCs overlaid",
                 fontsize=13, fontweight="bold")

    out_pdf = OUT_DIR / "fig_subroi_mean_tof_rest.pdf"
    out_png = OUT_DIR / "fig_subroi_mean_tof_rest.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nworst-case sigma_T (any band, any plateau) = {sigma_T_max_mC:.1f} mC")
    print(f"3 sigma_T envelope = +/- {thermal_3sig:.2f} ns")
    print(f"sigma_ROI noise floor = +/- {sigma_ROI:.2f} ns\n")

    rows = []
    for name, _, _ in SUBREGIONS:
        s = band_series[name]
        for soc in SOC_LABELS:
            t, y = s[soc]
            de = y[-1] - y[0]
            rows.append({
                "subregion": name, "SOC_pct": soc,
                "dToF_ns": round(de, 2),
                "thermal_x": round(de / thermal_3sig, 2),
                "n_sigma_ROI": round(de / sigma_ROI, 2),
            })
    df = pd.DataFrame(rows)
    csv = OUT_DIR / "fig_subroi_mean_tof_rest.csv"
    df.to_csv(csv, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
