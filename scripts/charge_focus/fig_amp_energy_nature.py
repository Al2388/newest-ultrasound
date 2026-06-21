"""Nature-style figure: amplitude + energy drift during rest, all on one axis.

Both features are normalised to their own sigma_ROI so they sit on a common
y-scale (units of sigma). 2x2 panels, one per SOC. Minimal chartjunk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import siegelslopes

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

CACHE = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "fig_evolution_amp_energy.pdf"
OUT_PNG = OUT_DIR / "fig_evolution_amp_energy.png"
OUT_CSV = OUT_DIR / "fig_evolution_amp_energy_data.csv"

ROI_X_MM = (14.6, 64.5)
ROI_Y_MM = (16.1, 55.4)
REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]
PANEL_LETTER = {20: "a", 40: "b", 60: "c", 80: "d"}

SIG_AMP = NOISE_FLOOR["amp_sigma_roi_mv"]
SIG_ENG = NOISE_FLOOR["energy_sigma_roi"]

# Restrained Nature-ish palette
COLOR_AMP    = "#c34141"   # warm red, for amp
COLOR_ENERGY = "#3b75c6"   # blue, for energy
COLOR_BAND   = "#bdbdbd"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",       # Arial-substitute
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      9,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "legend.frameon":     False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "savefig.dpi":        300,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
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
    roi = build_roi(d["x_mm"], d["y_mm"])
    amp_mV = d["amplitude"] * 1e3
    eng_au = d["energy"]

    # Collect per-SOC data
    by_soc = {}
    csv_rows = []
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        t = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        idx = sel.index.values
        amp = np.array([float(np.nanmedian(amp_mV[i][roi])) for i in idx])
        eng = np.array([float(np.nanmedian(eng_au[i][roi])) for i in idx])

        d_amp = amp - amp[0]      # ΔmV
        d_eng = eng - eng[0]      # Δa.u.
        z_amp = d_amp / SIG_AMP   # sigma units
        z_eng = d_eng / SIG_ENG

        # Theil-Sen on the absolute (signed mV / a.u.)
        s_amp, b_amp = siegelslopes(amp, t)
        s_eng, b_eng = siegelslopes(eng, t)
        d120_amp = s_amp * 120
        d120_eng = s_eng * 120

        by_soc[soc] = {
            "t": t, "amp": amp, "eng": eng,
            "z_amp": z_amp, "z_eng": z_eng,
            "fit_amp": (s_amp, b_amp), "fit_eng": (s_eng, b_eng),
            "d120_amp": d120_amp, "d120_eng": d120_eng,
        }
        for k in range(len(t)):
            csv_rows.append({
                "SOC%": soc, "t_rest_min": float(t[k]),
                "amp_med_mV": float(amp[k]),
                "energy_med_au": float(eng[k]),
                "d_amp_mV": float(d_amp[k]),
                "d_energy_au": float(d_eng[k]),
                "z_amp_sigmaROI": float(z_amp[k]),
                "z_energy_sigmaROI": float(z_eng[k]),
            })

    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False, float_format="%.6g")

    # Y range from all 8 traces
    pooled_z = np.concatenate([np.r_[by_soc[s]["z_amp"], by_soc[s]["z_eng"]]
                                for s in SOC_LABELS])
    ymax = max(35, np.nanmax(np.abs(pooled_z)) * 1.1)
    ylim = (-ymax, ymax)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), constrained_layout=True,
                              sharex=True, sharey=True)

    for ax, soc in zip(axes.ravel(), SOC_LABELS):
        bs = by_soc[soc]
        # ±1σ_ROI band on the normalised axis
        ax.fill_between([-3, 130], -1, +1, color=COLOR_BAND, alpha=0.30, lw=0,
                        zorder=0)
        ax.axhline(0, color="#444", lw=0.5, zorder=1)

        # amplitude scatter + fit
        ax.scatter(bs["t"], bs["z_amp"],
                   color=COLOR_AMP, s=22, zorder=4,
                   edgecolors="white", linewidths=0.6,
                   label="amplitude")
        s, b = bs["fit_amp"]
        tline = np.array([bs["t"].min(), bs["t"].max()])
        # convert fit (absolute amp) to z: (a*t+b - amp[0]) / σ
        z_amp_fit = (s*tline + b - bs["amp"][0]) / SIG_AMP
        ax.plot(tline, z_amp_fit, "-", color=COLOR_AMP, lw=1.6, zorder=3)

        # energy
        ax.scatter(bs["t"], bs["z_eng"],
                   color=COLOR_ENERGY, s=22, marker="s", zorder=4,
                   edgecolors="white", linewidths=0.6,
                   label="energy")
        s2, b2 = bs["fit_eng"]
        z_eng_fit = (s2*tline + b2 - bs["eng"][0]) / SIG_ENG
        ax.plot(tline, z_eng_fit, "-", color=COLOR_ENERGY, lw=1.6, zorder=3)

        ax.set_xlim(-2, 125)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 30, 60, 90, 120])

        # Panel label (a, b, c, d) -- Nature convention, top-left, bold
        ax.text(-0.18, 1.06, PANEL_LETTER[soc], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="left")
        # SOC subtitle, top-center small
        ax.text(0.5, 1.06, f"SOC {soc} %", transform=ax.transAxes,
                fontsize=10, va="bottom", ha="center", color="#222")

        # Annotation: endpoint deltas in physical units
        ax.text(0.97, 0.04,
                f"Δamp = {bs['d120_amp']:+.1f} mV\n"
                f"Δenergy = {bs['d120_eng']:+.2f} a.u.",
                transform=ax.transAxes, va="bottom", ha="right",
                fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#888", lw=0.4))

    # Common axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Rest time (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ feature  /  σ_ROI")

    # Single legend at the top centre
    handles = [
        matplotlib.lines.Line2D([0], [0], marker="o", color=COLOR_AMP,
                                lw=1.6, ms=6, markeredgecolor="white",
                                markeredgewidth=0.6, label="amplitude  (σ_ROI = 2.22 mV)"),
        matplotlib.lines.Line2D([0], [0], marker="s", color=COLOR_ENERGY,
                                lw=1.6, ms=6, markeredgecolor="white",
                                markeredgewidth=0.6, label="energy  (σ_ROI = 0.0814 a.u.)"),
        matplotlib.patches.Patch(facecolor=COLOR_BAND, alpha=0.30,
                                 label="±1 σ_ROI noise band"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
