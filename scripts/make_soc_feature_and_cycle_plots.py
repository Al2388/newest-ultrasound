"""
Two presentation deliverables:
  A) SOC-vs-temperature sensitivity bars for the features where SOC dominates
     temperature AND the feature carries real SOC signal.
  B) 18-5 full-cycle overview: SOC, ToF, amplitude, energy, voltage, temperature
     on a shared elapsed-time axis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/experiments/repeatability")
TABLE_18 = "reports/experiments/19-5_feature_exploration/ascan_feature_table.csv"
RANK = "reports/experiments/repeatability/feature_ranking.csv"

# Curated list: SOC selectivity > 0.7 AND meaningful SOC information
# (partial_r2_soc > 0.25). Avoids "selective but flat" features.
MIN_SOC_R2 = 0.25
MIN_SELECTIVITY = 0.70


def sensitivity_bars():
    r = pd.read_csv(RANK)
    r = r[r.cross_session_comparable
          & (r.soc_selectivity > MIN_SELECTIVITY)
          & (r.partial_r2_soc > MIN_SOC_R2)].copy()
    r = r.sort_values("partial_r2_soc", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    y = np.arange(len(r))
    ax.barh(y, r["partial_r2_soc"], color="#2563eb", label="SOC sensitivity (partial R2)")
    ax.barh(y, -r["partial_r2_temp"], color="#dc2626", label="temperature sensitivity (partial R2)")
    ax.set_yticks(y); ax.set_yticklabels(r["feature"])
    ax.axvline(0, color="k", lw=0.8)
    for yi, (s, t) in enumerate(zip(r["partial_r2_soc"], r["partial_r2_temp"])):
        ax.text(s + 0.01, yi, f"{s:.2f}", va="center", fontsize=8, color="#1e3a8a")
        ax.text(-t - 0.01, yi, f"{t:.2f}", va="center", ha="right", fontsize=8, color="#7f1d1d")
    ax.set_xlabel("partial R2  (left = temperature, right = SOC)")
    ax.set_title("Features more sensitive to SOC than temperature\n"
                 "(pooled both repeat sessions; SOC block = cubic + branch)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "7_soc_vs_temp_sensitivity_bars.png")
    plt.close(fig)
    return r.sort_values("partial_r2_soc", ascending=False)


def cycle_overview():
    df = pd.read_csv(TABLE_18).sort_values("time_h")
    t = df["time_h"].to_numpy()
    panels = [
        ("Voltage (V)",      df["Voltage"],          "#1d4ed8"),
        ("SOC (%)",          df["soc_pct_clipped"],  "#047857"),
        ("Temperature (C)",  df["temperature_c"],    "#c2410c"),
        ("ToF abs (us)",     df["h5_tof_us_absolute"], "#b91c1c"),
        ("Amplitude (V)",    df["h5_amplitude"],     "#15803d"),
        ("Energy",           df["h5_energy"],        "#7c3aed"),
    ]
    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 12), dpi=150, sharex=True)
    for ax, (label, y, c) in zip(axs, panels):
        ax.plot(t, y, color=c, lw=0.7)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.25)
    axs[-1].set_xlabel("elapsed time (hours)")
    fig.suptitle("18-5 cycle overview  (full charge -> discharge protocol)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "8_cycle_overview_18-5.png")
    plt.close(fig)


def main():
    top = sensitivity_bars()
    cycle_overview()
    print("Wrote", OUT / "7_soc_vs_temp_sensitivity_bars.png")
    print("Wrote", OUT / "8_cycle_overview_18-5.png")
    print("\nTop SOC-dominant, informative features:")
    print(top[["feature", "partial_r2_soc", "partial_r2_temp",
               "soc_selectivity", "discriminability"]].to_string(index=False))


if __name__ == "__main__":
    main()
