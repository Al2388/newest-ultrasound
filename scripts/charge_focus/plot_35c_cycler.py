"""Plot Maccor cycler data for the 35C LFP run.

V / I / Q / mode-band vs time, with the 14h charge-focus analysis window
highlighted so it's clear which segment maps to our acoustic analysis.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CYCLER = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
OUT = PROJ / "reports/longrun_cycling_35c_charge_focus/cycler_overview.png"

# Charge-focus analysis window (locked-in across the conversation)
FOCUS_T0 = datetime(2026, 5, 31,  7,  7, 49)
FOCUS_T1 = datetime(2026, 5, 31, 21,  7, 51)

MODE_COLOR = {"R": "#bbb", "C": "tab:red", "D": "tab:blue", "P": "#eee"}
MODE_LABEL = {"R": "rest", "C": "charge", "D": "discharge", "P": "pause"}


def main():
    df = pd.read_csv(CYCLER, sep="\t", skiprows=6, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DPT Time"] = pd.to_datetime(df["DPT Time"].astype(str),
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for c in ["Step", "Capacity", "Voltage", "Current"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MD"] = df["MD"].astype(str).str.strip()
    df = df.dropna(subset=["DPT Time"]).reset_index(drop=True)

    t = df["DPT Time"]
    V_mV = df["Voltage"].values * 1000
    I_mA = df["Current"].values * 1000
    Q_mAh = df["Capacity"].values * 1000   # Maccor capacity is per-step, in Ah -> mAh

    # Build cumulative SOC: Maccor Capacity column resets each step and is
    # absolute (|I|·dt accumulated). Sign is taken from MD (R=0, C=+, D=-).
    mode_sign = {"C": +1.0, "D": -1.0, "R": 0.0, "P": 0.0}
    dq = np.zeros(len(df))
    for step_id, sub in df.groupby("Step", sort=False):
        idx = sub.index.values
        q_step = sub["Capacity"].values * 1000
        d_abs = np.diff(q_step, prepend=q_step[0])
        d_abs = np.clip(d_abs, 0, None)       # only forward |Q| accumulation
        # sign by majority MD of the step
        md_majority = sub["MD"].mode().iloc[0]
        sign = mode_sign.get(md_majority, 0.0)
        dq[idx] = d_abs * sign
    Q_cum = np.cumsum(dq)
    SOC_pct = (Q_cum / 860.0) * 100.0
    SOC_pct = SOC_pct - SOC_pct.min()        # zero at minimum

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 0.6]})

    # ---- voltage ----
    ax = axes[0]
    ax.plot(t, V_mV, "-", color="tab:purple", lw=1.0)
    ax.set_ylabel("voltage [mV]")
    ax.grid(alpha=0.3)
    ax.set_title(f"35°C cycler overview  |  source: {CYCLER.name}", fontsize=11)
    ax.axvspan(FOCUS_T0, FOCUS_T1, color="orange", alpha=0.12,
               label="charge-focus 14h window")
    ax.legend(loc="upper right", fontsize=9)

    # ---- current ----
    ax = axes[1]
    ax.plot(t, I_mA, "-", color="tab:green", lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("current [mA]")
    ax.grid(alpha=0.3)
    ax.axvspan(FOCUS_T0, FOCUS_T1, color="orange", alpha=0.12)

    # ---- SOC (running coulomb-count) ----
    ax = axes[2]
    ax.plot(t, SOC_pct, "-", color="tab:red", lw=1.2)
    ax.set_ylabel("SOC [%]\n(coulomb-counted)")
    ax.grid(alpha=0.3)
    ax.axvspan(FOCUS_T0, FOCUS_T1, color="orange", alpha=0.12)
    for soc in [20, 40, 60, 80]:
        ax.axhline(soc, color="gray", lw=0.4, ls=":")

    # ---- mode band ----
    ax = axes[3]
    cur_mode = df["MD"].iloc[0]
    seg_t0 = t.iloc[0]
    for i in range(1, len(df)):
        if df["MD"].iloc[i] != cur_mode:
            ax.axvspan(seg_t0, t.iloc[i], color=MODE_COLOR.get(cur_mode, "#fff"),
                       ymin=0.2, ymax=0.8)
            cur_mode = df["MD"].iloc[i]
            seg_t0 = t.iloc[i]
    ax.axvspan(seg_t0, t.iloc[-1], color=MODE_COLOR.get(cur_mode, "#fff"),
               ymin=0.2, ymax=0.8)
    ax.set_yticks([])
    ax.set_ylabel("mode")
    ax.set_xlabel("time")
    # legend for modes
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=MODE_COLOR[m], label=MODE_LABEL[m])
               for m in ["R", "C", "D"]]
    ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))

    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT}")

    # quick text summary
    print(f"\nrun duration: {t.iloc[0]} -> {t.iloc[-1]}  "
          f"({(t.iloc[-1]-t.iloc[0]).total_seconds()/3600:.1f} h)")
    print(f"voltage span: {V_mV.min():.1f} -> {V_mV.max():.1f} mV")
    print(f"current span: {I_mA.min():.1f} -> {I_mA.max():.1f} mA")
    print(f"SOC span:     {SOC_pct.min():.1f}% -> {SOC_pct.max():.1f}%")
    print(f"steps:        {df['Step'].min():.0f}-{df['Step'].max():.0f}")


if __name__ == "__main__":
    main()
