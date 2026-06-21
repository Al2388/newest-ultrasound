"""35C: single twinx figure showing OCV drift vs (flat) cell temperature.

Narrative: voltage drifts throughout the 2 h rest at every SOC plateau,
while cell temperature is essentially flat. Excludes any "OCV drift is
just temperature" interpretation.

Output:
  ocv_35c_voltage_vs_temperature.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]

CYCLER = PROJ / "data" / "raw" / "cycler" / "LFP860_35degrees.002.txt"
CACHE_DIR = PROJ / "reports" / "longrun_cycling_35c_charge_focus" / "_cache"
OUT_PNG = PROJ / "reports" / "longrun_cycling_35c_charge_focus" / "ocv_35c_voltage_vs_temperature.png"

REST_PLATEAUS = [
    # (step, SOC label, rest start, color)
    (5,  20, datetime(2026, 5, 31,  7, 7, 49), "tab:blue"),
    (7,  40, datetime(2026, 5, 31, 11, 7, 49), "tab:green"),
    (9,  60, datetime(2026, 5, 31, 15, 7, 50), "tab:orange"),
    (11, 80, datetime(2026, 5, 31, 19, 7, 51), "tab:red"),
]
REST_DURATION = timedelta(hours=2)


def _load_cycler():
    df = pd.read_csv(CYCLER, sep="\t", skiprows=6, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DPT Time"] = pd.to_datetime(df["DPT Time"].astype(str),
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for c in ["Step", "Voltage", "Current"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    cyc = _load_cycler()
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])

    rests = []
    for step, soc, t0, color in REST_PLATEAUS:
        t1 = t0 + REST_DURATION
        # Voltage: high-res (5 s) from cycler
        v_trace = cyc[(cyc["DPT Time"] >= t0) & (cyc["DPT Time"] <= t1)].copy()
        v_trace["t_min"] = (v_trace["DPT Time"] - t0).dt.total_seconds() / 60.0
        # Temperature: per-scan (~9 min) from meta (line_T_mean_c)
        t_trace = meta[meta["step"] == step].sort_values("timestamp").copy()
        t_trace["t_min"] = (t_trace["timestamp"] - t_trace["timestamp"].iloc[0]).dt.total_seconds() / 60.0
        rests.append({
            "step": step, "soc": soc, "color": color,
            "v": v_trace, "T": t_trace,
        })

    # ----- aggregated stats for caption -----
    all_T = np.concatenate([r["T"]["line_T_mean_c"].values for r in rests])
    T_span_mC = (all_T.max() - all_T.min()) * 1000
    T_std_mC = np.std(all_T) * 1000
    T_mean = float(np.mean(all_T))
    v_spans = []
    for r in rests:
        v_spans.append(r["v"]["Voltage"].iloc[-1] - r["v"]["Voltage"].iloc[0])
    v_drift_mean_mV = float(np.mean(v_spans) * 1000)
    v_drift_max_mV = float(np.max(np.abs(v_spans)) * 1000)

    # ============== Figure ==============
    fig, axL = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    axR = axL.twinx()

    # Voltage curves (left axis, solid)
    for r in rests:
        axL.plot(r["v"]["t_min"], r["v"]["Voltage"] * 1000,
                 "-", color=r["color"], lw=1.6,
                 label=f"V  -  SOC {r['soc']}% (drift {(r['v']['Voltage'].iloc[-1]-r['v']['Voltage'].iloc[0])*1000:+.1f} mV)")

    # Temperature curves (right axis, dashed, muted)
    for r in rests:
        axR.plot(r["T"]["t_min"], r["T"]["line_T_mean_c"],
                 "s--", color=r["color"], lw=0.9, ms=4, alpha=0.45,
                 label=f"T  -  SOC {r['soc']}%")

    # ----- axis cosmetics -----
    axL.set_xlabel("rest time [min]", fontsize=11)
    axL.set_ylabel("cell voltage [mV]   (left axis, solid)", fontsize=11)
    axR.set_ylabel("cell temperature TC08 [degC]   (right axis, dashed)", fontsize=11, color="#444")

    axL.set_xlim(0, 120)
    # T axis: deliberately exaggerated so the flatness is visually obvious.
    # All TC08 readings fall in ~35.80-35.96, so a +/- 1 degC window around the
    # campaign mean makes them visually collapse to a near-horizontal band.
    axR.set_ylim(T_mean - 1.0, T_mean + 1.0)
    axR.tick_params(axis="y", colors="#444")

    axL.grid(alpha=0.3)

    # Two-column legend split between axes
    h_L, l_L = axL.get_legend_handles_labels()
    h_R, l_R = axR.get_legend_handles_labels()
    axL.legend(h_L + h_R, l_L + l_R, loc="lower right", fontsize=8, ncol=2,
               framealpha=0.9, title="(solid = V, dashed = TC08)")

    # ----- title + caption -----
    fig.suptitle(
        "35 degC LFP rest: cell voltage drifts continuously; cell temperature is flat\n"
        f"V drift per 2 h rest = {v_drift_mean_mV:+.1f} mV (mean), up to {v_drift_max_mV:.1f} mV max  |  "
        f"T span across all 4 rests = {T_span_mC:.0f} mC,  sigma_T = {T_std_mC:.0f} mC",
        fontsize=11,
    )

    # Side annotation block explaining what to look at
    annot = (
        "Reading guide:\n"
        "  - 4 SOLID lines (left axis) = cycler voltage at 5 s resolution\n"
        "  - 4 DASHED lines (right axis) = TC08 line-mean cell temperature, ~9 min spacing\n"
        "  - right-axis window is +/-1 degC around campaign mean, so the\n"
        "    < 0.2 degC envelope of measured T appears nearly flat\n"
        "  - voltage continues to drift even at t = 100-120 min:\n"
        "    end-of-rest is a pseudo-OCV, not a true equilibrium"
    )
    axL.text(0.015, 0.985, annot, transform=axL.transAxes, fontsize=8,
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")

    print(f"\nSummary:")
    print(f"  V drift per rest (signed end-start, mV): "
          f"{[f'{v*1000:+.1f}' for v in v_spans]}")
    print(f"  T span across all 4 rests = {T_span_mC:.1f} mC")
    print(f"  T std across all 4 rests  = {T_std_mC:.1f} mC")
    print(f"  T mean = {T_mean:.3f} degC")


if __name__ == "__main__":
    main()
