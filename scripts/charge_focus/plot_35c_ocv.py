"""35C end-of-rest pseudo-OCV plot.

Each of the 4 rest plateaus (SOC 20/40/60/80%) is 2 h, so end-of-rest voltage
is a good OCV proxy (LFP relaxes >95% within ~30-60 min after CC cutoff).

Outputs:
  ocv_35c_endrest.png  - 2-panel figure: (a) end-of-rest V vs SOC,
                                          (b) V(t) relaxation during each rest
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
OUT_PNG = PROJ / "reports" / "longrun_cycling_35c_charge_focus" / "ocv_35c_endrest.png"

REST_PLATEAUS = [
    # (step, SOC label, rest start)
    (5,  20, datetime(2026, 5, 31,  7, 7, 49)),
    (7,  40, datetime(2026, 5, 31, 11, 7, 49)),
    (9,  60, datetime(2026, 5, 31, 15, 7, 50)),
    (11, 80, datetime(2026, 5, 31, 19, 7, 51)),
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

    # ----- per-plateau cycler trace (high-res V) and scan-time V (low-res) -----
    rest_data = []
    for step, soc, t0 in REST_PLATEAUS:
        t1 = t0 + REST_DURATION
        # cycler high-res trace (~5 s spacing)
        sub_cyc = cyc[(cyc["DPT Time"] >= t0) & (cyc["DPT Time"] <= t1)].copy()
        sub_cyc["t_min"] = (sub_cyc["DPT Time"] - t0).dt.total_seconds() / 60.0
        # scans (~9 min spacing) - voltage_at_scan from meta
        sub_meta = meta[meta["step"] == step].sort_values("timestamp").copy()
        sub_meta["t_min"] = (sub_meta["timestamp"] - sub_meta["timestamp"].iloc[0]).dt.total_seconds() / 60.0
        # end-of-rest V = mean of last 5 cycler samples (~25 s before rest end)
        v_end = float(sub_cyc["Voltage"].iloc[-5:].mean())
        v_start = float(sub_cyc["Voltage"].iloc[0])
        rest_data.append({
            "step": step, "soc": soc, "t0": t0,
            "v_end": v_end, "v_start": v_start,
            "cyc": sub_cyc, "meta": sub_meta,
        })

    # ============== Figure ==============
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.0, 1.4]})

    # ---- Panel A: end-of-rest V vs SOC (the "OCV") ----
    socs = np.array([r["soc"] for r in rest_data])
    v_ends = np.array([r["v_end"] for r in rest_data])
    axA.plot(socs, v_ends, "o-", color="tab:red", ms=10, lw=1.6, label="end-of-rest V (2 h rest)")
    for s, v in zip(socs, v_ends):
        axA.annotate(f"  {v*1000:.1f} mV", (s, v), fontsize=9,
                     xytext=(6, -3), textcoords="offset points")
    # plateau slope annotation
    slope = float(np.polyfit(socs, v_ends, 1)[0]) * 1000  # mV/%SoC
    axA.set_xlabel("SOC [%]")
    axA.set_ylabel("end-of-rest cell voltage [V]")
    axA.set_title(f"(a) 35C pseudo-OCV  (slope = {slope:+.2f} mV/%SoC across plateau)", fontsize=10)
    axA.grid(alpha=0.3)
    axA.legend(loc="best", fontsize=9)
    axA.set_xticks([0, 20, 40, 60, 80, 100])
    axA.set_xlim(10, 90)

    # ---- Panel B: relaxation V(t) during each rest ----
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    for r, color in zip(rest_data, colors):
        cyc = r["cyc"]
        axB.plot(cyc["t_min"], cyc["Voltage"]*1000, "-", color=color, lw=1.2,
                 label=f"SOC {r['soc']}% (V_end = {r['v_end']*1000:.1f} mV)")
        # mark final voltage with a dot
        axB.plot([cyc["t_min"].iloc[-1]], [r["v_end"]*1000], "o", color=color, ms=7)
    axB.set_xlabel("rest time [min]")
    axB.set_ylabel("cell voltage [mV]")
    axB.set_title("(b) voltage relaxation during each rest (5 s resolution from cycler)", fontsize=10)
    axB.grid(alpha=0.3)
    axB.legend(loc="best", fontsize=8)
    axB.set_xlim(0, 120)

    fig.suptitle("35C subset (SOC 20 -> 80% with 2 h rests)  -  pseudo-OCV from end-of-rest voltage",
                 fontsize=11)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")

    # short text summary
    print("\nend-of-rest voltages (LFP plateau is very flat):")
    for r in rest_data:
        relax_mV = (r["v_start"] - r["v_end"]) * 1000
        print(f"  SOC {r['soc']:2d}%  V_end = {r['v_end']*1000:7.2f} mV   "
              f"(relaxed {relax_mV:+.1f} mV during the 2 h rest)")
    print(f"\nslope across plateau = {slope:+.2f} mV/%SoC  "
          f"(LFP signature: near-flat, hence the need for ultrasound)")


if __name__ == "__main__":
    main()
