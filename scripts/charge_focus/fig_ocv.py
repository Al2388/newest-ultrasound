"""Figure: OCV vs rest time for all 5 rest plateaus of the 35C run.

Publication-quality style matching the rest of the thesis figure suite.
Two panels:
  (a) full V(t) traces, 5 sec resolution from the Maccor cycler
  (b) zoomed early relaxation (first 30 min) so the small SOC 20-80%
      drifts are legible
Annotations: endpoint drift in mV per plateau.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CYCLER = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "fig_ocv.pdf"
OUT_PNG = OUT_DIR / "fig_ocv.png"

REST_PLATEAUS = [
    # (step, SOC label, color)
    (3,  "floor (0 %)", "#7a7a7a"),
    (5,  "SOC 20 %",    "#3b75c6"),
    (7,  "SOC 40 %",    "#3aa86c"),
    (9,  "SOC 60 %",    "#e08e1c"),
    (11, "SOC 80 %",    "#c34141"),
]

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


def main():
    df = pd.read_csv(CYCLER, sep="\t", skiprows=6, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DPT Time"] = pd.to_datetime(df["DPT Time"].astype(str),
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
    df = df.dropna(subset=["DPT Time"]).reset_index(drop=True)

    # Per-plateau (t_min, V_mV) traces
    traces = []
    for step, label, color in REST_PLATEAUS:
        sub = df[df["Step"] == step].copy()
        sub["t_min"] = (sub["DPT Time"] - sub["DPT Time"].iloc[0]).dt.total_seconds() / 60
        sub["V_mV"] = sub["Voltage"] * 1000
        traces.append({"step": step, "label": label, "color": color,
                       "t": sub["t_min"].values, "v": sub["V_mV"].values})

    # ----- Figure -----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.1, 1.0]})

    # ===== Panel A: full 2h OCV, all plateaus on log-like dual scale =====
    # Use two y-axes so floor and SOC 20-80% are both visible.
    # Floor is ~2.5-2.9 V, SOC 20-80% is 3.27-3.36 V
    axA_top = axA
    axA_top.spines["right"].set_visible(False)
    axA_bot = axA.twinx()
    axA_bot.spines["right"].set_visible(False)
    axA_bot.spines["top"].set_visible(False)
    # Actually a single shared axis is fine -- both ranges visible
    # but it'll compress. Use the single axis.

    fig.clear()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.1, 1.0]})
    for ax in (axA, axB):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel A: full 2h trace, all 5 plateaus
    for tr in traces:
        axA.plot(tr["t"], tr["v"], "-", color=tr["color"], lw=1.6, label=tr["label"])
    axA.set_xlim(0, 120)
    axA.set_xlabel("rest time [min]")
    axA.set_ylabel("cell voltage [mV]")
    axA.set_title("(a) Full 2 h OCV traces  —  all 5 rest plateaus", fontweight="bold")
    axA.grid(alpha=0.20, linewidth=0.7)
    axA.legend(loc="center right", fontsize=8.5, framealpha=0.92)
    # Endpoint Δ annotation, right edge
    for tr in traces:
        dV = float(tr["v"][-1] - tr["v"][0])
        axA.annotate(f"Δ = {dV:+.1f} mV",
                     xy=(120, tr["v"][-1]),
                     xytext=(4, 0), textcoords="offset points",
                     ha="left", va="center",
                     fontsize=7.5, color=tr["color"], fontweight="bold")
    axA.set_xlim(0, 145)   # leave room for annotations

    # Panel B: zoomed -- SOC 20-80% only, first 30 minutes (where most drift happens)
    for tr in traces:
        if tr["step"] == 3:
            continue   # skip floor for zoomed view
        # normalize to V_start so zoom shows DRIFT, not absolute
        v_norm = tr["v"] - tr["v"][0]
        axB.plot(tr["t"], v_norm, "o-", color=tr["color"],
                 lw=1.6, ms=4, markevery=10, label=tr["label"])
    axB.set_xlim(0, 120)
    axB.set_xlabel("rest time [min]")
    axB.set_ylabel("voltage drift since rest start [mV]")
    axB.set_title("(b) SOC 20–80 % plateaus, drift only  (V(t) − V(0))", fontweight="bold")
    axB.axhline(0, color="#444", lw=0.5)
    axB.grid(alpha=0.20, linewidth=0.7)
    axB.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # Δ_120min annotations on right
    for tr in traces:
        if tr["step"] == 3:
            continue
        dV = float(tr["v"][-1] - tr["v"][0])
        axB.annotate(f"{dV:+.1f} mV",
                     xy=(tr["t"][-1], dV),
                     xytext=(2, 0), textcoords="offset points",
                     fontsize=8, color=tr["color"], fontweight="bold",
                     va="center")
    axB.set_xlim(0, 135)

    fig.suptitle("Figure   OCV evolution during each 2 h rest plateau  (35 °C cell;  5 s cycler resolution)",
                 fontsize=11, y=1.04)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")

    # Also write companion CSV with endpoint deltas
    summary = []
    for tr in traces:
        summary.append({
            "step": tr["step"], "plateau": tr["label"],
            "V_start_mV": float(tr["v"][0]),
            "V_end_mV":   float(tr["v"][-1]),
            "dV_120min_mV": float(tr["v"][-1] - tr["v"][0]),
            "n_samples": int(len(tr["v"])),
        })
    pd.DataFrame(summary).to_csv(OUT_DIR / "fig_ocv_data.csv", index=False)
    print(f"wrote {OUT_DIR/'fig_ocv_data.csv'}")
    for r in summary:
        print(f"  {r['plateau']:<12s}  Vstart={r['V_start_mV']:>7.1f}  "
              f"Vend={r['V_end_mV']:>7.1f}  dV_120min={r['dV_120min_mV']:>+7.1f} mV")


if __name__ == "__main__":
    main()
