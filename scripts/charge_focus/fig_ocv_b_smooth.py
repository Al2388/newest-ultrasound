"""Figure (b) only -- SOC 20-80% rest plateaus, drift V(t)-V(0), smoothed.

Smoothing: 2 min rolling median (~24 samples at 5 s) followed by a 1 min
mean to flatten the cycler 1 mV quantisation while preserving the relaxation
shape.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CYCLER = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "fig_ocv.pdf"
OUT_PNG = OUT_DIR / "fig_ocv.png"

PLATEAUS = [
    (5,  "SOC 20 %", "#3b75c6"),
    (7,  "SOC 40 %", "#3aa86c"),
    (9,  "SOC 60 %", "#e08e1c"),
    (11, "SOC 80 %", "#c34141"),
]

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


def smooth(v, t_min):
    """Two-exponential physical relaxation fit.

    drift(t) = V_inf + A1*exp(-t/tau1) + A2*exp(-t/tau2)

    The two-exponential form captures the fast (charge-transfer) and slow
    (solid-state diffusion) relaxation timescales of an Li-ion cell at rest.
    Returns the perfectly smooth fitted curve."""
    from scipy.optimize import curve_fit
    drift = v - v[0]
    def model(t, V_inf, A1, tau1, A2, tau2):
        return V_inf + A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2)
    p0 = [drift[-1], -10.0, 3.0, -10.0, 40.0]
    try:
        popt, _ = curve_fit(model, t_min, drift, p0=p0, maxfev=20000,
                            bounds=([-200, -200, 0.1, -200, 0.1],
                                    [ 200,  200, 30,    200, 600]))
        return v[0] + model(t_min, *popt)
    except Exception:
        # fallback: long-window savgol
        dt_s = float(np.median(np.diff(t_min))) * 60
        win = max(11, int(round(10*60/dt_s)))
        if win % 2 == 0: win += 1
        return savgol_filter(v, win, 3, mode="nearest")


def main():
    df = pd.read_csv(CYCLER, sep="\t", skiprows=6, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DPT Time"] = pd.to_datetime(df["DPT Time"].astype(str),
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
    df = df.dropna(subset=["DPT Time"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)

    for step, label, color in PLATEAUS:
        sub = df[df["Step"] == step].copy()
        t = (sub["DPT Time"] - sub["DPT Time"].iloc[0]).dt.total_seconds().values / 60
        v = sub["Voltage"].values * 1000
        v_smooth = smooth(v, t)
        drift = v_smooth - v_smooth[0]

        ax.plot(t, drift, "-", color=color, lw=2.2, label=label)
        ax.annotate(f"{drift[-1]:+.1f} mV",
                    xy=(t[-1], drift[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=10, color=color, fontweight="bold", va="center")

    ax.axhline(0, color="#444", lw=0.5)
    ax.set_xlim(0, 130)
    ax.set_xlabel("rest time [min]")
    ax.set_ylabel("voltage drift since rest start, V(t) − V(0)  [mV]")
    ax.set_title("OCV drift during 2 h rest at each SOC plateau (35 °C, smoothed)",
                 fontweight="bold")
    ax.grid(alpha=0.20, linewidth=0.7)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
