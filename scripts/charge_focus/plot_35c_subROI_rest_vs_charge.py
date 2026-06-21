"""35C: sub-ROI band evolution as a function of SOC, split into REST and CHARGE.

2 x 3 figure (top: REST, bottom: CHARGE; columns: ToF / amplitude / energy).
Each column shows the three sub-bands (tab-distal blue, interior gray,
tab-proximal red) with linear fits.

Companion: beta_S(x) profile across X with sub-region shading (separate fig).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
CYCLER = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
OUT_PNG = PROJ / "reports/longrun_cycling_35c_charge_focus/subROI_rest_charge_evolution.png"

# Sub-region X bands
SUBREGIONS = {
    "tab-distal (left)":    (14.0, 30.0, "tab:blue"),
    "interior (middle)":    (30.0, 50.0, "tab:gray"),
    "tab-proximal (right)": (50.0, 64.5, "tab:red"),
}

NOMINAL_AH = 0.860


def _xband(roi, x_mm, lo, hi):
    cl = int(np.searchsorted(x_mm, lo)); ch = int(np.searchsorted(x_mm, hi))
    m = np.zeros_like(roi, dtype=bool); m[:, cl:ch] = True
    return m & roi


def _build_soc():
    df = pd.read_csv(CYCLER, sep="\t", skiprows=6, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DPT Time"] = pd.to_datetime(df["DPT Time"].astype(str),
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for c in ["Step", "Capacity", "Voltage", "Current"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MD"] = df["MD"].astype(str).str.strip()
    df = df.dropna(subset=["DPT Time"]).reset_index(drop=True)
    mode_sign = {"C": +1.0, "D": -1.0, "R": 0.0, "P": 0.0}
    dq = np.zeros(len(df))
    for _, sub in df.groupby("Step", sort=False):
        idx = sub.index.values
        q = sub["Capacity"].values
        d_abs = np.clip(np.diff(q, prepend=q[0]), 0, None)
        md = sub["MD"].mode().iloc[0]
        dq[idx] = d_abs * mode_sign.get(md, 0.0)
    cum = np.cumsum(dq)
    soc = (cum / NOMINAL_AH) * 100
    soc -= soc.min()
    return pd.DataFrame({"time": df["DPT Time"], "SOC_pct": soc})


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    roi = d["roi_mask"]; x_mm = d["x_mm"]
    arrs = {"tof": d["tof"], "amplitude": d["amplitude"], "energy": d["energy"]}
    scales = {"tof": 1e3, "amplitude": 1e3, "energy": 1.0}      # to ns / mV / a.u.
    units  = {"tof": "ns", "amplitude": "mV", "energy": "a.u."}

    # Per-scan SOC
    soc_df = _build_soc()
    t_ref = soc_df["time"].values.astype("datetime64[s]").astype(np.int64)
    ts_int = meta["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    soc_per_scan = np.interp(ts_int, t_ref, soc_df["SOC_pct"].values)

    # Sub-region masks
    sub_masks = {name: _xband(roi, x_mm, lo, hi)
                 for name, (lo, hi, _) in SUBREGIONS.items()}

    # Per-scan band means (n_scans, 3 bands) per modality
    band_means = {mod: {name: None for name in SUBREGIONS} for mod in arrs}
    for mod, arr in arrs.items():
        for name, mask in sub_masks.items():
            band_means[mod][name] = (np.nanmean(arr.reshape(arr.shape[0], -1)[:, mask.reshape(-1)],
                                                axis=1) * scales[mod])

    # Step tags
    is_rest = meta["step_tag"].values == "rest"
    is_charge = meta["step_tag"].values == "charge"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    cols = ["tof", "amplitude", "energy"]

    for col_i, mod in enumerate(cols):
        unit = units[mod]
        # ---- REST row ----
        ax = axes[0, col_i]
        for name, (lo, hi, color) in SUBREGIONS.items():
            y_rest = band_means[mod][name][is_rest]
            x_rest = soc_per_scan[is_rest]
            ax.scatter(x_rest, y_rest, color=color, s=30, alpha=0.85, label=name)
            # linear fit through rest scans
            if len(x_rest) >= 2:
                slope, intercept = np.polyfit(x_rest, y_rest, 1)
                xs = np.linspace(0, 80, 50)
                ax.plot(xs, slope*xs + intercept, "-", color=color, lw=1.4, alpha=0.85)
        ax.set_xlabel("SOC [%]")
        ax.set_ylabel(f"band-mean {mod} [{unit}]")
        ax.set_title(f"REST — {mod}", fontsize=11)
        ax.set_xlim(15, 85)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        # ---- CHARGE row ----
        ax = axes[1, col_i]
        for name, (lo, hi, color) in SUBREGIONS.items():
            y_ch = band_means[mod][name][is_charge]
            x_ch = soc_per_scan[is_charge]
            order = np.argsort(x_ch)
            ax.scatter(x_ch[order], y_ch[order], color=color, s=22, alpha=0.85, label=name)
            if len(x_ch) >= 2:
                slope, intercept = np.polyfit(x_ch, y_ch, 1)
                xs = np.linspace(20, 80, 50)
                ax.plot(xs, slope*xs + intercept, "-", color=color, lw=1.4, alpha=0.85)
        ax.set_xlabel("SOC [%]")
        ax.set_ylabel(f"band-mean {mod} [{unit}]")
        ax.set_title(f"CHARGE — {mod}", fontsize=11)
        ax.set_xlim(15, 85)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("35°C  Sub-ROI band evolution: tab-distal (blue) → interior (gray) → tab-proximal (red)\n"
                 "Top row: REST (4 plateaus at SOC = 20/40/60/80%);  bottom: CHARGE (continuous 20→80%)",
                 fontsize=12)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")

    # Print quantitative summary
    print(f"\n=== REST-only sub-band linear fit (slope per %SoC) ===")
    print(f"{'modality':<10s}  {'band':<25s}  {'slope':>14s}  {'intercept':>14s}")
    for mod in cols:
        unit = units[mod]
        for name, (lo, hi, color) in SUBREGIONS.items():
            y_rest = band_means[mod][name][is_rest]
            x_rest = soc_per_scan[is_rest]
            slope, intercept = np.polyfit(x_rest, y_rest, 1)
            print(f"{mod:<10s}  {name:<25s}  {slope:>+10.4f} {unit}/%  {intercept:>10.2f} {unit}")


if __name__ == "__main__":
    main()
