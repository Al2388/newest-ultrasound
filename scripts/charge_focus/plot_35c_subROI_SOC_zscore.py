"""35C: significance (z-score) of band-mean change vs SOC.

Three panels (ToF / amplitude / energy).  For each sub-band, at each 5%-wide
SOC bin we plot

      z(bin)  =  ( feature(bin) - feature(first bin) )  /  sigma_combined(bin)

where

      sigma_combined^2  =  sigma_ROI^2   (always)
                         + ( 73.6 ns/C * |T(bin) - T(first bin)| )^2   (ToF only)

Horizontal cutoffs at z = +/-2 (significant) and z = +/-3 (highly significant)
are drawn.  Any point outside the +/-2 grey band exceeds the combined noise
+ thermal envelope used in the within-rest figure.
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

CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
CYCLER    = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
OUT_PNG   = PROJ / "reports/longrun_cycling_35c_charge_focus/subROI_SOC_zscore.png"

SUBREGIONS = {
    "tab-distal (left)":    (14.0, 30.0, "tab:blue"),
    "interior (middle)":    (30.0, 50.0, "tab:gray"),
    "tab-proximal (right)": (50.0, 64.5, "tab:red"),
}
NOMINAL_AH = 0.860
BIN_W      = 5.0

TOF_THERMAL_NS_PER_C = 73.6
SIGMA = {
    "tof":       NOISE_FLOOR["tof_sigma_roi_ns"],
    "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"],
    "energy":    NOISE_FLOOR["energy_sigma_roi"],
}
UNITS  = {"tof": "ns", "amplitude": "mV", "energy": "a.u."}
SCALES = {"tof": 1e3, "amplitude": 1e3, "energy": 1.0}


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
        idx = sub.index.values; q = sub["Capacity"].values
        d_abs = np.clip(np.diff(q, prepend=q[0]), 0, None)
        md = sub["MD"].mode().iloc[0]
        dq[idx] = d_abs * mode_sign.get(md, 0.0)
    soc = np.cumsum(dq) / NOMINAL_AH * 100
    soc -= soc.min()
    return pd.DataFrame({"time": df["DPT Time"], "SOC_pct": soc})


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    roi, x_mm = d["roi_mask"], d["x_mm"]
    arrs   = {"tof": d["tof"], "amplitude": d["amplitude"], "energy": d["energy"]}

    soc_df = _build_soc()
    t_ref = soc_df["time"].values.astype("datetime64[s]").astype(np.int64)
    ts_int = meta["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    soc = np.interp(ts_int, t_ref, soc_df["SOC_pct"].values)
    T_C = meta["line_T_mean_c"].values.astype(float)

    sub_masks = {name: _xband(roi, x_mm, lo, hi)
                 for name, (lo, hi, _) in SUBREGIONS.items()}
    band_means = {mod: {} for mod in arrs}
    for mod, arr in arrs.items():
        flat = arr.reshape(arr.shape[0], -1)
        for name, mask in sub_masks.items():
            band_means[mod][name] = (np.nanmean(flat[:, mask.reshape(-1)], axis=1)
                                      * SCALES[mod])

    edges   = np.arange(0, 100 + BIN_W, BIN_W)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(soc, edges) - 1, 0, len(centers) - 1)

    # Per-bin mean T and Δfeature
    T_bin = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        sel = bin_idx == k
        if sel.sum() >= 1:
            T_bin[k] = np.nanmean(T_C[sel])

    plt.rcParams.update({
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":   "#444444",
        "xtick.color":      "#444444",
        "ytick.color":      "#444444",
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
    cols = ["tof", "amplitude", "energy"]

    z_summary = []   # for terminal summary

    for ax, mod in zip(axes, cols):
        sigma = SIGMA[mod]
        unit  = UNITS[mod]

        # Find the first bin with valid data (anchor)
        any_band_valid = np.zeros(len(centers), dtype=bool)
        for name in SUBREGIONS:
            y = band_means[mod][name]
            for k in range(len(centers)):
                if (bin_idx == k).sum() >= 1:
                    any_band_valid[k] = True
        if not any_band_valid.any():
            continue
        k0 = int(np.argmax(any_band_valid))   # first valid bin index
        T_ref = T_bin[k0]

        # Significance bands
        ax.axhspan(-2, 2, color="#888", alpha=0.13, lw=0, zorder=0,
                    label="|z| ≤ 2 (within noise + thermal)")
        for z_cut, ls, c, lbl in [
            (+2.0, "--", "#888", "z = ±2 (2σ)"),
            (-2.0, "--", "#888", None),
            (+3.0, ":",  "#444", "z = ±3 (3σ)"),
            (-3.0, ":",  "#444", None),
        ]:
            ax.axhline(z_cut, color=c, ls=ls, lw=1.0, alpha=0.85, zorder=1,
                        label=lbl)
        ax.axhline(0, color="#444", lw=0.7, alpha=0.7, zorder=1)

        for name, (_, _, color) in SUBREGIONS.items():
            y = band_means[mod][name]
            mu = np.full(len(centers), np.nan)
            for k in range(len(centers)):
                sel = bin_idx == k
                if sel.sum() >= 1:
                    mu[k] = np.nanmean(y[sel])

            d_feat = mu - mu[k0]
            if mod == "tof":
                thermal = TOF_THERMAL_NS_PER_C * np.abs(T_bin - T_ref)
                sigma_comb = np.sqrt(sigma ** 2 + thermal ** 2)
            else:
                sigma_comb = np.full_like(d_feat, sigma, dtype=float)
            z = d_feat / sigma_comb
            ok = np.isfinite(z)

            ax.plot(centers[ok], z[ok], "o-", color=color, lw=2.0, ms=6,
                     alpha=0.95, label=name, zorder=4)
            # Highlight points above |z|=2 with a white-edged ring
            sig = ok & (np.abs(z) >= 2.0)
            ax.scatter(centers[sig], z[sig], facecolors="none", edgecolors=color,
                        s=120, lw=1.4, zorder=5)

            # Summary line: largest |z|
            if ok.any():
                i_pk = int(np.nanargmax(np.abs(np.where(ok, z, np.nan))))
                z_summary.append((mod, name, centers[i_pk], z[i_pk], d_feat[i_pk], unit))

        ax.set_xlabel("SOC [%]")
        ax.set_ylabel(f"z-score  (Δ {mod} normalised by noise"
                       + (" + thermal)" if mod == "tof" else ")"))
        ax.set_title(mod.upper(), fontsize=12, weight="bold")
        ax.set_xlim(-2, 100)
        ax.grid(alpha=0.3)
        # Clamp y if z explodes (rare)
        z_extreme = 15
        ax.set_ylim(-z_extreme, z_extreme)
        ax.legend(fontsize=8.5, loc="best")

    fig.suptitle("35°C  Significance of sub-band departures vs SOC  ·  "
                 "anchor = first SOC bin  ·  rings = |z| ≥ 2",
                 fontsize=12)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")

    print(f"\n=== peak |z| per (modality, band) ===")
    print(f"{'modality':<10s}  {'band':<25s}  {'SOC':>6s}  {'z':>8s}  {'d feat':>12s}")
    for mod, name, soc_at, z_at, df_at, u in z_summary:
        print(f"{mod:<10s}  {name:<25s}  {soc_at:>5.1f}%  {z_at:>+7.2f}  {df_at:>+8.2f} {u}")


if __name__ == "__main__":
    main()
