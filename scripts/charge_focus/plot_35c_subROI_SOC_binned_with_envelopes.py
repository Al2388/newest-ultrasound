"""35C: per-sub-band feature vs SOC (5% bins) with noise + thermal envelopes.

Three panels (ToF / amplitude / energy) showing the band-mean of each
sub-region binned to 5%-wide SOC bins (mean ± SD), with the linear fit
overlaid (dashed) and two envelopes:

  - Grey ±BUFFER * sigma_ROI noise floor band, shaded around each band's
    binned mean line.
  - Pink thermal envelope on the ToF panel only:
      half-width = BUFFER * 73.6 ns/C * |T_bin - T_ref|
    drawn around the linear fit of each band.  T_ref is the campaign-mean
    line temperature.

All rest + charge scans are pooled (consistent with the existing
subROI_rest_charge_overlay figure).
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
CYCLER = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
OUT_PNG = PROJ / "reports/longrun_cycling_35c_charge_focus/subROI_SOC_binned_with_envelopes.png"

SUBREGIONS = {
    "tab-distal (left)":    (14.0, 30.0, "tab:blue"),
    "interior (middle)":    (30.0, 50.0, "tab:gray"),
    "tab-proximal (right)": (50.0, 64.5, "tab:red"),
}
NOMINAL_AH = 0.860
BIN_W      = 5.0

BUFFER = 2.0
TOF_THERMAL_NS_PER_C = 73.6
THERMAL_BAND_COLOR   = "#b03060"
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

    # Per-scan SOC
    soc_df = _build_soc()
    t_ref = soc_df["time"].values.astype("datetime64[s]").astype(np.int64)
    ts_int = meta["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    soc = np.interp(ts_int, t_ref, soc_df["SOC_pct"].values)
    T_C = meta["line_T_mean_c"].values.astype(float)
    T_ref = float(np.nanmean(T_C))

    # Per-scan band means
    sub_masks = {name: _xband(roi, x_mm, lo, hi)
                 for name, (lo, hi, _) in SUBREGIONS.items()}
    band_means = {mod: {} for mod in arrs}
    for mod, arr in arrs.items():
        flat = arr.reshape(arr.shape[0], -1)
        for name, mask in sub_masks.items():
            band_means[mod][name] = (np.nanmean(flat[:, mask.reshape(-1)], axis=1)
                                      * SCALES[mod])

    # SOC bins
    edges   = np.arange(0, 100 + BIN_W, BIN_W)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(soc, edges) - 1, 0, len(centers) - 1)

    # Per-bin mean T (for the thermal envelope)
    T_bin = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        sel = bin_idx == k
        if sel.sum() >= 1:
            T_bin[k] = np.nanmean(T_C[sel])
    dT_bin_abs = np.abs(T_bin - T_ref)

    plt.rcParams.update({
        "font.size":        11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":   "#444444",
        "xtick.color":      "#444444",
        "ytick.color":      "#444444",
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
    cols = ["tof", "amplitude", "energy"]

    for ax, mod in zip(axes, cols):
        unit  = UNITS[mod]
        sigma = SIGMA[mod]
        band_step = BUFFER * sigma

        # Linear fits per band (so the thermal envelope has a center line)
        band_fits = {}
        for name, (_, _, color) in SUBREGIONS.items():
            y_all = band_means[mod][name]
            sl, ic = np.polyfit(soc, y_all, 1)
            band_fits[name] = (sl, ic)

        # ----- Thermal envelope (TOF only) -----
        # Drawn as a single shaded band centered on the *first* band's
        # linear fit (just to give the eye a visible context strip);
        # the half-width is the same at each SOC bin regardless of band
        # since the temperature is a campaign-wide quantity.
        if mod == "tof":
            xs = np.linspace(0, 100, 200)
            # mean temperature per SOC at the panel's resolution
            dT_curve = np.interp(xs, centers, dT_bin_abs, left=np.nan, right=np.nan)
            therm = BUFFER * TOF_THERMAL_NS_PER_C * dT_curve
            for name, (_, _, color) in SUBREGIONS.items():
                sl, ic = band_fits[name]
                ys_fit = sl * xs + ic
                ax.fill_between(xs, ys_fit - therm, ys_fit + therm,
                                 color=THERMAL_BAND_COLOR, alpha=0.10, lw=0,
                                 zorder=1)
            # Stand-alone legend entry
            ax.fill_between([], [], [], color=THERMAL_BAND_COLOR, alpha=0.18,
                             label=f"±{BUFFER:.0f}·73.6 ns/°C·|ΔT(SOC)|  "
                                    "(thermal envelope)")

        # ----- Per band -----
        for name, (_, _, color) in SUBREGIONS.items():
            y = band_means[mod][name]
            mu = np.full(len(centers), np.nan)
            sd = np.full(len(centers), np.nan)
            for k in range(len(centers)):
                sel = bin_idx == k
                if sel.sum() >= 1:
                    mu[k] = np.nanmean(y[sel])
                    sd[k] = np.nanstd(y[sel]) if sel.sum() > 1 else 0.0
            ok = np.isfinite(mu)

            # Noise floor band shaded around the binned line
            ax.fill_between(centers[ok], mu[ok] - band_step, mu[ok] + band_step,
                             color=color, alpha=0.10, lw=0, zorder=2)

            # Binned mean line + std error bars
            ax.errorbar(centers[ok], mu[ok], yerr=sd[ok], fmt="o-", color=color,
                         lw=1.9, ms=5, capsize=2.5, alpha=0.95, label=name,
                         zorder=4)

            # Linear fit
            sl, ic = band_fits[name]
            xs = np.linspace(0, 100, 50)
            ax.plot(xs, sl * xs + ic, "--", color=color, lw=1.0, alpha=0.55,
                     zorder=3)

        # Noise-floor legend entry
        ax.fill_between([], [], [], color="#666", alpha=0.18,
                         label=f"±{BUFFER:.0f}·σ_ROI = ±{band_step:.2g} {unit}  "
                                "(noise floor)")

        ax.set_xlabel("SOC [%]")
        ax.set_ylabel(f"band-mean {mod} [{unit}]")
        ax.set_title(mod.upper(), fontsize=12, weight="bold")
        ax.set_xlim(-2, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="best")

    fig.suptitle(f"35°C  Sub-ROI band evolution vs SOC  ·  "
                 f"binned to {int(BIN_W)} %-wide SOC bins (mean ± SD)  ·  "
                 "noise + thermal envelopes overlaid",
                 fontsize=12)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
