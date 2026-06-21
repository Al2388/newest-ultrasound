"""35C β_S(x) profile along X — matches the 25C style.

For each X column:
  1. fit β_S = ΔX/ΔSOC at every pixel (Y, x) using all rest scans
  2. take mean and std of β_S across Y inside ROI
  3. plot mean as line, ±std as shaded band

Subregion backgrounds:
  blue   = tab-distal  (X 14-30)
  brown  = interior    (X 30-50)
  pink   = tab-proximal (X 50-64.5)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))

CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT_PNG = PROJ / "reports/longrun_cycling_35c_charge_focus/07_tab_heterogeneity/betaS_x_profile.png"

SUBREGIONS = {
    "tab-distal (left)":    (14.0, 30.0, "#cfe2f3"),   # light blue
    "interior (middle)":    (30.0, 50.0, "#e3dcd0"),   # light brown
    "tab-proximal (right)": (50.0, 64.5, "#f4cccc"),   # light pink
}

REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20.0, 40.0, 60.0, 80.0]

UNITS = {
    "tof":       ("[ns/%SoC]", 1e3),
    "amplitude": ("[mV/%SoC]", 1e3),
    "energy":    ("[/%SoC]",   1.0),
}

# Fixed Y-limits, matched to the 25C-vs-35C comparison figure
YLIMS = {
    "tof":       (-9.0, 3.0),
    "amplitude": (-9.0, 9.0),
    "energy":    (-0.3, 0.35),
}


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    roi = d["roi_mask"]
    x_mm = d["x_mm"]
    arrs = {"tof": d["tof"], "amplitude": d["amplitude"], "energy": d["energy"]}

    # Per-rest-scan SOC label (one per plateau)
    soc_per_scan = np.zeros(len(meta))
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        soc_per_scan[meta["step"] == step] = soc
    rest_mask = meta["step"].isin(REST_STEPS).values
    rest_idx = np.where(rest_mask)[0]
    soc_rest = soc_per_scan[rest_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, mod in zip(axes, ["tof", "amplitude", "energy"]):
        unit, scale = UNITS[mod]
        arr = arrs[mod] * scale     # already in plot units

        # Per-pixel β_S via linear fit through 4 SOC plateau means (all rest scans)
        # arr_rest shape: (n_rest, 144, 500)
        arr_rest = arr[rest_idx]
        nz, ny, nx = arr_rest.shape
        # Design: SOC -> beta_S * SOC + c
        X = np.column_stack([soc_rest, np.ones_like(soc_rest)]).astype(np.float64)
        flat = arr_rest.reshape(nz, -1).astype(np.float64)
        mask_flat = roi.reshape(-1)
        Y_roi = flat[:, mask_flat]
        finite = np.isfinite(Y_roi).all(axis=0)
        coeffs, *_ = np.linalg.lstsq(X, Y_roi[:, finite], rcond=None)
        beta_S_finite = coeffs[0]
        full = np.full(int(mask_flat.sum()), np.nan)
        full[finite] = beta_S_finite
        beta_map = np.full(roi.shape, np.nan)
        beta_map[roi] = full

        # Per-X column: mean and std across Y inside ROI
        mean_x = np.full(nx, np.nan)
        std_x = np.full(nx, np.nan)
        for c in range(nx):
            vals = beta_map[roi[:, c], c]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 5:
                mean_x[c] = float(np.mean(vals))
                std_x[c] = float(np.std(vals))

        # ---- background shading for sub-regions ----
        for name, (lo, hi, color) in SUBREGIONS.items():
            ax.axvspan(lo, hi, color=color, alpha=0.55,
                       label=name if mod == "tof" else None)

        # ---- mean line + ±std band ----
        valid = np.isfinite(mean_x) & np.isfinite(std_x)
        ax.fill_between(x_mm[valid],
                        (mean_x - std_x)[valid],
                        (mean_x + std_x)[valid],
                        color="#7a5e44", alpha=0.30, linewidth=0)
        ax.plot(x_mm[valid], mean_x[valid], "-", color="black", lw=1.6)

        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("X [mm]   (tabs at X ≈ 64+ →)")
        ax.set_ylabel(f"β_S along X-row mean {unit}")
        ax.set_title(mod, fontsize=12)
        ax.set_xlim(14, 65)
        ax.set_ylim(*YLIMS[mod])
        ax.grid(alpha=0.25)
        if mod == "tof":
            ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    fig.suptitle("β_S profile along X  (rest-only pixel fit, Y-averaged inside ROI)",
                 fontsize=13)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
