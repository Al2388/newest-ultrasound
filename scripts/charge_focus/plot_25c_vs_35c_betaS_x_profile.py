"""25C vs 35C beta_S(x) profile side-by-side -- 2 rows x 3 columns.

Top row: 25C charge-focus subset (147 scans, 60 rest)
Bottom row: 35C charge-focus subset (93 scans, 52 rest)
Columns: ToF / amplitude / energy
Same Y-axis scale within each column so the two temperatures are directly
comparable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")

# 25C uses the existing pre-computed beta_S maps (which include both SOC and T
# in the pixelwise regression -- the cleanest reference)
PRECOMPUTED_25C = PROJ / "reports/longrun_cycling_22h_charge_focus/03_beta_soc_map"
CACHE_25C_STACK = PROJ / "reports/longrun_cycling_22h_charge_focus/_cache/stack.npz"
CACHE_35C = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT = PROJ / "reports/longrun_cycling_35c_charge_focus/07_tab_heterogeneity/betaS_x_profile_25C_vs_35C.png"

SUBREGIONS = {
    "tab-distal (left)":    (14.0, 30.0, "#cfe2f3"),
    "interior (middle)":    (30.0, 50.0, "#e3dcd0"),
    "tab-proximal (right)": (50.0, 64.5, "#f4cccc"),
}

UNITS = {
    "tof":       ("[ns/%SoC]", 1e3),
    "amplitude": ("[mV/%SoC]", 1e3),
    "energy":    ("[/%SoC]",   1.0),
}

# Fixed Y-limits per modality, matched to the 25C reference figure
YLIMS = {
    "tof":       (-9.0, 3.0),
    "amplitude": (-9.0, 9.0),
    "energy":    (-0.3, 0.35),
}


def betaS_map_to_x_profile(beta_map, roi, x_mm, scale):
    """Per-X column: Y-mean and Y-std of beta_S inside ROI."""
    nx = roi.shape[1]
    mean_x = np.full(nx, np.nan)
    std_x = np.full(nx, np.nan)
    for c in range(nx):
        vals = beta_map[roi[:, c], c]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 5:
            mean_x[c] = float(np.mean(vals)) * scale
            std_x[c] = float(np.std(vals)) * scale
    return {"x_mm": x_mm, "mean": mean_x, "std": std_x}


def load_25C_profile():
    """Load pre-computed 25C beta_S NPZ (SOC + T regression)."""
    d_stack = np.load(CACHE_25C_STACK)
    roi = d_stack["roi_mask"]; x_mm = d_stack["x_mm"]
    out = {}
    for mod in ["tof", "amplitude", "energy"]:
        npz = np.load(PRECOMPUTED_25C / f"beta_S_{mod}.npz")
        beta_map = npz["beta_S"]
        scale = UNITS[mod][1]
        out[mod] = betaS_map_to_x_profile(beta_map, roi, x_mm, scale)
    return out


def compute_35C_profile():
    d = np.load(CACHE_35C / "stack.npz")
    meta = pd.read_csv(CACHE_35C / "meta.csv")
    roi = d["roi_mask"]; x_mm = d["x_mm"]
    arrs = {"tof": d["tof"], "amplitude": d["amplitude"], "energy": d["energy"]}
    rest_idx = np.where(meta["step_tag"].values == "rest")[0]
    soc = meta["soc_plateau_label"].values[rest_idx].astype(float)

    out = {}
    for mod, arr in arrs.items():
        a = arr[rest_idx].astype(np.float64)
        nz, ny, nx = a.shape
        X = np.column_stack([soc, np.ones_like(soc)]).astype(np.float64)
        flat = a.reshape(nz, -1)
        mask_flat = roi.reshape(-1)
        Y = flat[:, mask_flat]
        finite = np.isfinite(Y).all(axis=0)
        coeffs, *_ = np.linalg.lstsq(X, Y[:, finite], rcond=None)
        full = np.full(int(mask_flat.sum()), np.nan)
        full[finite] = coeffs[0]
        beta_map = np.full(roi.shape, np.nan)
        beta_map[roi] = full
        scale = UNITS[mod][1]
        out[mod] = betaS_map_to_x_profile(beta_map, roi, x_mm, scale)
    return out


def main():
    print("loading 25C precomputed beta_S maps ...")
    prof_25 = load_25C_profile()
    print("computing 35C beta_S(x) ...")
    prof_35 = compute_35C_profile()

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    for row, (label, prof) in enumerate([("25 °C", prof_25), ("35 °C", prof_35)]):
        for col, mod in enumerate(["tof", "amplitude", "energy"]):
            unit, _ = UNITS[mod]
            ax = axes[row, col]

            for name, (lo, hi, color) in SUBREGIONS.items():
                ax.axvspan(lo, hi, color=color, alpha=0.55,
                           label=name if (row == 0 and col == 0) else None)

            x_mm = prof[mod]["x_mm"]
            mean_x = prof[mod]["mean"]
            std_x = prof[mod]["std"]
            valid = np.isfinite(mean_x) & np.isfinite(std_x)

            ax.fill_between(x_mm[valid],
                            (mean_x - std_x)[valid],
                            (mean_x + std_x)[valid],
                            color="#7a5e44", alpha=0.30, linewidth=0)
            ax.plot(x_mm[valid], mean_x[valid], "-", color="black", lw=1.6)
            ax.axhline(0, color="k", lw=0.7)

            ax.set_xlim(14, 65)
            ax.set_ylim(*YLIMS[mod])
            if row == 1:
                ax.set_xlabel("X [mm]   (tabs at X ≈ 64+ →)")
            ax.set_ylabel(f"{label}  β_S {unit}")
            if row == 0:
                ax.set_title(mod, fontsize=12)
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    fig.suptitle("25 °C vs 35 °C: β_S profile along X  "
                 "(rest-only pixel fit, Y-averaged inside ROI)",
                 fontsize=13)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
