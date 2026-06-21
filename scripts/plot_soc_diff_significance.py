"""Significance-aware SoC difference maps.

For each pixel and each ΔSoC, compute z = (Δfeature) / σ_per_pixel using
the per-pixel σ from the 6-repeat noise floor. Plot two views:

  (a) Δ value but with non-significant pixels (|z| < threshold) greyed out
  (b) z-score map (Δ in units of σ) with ±5σ colour limits

These visualisations answer "is this pixel's change above the noise floor?"
properly — subtracting σ from |Δ| per pixel would not, since pixel noise
direction is unknown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"

# Canonical ROI overlay
ROI_W_MM, ROI_H_MM = 50.02, 39.78
ROI_CX_MM, ROI_CY_MM = 39.59, 36.00

# Sigma maps from the 6-repeat noise floor analysis
SIGMA_DIR = PROJECT / "reports/experiments/mask_noisefloor_v3.238_2026-05-29_16-20-46"


def find_scan_dir(scan_id: str) -> Path | None:
    suffix = scan_id.replace("scan_", "", 1)
    candidates = list(CSCAN_ROOT.glob(f"cscan_*{suffix}"))
    if not candidates:
        candidates = [p for p in CSCAN_ROOT.iterdir()
                      if p.is_dir() and p.name.endswith(suffix)]
    return candidates[0] if candidates else None


def group_rest_plateaus(rest_df: pd.DataFrame, max_gap_s: float = 1200.0):
    rest = rest_df.sort_values("time_utc").reset_index(drop=True).copy()
    t = rest["time_utc"].astype("int64") / 1e9
    plateau_id = (t.diff().fillna(0) > max_gap_s).cumsum()
    return [g.reset_index(drop=True) for _, g in rest.groupby(plateau_id)]


def load_scan_arrays(scan_id: str):
    sd = find_scan_dir(scan_id)
    if sd is None:
        return None
    npz = next(sd.glob("scan_*.npz"), None)
    if npz is None:
        return None
    d = np.load(npz)
    return {
        "amp": np.asarray(d["amplitude"], dtype=np.float64),
        "tof": np.asarray(d["tof"],       dtype=np.float64),
        "eng": np.asarray(d["energy"],    dtype=np.float64),
        "x_mm": np.asarray(d["x_mm"]),
        "y_mm": np.asarray(d["y_mm"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--socs", nargs="+", type=float, default=[0, 20, 40, 60])
    ap.add_argument("--z-threshold", type=float, default=2.0,
                    help="|z| threshold for significance (default 2σ)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)

    # ---- load σ maps ----
    sigma_amp = np.load(SIGMA_DIR / "sigma_amp.npy").astype(np.float64)
    sigma_tof = np.load(SIGMA_DIR / "sigma_tof.npy").astype(np.float64)
    sigma_eng = np.load(SIGMA_DIR / "sigma_eng.npy").astype(np.float64)
    sigma_maps = {"amp": sigma_amp, "tof": sigma_tof, "eng": sigma_eng}
    print(f"loaded sigma maps: shape {sigma_tof.shape}, "
          f"sigma_tof median = {np.median(sigma_tof)*1000:.2f} ns")

    # Differences in Δ require σ_Δ = √2 × σ (sum of two scan variances)
    sigma_delta = {k: np.sqrt(2) * v for k, v in sigma_maps.items()}

    # ---- pick equilibrium scans ----
    feat = pd.read_csv(batch_dir / "overview" / "scans_tagged.csv")
    feat["time_utc"] = pd.to_datetime(feat["time_utc"], utc=True)
    rest = feat[feat["step_tag"] == "rest"].copy()
    plateaus = group_rest_plateaus(rest)
    plateau_socs = np.array([p["soc_pct"].mean() for p in plateaus])

    picks = []
    for target in args.socs:
        idx = int(np.argmin(np.abs(plateau_socs - target)))
        last = plateaus[idx].iloc[-1]
        picks.append({
            "target_soc": target,
            "actual_soc": float(last["soc_pct"]),
            "scan_id":    last["scan_id"],
        })
    for p in picks:
        print(f"  target {p['target_soc']}%  -> actual {p['actual_soc']:+.1f}%  "
              f"({p['scan_id']})")

    data = []
    for p in picks:
        arr = load_scan_arrays(p["scan_id"])
        if arr is None:
            print(f"  MISSING: {p['scan_id']}")
            continue
        data.append({**p, **arr})

    if len(data) < 2:
        raise SystemExit("need at least 2 scans")
    baseline = data[0]
    diffs = data[1:]
    extent = [float(baseline["x_mm"][0]), float(baseline["x_mm"][-1]),
              float(baseline["y_mm"][-1]), float(baseline["y_mm"][0])]

    # Refine SoC reference: the user said the discharge plateau = true 0%.
    # That's baseline.actual_soc in the script. Compute "true SoC" labels.
    soc_offset = -baseline["actual_soc"]   # add this to get true SoC

    out_dir = Path(args.out) if args.out else batch_dir / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_specs = [
        ("amp", "amplitude", "V",  1.0,   0.018),
        ("tof", "ToF",       "ns", 1000,  26.5),    # ns
        ("eng", "energy",    "",   1.0,   0.58),
    ]

    # ========== Figure 1: Δ-map with non-significant pixels greyed ==========
    fig, axes = plt.subplots(3, len(diffs),
                              figsize=(3.6 * len(diffs) + 0.6, 11),
                              dpi=160,
                              gridspec_kw={"wspace": 0.10, "hspace": 0.18,
                                           "top": 0.92, "bottom": 0.05,
                                           "left": 0.06, "right": 0.93})
    if len(diffs) == 1:
        axes = axes.reshape(3, 1)

    for row, (key, ylabel, unit, scale, vlim) in enumerate(feature_specs):
        last_im = None
        for col, d in enumerate(diffs):
            ax = axes[row, col]
            delta = (d[key] - baseline[key]) * scale
            # z-score using per-pixel σ_Δ
            with np.errstate(invalid="ignore", divide="ignore"):
                z = ((d[key] - baseline[key]) / sigma_delta[key])
            significant = np.abs(z) >= args.z_threshold

            # Show Δ, mask non-significant
            delta_masked = np.where(significant, delta, np.nan)
            last_im = ax.imshow(delta_masked, cmap="RdBu_r",
                                 vmin=-vlim, vmax=vlim,
                                 extent=extent, origin="upper", aspect="equal")
            # Grey out non-significant
            grey = np.where(significant, np.nan, 1.0)
            ax.imshow(grey, cmap=mcolors.ListedColormap([[0.78, 0.78, 0.78, 0.85]]),
                      extent=extent, origin="upper", aspect="equal",
                      interpolation="nearest")
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f"Δ {ylabel} ({unit})\nY (mm)", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("X (mm)", fontsize=9)
            else:
                ax.set_xticklabels([])
            if row == 0:
                true_soc_diff = d["actual_soc"] - baseline["actual_soc"]
                ax.set_title(
                    f"SoC {d['actual_soc']+soc_offset:.0f}%  −  "
                    f"{baseline['actual_soc']+soc_offset:.0f}%\n"
                    f"ΔSoC = {true_soc_diff:+.0f}%   "
                    f"({(significant.sum()/significant.size)*100:.1f}% sig)",
                    fontsize=10, fontweight="bold")
            rect = mpatches.Rectangle(
                (ROI_CX_MM - ROI_W_MM/2, ROI_CY_MM - ROI_H_MM/2),
                ROI_W_MM, ROI_H_MM,
                fill=False, edgecolor="#16a34a", linewidth=1.2, alpha=0.85)
            ax.add_patch(rect)

        plt.colorbar(last_im, ax=axes[row, :].tolist(),
                     fraction=0.018, pad=0.02, shrink=0.92,
                     label=f"Δ {ylabel} ({unit})")

    fig.suptitle(
        f"SoC difference maps with significance filter  —  grey = |z| < "
        f"{args.z_threshold}σ (pixel-wise noise floor)",
        fontsize=11.5, y=0.97)
    fig.savefig(out_dir / "soc_diff_significant.png", bbox_inches="tight")
    fig.savefig(out_dir / "soc_diff_significant.pdf", bbox_inches="tight")
    plt.close(fig)

    # ========== Figure 2: z-score maps directly ==========
    fig, axes = plt.subplots(3, len(diffs),
                              figsize=(3.6 * len(diffs) + 0.6, 11),
                              dpi=160,
                              gridspec_kw={"wspace": 0.10, "hspace": 0.18,
                                           "top": 0.92, "bottom": 0.05,
                                           "left": 0.06, "right": 0.93})
    if len(diffs) == 1:
        axes = axes.reshape(3, 1)
    for row, (key, ylabel, _, _, _) in enumerate(feature_specs):
        last_im = None
        for col, d in enumerate(diffs):
            ax = axes[row, col]
            with np.errstate(invalid="ignore", divide="ignore"):
                z = ((d[key] - baseline[key]) / sigma_delta[key])
            last_im = ax.imshow(z, cmap="RdBu_r", vmin=-5, vmax=5,
                                 extent=extent, origin="upper", aspect="equal")
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f"z ({ylabel})\nY (mm)", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("X (mm)", fontsize=9)
            else:
                ax.set_xticklabels([])
            if row == 0:
                true_soc_diff = d["actual_soc"] - baseline["actual_soc"]
                ax.set_title(
                    f"ΔSoC = {true_soc_diff:+.0f}%",
                    fontsize=10, fontweight="bold")
            rect = mpatches.Rectangle(
                (ROI_CX_MM - ROI_W_MM/2, ROI_CY_MM - ROI_H_MM/2),
                ROI_W_MM, ROI_H_MM,
                fill=False, edgecolor="#16a34a", linewidth=1.2, alpha=0.85)
            ax.add_patch(rect)
        plt.colorbar(last_im, ax=axes[row, :].tolist(),
                     fraction=0.018, pad=0.02, shrink=0.92,
                     label=f"z ({ylabel}) = Δ / σ_Δ")

    fig.suptitle(
        "Z-score maps  —  per-pixel Δ divided by per-pixel σ_Δ "
        "(σ_Δ = √2 × σ_noise_floor); ±5σ saturated",
        fontsize=11.5, y=0.97)
    fig.savefig(out_dir / "soc_diff_zscore.png", bbox_inches="tight")
    fig.savefig(out_dir / "soc_diff_zscore.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved: {out_dir}")
    print("  soc_diff_significant.* — Δ map, |z|<threshold pixels greyed")
    print("  soc_diff_zscore.*       — direct z-score map (Δ/σ_Δ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
