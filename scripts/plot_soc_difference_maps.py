"""Difference maps: scan(higher SoC) − scan(lowest SoC) for each feature.

For each equilibrium scan (rest-end), subtract the lowest-SoC scan and plot
the residual on a divergent (red–white–blue) colour scale centred on zero.
The colour range is ±3× per-pixel noise floor (median σ from the 6-repeat
noise-floor analysis), so saturated pixels are reliably > 3σ above noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"

# Canonical ROI overlay
ROI_W_MM, ROI_H_MM = 50.02, 39.78
ROI_CX_MM, ROI_CY_MM = 39.59, 36.00

# Per-pixel noise floor σ (median inside ROI from the 6-repeat baseline)
PER_PIXEL_SIGMA = {
    "amp": 0.00608,    # V (median per-pixel σ across 6 repeats)
    "tof": 0.00882,    # µs (= 8.82 ns)
    "eng": 0.194,
}


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
    gaps = t.diff().fillna(0)
    plateau_id = (gaps > max_gap_s).cumsum()
    return [g.reset_index(drop=True) for _, g in rest.groupby(plateau_id)]


def load_features(scan_id: str) -> dict | None:
    scan_dir = find_scan_dir(scan_id)
    if scan_dir is None:
        return None
    npz = next(scan_dir.glob("scan_*.npz"), None)
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
    ap.add_argument("--socs", nargs="+", type=float,
                    default=[0, 20, 40, 60],
                    help="Target SoCs; the LAST rest scan of the plateau "
                         "closest to each is used. First entry = baseline.")
    ap.add_argument("--n-sigma", type=float, default=3.0,
                    help="Divergent colour range = ±n-sigma × per-pixel σ")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
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
            "target_soc":  target,
            "actual_soc":  float(last["soc_pct"]),
            "scan_id":     last["scan_id"],
            "voltage":     float(last["voltage_at_scan"]),
            "temp":        float(last["temp_at_scan"])
                           if "temp_at_scan" in last else float("nan"),
        })

    print("Picks:")
    for p in picks:
        print(f"  target {p['target_soc']}%  -> actual {p['actual_soc']:+.1f}%  "
              f"V={p['voltage']:.3f}  T={p['temp']:.2f}°C  ({p['scan_id']})")

    # Load all
    data = []
    for p in picks:
        d = load_features(p["scan_id"])
        if d is None:
            print(f"  MISSING: {p['scan_id']}")
            continue
        data.append({**p, **d})
    if len(data) < 2:
        raise SystemExit("need at least 2 scans")

    baseline = data[0]
    diffs = data[1:]   # the ones we'll subtract baseline from
    extent = [float(baseline["x_mm"][0]), float(baseline["x_mm"][-1]),
              float(baseline["y_mm"][-1]), float(baseline["y_mm"][0])]

    # ---- 3 × N grid: rows = feature, cols = SoC delta ----
    fig, axes = plt.subplots(3, len(diffs),
                              figsize=(3.6 * len(diffs) + 0.6, 11),
                              dpi=160,
                              gridspec_kw={"wspace": 0.10, "hspace": 0.18,
                                           "top": 0.92, "bottom": 0.05,
                                           "left": 0.06, "right": 0.93})
    if len(diffs) == 1:
        axes = axes.reshape(3, 1)

    features = [
        ("amp", "Δ amplitude (V)",  "V",  1.0,   "RdBu_r"),
        ("tof", "Δ ToF (ns)",        "ns", 1000,  "RdBu_r"),
        ("eng", "Δ energy",          "",   1.0,   "RdBu_r"),
    ]
    for row, (key, ylabel, unit, scale, cmap) in enumerate(features):
        sigma_px = PER_PIXEL_SIGMA[key]
        vlim = args.n_sigma * sigma_px * scale
        last_im = None
        for col, d in enumerate(diffs):
            ax = axes[row, col]
            delta = (d[key] - baseline[key]) * scale
            last_im = ax.imshow(delta, cmap=cmap, vmin=-vlim, vmax=vlim,
                                 extent=extent, origin="upper", aspect="equal")
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f"{ylabel}\nY (mm)", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("X (mm)", fontsize=9)
            else:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(
                    f"SoC {d['actual_soc']:+.1f}%  −  {baseline['actual_soc']:+.1f}%\n"
                    f"ΔSoC = {d['actual_soc'] - baseline['actual_soc']:+.1f}%",
                    fontsize=10, fontweight="bold")
            # ROI overlay
            rect = mpatches.Rectangle(
                (ROI_CX_MM - ROI_W_MM/2, ROI_CY_MM - ROI_H_MM/2),
                ROI_W_MM, ROI_H_MM,
                fill=False, edgecolor="#16a34a", linewidth=1.2, alpha=0.85,
            )
            ax.add_patch(rect)
        plt.colorbar(last_im, ax=axes[row, :].tolist(),
                     fraction=0.018, pad=0.02, shrink=0.92,
                     label=f"{ylabel}  (±{args.n_sigma:.0f}σ = ±{vlim:.3g} {unit})")

    fig.suptitle(
        f"SoC difference maps (rest scan at given SoC − rest scan at SoC = "
        f"{baseline['actual_soc']:+.1f}%)  —  colour = ±{args.n_sigma:.0f}× per-pixel σ "
        f"from 6-repeat noise floor",
        fontsize=11.5, y=0.97)

    out_dir = Path(args.out) if args.out else batch_dir / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "soc_difference_maps.png", bbox_inches="tight")
    fig.savefig(out_dir / "soc_difference_maps.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Bonus: ROI-stats on each Δ map ----
    print("\nROI-mean delta per feature × delta-SoC:")
    print(f"{'feature':>9} " + "  ".join(f"d@{d['actual_soc']:+.1f}%".rjust(14)
                                          for d in diffs))
    for key, ylabel, unit, scale, _ in features:
        roi_x = (slice(None), slice(None))   # use full footprint here for context
        vals = []
        for d in diffs:
            delta = (d[key] - baseline[key]) * scale
            # restrict to canonical ROI bounds
            x_mm = baseline["x_mm"]; y_mm = baseline["y_mm"]
            dx = float(x_mm[1] - x_mm[0]); dy = float(y_mm[1] - y_mm[0])
            col_lo = int(round((ROI_CX_MM - ROI_W_MM/2 - x_mm[0]) / dx))
            col_hi = int(round((ROI_CX_MM + ROI_W_MM/2 - x_mm[0]) / dx))
            row_lo = int(round((ROI_CY_MM - ROI_H_MM/2 - y_mm[0]) / dy))
            row_hi = int(round((ROI_CY_MM + ROI_H_MM/2 - y_mm[0]) / dy))
            roi_d = delta[row_lo:row_hi, col_lo:col_hi]
            vals.append(float(np.nanmean(roi_d)))
        print(f"{key:>9} " +
              "  ".join(f"{v:+.4g} {unit}".rjust(14) for v in vals))

    print(f"\nsaved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
