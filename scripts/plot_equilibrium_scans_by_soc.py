"""For each target SoC, find the LAST rest scan in that plateau and plot
amplitude / ToF / energy maps side by side.

Outputs a 3-row × N-col panel (rows = feature, cols = target SoC) with shared
per-row colour scale. Each panel labelled with the actual SoC% of the chosen
scan.
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

# Canonical ROI for overlay
ROI_W_MM, ROI_H_MM = 50.02, 39.78
ROI_CX_MM, ROI_CY_MM = 39.59, 36.00


def group_rest_plateaus(rest_df: pd.DataFrame, max_gap_s: float = 1200.0) -> list[pd.DataFrame]:
    """Split rest scans into plateaus by time-gap clustering."""
    rest = rest_df.sort_values("time_utc").reset_index(drop=True).copy()
    t = rest["time_utc"].astype("int64") / 1e9
    gaps = t.diff().fillna(0)
    plateau_id = (gaps > max_gap_s).cumsum()
    return [g.reset_index(drop=True) for _, g in rest.groupby(plateau_id)]


def find_scan_dir(scan_id: str) -> Path | None:
    # scan_id is like "scan_2026-05-30_03-11-29" — the timestamp suffix is the
    # unique tail of the scan directory's name. Match on the timestamp only.
    suffix = scan_id.replace("scan_", "", 1)
    candidates = list(CSCAN_ROOT.glob(f"cscan_*{suffix}"))
    if not candidates:
        # fallback: directory ending with the suffix (in case glob misses)
        candidates = [p for p in CSCAN_ROOT.iterdir()
                      if p.is_dir() and p.name.endswith(suffix)]
    return candidates[0] if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--target-socs", nargs="+", type=float,
                    default=[20.0, 40.0, 60.0, 80.0],
                    help="Target SoC values (%); we pick the plateau whose SoC "
                         "is closest to each, then the LAST scan in that plateau")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    feat = pd.read_csv(batch_dir / "overview" / "scans_tagged.csv")
    feat["time_utc"] = pd.to_datetime(feat["time_utc"], utc=True)
    rest = feat[feat["step_tag"] == "rest"].copy()
    print(f"rest scans: {len(rest)}")

    plateaus = group_rest_plateaus(rest)
    print(f"plateaus: {len(plateaus)}")
    for i, p in enumerate(plateaus):
        last = p.iloc[-1]
        print(f"  plateau {i}: n={len(p)}, SoC~{p['soc_pct'].mean():.1f}%, "
              f"last scan = {last['scan_id']} @ {last['soc_pct']:.1f}%")

    # Find the plateau closest to each target SoC
    plateau_socs = np.array([p["soc_pct"].mean() for p in plateaus])
    picks = []
    for target in args.target_socs:
        idx = int(np.argmin(np.abs(plateau_socs - target)))
        last_scan = plateaus[idx].iloc[-1]
        picks.append({
            "target_soc": target,
            "actual_soc": float(last_scan["soc_pct"]),
            "plateau_idx": idx,
            "scan_id":      last_scan["scan_id"],
            "time_utc":     last_scan["time_utc"],
            "voltage":      float(last_scan["voltage_at_scan"]),
            "temp":         float(last_scan["temp_at_scan"])
                            if "temp_at_scan" in last_scan else float("nan"),
        })
        print(f"target SoC {target}% -> plateau {idx} (actual {last_scan['soc_pct']:.1f}%, "
              f"V={last_scan['voltage_at_scan']:.3f}, scan={last_scan['scan_id']})")

    # Load each scan's 2D maps
    panels = []
    for pick in picks:
        scan_dir = find_scan_dir(pick["scan_id"])
        if scan_dir is None:
            print(f"  WARNING: no scan dir for {pick['scan_id']}")
            continue
        npz_p = next(scan_dir.glob("scan_*.npz"), None)
        if npz_p is None:
            continue
        d = np.load(npz_p)
        panels.append({
            **pick,
            "amp": np.asarray(d["amplitude"], dtype=np.float64),
            "tof": np.asarray(d["tof"],       dtype=np.float64),
            "eng": np.asarray(d["energy"],    dtype=np.float64),
            "x_mm": np.asarray(d["x_mm"]),
            "y_mm": np.asarray(d["y_mm"]),
        })

    n = len(panels)
    if n == 0:
        raise SystemExit("no panels loaded")
    extent = [float(panels[0]["x_mm"][0]), float(panels[0]["x_mm"][-1]),
              float(panels[0]["y_mm"][-1]), float(panels[0]["y_mm"][0])]

    # Shared per-row colour scale (1-99 percentile across all picks)
    def joint_range(key):
        arrs = [p[key] for p in panels]
        joint = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs])
        return float(np.percentile(joint, 1)), float(np.percentile(joint, 99))

    feature_specs = [
        ("amp", "amplitude (V)", "turbo",   "gray"),
        ("tof", "ToF (µs)",       "viridis", "magma"),
        ("eng", "energy",          "cividis", "viridis"),
    ]

    fig, axes = plt.subplots(3, n, figsize=(3.6 * n + 0.8, 11), dpi=160,
                             gridspec_kw={"wspace": 0.10, "hspace": 0.20,
                                          "top": 0.93, "bottom": 0.05,
                                          "left": 0.06, "right": 0.94})
    if n == 1:
        axes = axes.reshape(3, 1)

    for row, (key, ylabel, cmap_main, _) in enumerate(feature_specs):
        vmin, vmax = joint_range(key)
        last_im = None
        for col, p in enumerate(panels):
            ax = axes[row, col]
            last_im = ax.imshow(p[key], cmap=cmap_main, vmin=vmin, vmax=vmax,
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
                    f"SoC = {p['actual_soc']:.1f}%\n"
                    f"V = {p['voltage']:.3f} V,  T = {p['temp']:.2f}°C",
                    fontsize=10, fontweight="bold")
            # ROI overlay
            rect = mpatches.Rectangle(
                (ROI_CX_MM - ROI_W_MM/2, ROI_CY_MM - ROI_H_MM/2),
                ROI_W_MM, ROI_H_MM,
                fill=False, edgecolor="#16a34a", linewidth=1.3, alpha=0.9,
            )
            ax.add_patch(rect)
        # one colourbar per row, attached to last panel
        plt.colorbar(last_im, ax=axes[row, :].tolist(), fraction=0.018,
                     pad=0.02, shrink=0.92, label=ylabel)

    fig.suptitle(
        "Quasi-equilibrium C-scans at four SoC levels "
        "(last rest scan in each plateau)  —  rows: amp / ToF / energy, "
        "shared per-row colour scale",
        fontsize=12, y=0.985)

    out_dir = Path(args.out) if args.out else batch_dir / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "equilibrium_scans_by_soc.png", bbox_inches="tight")
    fig.savefig(out_dir / "equilibrium_scans_by_soc.pdf", bbox_inches="tight")
    plt.close(fig)

    # Save metadata
    with (out_dir / "equilibrium_picks.json").open("w", encoding="utf-8") as f:
        json.dump([{
            "target_soc": p["target_soc"],
            "actual_soc": p["actual_soc"],
            "plateau_idx": p["plateau_idx"],
            "scan_id":     p["scan_id"],
            "time_utc":    str(p["time_utc"]),
            "voltage_v":   p["voltage"],
            "temp_c":      p["temp"],
        } for p in panels], f, indent=2)

    print(f"\nsaved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
