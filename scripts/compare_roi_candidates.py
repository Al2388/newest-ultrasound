"""Compare multiple ROI candidates: for each (W × H) pair, compute the
ROI-aggregated σ for amp / ToF / energy, then plot all candidates on one
sigma_ToF map plus a bar-chart panel so the trade-off between area and
noise is visible at a glance.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"

EXPECTED = {"roi_w_mm": 80.0, "roi_h_mm": 72.0, "pitch_mm": 0.5,
            "speed_mm_s": 25.0, "accel_mm_s2": 800.0}


def matches_params(meta: dict, tol: float = 1e-3) -> bool:
    for k, v in EXPECTED.items():
        if k not in meta or abs(float(meta[k]) - v) > tol:
            return False
    return True


def find_runs_by_prefix(prefix: str) -> list[Path]:
    candidates = []
    for d in CSCAN_ROOT.iterdir():
        if not d.is_dir() or not d.name.startswith(f"cscan_{prefix}"):
            continue
        if next(d.glob("scan_*.npz"), None) is None:
            continue
        meta_p = next(d.glob("scan_*_meta.json"), None)
        if meta_p is None:
            continue
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        if not matches_params(meta):
            continue
        if len(list((d / "lines_raw").glob("line_*.npz"))) < int(meta.get("nlines", 0)):
            continue
        candidates.append(d)
    batch_tag = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_r\d{2}_")
    by_batch: dict[str, list[Path]] = {}
    for d in candidates:
        m = batch_tag.search(d.name)
        if m:
            by_batch.setdefault(m.group(1), []).append(d)
    return sorted(by_batch[max(by_batch.keys())]) if by_batch else sorted(candidates)


def load_stack(scan_dirs: list[Path]) -> dict:
    npzs = [next(d.glob("scan_*.npz")) for d in scan_dirs]
    first = np.load(npzs[0])
    nrows, ncols = first["amplitude"].shape
    amp = np.empty((len(npzs), nrows, ncols), dtype=np.float64)
    tof = np.empty_like(amp); eng = np.empty_like(amp)
    amp[0] = first["amplitude"]; tof[0] = first["tof"]; eng[0] = first["energy"]
    for i, p in enumerate(npzs[1:], start=1):
        d = np.load(p)
        amp[i] = d["amplitude"]; tof[i] = d["tof"]; eng[i] = d["energy"]
    return {
        "amp": amp, "tof": tof, "eng": eng,
        "x_mm": first["x_mm"], "y_mm": first["y_mm"],
    }


def detect_footprint(mean_amp: np.ndarray, rel_median: float = 0.5) -> np.ndarray:
    finite = mean_amp[np.isfinite(mean_amp)]
    if finite.size == 0:
        return np.zeros_like(mean_amp, dtype=bool)
    return np.isfinite(mean_amp) & (
        mean_amp >= rel_median * float(np.percentile(finite, 50)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--candidates", nargs="+",
                    default=["20x15", "30x25", "40x30", "45x35", "50x40", "55x45"],
                    help="ROI candidates as 'WxH' in mm")
    ap.add_argument("--cx-mm", type=float, default=None,
                    help="ROI centre X (default: footprint centroid)")
    ap.add_argument("--cy-mm", type=float, default=None,
                    help="ROI centre Y")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cands: list[tuple[float, float]] = []
    for spec in args.candidates:
        try:
            w_s, h_s = spec.lower().replace(" ", "").split("x")
            cands.append((float(w_s), float(h_s)))
        except Exception:
            print(f"skip unparseable '{spec}'")
    cands.sort(key=lambda wh: wh[0] * wh[1])  # smallest area first

    runs = find_runs_by_prefix(args.prefix)
    if not runs:
        raise SystemExit(f"no runs match '{args.prefix}'")
    print(f"loaded {len(runs)} scans")
    stk = load_stack(runs)
    nrows, ncols = stk["amp"].shape[1:]
    dx_mm = float(stk["x_mm"][1] - stk["x_mm"][0])
    dy_mm = float(stk["y_mm"][1] - stk["y_mm"][0])

    mean_amp = np.nanmean(stk["amp"], axis=0)
    footprint = detect_footprint(mean_amp)
    sigma_tof_map = np.nanstd(stk["tof"], axis=0, ddof=1)

    if args.cx_mm is None or args.cy_mm is None:
        ys, xs = np.where(footprint)
        cx_mm = float(xs.mean() * dx_mm + stk["x_mm"][0])
        cy_mm = float(ys.mean() * dy_mm + stk["y_mm"][0])
    else:
        cx_mm = float(args.cx_mm); cy_mm = float(args.cy_mm)
    print(f"centre = ({cx_mm:.2f}, {cy_mm:.2f}) mm")

    # ----- compute σ_ROI for each candidate -----
    results = []
    for w_mm, h_mm in cands:
        col_lo = max(0, int(round((cx_mm - w_mm / 2 - stk["x_mm"][0]) / dx_mm)))
        col_hi = min(ncols, int(round((cx_mm + w_mm / 2 - stk["x_mm"][0]) / dx_mm)))
        row_lo = max(0, int(round((cy_mm - h_mm / 2 - stk["y_mm"][0]) / dy_mm)))
        row_hi = min(nrows, int(round((cy_mm + h_mm / 2 - stk["y_mm"][0]) / dy_mm)))
        roi_mask = np.zeros((nrows, ncols), dtype=bool)
        roi_mask[row_lo:row_hi, col_lo:col_hi] = True

        # ROI-aggregated SD across 6 scans
        sigmas = {}
        for key in ("amp", "tof", "eng"):
            per_scan = np.array([float(np.nanmean(stk[key][i][roi_mask]))
                                  for i in range(stk[key].shape[0])])
            sigmas[key] = float(np.std(per_scan, ddof=1))
        # Per-pixel σ statistics inside ROI
        pp = sigma_tof_map[roi_mask & np.isfinite(sigma_tof_map)]
        results.append({
            "w_mm": w_mm, "h_mm": h_mm,
            "area_mm2": w_mm * h_mm,
            "n_px": int(roi_mask.sum()),
            "sigma_amp": sigmas["amp"],
            "sigma_tof": sigmas["tof"],
            "sigma_eng": sigmas["eng"],
            "ppx_tof_median": float(np.median(pp)) if pp.size else float("nan"),
            "ppx_tof_p95":    float(np.percentile(pp, 95)) if pp.size else float("nan"),
            "x0_mm": cx_mm - w_mm / 2, "x1_mm": cx_mm + w_mm / 2,
            "y0_mm": cy_mm - h_mm / 2, "y1_mm": cy_mm + h_mm / 2,
        })

    # ----- output dir -----
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args.label}" if args.label else ""
    out_dir = PROJECT / "reports" / "experiments" / f"roi_compare_{ts}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- figure: σ_ToF map with rectangles + bar chart on right -----
    extent = [float(stk["x_mm"][0]), float(stk["x_mm"][-1]),
              float(stk["y_mm"][-1]), float(stk["y_mm"][0])]

    fig = plt.figure(figsize=(15, 7.5), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0],
                          wspace=0.18, left=0.05, right=0.97,
                          top=0.92, bottom=0.10)

    # --- left: noise map with all candidates ---
    ax_map = fig.add_subplot(gs[0, 0])
    sig_fp = sigma_tof_map[footprint & np.isfinite(sigma_tof_map)]
    vmax = float(np.percentile(sig_fp, 99))
    im = ax_map.imshow(sigma_tof_map, cmap="magma", vmin=0, vmax=max(vmax, 1e-9),
                       extent=extent, origin="upper", aspect="equal")
    grey = np.where(footprint, np.nan, 1.0)
    ax_map.imshow(grey, cmap=mcolors.ListedColormap([[0.55, 0.55, 0.55, 0.8]]),
                  extent=extent, origin="upper", aspect="equal",
                  interpolation="nearest")
    plt.colorbar(im, ax=ax_map, label="σ ToF (µs)", fraction=0.046, pad=0.04)

    palette = plt.colormaps["viridis"](np.linspace(0.05, 0.95, len(results)))
    for i, r in enumerate(results):
        col = palette[i]
        rect = mpatches.Rectangle(
            (r["x0_mm"], r["y0_mm"]), r["w_mm"], r["h_mm"],
            fill=False, edgecolor=col, linewidth=2.0, linestyle="-",
            alpha=0.95,
        )
        ax_map.add_patch(rect)
        # Label outside on the top-right corner of each
        ax_map.text(r["x1_mm"] - 0.5, r["y0_mm"] - 0.5,
                    f"{int(r['w_mm'])}x{int(r['h_mm'])}: σ={r['sigma_tof']*1000:.1f}ns",
                    color="white", fontsize=8, fontweight="bold",
                    ha="right", va="bottom",
                    bbox=dict(facecolor=col, edgecolor="none", pad=2, alpha=0.85))

    ax_map.set_title("σ ToF noise field with candidate ROIs", fontsize=11)
    ax_map.set_xlabel("X (mm)"); ax_map.set_ylabel("Y (mm)")
    ax_map.tick_params(labelsize=8)

    # --- right: TWO bar charts stacked, showing two complementary noise metrics ---
    gs_r = gs[0, 1].subgridspec(2, 1, hspace=0.45)
    ax_bar1 = fig.add_subplot(gs_r[0, 0])
    ax_bar2 = fig.add_subplot(gs_r[1, 0])

    sizes = [f"{int(r['w_mm'])}x{int(r['h_mm'])}\n({r['n_px']:,} px)" for r in results]
    x = np.arange(len(results))

    # --- (top) ROI-aggregated σ_ToF (scan-to-scan correlated component) ---
    sig_tof = np.array([r["sigma_tof"] for r in results]) * 1000  # ns
    ax_bar1.bar(x, sig_tof, color="#16a34a", alpha=0.85)
    for i, v in enumerate(sig_tof):
        ax_bar1.text(x[i], v + 0.05, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold", color="#16a34a")
    ax_bar1.set_xticks(x); ax_bar1.set_xticklabels(sizes, fontsize=8)
    ax_bar1.set_ylabel("σ_ROI (ToF, ns)", fontsize=9)
    ax_bar1.set_title("(top) σ_ROI(ToF) — scan-to-scan correlated drift "
                      "(roughly flat in size)", fontsize=9.5, loc="left")
    ax_bar1.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax_bar1.set_ylim(0, max(sig_tof) * 1.20)
    ax_bar1.tick_params(labelsize=8)

    # --- (bottom) Per-pixel σ_ToF inside ROI (median + p95) ---
    ppx_med = np.array([r["ppx_tof_median"] for r in results]) * 1000
    ppx_p95 = np.array([r["ppx_tof_p95"] for r in results]) * 1000
    width = 0.4
    ax_bar2.bar(x - width / 2, ppx_med, width, color="#16a34a", alpha=0.85,
                label="median")
    ax_bar2.bar(x + width / 2, ppx_p95, width, color="#f59e0b", alpha=0.85,
                label="p95")
    for i in range(len(results)):
        ax_bar2.text(x[i] - width / 2, ppx_med[i] + 0.3, f"{ppx_med[i]:.1f}",
                     ha="center", va="bottom", fontsize=7.5, color="#15803d")
        ax_bar2.text(x[i] + width / 2, ppx_p95[i] + 0.3, f"{ppx_p95[i]:.1f}",
                     ha="center", va="bottom", fontsize=7.5, color="#b45309")
    ax_bar2.set_xticks(x); ax_bar2.set_xticklabels(sizes, fontsize=8)
    ax_bar2.set_ylabel("per-pixel σ_ToF (ns)", fontsize=9)
    ax_bar2.set_title("(bottom) per-pixel σ_ToF inside ROI — what shrinking removes",
                      fontsize=9.5, loc="left")
    ax_bar2.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax_bar2.legend(fontsize=8.5, loc="upper left")
    ax_bar2.set_ylim(0, max(ppx_p95) * 1.20)
    ax_bar2.tick_params(labelsize=8)

    fig.suptitle(f"ROI candidate comparison  —  {len(runs)} scans  ·  centre = "
                 f"({cx_mm:.1f}, {cy_mm:.1f}) mm",
                 fontsize=12, y=0.98)
    fig.savefig(out_dir / "roi_compare.png", bbox_inches="tight")
    fig.savefig(out_dir / "roi_compare.pdf", bbox_inches="tight")
    plt.close(fig)

    # ----- summary CSV -----
    with (out_dir / "candidates_summary.csv").open("w", encoding="utf-8") as f:
        f.write("w_mm,h_mm,area_mm2,n_px,sigma_amp_V,sigma_tof_us,sigma_tof_ns,sigma_eng,"
                "ppx_tof_median_us,ppx_tof_p95_us,x0_mm,x1_mm,y0_mm,y1_mm\n")
        for r in results:
            f.write(f"{r['w_mm']:.1f},{r['h_mm']:.1f},{r['area_mm2']:.0f},{r['n_px']},"
                    f"{r['sigma_amp']:.6f},{r['sigma_tof']:.6f},{r['sigma_tof']*1000:.2f},"
                    f"{r['sigma_eng']:.4f},{r['ppx_tof_median']:.6f},"
                    f"{r['ppx_tof_p95']:.6f},"
                    f"{r['x0_mm']:.2f},{r['x1_mm']:.2f},"
                    f"{r['y0_mm']:.2f},{r['y1_mm']:.2f}\n")

    # Print table to stdout
    print(f"\n{'size':>10} {'n_px':>7} {'sig_ROI ToF (ns)':>18} "
          f"{'pp med (ns)':>13} {'pp p95 (ns)':>13}")
    print("-" * 72)
    for r in results:
        print(f"{int(r['w_mm'])}x{int(r['h_mm']):<6} "
              f"{r['n_px']:>7,} "
              f"{r['sigma_tof']*1000:>18.2f} "
              f"{r['ppx_tof_median']*1000:>13.2f} "
              f"{r['ppx_tof_p95']*1000:>13.2f}")

    print(f"\nsaved: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
