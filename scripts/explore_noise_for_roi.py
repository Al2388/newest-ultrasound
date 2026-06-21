"""Visualise the 6-repeat noise field so the operator can pick the ROI rectangle.

Outputs one figure with four panels:
  (a) Mean amplitude (where the cell is)
  (b) σ_ToF heatmap (the primary noise field)
  (c) σ_ToF inside footprint only (rest greyed) — easier to see "where it's quiet"
  (d) Composite low-σ mask: pixels with σ_ToF below 30/50/70 percentile

Also overlays three candidate central rectangles (20×20, 30×30, 40×40 mm) so the
operator can read off mm-coordinates of the cell-body region they want.
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
        "x_mm": first["x_mm"], "y_mm": first["y_mm"],
        "roi_w": float(first["x_mm"][-1] - first["x_mm"][0]),
        "roi_h": float(first["y_mm"][-1] - first["y_mm"][0]),
        "mean_amp": np.nanmean(amp, axis=0),
        "sigma_amp": np.nanstd(amp, axis=0, ddof=1),
        "sigma_tof": np.nanstd(tof, axis=0, ddof=1),
        "sigma_eng": np.nanstd(eng, axis=0, ddof=1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="noisefloor_v3.238")
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="Manual ROI candidates as 'WxH' in mm. If omitted, "
                         "candidates are auto-derived from noise thresholds.")
    ap.add_argument("--noise-percentiles", nargs="+", type=float,
                    default=[30.0, 50.0, 70.0],
                    help="If --candidates is not given: for each percentile of "
                         "footprint sigma_ToF, find the largest centred "
                         "rectangle (cell aspect ratio) whose mean σ stays "
                         "below that percentile.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = find_runs_by_prefix(args.prefix)
    if not runs:
        raise SystemExit(f"no runs match prefix '{args.prefix}'")
    print(f"using {len(runs)} runs")
    d = load_stack(runs)

    nrows, ncols = d["mean_amp"].shape
    extent = [float(d["x_mm"][0]), float(d["x_mm"][-1]),
              float(d["y_mm"][-1]), float(d["y_mm"][0])]
    px_x_mm = d["roi_w"] / ncols   # mm per pixel along X
    px_y_mm = d["roi_h"] / nrows   # mm per pixel along Y

    # Cell footprint detection (echo-strength threshold)
    finite_amp = d["mean_amp"][np.isfinite(d["mean_amp"])]
    p95_amp = float(np.percentile(finite_amp, 95))
    median_amp = float(np.percentile(finite_amp, 50))
    threshold_amp = 0.5 * median_amp   # the "in-cell" criterion
    footprint = np.isfinite(d["mean_amp"]) & (d["mean_amp"] >= threshold_amp)

    # Centroid of the footprint (for drawing centered candidate rectangles)
    ys, xs = np.where(footprint)
    if ys.size == 0:
        raise SystemExit("empty footprint")
    cy_px = float(ys.mean()); cx_px = float(xs.mean())
    cy_mm = cy_px * px_y_mm
    cx_mm = cx_px * px_x_mm
    print(f"footprint centroid: ({cx_mm:.2f}, {cy_mm:.2f}) mm")

    # σ statistics inside footprint
    sig_in = d["sigma_tof"][footprint & np.isfinite(d["sigma_tof"])]
    p30 = float(np.percentile(sig_in, 30))
    p50 = float(np.percentile(sig_in, 50))
    p70 = float(np.percentile(sig_in, 70))
    print(f"sigma_ToF in footprint (us): p30={p30:.4g}, p50={p50:.4g}, p70={p70:.4g}")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=160,
                             gridspec_kw={"wspace": 0.20, "hspace": 0.28,
                                          "left": 0.07, "right": 0.93,
                                          "top": 0.94, "bottom": 0.05})

    # ----- candidate rectangles -----
    candidates_wh: list[tuple[float, float, str]] = []   # (W, H, label)

    if args.candidates:
        # Manual mode
        for spec in args.candidates:
            try:
                w_s, h_s = spec.lower().replace(" ", "").split("x")
                w_mm, h_mm = float(w_s), float(h_s)
                candidates_wh.append((w_mm, h_mm, f"{w_mm:.0f}x{h_mm:.0f} mm"))
            except Exception:
                print(f"could not parse candidate '{spec}', skipping")
    else:
        # Auto: cell aspect ratio is taken from the cell-BODY bbox, defined as
        # footprint pixels whose sigma_tof is below p70 — this excludes both
        # the fixture region and the high-amplitude rim around the cell which
        # itself has high σ.
        core = footprint & np.isfinite(d["sigma_tof"]) & (d["sigma_tof"] <= p70)
        ys, xs = np.where(core)
        if ys.size == 0:
            ys, xs = np.where(footprint)
        cell_y0 = float(np.percentile(ys, 5)); cell_y1 = float(np.percentile(ys, 95))
        cell_x0 = float(np.percentile(xs, 5)); cell_x1 = float(np.percentile(xs, 95))
        cell_w_mm = (cell_x1 - cell_x0) * px_x_mm
        cell_h_mm = (cell_y1 - cell_y0) * px_y_mm
        aspect = cell_w_mm / cell_h_mm
        print(f"cell-body bbox (low-sigma core, p5-p95): "
              f"X=[{cell_x0*px_x_mm:.1f}, {cell_x1*px_x_mm:.1f}] mm, "
              f"Y=[{cell_y0*px_y_mm:.1f}, {cell_y1*px_y_mm:.1f}] mm, "
              f"aspect W/H = {aspect:.3f}")

        # Sweep heights, compute mean σ inside the centred rectangle
        h_values = np.linspace(5.0, cell_h_mm * 0.96, 120)
        mean_sigmas = []
        valid_h = []
        for h_mm in h_values:
            w_mm = h_mm * aspect
            h_px = max(2, int(round(h_mm / px_y_mm)))
            w_px = max(2, int(round(w_mm / px_x_mm)))
            cy_px = int(round(cy_mm / px_y_mm))
            cx_px = int(round(cx_mm / px_x_mm))
            y_s = cy_px - h_px // 2; x_s = cx_px - w_px // 2
            y_e = y_s + h_px;        x_e = x_s + w_px
            if y_s < 0 or x_s < 0 or y_e > nrows or x_e > ncols:
                continue
            sub_sig = d["sigma_tof"][y_s:y_e, x_s:x_e]
            sub_fp  = footprint[y_s:y_e, x_s:x_e]
            valid = sub_fp & np.isfinite(sub_sig)
            if not valid.any():
                continue
            mean_sigmas.append(float(np.nanmean(sub_sig[valid])))
            valid_h.append(h_mm)
        mean_sigmas = np.array(mean_sigmas)
        valid_h = np.array(valid_h)

        # For each noise percentile target, take the LARGEST h whose mean σ
        # is still ≤ that target.
        for pct in args.noise_percentiles:
            thr = float(np.percentile(sig_in, pct))
            ok = mean_sigmas <= thr
            if not ok.any():
                print(f"target p{pct:.0f} (sigma <= {thr*1000:.2f} ns): no rect fits")
                continue
            best_h = float(valid_h[ok].max())
            best_w = best_h * aspect
            label = f"sigma <= p{pct:.0f} ({thr*1000:.1f} ns)"
            candidates_wh.append((best_w, best_h, label))
            print(f"  target p{pct:.0f} (sigma <= {thr*1000:.2f} ns): "
                  f"largest rect = {best_w:.1f} x {best_h:.1f} mm")

    rect_colors = ["#22c55e", "#f59e0b", "#ef4444", "#a855f7"]   # green, amber, red, purple

    def draw_candidate_rectangles(ax):
        for i, (w_mm, h_mm, label) in enumerate(candidates_wh):
            col = rect_colors[i % len(rect_colors)]
            rect = mpatches.Rectangle(
                (cx_mm - w_mm / 2, cy_mm - h_mm / 2),
                w_mm, h_mm,
                fill=False, edgecolor=col, linewidth=1.8, linestyle="-",
                alpha=0.95,
            )
            ax.add_patch(rect)
            ax.text(cx_mm + w_mm / 2 - 0.5, cy_mm - h_mm / 2 + 1.6,
                    f"{w_mm:.0f}x{h_mm:.0f}  ({label})",
                    color="white", fontsize=8, ha="right", va="bottom",
                    bbox=dict(facecolor=col, alpha=0.85, pad=2,
                              edgecolor="none"))

    # (a) Mean amplitude
    ax = axes[0, 0]
    vmin = float(np.percentile(finite_amp, 1))
    vmax = float(np.percentile(finite_amp, 99))
    im = ax.imshow(d["mean_amp"], cmap="turbo", vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="mean amplitude (V)", fraction=0.046, pad=0.04)
    ax.set_title("(a) 6-scan mean amplitude   —   shows where the cell is",
                 fontsize=11, loc="left")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    draw_candidate_rectangles(ax)

    # (b) σ ToF — full footprint range
    ax = axes[0, 1]
    sig_finite = d["sigma_tof"][np.isfinite(d["sigma_tof"])]
    s_vmax = float(np.percentile(sig_finite, 98)) if sig_finite.size else 1.0
    im = ax.imshow(d["sigma_tof"], cmap="magma", vmin=0, vmax=max(s_vmax, 1e-9),
                   extent=extent, origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="σ ToF (µs)", fraction=0.046, pad=0.04)
    ax.set_title("(b) σ ToF — full noise field",
                 fontsize=11, loc="left")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    draw_candidate_rectangles(ax)

    # (c) σ ToF — only inside footprint, rest greyed; range trimmed to footprint p99
    ax = axes[1, 0]
    sig_fp = np.where(footprint, d["sigma_tof"], np.nan)
    vmax_fp = float(np.percentile(sig_in, 99))
    im = ax.imshow(sig_fp, cmap="magma", vmin=0, vmax=max(vmax_fp, 1e-9),
                   extent=extent, origin="upper", aspect="equal")
    # Grey overlay for masked-out
    grey = np.where(footprint, np.nan, 1.0)
    ax.imshow(grey, cmap=mcolors.ListedColormap([[0.6, 0.6, 0.6, 0.85]]),
              extent=extent, origin="upper", aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="σ ToF (µs) — footprint only", fraction=0.046, pad=0.04)
    ax.set_title("(c) σ ToF inside cell footprint   —   where to put the ROI",
                 fontsize=11, loc="left")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    draw_candidate_rectangles(ax)

    # (d) Low-σ mask quantiles
    ax = axes[1, 1]
    base = np.zeros_like(d["sigma_tof"])
    base[footprint & (d["sigma_tof"] <= p70)] = 1
    base[footprint & (d["sigma_tof"] <= p50)] = 2
    base[footprint & (d["sigma_tof"] <= p30)] = 3
    cmap_quant = mcolors.ListedColormap(
        ["#222222",      # 0 — non-footprint
         "#777777",      # 1 — high σ (footprint but > p70)
         "#fde68a",      # 2 — medium (p50–p70)
         "#fb923c",      # 3 — low (p30–p50)
         "#16a34a",      # 4 — ultra-low (≤ p30)
        ])
    # Re-encode so that levels 1..4 represent the bands
    band = np.where(footprint, 0, 0).astype(int)
    band[footprint & (d["sigma_tof"] >  p70)] = 1
    band[footprint & (d["sigma_tof"] <= p70)] = 2
    band[footprint & (d["sigma_tof"] <= p50)] = 3
    band[footprint & (d["sigma_tof"] <= p30)] = 4
    ax.imshow(band, cmap=cmap_quant, vmin=0, vmax=4,
              extent=extent, origin="upper", aspect="equal", interpolation="nearest")
    # Legend
    legend_patches = [
        mpatches.Patch(facecolor="#222222", edgecolor="#444",
                       label="outside footprint"),
        mpatches.Patch(facecolor="#777777", edgecolor="#444",
                       label="footprint, σ > p70 (noisy)"),
        mpatches.Patch(facecolor="#fde68a", edgecolor="#444",
                       label=f"σ ≤ p70 = {p70:.4g} µs"),
        mpatches.Patch(facecolor="#fb923c", edgecolor="#444",
                       label=f"σ ≤ p50 = {p50:.4g} µs"),
        mpatches.Patch(facecolor="#16a34a", edgecolor="#444",
                       label=f"σ ≤ p30 = {p30:.4g} µs"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="lower right",
              framealpha=0.95)
    ax.set_title("(d) σ_ToF quantile bands inside footprint — green = cleanest",
                 fontsize=11, loc="left")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    draw_candidate_rectangles(ax)

    fig.suptitle(
        f"Noise field for ROI selection  —  prefix '{args.prefix}', n={len(runs)} scans\n"
        f"echo threshold = 0.5 × median(mean amplitude) = {threshold_amp:.3g} V  ·  "
        f"footprint centroid = ({cx_mm:.1f}, {cy_mm:.1f}) mm",
        fontsize=11.5, y=0.985,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) if args.out else (
        PROJECT / "reports" / "experiments" / f"noise_explore_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "noise_for_roi.png", bbox_inches="tight")
    fig.savefig(out_dir / "noise_for_roi.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Print key coordinates for the user to read off
    print(f"\n=== saved to: {out_dir} ===")
    print(f"  candidate rectangles centred at ({cx_mm:.2f}, {cy_mm:.2f}) mm")
    for w_mm, h_mm, label in candidates_wh:
        x0 = cx_mm - w_mm / 2; y0 = cy_mm - h_mm / 2
        x1 = cx_mm + w_mm / 2; y1 = cy_mm + h_mm / 2
        print(f"    {w_mm:.1f}x{h_mm:.1f} mm  ({label}): "
              f"X = [{x0:.2f}, {x1:.2f}] mm, Y = [{y0:.2f}, {y1:.2f}] mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
