"""Two clean figures for the chosen ROI:
  Fig A: 1×3 of 6-scan MEAN amplitude, ToF, energy maps + ROI overlay.
  Fig B: 1×3 of per-pixel σ amplitude, ToF, energy maps + ROI overlay,
         with per-feature ROI σ statistics annotated on each panel.

Only the ROI is annotated. No "excluded cell band" overlay.
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


def add_roi_box(ax, x0, y0, w, h, color="#16a34a", label="ROI"):
    rect = mpatches.Rectangle((x0, y0), w, h,
                              fill=False, edgecolor=color, linewidth=2.0,
                              clip_on=False)
    ax.add_patch(rect)
    ax.text(x0 + 1, y0 + 2, label, color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.9),
            va="top", ha="left")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--roi-w-mm", type=float, required=True)
    ap.add_argument("--roi-h-mm", type=float, required=True)
    ap.add_argument("--roi-cx-mm", type=float, default=None)
    ap.add_argument("--roi-cy-mm", type=float, default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    runs = find_runs_by_prefix(args.prefix)
    if not runs:
        raise SystemExit(f"no runs match '{args.prefix}'")
    print(f"loaded {len(runs)} scans")
    stk = load_stack(runs)
    nrows, ncols = stk["amp"].shape[1:]
    dx_mm = float(stk["x_mm"][1] - stk["x_mm"][0])
    dy_mm = float(stk["y_mm"][1] - stk["y_mm"][0])

    mean_amp = np.nanmean(stk["amp"], axis=0)
    mean_tof = np.nanmean(stk["tof"], axis=0)
    mean_eng = np.nanmean(stk["eng"], axis=0)
    sigma_amp = np.nanstd(stk["amp"], axis=0, ddof=1)
    sigma_tof = np.nanstd(stk["tof"], axis=0, ddof=1)
    sigma_eng = np.nanstd(stk["eng"], axis=0, ddof=1)

    footprint = detect_footprint(mean_amp)

    # ROI placement
    cx = args.roi_cx_mm
    cy = args.roi_cy_mm
    if cx is None or cy is None:
        ys, xs = np.where(footprint)
        if cx is None:
            cx = float(xs.mean() * dx_mm + stk["x_mm"][0])
        if cy is None:
            cy = float(ys.mean() * dy_mm + stk["y_mm"][0])
    x0 = cx - args.roi_w_mm / 2; x1 = cx + args.roi_w_mm / 2
    y0 = cy - args.roi_h_mm / 2; y1 = cy + args.roi_h_mm / 2

    col_lo = max(0, int(round((x0 - float(stk["x_mm"][0])) / dx_mm)))
    col_hi = min(ncols, int(round((x1 - float(stk["x_mm"][0])) / dx_mm)))
    row_lo = max(0, int(round((y0 - float(stk["y_mm"][0])) / dy_mm)))
    row_hi = min(nrows, int(round((y1 - float(stk["y_mm"][0])) / dy_mm)))
    roi_mask = np.zeros_like(mean_amp, dtype=bool)
    roi_mask[row_lo:row_hi, col_lo:col_hi] = True
    roi_w_actual = (col_hi - col_lo) * dx_mm
    roi_h_actual = (row_hi - row_lo) * dy_mm
    n_px = int(roi_mask.sum())

    # Per-pixel σ stats inside ROI
    sa = sigma_amp[roi_mask & np.isfinite(sigma_amp)]
    st = sigma_tof[roi_mask & np.isfinite(sigma_tof)]
    se = sigma_eng[roi_mask & np.isfinite(sigma_eng)]

    # ROI-aggregated σ (SD of ROI-mean across 6 scans)
    def roi_sd(stack):
        per_scan = np.array(
            [float(np.nanmean(stack[i][roi_mask])) for i in range(stack.shape[0])]
        )
        return float(np.std(per_scan, ddof=1))

    sigma_amp_roi = roi_sd(stk["amp"])
    sigma_tof_roi = roi_sd(stk["tof"])
    sigma_eng_roi = roi_sd(stk["eng"])

    print(f"\nROI: {roi_w_actual:.2f} x {roi_h_actual:.2f} mm   centre = ({cx:.2f}, {cy:.2f}) mm")
    print(f"     X = [{stk['x_mm'][col_lo]:.2f}, {stk['x_mm'][col_hi-1]:.2f}] mm")
    print(f"     Y = [{stk['y_mm'][row_lo]:.2f}, {stk['y_mm'][row_hi-1]:.2f}] mm")
    print(f"     n_pixels = {n_px:,}")
    print(f"\nPer-pixel sigma inside ROI:")
    print(f"  amp:    median = {sa.mean()*1000:.2f} mV  (mean)  med = {np.median(sa)*1000:.2f} mV   p95 = {np.percentile(sa,95)*1000:.2f} mV")
    print(f"  ToF:    median = {np.median(st)*1000:.2f} ns                  p95 = {np.percentile(st,95)*1000:.2f} ns")
    print(f"  energy: median = {np.median(se):.4f}                          p95 = {np.percentile(se,95):.4f}")
    print(f"\nROI-aggregated sigma (SD of ROI-mean across {stk['amp'].shape[0]} scans):")
    print(f"  amp:    {sigma_amp_roi*1000:.3f} mV")
    print(f"  ToF:    {sigma_tof_roi*1000:.3f} ns")
    print(f"  energy: {sigma_eng_roi:.5f}")

    extent = [float(stk["x_mm"][0]), float(stk["x_mm"][-1]),
              float(stk["y_mm"][-1]), float(stk["y_mm"][0])]

    # ----------------------------------------------------------------- Fig A
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), dpi=160,
                             gridspec_kw={"wspace": 0.22, "top": 0.90,
                                          "left": 0.05, "right": 0.97,
                                          "bottom": 0.10})
    panels_a = [
        (mean_amp, "(a) Mean amplitude", "amplitude (V)", "gray"),
        (mean_tof, "(b) Mean ToF",        "ToF (µs)",      "viridis"),
        (mean_eng, "(c) Mean energy",     "energy",         "cividis"),
    ]
    for ax, (data, title, cbar_lbl, cmap) in zip(axes, panels_a):
        finite = data[np.isfinite(data)]
        vmin = float(np.percentile(finite, 1))
        vmax = float(np.percentile(finite, 99))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, origin="upper", aspect="equal")
        plt.colorbar(im, ax=ax, label=cbar_lbl, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("X (mm)", fontsize=9)
        ax.set_ylabel("Y (mm)", fontsize=9)
        ax.tick_params(labelsize=8)
        add_roi_box(ax, x0, y0, roi_w_actual, roi_h_actual)
    fig.suptitle(f"6-scan mean baseline maps  —  "
                 f"ROI = {roi_w_actual:.1f} × {roi_h_actual:.1f} mm "
                 f"({n_px:,} px),  centre = ({cx:.1f}, {cy:.1f}) mm",
                 fontsize=12, y=0.99)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args.label}" if args.label else ""
    out_dir = PROJECT / "reports" / "experiments" / f"roi_clean_figs_{ts}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figA_mean_maps.png", bbox_inches="tight")
    fig.savefig(out_dir / "figA_mean_maps.pdf", bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------- Fig B
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=160,
                             gridspec_kw={"wspace": 0.22, "top": 0.88,
                                          "left": 0.05, "right": 0.97,
                                          "bottom": 0.10})
    panels_b = [
        (sigma_amp, sa, "(a) σ amplitude (V)",    "σ amplitude (V)",  "mV",  1000),
        (sigma_tof, st, "(b) σ ToF (µs)",         "σ ToF (µs)",       "ns",  1000),
        (sigma_eng, se, "(c) σ energy",            "σ energy",          "",    1),
    ]
    for ax, (sig_map, sig_in_roi, title, cbar_lbl, units, scale) in zip(axes, panels_b):
        sig_fp = sig_map[footprint & np.isfinite(sig_map)]
        vmax = float(np.percentile(sig_fp, 98)) if sig_fp.size else 1.0
        im = ax.imshow(sig_map, cmap="magma", vmin=0, vmax=max(vmax, 1e-9),
                       extent=extent, origin="upper", aspect="equal")
        # Grey outside footprint
        grey = np.where(footprint, np.nan, 1.0)
        ax.imshow(grey, cmap=mcolors.ListedColormap([[0.55, 0.55, 0.55, 0.8]]),
                  extent=extent, origin="upper", aspect="equal",
                  interpolation="nearest")
        plt.colorbar(im, ax=ax, label=cbar_lbl, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("X (mm)", fontsize=9)
        ax.set_ylabel("Y (mm)", fontsize=9)
        ax.tick_params(labelsize=8)
        add_roi_box(ax, x0, y0, roi_w_actual, roi_h_actual)
        # ROI σ stats annotation
        med = float(np.median(sig_in_roi)) * scale
        p95 = float(np.percentile(sig_in_roi, 95)) * scale
        ax.text(0.98, 0.97,
                f"ROI per-pixel σ\nmed = {med:.2f} {units}\np95 = {p95:.2f} {units}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color="white", fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="white", linewidth=0.5,
                          pad=4, alpha=0.75))

    fig.suptitle(f"Per-pixel σ maps from 6 repeat scans  —  "
                 f"ROI = {roi_w_actual:.1f} × {roi_h_actual:.1f} mm ({n_px:,} px)",
                 fontsize=12, y=0.99)
    fig.savefig(out_dir / "figB_sigma_maps.png", bbox_inches="tight")
    fig.savefig(out_dir / "figB_sigma_maps.pdf", bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------- summary
    summary = {
        "roi": {
            "w_mm": roi_w_actual, "h_mm": roi_h_actual,
            "cx_mm": cx, "cy_mm": cy,
            "x0_mm": float(stk["x_mm"][col_lo]), "x1_mm": float(stk["x_mm"][col_hi-1]),
            "y0_mm": float(stk["y_mm"][row_lo]), "y1_mm": float(stk["y_mm"][row_hi-1]),
            "n_pixels": n_px,
        },
        "per_pixel_sigma_inside_roi": {
            "amplitude_V":      {"median": float(np.median(sa)),
                                  "p95":    float(np.percentile(sa, 95)),
                                  "max":    float(sa.max())},
            "tof_us":           {"median": float(np.median(st)),
                                  "p95":    float(np.percentile(st, 95)),
                                  "max":    float(st.max())},
            "energy":           {"median": float(np.median(se)),
                                  "p95":    float(np.percentile(se, 95)),
                                  "max":    float(se.max())},
        },
        "roi_aggregated_sigma": {
            "amplitude_V": sigma_amp_roi,
            "tof_us":      sigma_tof_roi,
            "energy":      sigma_eng_roi,
        },
        "n_scans": int(stk["amp"].shape[0]),
        "prefix":  args.prefix,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "roi_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "roi_mask.npy", roi_mask)

    print(f"\nsaved: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
