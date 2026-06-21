"""
Gaussian-beam-kernel sigma sweep for the M6 reconstruction method.

Runs the M6 Gaussian-kernel reconstruction (from reprocess_cscan_methods.py)
across a range of sigma values on one C-scan, and reports:

  - Noise floor (row-to-row sigma inside the cell interior)
  - Edge sharpness  (median gradient magnitude at a horizontal cross-section
                    across the cell-to-background transition)
  - Visual panels for amplitude / ToF / energy at several sigmas

A good sigma is the smallest value that flattens the noise floor without
visibly softening the cell-edge transition. Two reported curves let you
pick that knee directly from data, instead of guessing from the beam spec.

Usage
-----
  python scripts/sweep_gauss_sigma.py --scan data/raw/cscan/<run-folder>
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Local copies of the M6 kernel & helpers (avoid importing the comparison
# script as a module — it has CLI side-effects).
# ---------------------------------------------------------------------------

def _bin_indices(x_mm, roi_w, ncols):
    idx = np.floor(x_mm / roi_w * ncols).astype(np.int64)
    return np.clip(idx, 0, ncols - 1)


def m6_gauss_kernel(aa, x_mm, roi_w, ncols, sigma_mm):
    """Gaussian-weighted scatter with beam-radius sigma in mm."""
    out = np.full(ncols, np.nan, dtype=np.float32)
    valid = np.isfinite(aa)
    if not np.any(valid):
        return out
    if sigma_mm <= 0:
        # Fall back to bilinear scatter (point-footprint baseline)
        a = aa[valid].astype(np.float64)
        x = x_mm[valid].astype(np.float64) / roi_w * ncols
        i0 = np.floor(x).astype(np.int64); w1 = x - i0; w0 = 1.0 - w1
        i1 = i0 + 1
        sumw = np.zeros(ncols); acc = np.zeros(ncols)
        m = (i0 >= 0) & (i0 < ncols)
        np.add.at(sumw, i0[m], w0[m]); np.add.at(acc, i0[m], w0[m] * a[m])
        m = (i1 >= 0) & (i1 < ncols)
        np.add.at(sumw, i1[m], w1[m]); np.add.at(acc, i1[m], w1[m] * a[m])
        nz = sumw > 1e-12
        out[nz] = (acc[nz] / sumw[nz]).astype(np.float32)
        return out

    a = aa[valid].astype(np.float64)
    x = x_mm[valid].astype(np.float64) / roi_w * ncols
    sigma_cols = sigma_mm / roi_w * ncols
    half = int(np.ceil(3.0 * sigma_cols))
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    sumw = np.zeros(ncols); acc = np.zeros(ncols)
    i_centre = np.round(x).astype(np.int64)
    for off in offsets:
        cols = i_centre + off
        m = (cols >= 0) & (cols < ncols)
        if not np.any(m):
            continue
        d = (cols[m] - x[m]).astype(np.float64)
        w = np.exp(-(d ** 2) / (2.0 * sigma_cols ** 2))
        np.add.at(sumw, cols[m], w)
        np.add.at(acc,  cols[m], w * a[m])
    nz = sumw > 1e-12
    out[nz] = (acc[nz] / sumw[nz]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def noise_row_diff_sigma(img: np.ndarray) -> float:
    """sigma of one row, recovered from sigma of (row[i+1] - row[i]) in a
    central block presumed uniform (deep cell interior).
    """
    r0, r1 = int(0.40 * img.shape[0]), int(0.60 * img.shape[0])
    c0, c1 = int(0.40 * img.shape[1]), int(0.60 * img.shape[1])
    diff = np.diff(img[r0:r1, c0:c1], axis=0)
    diff = diff[np.isfinite(diff)]
    if diff.size < 50:
        return float("nan")
    return float(np.nanstd(diff) / np.sqrt(2))


def edge_sharpness(img: np.ndarray) -> float:
    """
    Median magnitude of the lateral gradient (|d/dx|) in regions known to
    contain cell edges. Larger = sharper transitions = better lateral
    resolution. Computed on the band of rows that intersect the cell.

    We pick the 99th-percentile pixels of |gradient| from the central rows —
    these are the cell-edge pixels — and report their median magnitude. That
    isolates resolution at the edges and ignores the interior.
    """
    r0, r1 = int(0.30 * img.shape[0]), int(0.70 * img.shape[0])
    band = img[r0:r1, :]
    grad = np.abs(np.diff(band, axis=1))
    grad = grad[np.isfinite(grad)]
    if grad.size < 50:
        return float("nan")
    thr = np.percentile(grad, 99)
    edge_pix = grad[grad >= thr]
    return float(np.median(edge_pix))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def load_meta(scan_dir: Path) -> dict:
    import json
    matches = list(scan_dir.glob("scan_*_meta.json"))
    if matches:
        return json.loads(matches[0].read_text())
    cfg = json.loads((scan_dir / "session_manifest.json").read_text())["config"]
    return {"roi_w_mm": cfg["roi_w"], "roi_h_mm": cfg["roi_h"],
            "pitch_mm": cfg["pitch"], "speed_mm_s": cfg["speed"],
            "ncols": cfg["cols"], "fs_hz": 20_000_000, "gate_us": [25.0, 50.0]}


def run_sweep(scan_dir: Path, out_dir: Path, sigmas_mm: list[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(scan_dir)
    roi_w = float(meta["roi_w_mm"]); roi_h = float(meta["roi_h_mm"])
    ncols = int(meta["ncols"])

    line_files = sorted((scan_dir / "lines_raw").glob("line_*.npz"))
    nlines = len(line_files)
    print(f"Scan: {scan_dir.name}  speed={meta.get('speed_mm_s','?')} mm/s  "
          f"nlines={nlines} ncols={ncols}")

    # Pre-load all per-line pulse arrays into memory so each sigma re-uses them
    # (only the binning changes; the underlying pulse data is identical).
    print("Loading raw line data...")
    lines = []
    for lf in line_files:
        d = np.load(lf)
        lines.append({
            "amp":  d["amplitude"].astype(np.float32),
            "tof":  d["tof_us"].astype(np.float32),
            "eng":  d["energy"].astype(np.float32),
            "x_mm": d["x_mm"].astype(np.float32),
        })

    feature_keys = ("amp", "tof", "eng")
    feature_labels = {"amp": ("Amplitude", "V"),
                      "tof": ("ToF", "us"),
                      "eng": ("Energy", "V^2*samples")}

    # results[sigma][feature] -> 2D image
    results: dict[float, dict[str, np.ndarray]] = {}
    metrics_rows = [("sigma_mm", "feature", "row_diff_sigma", "edge_grad")]

    for sigma in sigmas_mm:
        print(f"sigma = {sigma:.3f} mm ...")
        imgs = {f: np.full((nlines, ncols), np.nan, dtype=np.float32)
                for f in feature_keys}
        for li, ln in enumerate(lines):
            for fk in feature_keys:
                imgs[fk][li] = m6_gauss_kernel(ln[fk], ln["x_mm"], roi_w, ncols, sigma)
        results[sigma] = imgs
        for fk in feature_keys:
            metrics_rows.append((
                f"{sigma:.3f}", fk,
                f"{noise_row_diff_sigma(imgs[fk]):.5g}",
                f"{edge_sharpness(imgs[fk]):.5g}",
            ))

    # ---- write metrics CSV ----
    with open(out_dir / "sigma_sweep_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(metrics_rows)
    print(f"wrote {out_dir / 'sigma_sweep_metrics.csv'}")

    # ---- noise vs sigma curves (one panel per feature) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=110)
    for ax, fk in zip(axes, feature_keys):
        label, unit = feature_labels[fk]
        noise = [noise_row_diff_sigma(results[s][fk]) for s in sigmas_mm]
        edge  = [edge_sharpness(results[s][fk])     for s in sigmas_mm]
        ax2 = ax.twinx()
        ax.plot(sigmas_mm, noise, "o-", color="C0", label="noise (lower=cleaner)")
        ax2.plot(sigmas_mm, edge, "s--", color="C3", label="edge grad (higher=sharper)")
        ax.set_xlabel("Gaussian sigma (mm)")
        ax.set_ylabel(f"row-diff sigma ({unit})", color="C0")
        ax2.set_ylabel(f"edge grad ({unit}/col)", color="C3")
        ax.set_title(f"{label}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Gaussian-kernel sigma sweep — {scan_dir.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'tradeoff_curves.png'}")

    # ---- visual: amplitude maps across sigma values ----
    # Pick a sparse set so the figure stays readable
    visual_sigmas = [sigmas_mm[i] for i in
                     [0, len(sigmas_mm)//4, len(sigmas_mm)//2,
                      3*len(sigmas_mm)//4, len(sigmas_mm)-1]]
    for fk in feature_keys:
        label, unit = feature_labels[fk]
        all_vals = np.concatenate([results[s][fk][np.isfinite(results[s][fk])].ravel()
                                   for s in visual_sigmas])
        vmin, vmax = np.percentile(all_vals, [3, 97])
        cmap = matplotlib.colormaps.get_cmap("turbo").copy(); cmap.set_bad("white")

        fig, axes = plt.subplots(1, len(visual_sigmas), figsize=(4 * len(visual_sigmas), 5), dpi=110)
        for ax, s in zip(axes, visual_sigmas):
            im = ax.imshow(results[s][fk], cmap=cmap, origin="upper",
                           aspect="equal", extent=[0, roi_w, roi_h, 0],
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"sigma = {s:.2f} mm")
            ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        fig.suptitle(f"{label} ({unit}) — sigma sweep — {scan_dir.name}", fontsize=11)
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.012, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(out_dir / f"maps_{fk}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_dir / f'maps_{fk}.png'}")

    # ---- recommend a sigma ----
    # Heuristic: the smallest sigma at which noise has reached 95% of the
    # asymptotic reduction, but edge sharpness is still >= 80% of the
    # max-sharpness value (sigma = 0).
    print()
    print("Recommended sigma per feature:")
    for fk in feature_keys:
        noise = np.array([noise_row_diff_sigma(results[s][fk]) for s in sigmas_mm])
        edge  = np.array([edge_sharpness(results[s][fk])      for s in sigmas_mm])
        noise_floor = np.nanmin(noise)
        noise_top   = np.nanmax(noise)
        edge_max    = np.nanmax(edge)
        target_noise = noise_top - 0.95 * (noise_top - noise_floor)
        edge_floor   = 0.80 * edge_max
        ok = (noise <= target_noise) & (edge >= edge_floor)
        if np.any(ok):
            rec = float(np.array(sigmas_mm)[ok][0])
        else:
            # No sigma satisfies both — fall back to argmin(noise/edge ratio)
            rec = float(sigmas_mm[int(np.argmin(noise / np.maximum(edge, 1e-9)))])
        print(f"  {fk}: sigma = {rec:.2f} mm")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
    args = ap.parse_args()

    scan = args.scan
    tag = re.sub(r"^cscan_scan_", "", scan.name)
    out = args.out or Path("reports/experiments") / f"sigma_sweep_{tag}"
    run_sweep(scan, out, args.sigmas)


if __name__ == "__main__":
    main()
