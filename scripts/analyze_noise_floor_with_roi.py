"""Full noise-floor analysis with a user-chosen rectangular ROI.

Pipeline (matches the report plan §4.9):
  0. Registration check across the 6 repeats (FFT phase correlation in the
     fixture region — sub-pixel offsets reported but not corrected by default,
     since they should be near zero with the gauging origin).
  1. Per-pixel statistics: mean and σ across the 6 repeats for amplitude,
     ToF, and energy.
  2. Echo-strength mask (cell footprint).
  3. ROI rectangle (user supplies W × H mm centred on the footprint
     centroid, or explicit X/Y bounds).
  4. Scalar σ — both ROI-aggregated (the working noise floor) and per-pixel
     median (diagnostic).
  5. Drift check — linear fit of ROI-mean vs scan index, separately for the
     cell ROI and the fixture reference region.

Outputs (single timestamped folder):
  figA_baseline_maps.pdf  — 1×3: mean amplitude, ToF, energy + ROI overlay
  figB_noise_map_roi.pdf  — σ_ToF map with ROI rectangle (the key figure)
  figC_drift_strip.pdf    — ROI-mean vs scan index, cell + fixture lines
  roi_mask.npy            — boolean array, the ROI
  roi_bounds.json         — pixel + mm bounds
  registration_offsets.csv
  noise_floor_summary.csv
  REPORT.md               — human-readable summary + caption candidates
"""
from __future__ import annotations

import argparse
import csv
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
from scipy import stats as sstats


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"

EXPECTED = {"roi_w_mm": 80.0, "roi_h_mm": 72.0, "pitch_mm": 0.5,
            "speed_mm_s": 25.0, "accel_mm_s2": 800.0}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
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
            continue   # partial
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
        "roi_w": float(first["x_mm"][-1] - first["x_mm"][0]),
        "roi_h": float(first["y_mm"][-1] - first["y_mm"][0]),
    }


# ---------------------------------------------------------------------------
# Registration: FFT phase correlation
# ---------------------------------------------------------------------------
def phase_correlation(ref: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    """Return (shift_y_px, shift_x_px) of `moving` relative to `ref`, using
    phase-only normalisation. Sub-pixel refinement via 3-pt parabolic fit on
    the correlation surface around the integer-peak."""
    if ref.shape != moving.shape:
        return float("nan"), float("nan")
    ref = np.nan_to_num(ref - np.nanmean(ref))
    moving = np.nan_to_num(moving - np.nanmean(moving))
    F1 = np.fft.fft2(ref)
    F2 = np.fft.fft2(moving)
    R = F1 * np.conj(F2)
    R /= np.abs(R) + 1e-12
    corr = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    # Parabolic sub-pixel refinement
    def parab(a, b, c):
        denom = (a - 2 * b + c)
        return 0.5 * (a - c) / denom if denom != 0 else 0.0

    H, W = corr.shape
    dy = parab(corr[(py - 1) % H, px],
               corr[py, px],
               corr[(py + 1) % H, px])
    dx = parab(corr[py, (px - 1) % W],
               corr[py, px],
               corr[py, (px + 1) % W])
    sy = py + dy; sx = px + dx
    # Wrap to (-H/2, H/2]
    if sy > H / 2: sy -= H
    if sx > W / 2: sx -= W
    return float(sy), float(sx)


# ---------------------------------------------------------------------------
# Detection / ROI
# ---------------------------------------------------------------------------
def detect_footprint(mean_amp: np.ndarray, rel_median: float = 0.5) -> np.ndarray:
    finite = mean_amp[np.isfinite(mean_amp)]
    if finite.size == 0:
        return np.zeros_like(mean_amp, dtype=bool)
    threshold = rel_median * float(np.percentile(finite, 50))
    return np.isfinite(mean_amp) & (mean_amp >= threshold)


def build_roi_mask(shape: tuple[int, int],
                   x_mm: np.ndarray, y_mm: np.ndarray,
                   x0_mm: float, x1_mm: float,
                   y0_mm: float, y1_mm: float) -> tuple[np.ndarray, dict]:
    """Return (mask, bounds_dict). Bounds reported in both pixel and mm units."""
    nrows, ncols = shape
    # Convert mm -> pixel index. x_mm is the column-centre array (length ncols).
    dx_mm = float(x_mm[1] - x_mm[0])
    dy_mm = float(y_mm[1] - y_mm[0])
    col_lo = int(round((x0_mm - float(x_mm[0])) / dx_mm))
    col_hi = int(round((x1_mm - float(x_mm[0])) / dx_mm))
    row_lo = int(round((y0_mm - float(y_mm[0])) / dy_mm))
    row_hi = int(round((y1_mm - float(y_mm[0])) / dy_mm))
    col_lo = max(0, min(col_lo, ncols))
    col_hi = max(0, min(col_hi, ncols))
    row_lo = max(0, min(row_lo, nrows))
    row_hi = max(0, min(row_hi, nrows))
    if col_lo > col_hi: col_lo, col_hi = col_hi, col_lo
    if row_lo > row_hi: row_lo, row_hi = row_hi, row_lo
    mask = np.zeros(shape, dtype=bool)
    mask[row_lo:row_hi, col_lo:col_hi] = True
    bounds = {
        "x0_mm": float(x_mm[col_lo]) if col_lo < ncols else float(x_mm[-1]),
        "x1_mm": float(x_mm[col_hi - 1]) if col_hi >= 1 else float(x_mm[0]),
        "y0_mm": float(y_mm[row_lo]) if row_lo < nrows else float(y_mm[-1]),
        "y1_mm": float(y_mm[row_hi - 1]) if row_hi >= 1 else float(y_mm[0]),
        "col_lo": col_lo, "col_hi": col_hi,
        "row_lo": row_lo, "row_hi": row_hi,
        "n_pixels": int(mask.sum()),
        "w_mm": (col_hi - col_lo) * dx_mm,
        "h_mm": (row_hi - row_lo) * dy_mm,
    }
    return mask, bounds


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def roi_drift(stack: np.ndarray, mask: np.ndarray) -> dict:
    per_scan = np.array(
        [float(np.nanmean(stack[i][mask])) for i in range(stack.shape[0])]
    )
    sigma_roi = float(np.std(per_scan, ddof=1))
    x = np.arange(stack.shape[0], dtype=float)
    res = sstats.linregress(x, per_scan)
    return {
        "per_scan": per_scan,
        "sigma_roi": sigma_roi,
        "drift_slope": float(res.slope),
        "drift_p": float(res.pvalue),
        "drift_intercept": float(res.intercept),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def add_roi_overlay(ax, bounds: dict, label: str = "ROI",
                    color: str = "#16a34a", lw: float = 1.8) -> None:
    rect = mpatches.Rectangle(
        (bounds["x0_mm"], bounds["y0_mm"]),
        bounds["w_mm"], bounds["h_mm"],
        fill=False, edgecolor=color, linewidth=lw, linestyle="-",
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(bounds["x0_mm"] + 1, bounds["y0_mm"] + 1.5,
            label, color="white", fontsize=8, fontweight="bold",
            bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.85),
            va="top", ha="left")


def add_fixture_overlay(ax, fixture_box_mm: dict, color: str = "#a855f7") -> None:
    """Kept name for compatibility but now annotates the excluded cell band."""
    if "w_mm" not in fixture_box_mm:
        return   # nothing to draw for masks without a bounding box
    rect = mpatches.Rectangle(
        (fixture_box_mm["x0_mm"], fixture_box_mm["y0_mm"]),
        fixture_box_mm["w_mm"], fixture_box_mm["h_mm"],
        fill=False, edgecolor=color, linewidth=1.0, linestyle=":",
        alpha=0.5, clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(fixture_box_mm["x0_mm"] + 1,
            fixture_box_mm["y0_mm"] + fixture_box_mm["h_mm"] - 1.5,
            "excluded cell band", color="white", fontsize=7, fontweight="bold",
            bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.85),
            va="bottom", ha="left")


def fig_A_baseline_maps(stk: dict, roi_bounds: dict, fix_bounds: dict,
                        out_path: Path) -> None:
    mean_amp = np.nanmean(stk["amp"], axis=0)
    mean_tof = np.nanmean(stk["tof"], axis=0)
    mean_eng = np.nanmean(stk["eng"], axis=0)
    extent = [float(stk["x_mm"][0]), float(stk["x_mm"][-1]),
              float(stk["y_mm"][-1]), float(stk["y_mm"][0])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), dpi=160,
                             gridspec_kw={"wspace": 0.22, "top": 0.90,
                                          "left": 0.05, "right": 0.97,
                                          "bottom": 0.10})
    panels = [
        (mean_amp, "(a) Mean amplitude", "amplitude (V)", "turbo"),
        (mean_tof, "(b) Mean ToF",       "ToF (µs)",      "viridis"),
        (mean_eng, "(c) Mean energy",    "energy",         "cividis"),
    ]
    for ax, (data, title, cbar_lbl, cmap) in zip(axes, panels):
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
        add_roi_overlay(ax, roi_bounds)
        add_fixture_overlay(ax, fix_bounds)
    fig.suptitle(f"6-scan mean baseline maps  —  "
                 f"ROI = {roi_bounds['w_mm']:.1f} × {roi_bounds['h_mm']:.1f} mm  "
                 f"({roi_bounds['n_pixels']:,} px)",
                 fontsize=12, y=0.99)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_B_noise_map_with_roi(stk: dict, footprint: np.ndarray,
                              roi_bounds: dict, fix_bounds: dict,
                              out_path: Path) -> None:
    sigma_tof = np.nanstd(stk["tof"], axis=0, ddof=1)
    extent = [float(stk["x_mm"][0]), float(stk["x_mm"][-1]),
              float(stk["y_mm"][-1]), float(stk["y_mm"][0])]

    fig = plt.figure(figsize=(13.5, 6.0), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0],
                          wspace=0.22, left=0.06, right=0.96,
                          top=0.92, bottom=0.10)

    # Left: σ_ToF map with ROI overlay
    ax = fig.add_subplot(gs[0, 0])
    sig_in_fp = sigma_tof[footprint & np.isfinite(sigma_tof)]
    vmax = float(np.percentile(sig_in_fp, 99)) if sig_in_fp.size else 1.0
    im = ax.imshow(sigma_tof, cmap="magma", vmin=0, vmax=max(vmax, 1e-9),
                   extent=extent, origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="σ ToF (µs)", fraction=0.046, pad=0.04)
    # Grey out non-footprint
    grey = np.where(footprint, np.nan, 1.0)
    ax.imshow(grey, cmap=mcolors.ListedColormap([[0.55, 0.55, 0.55, 0.85]]),
              extent=extent, origin="upper", aspect="equal",
              interpolation="nearest")
    add_roi_overlay(ax, roi_bounds)
    add_fixture_overlay(ax, fix_bounds)
    ax.set_title("(a) Per-pixel σ ToF inside cell footprint", fontsize=11, loc="left")
    ax.set_xlabel("X (mm)", fontsize=9); ax.set_ylabel("Y (mm)", fontsize=9)
    ax.tick_params(labelsize=8)

    # Right: histogram comparing ROI vs excluded cell band
    ax2 = fig.add_subplot(gs[0, 1])
    roi_mask = np.zeros_like(sigma_tof, dtype=bool)
    roi_mask[roi_bounds["row_lo"]:roi_bounds["row_hi"],
             roi_bounds["col_lo"]:roi_bounds["col_hi"]] = True
    roi_vals = sigma_tof[roi_mask & np.isfinite(sigma_tof)]
    excluded_vals = sigma_tof[footprint & ~roi_mask & np.isfinite(sigma_tof)]
    bins = np.linspace(0, float(np.percentile(excluded_vals, 99))
                        if excluded_vals.size else 1.0, 70)
    ax2.hist(excluded_vals, bins=bins, color="#a855f7", alpha=0.55,
             label=(f"excluded cell band (n = {excluded_vals.size:,})\n"
                    f"med = {np.median(excluded_vals)*1000:.1f} ns"))
    ax2.hist(roi_vals, bins=bins, color="#16a34a", alpha=0.75,
             label=(f"chosen ROI (n = {roi_vals.size:,})\n"
                    f"med = {np.median(roi_vals)*1000:.1f} ns"))
    ax2.axvline(np.median(excluded_vals), color="#a855f7", linewidth=1.4, linestyle="--")
    ax2.axvline(np.median(roi_vals), color="#16a34a", linewidth=1.6, linestyle="--")
    ax2.set_xlabel("σ ToF (µs)", fontsize=9)
    ax2.set_ylabel("pixel count", fontsize=9)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_title("(b) σ ToF distribution: ROI vs excluded cell band",
                  fontsize=11, loc="left")
    ax2.tick_params(labelsize=8)
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.4)

    fig.suptitle(f"σ ToF noise map and the selected ROI "
                 f"({roi_bounds['w_mm']:.1f}×{roi_bounds['h_mm']:.1f} mm)",
                 fontsize=12, y=0.98)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_C_drift_strip(roi_drift_dict: dict, fix_drift_dict: dict,
                       out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=160,
                             gridspec_kw={"wspace": 0.28, "top": 0.86,
                                          "bottom": 0.14, "left": 0.06,
                                          "right": 0.97})
    features = [
        ("amp", "amplitude", "V",  axes[0]),
        ("tof", "ToF",       "µs", axes[1]),
        ("eng", "energy",    "",   axes[2]),
    ]
    for key, name, units, ax in features:
        roi_d = roi_drift_dict[key]
        fix_d = fix_drift_dict[key]
        roi_dev = roi_d["per_scan"] - roi_d["per_scan"].mean()
        fix_dev = fix_d["per_scan"] - fix_d["per_scan"].mean()
        x = np.arange(roi_dev.size)

        ax.axhspan(-roi_d["sigma_roi"], roi_d["sigma_roi"],
                   color="#16a34a", alpha=0.12, linewidth=0)

        ax.plot(x, roi_dev, "o-", color="#16a34a", linewidth=1.7, markersize=7,
                label=f"chosen ROI  σ = {roi_d['sigma_roi']:.3g}{(' '+units) if units else ''}")
        ax.plot(x, fix_dev, "s--", color="#a855f7", linewidth=1.3, markersize=5,
                alpha=0.85, markerfacecolor="white",
                label=f"excluded cell band  σ = {fix_d['sigma_roi']:.3g}{(' '+units) if units else ''}")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.6)
        ax.set_title(f"{name}  —  drift slope (ROI) = {roi_d['drift_slope']:+.3g}/scan, "
                     f"p = {roi_d['drift_p']:.2f}",
                     fontsize=10)
        ax.set_xlabel("scan index")
        ax.set_ylabel(f"deviation from batch mean ({units})" if units
                      else "deviation from batch mean")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(labelsize=8)

    fig.suptitle("Drift / repeatability strip — ROI-aggregated feature deviation across 6 scans",
                 fontsize=12, y=0.99)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--roi-cx-mm", type=float, default=None,
                    help="ROI centre X (mm). Defaults to footprint centroid.")
    ap.add_argument("--roi-cy-mm", type=float, default=None,
                    help="ROI centre Y (mm). Defaults to footprint centroid.")
    ap.add_argument("--roi-w-mm", type=float, required=True,
                    help="ROI width (mm)")
    ap.add_argument("--roi-h-mm", type=float, required=True,
                    help="ROI height (mm)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    runs = find_runs_by_prefix(args.prefix)
    if not runs:
        raise SystemExit(f"no runs match '{args.prefix}'")
    print(f"loaded {len(runs)} runs")
    stk = load_stack(runs)
    nrows, ncols = stk["amp"].shape[1:]
    extent_mm = (float(stk["x_mm"][0]), float(stk["x_mm"][-1]),
                 float(stk["y_mm"][0]), float(stk["y_mm"][-1]))

    # ---- Step 0: registration ----
    mean_amp = np.nanmean(stk["amp"], axis=0)
    footprint = detect_footprint(mean_amp)
    fixture_global = ~footprint
    # Erode the global fixture mask slightly: drop pixels touching the image
    # boundary (artefact-prone) and any pixel within 2 px of the footprint.
    from scipy import ndimage
    fixture_clean = fixture_global & ~ndimage.binary_dilation(footprint, iterations=2)
    fixture_clean[:2, :] = False; fixture_clean[-2:, :] = False
    fixture_clean[:, :2] = False; fixture_clean[:, -2:] = False
    ref = stk["amp"][0]
    offsets = []
    for i in range(1, stk["amp"].shape[0]):
        moving = stk["amp"][i]
        sy, sx = phase_correlation(ref * fixture_clean, moving * fixture_clean)
        offsets.append((i, sy, sx))
        print(f"  scan {i} vs scan 0: dy = {sy:+.3f} px, dx = {sx:+.3f} px")
    max_abs = max((max(abs(sy), abs(sx)) for _, sy, sx in offsets), default=0.0)
    if max_abs > 0.5:
        print(f"WARNING: max sub-pixel offset {max_abs:.3f} > 0.5 px — registration drift may inflate noise")

    # ---- ROI ----
    cx = args.roi_cx_mm
    cy = args.roi_cy_mm
    if cx is None or cy is None:
        ys, xs = np.where(footprint)
        dx_mm = float(stk["x_mm"][1] - stk["x_mm"][0])
        dy_mm = float(stk["y_mm"][1] - stk["y_mm"][0])
        cx = cx if cx is not None else float(xs.mean() * dx_mm + stk["x_mm"][0])
        cy = cy if cy is not None else float(ys.mean() * dy_mm + stk["y_mm"][0])
    x0 = cx - args.roi_w_mm / 2; x1 = cx + args.roi_w_mm / 2
    y0 = cy - args.roi_h_mm / 2; y1 = cy + args.roi_h_mm / 2
    roi_mask, roi_bounds = build_roi_mask((nrows, ncols), stk["x_mm"], stk["y_mm"],
                                          x0, x1, y0, y1)
    print(f"ROI: {roi_bounds['w_mm']:.2f} x {roi_bounds['h_mm']:.2f} mm, "
          f"{roi_bounds['n_pixels']:,} px, X=[{roi_bounds['x0_mm']:.2f}, {roi_bounds['x1_mm']:.2f}], "
          f"Y=[{roi_bounds['y0_mm']:.2f}, {roi_bounds['y1_mm']:.2f}]")

    # ---- Reference region: the noisy cell band — pixels inside the cell
    # footprint but OUTSIDE the chosen ROI. This is the "rest of the cell" —
    # exactly the area the ROI is meant to exclude. It carries the same
    # nominal SoC signal as the ROI but suffers from the high-σ edge and
    # streak regions.
    cell_outside_mask = footprint & ~roi_mask
    cell_outside_bounds = {
        "n_pixels": int(cell_outside_mask.sum()),
        "description": "footprint AND NOT ROI — the noisy edge/streak band",
    }
    # For overlay we don't have a single rectangle, so use the footprint bbox
    fp_ys, fp_xs = np.where(cell_outside_mask)
    if fp_ys.size > 0:
        cell_outside_bounds.update({
            "x0_mm": float(stk["x_mm"][int(fp_xs.min())]),
            "x1_mm": float(stk["x_mm"][int(fp_xs.max())]),
            "y0_mm": float(stk["y_mm"][int(fp_ys.min())]),
            "y1_mm": float(stk["y_mm"][int(fp_ys.max())]),
        })
        cell_outside_bounds["w_mm"] = cell_outside_bounds["x1_mm"] - cell_outside_bounds["x0_mm"]
        cell_outside_bounds["h_mm"] = cell_outside_bounds["y1_mm"] - cell_outside_bounds["y0_mm"]
    fix_mask = cell_outside_mask
    fix_bounds = cell_outside_bounds
    print(f"cell-outside-ROI ref: {fix_bounds['n_pixels']:,} px "
          f"(inside footprint, outside chosen ROI)")

    # ---- Step 4-5: ROI σ + drift, for both cell ROI and fixture ----
    roi_drifts  = {k: roi_drift(stk[k], roi_mask) for k in ("amp", "tof", "eng")}
    fix_drifts  = {k: roi_drift(stk[k], fix_mask) for k in ("amp", "tof", "eng")}

    # ---- Output dir ----
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args.label}" if args.label else ""
    out_dir = PROJECT / "reports" / "experiments" / f"noise_floor_analysis_{ts}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figures ----
    fig_A_baseline_maps(stk, roi_bounds, fix_bounds, out_dir / "figA_baseline_maps")
    fig_B_noise_map_with_roi(stk, footprint, roi_bounds, fix_bounds,
                              out_dir / "figB_noise_map_roi")
    fig_C_drift_strip(roi_drifts, fix_drifts, out_dir / "figC_drift_strip")

    # ---- Save arrays + JSON + CSVs ----
    np.save(out_dir / "roi_mask.npy", roi_mask)
    np.save(out_dir / "fixture_mask.npy", fix_mask)
    np.save(out_dir / "footprint_mask.npy", footprint)
    (out_dir / "roi_bounds.json").write_text(json.dumps({
        "cell_roi": roi_bounds,
        "fixture_ref": fix_bounds,
        "footprint_n_pixels": int(footprint.sum()),
    }, indent=2))
    with (out_dir / "registration_offsets.csv").open("w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scan_index_vs_0", "dy_px", "dx_px",
                    "dy_mm", "dx_mm"])
        dx_mm = float(stk["x_mm"][1] - stk["x_mm"][0])
        dy_mm = float(stk["y_mm"][1] - stk["y_mm"][0])
        for i, sy, sx in offsets:
            w.writerow([i, f"{sy:.4f}", f"{sx:.4f}",
                        f"{sy * dy_mm:.4f}", f"{sx * dx_mm:.4f}"])

    # Summary CSV
    with (out_dir / "noise_floor_summary.csv").open("w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "feature", "sigma_roi", "drift_slope_per_scan",
                    "drift_p_value", "per_pixel_sigma_median"])
        for key, name in [("amp", "amplitude"), ("tof", "ToF"), ("eng", "energy")]:
            sigma_map = np.nanstd(stk[key], axis=0, ddof=1)
            ppm_roi = float(np.nanmedian(sigma_map[roi_mask]))
            ppm_fix = float(np.nanmedian(sigma_map[fix_mask]))
            w.writerow(["cell_roi", name,
                        roi_drifts[key]["sigma_roi"],
                        roi_drifts[key]["drift_slope"],
                        roi_drifts[key]["drift_p"], ppm_roi])
            w.writerow(["fixture_ref", name,
                        fix_drifts[key]["sigma_roi"],
                        fix_drifts[key]["drift_slope"],
                        fix_drifts[key]["drift_p"], ppm_fix])

    # REPORT.md
    md = [
        "# Noise-floor analysis (chosen ROI)\n",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Batch prefix: `{args.prefix}` ({len(runs)} scans)",
        f"",
        f"## Registration",
        f"Max sub-pixel offset across scans 1-5 vs scan 0: **{max_abs:.3f} px** "
        f"({'OK' if max_abs <= 0.5 else 'WARNING'}).",
        f"Per-scan offsets: see `registration_offsets.csv`.",
        f"",
        f"## ROI",
        f"- Cell ROI: **{roi_bounds['w_mm']:.2f} x {roi_bounds['h_mm']:.2f} mm**, "
        f"{roi_bounds['n_pixels']:,} pixels",
        f"  X ∈ [{roi_bounds['x0_mm']:.2f}, {roi_bounds['x1_mm']:.2f}] mm",
        f"  Y ∈ [{roi_bounds['y0_mm']:.2f}, {roi_bounds['y1_mm']:.2f}] mm",
        f"- Excluded cell band (cell footprint minus ROI): "
        f"{fix_bounds['n_pixels']:,} pixels",
        f"",
        f"## Scalar noise floor (ROI-aggregated SD across 6 scans)",
        f"",
        f"Each ROI's mean is computed per scan, then SD taken across the 6 ROI-mean "
        f"values. Comparing the chosen ROI against the *excluded cell band* "
        f"(rest of the cell footprint) shows how much noise the ROI selection "
        f"removed — both regions carry the same nominal SoC signal.",
        f"",
        f"| Feature | Chosen ROI σ_ROI | Excluded band σ_ROI | ratio |",
        f"|---|---:|---:|---:|",
    ]
    for key, name in [("amp", "amplitude (V)"),
                      ("tof", "ToF (µs)"),
                      ("eng", "energy")]:
        sr = roi_drifts[key]["sigma_roi"]
        sf = fix_drifts[key]["sigma_roi"]
        ratio = sr / sf if sf > 0 else float("nan")
        md.append(f"| {name} | **{sr:.4g}** | {sf:.4g} | {ratio:.2f} |")
    md += [
        f"",
        f"## Drift check (linear fit of ROI-mean vs scan index)",
        f"",
        f"| Feature | Region | slope/scan | p-value | verdict |",
        f"|---|---|---:|---:|---|",
    ]
    for key, name in [("amp", "amplitude"),
                      ("tof", "ToF"),
                      ("eng", "energy")]:
        for reg_name, dd in [("chosen ROI", roi_drifts[key]),
                              ("excluded cell band", fix_drifts[key])]:
            verdict = "drift" if dd["drift_p"] < 0.05 else "no drift"
            md.append(f"| {name} | {reg_name} | {dd['drift_slope']:+.3g} | "
                      f"{dd['drift_p']:.3f} | {verdict} |")
    md += [
        f"",
        f"## Caption candidates",
        f"",
        f"**Fig A (baseline maps):** \"6-scan mean amplitude, ToF and energy maps "
        f"at the chosen working distance. The selected ROI "
        f"({roi_bounds['w_mm']:.1f} × {roi_bounds['h_mm']:.1f} mm, solid green) "
        f"is overlaid on each panel.\"",
        f"",
        f"**Fig B (noise map):** \"Per-pixel σ ToF inside the cell footprint with the "
        f"chosen ROI overlaid. Median σ inside the ROI is "
        f"{1000*np.nanmedian(np.nanstd(stk['tof'], axis=0, ddof=1)[roi_mask]):.1f} ns vs "
        f"{1000*np.nanmedian(np.nanstd(stk['tof'], axis=0, ddof=1)[footprint]):.1f} ns across the full footprint.\"",
        f"",
        f"**Fig C (drift strip):** \"ROI-aggregated feature deviation across the 6 "
        f"repeat scans. Shaded band is ±σ_ROI for the chosen ROI. The dashed "
        f"trace is the same statistic computed for the *excluded cell band* "
        f"(footprint pixels outside the ROI) — both regions carry the same "
        f"nominal SoC signal, so the σ ratio quantifies how much noise the ROI "
        f"selection removes.\"",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n=== saved to: {out_dir} ===")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
