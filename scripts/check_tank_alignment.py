"""Compare a verification scan against a reference scan to detect tank
misalignment after physical movement.

Reports three quantities:
  1. Lateral shift (sub-pixel phase correlation in mm)
  2. Tilt change (amplitude / ToF plane-fit slope in V/mm and ns/mm)
  3. Working-distance drift (ROI-mean ToF shift in µs)

Then prints a verdict:
  OK     — continue without correction (lateral <0.5 mm, dToF <100 ns)
  MINOR  — apply registration shift to all future scans
  MAJOR  — re-align tank or re-do noise floor & ROI

Usage:
    python scripts/check_tank_alignment.py --new <scan_name_or_id> \
                                            --reference <pre-move scan name or id>
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


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"

ROI_W_MM, ROI_H_MM = 50.02, 39.78
ROI_CX_MM, ROI_CY_MM = 39.59, 36.00


def find_scan(query: str) -> Path | None:
    """Match by full scan name, by timestamp suffix, or by partial substring."""
    suffix = query.replace("scan_", "", 1)
    candidates = list(CSCAN_ROOT.glob(f"cscan_*{suffix}"))
    if not candidates:
        candidates = [p for p in CSCAN_ROOT.iterdir()
                      if p.is_dir() and query in p.name]
    if not candidates:
        candidates = [p for p in CSCAN_ROOT.iterdir()
                      if p.is_dir() and p.name.endswith(suffix)]
    if not candidates:
        return None
    # Prefer the most recent if multiple match
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_scan(scan_dir: Path) -> dict:
    npz = next(scan_dir.glob("scan_*.npz"))
    d = np.load(npz)
    return {
        "amp": np.asarray(d["amplitude"], dtype=np.float64),
        "tof": np.asarray(d["tof"],       dtype=np.float64),
        "eng": np.asarray(d["energy"],    dtype=np.float64),
        "x_mm": np.asarray(d["x_mm"]),
        "y_mm": np.asarray(d["y_mm"]),
        "scan_dir": scan_dir,
    }


def detect_footprint(mean_amp: np.ndarray, rel: float = 0.5) -> np.ndarray:
    finite = mean_amp[np.isfinite(mean_amp)]
    if finite.size == 0:
        return np.zeros_like(mean_amp, dtype=bool)
    thr = rel * float(np.percentile(finite, 50))
    return np.isfinite(mean_amp) & (mean_amp >= thr)


def phase_correlation(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    if ref.shape != mov.shape:
        return float("nan"), float("nan")
    r = np.nan_to_num(ref - np.nanmean(ref))
    m = np.nan_to_num(mov - np.nanmean(mov))
    F1 = np.fft.fft2(r); F2 = np.fft.fft2(m)
    R = F1 * np.conj(F2); R /= np.abs(R) + 1e-12
    corr = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    H, W = corr.shape

    def parab(a, b, c):
        d = a - 2 * b + c
        return 0.5 * (a - c) / d if d != 0 else 0.0
    dy = parab(corr[(py - 1) % H, px], corr[py, px], corr[(py + 1) % H, px])
    dx = parab(corr[py, (px - 1) % W], corr[py, px], corr[py, (px + 1) % W])
    sy = py + dy; sx = px + dx
    if sy > H / 2: sy -= H
    if sx > W / 2: sx -= W
    return float(sy), float(sx)


def fit_plane(arr: np.ndarray, mask: np.ndarray):
    """Return (a_y, b_x, c) such that arr ≈ a_y*y + b_x*x + c inside the mask
    (units: arr-units per pixel)."""
    ys, xs = np.where(mask & np.isfinite(arr))
    if ys.size < 10:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([ys, xs, np.ones_like(ys, dtype=float)])
    z = arr[ys, xs].astype(float)
    coefs, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(coefs[0]), float(coefs[1]), float(coefs[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new",       required=True,
                    help="New (post-move) scan: scan name, scan_id or substring")
    ap.add_argument("--reference", required=True,
                    help="Reference (pre-move) scan: scan name, scan_id or substring")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref_dir = find_scan(args.reference)
    new_dir = find_scan(args.new)
    if ref_dir is None:
        raise SystemExit(f"reference scan not found for '{args.reference}'")
    if new_dir is None:
        raise SystemExit(f"new scan not found for '{args.new}'")
    print(f"reference: {ref_dir.name}")
    print(f"new:       {new_dir.name}")

    ref = load_scan(ref_dir)
    new = load_scan(new_dir)
    if ref["amp"].shape != new["amp"].shape:
        raise SystemExit(f"shape mismatch: {ref['amp'].shape} vs {new['amp'].shape}")

    px_x_mm = float(ref["x_mm"][1] - ref["x_mm"][0])
    px_y_mm = float(ref["y_mm"][1] - ref["y_mm"][0])

    # -- 1. Phase correlation (lateral shift) -----------------------------
    dy_px, dx_px = phase_correlation(ref["amp"], new["amp"])
    dy_mm = dy_px * px_y_mm
    dx_mm = dx_px * px_x_mm
    shift_mag_mm = float(np.hypot(dx_mm, dy_mm))

    # -- 2. Plane-fit tilt change (within reference footprint) ------------
    fp = detect_footprint(ref["amp"])
    a_ref, b_ref, c_ref = fit_plane(ref["amp"], fp)
    a_new, b_new, c_new = fit_plane(new["amp"], fp)
    amp_slope_y_per_mm_ref = a_ref / px_y_mm    # V/mm
    amp_slope_x_per_mm_ref = b_ref / px_x_mm
    amp_slope_y_per_mm_new = a_new / px_y_mm
    amp_slope_x_per_mm_new = b_new / px_x_mm
    d_amp_slope_y = amp_slope_y_per_mm_new - amp_slope_y_per_mm_ref
    d_amp_slope_x = amp_slope_x_per_mm_new - amp_slope_x_per_mm_ref

    at_ref, bt_ref, _ = fit_plane(ref["tof"], fp)
    at_new, bt_new, _ = fit_plane(new["tof"], fp)
    tof_slope_y_per_mm_ref = at_ref / px_y_mm * 1000   # ns/mm
    tof_slope_x_per_mm_ref = bt_ref / px_x_mm * 1000
    tof_slope_y_per_mm_new = at_new / px_y_mm * 1000
    tof_slope_x_per_mm_new = bt_new / px_x_mm * 1000
    d_tof_slope_y = tof_slope_y_per_mm_new - tof_slope_y_per_mm_ref
    d_tof_slope_x = tof_slope_x_per_mm_new - tof_slope_x_per_mm_ref

    # -- 3. Working distance: ROI-mean ToF shift --------------------------
    col_lo = int(round((ROI_CX_MM - ROI_W_MM / 2 - ref["x_mm"][0]) / px_x_mm))
    col_hi = int(round((ROI_CX_MM + ROI_W_MM / 2 - ref["x_mm"][0]) / px_x_mm))
    row_lo = int(round((ROI_CY_MM - ROI_H_MM / 2 - ref["y_mm"][0]) / px_y_mm))
    row_hi = int(round((ROI_CY_MM + ROI_H_MM / 2 - ref["y_mm"][0]) / px_y_mm))
    roi_mask = np.zeros_like(ref["amp"], dtype=bool)
    roi_mask[row_lo:row_hi, col_lo:col_hi] = True

    tof_ref_roi = float(np.nanmean(ref["tof"][roi_mask]))
    tof_new_roi = float(np.nanmean(new["tof"][roi_mask]))
    delta_tof_us = tof_new_roi - tof_ref_roi
    delta_tof_ns = delta_tof_us * 1000

    amp_ref_roi = float(np.nanmean(ref["amp"][roi_mask]))
    amp_new_roi = float(np.nanmean(new["amp"][roi_mask]))
    delta_amp_v = amp_new_roi - amp_ref_roi
    amp_ratio = (amp_new_roi / amp_ref_roi) if amp_ref_roi > 0 else float("nan")

    # -- Verdict ---------------------------------------------------------
    verdict = "OK"
    notes = []
    if shift_mag_mm > 0.5:
        verdict = "MINOR"
        notes.append(f"lateral shift {shift_mag_mm:.2f} mm — apply registration")
    if shift_mag_mm > 3.0:
        verdict = "MAJOR"
        notes.append("lateral shift > 3 mm; ROI may no longer cover the same physical area")
    if abs(delta_tof_ns) > 100:
        verdict = "MINOR" if verdict == "OK" else verdict
        notes.append(f"dToF {delta_tof_ns:+.0f} ns — working-distance drift suspected")
    if abs(delta_tof_ns) > 300:
        verdict = "MAJOR"
        notes.append("dToF > 300 ns — focus likely shifted, redo noise floor")
    if (abs(d_tof_slope_x) > 3 or abs(d_tof_slope_y) > 3):
        verdict = "MINOR" if verdict == "OK" else verdict
        notes.append(f"ToF tilt change ({d_tof_slope_x:+.1f}, {d_tof_slope_y:+.1f}) ns/mm "
                     "— cell surface not parallel to scan plane")
    if (abs(d_tof_slope_x) > 8 or abs(d_tof_slope_y) > 8):
        verdict = "MAJOR"
        notes.append("ToF tilt > 8 ns/mm — physical re-alignment recommended")
    if amp_ratio < 0.7 or amp_ratio > 1.4:
        verdict = "MAJOR" if verdict != "MAJOR" else verdict
        notes.append(f"amplitude scale changed by factor {amp_ratio:.2f} — focus issue")

    print("\n=== ALIGNMENT METRICS ===")
    print(f"  lateral shift     dx = {dx_mm:+.3f} mm,  dy = {dy_mm:+.3f} mm  "
          f"(|shift| = {shift_mag_mm:.3f} mm)")
    print(f"  ROI-mean ToF      ref = {tof_ref_roi:.4f} us,  new = {tof_new_roi:.4f} us  "
          f"(delta = {delta_tof_ns:+.1f} ns)")
    print(f"  ROI-mean amp      ref = {amp_ref_roi:.4f} V,   new = {amp_new_roi:.4f} V  "
          f"(ratio = {amp_ratio:.3f})")
    print(f"  ToF tilt (X, Y) change: ({d_tof_slope_x:+.2f}, {d_tof_slope_y:+.2f}) ns/mm")
    print(f"  amp tilt (X, Y) change: ({d_amp_slope_x:+.4f}, {d_amp_slope_y:+.4f}) V/mm")
    print(f"\n=== VERDICT: {verdict} ===")
    for n in notes:
        print(f"  - {n}")
    if not notes:
        print("  - all metrics within continue-without-correction thresholds")

    # ----- figure -----
    extent = [float(ref["x_mm"][0]), float(ref["x_mm"][-1]),
              float(ref["y_mm"][-1]), float(ref["y_mm"][0])]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=160,
                             gridspec_kw={"wspace": 0.20, "hspace": 0.28,
                                          "top": 0.93, "bottom": 0.06,
                                          "left": 0.06, "right": 0.94})

    def panel(ax, data, cmap, vmin, vmax, title, cbar_lbl):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, origin="upper", aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_lbl)
        rect = mpatches.Rectangle(
            (ROI_CX_MM - ROI_W_MM/2, ROI_CY_MM - ROI_H_MM/2),
            ROI_W_MM, ROI_H_MM,
            fill=False, edgecolor="#16a34a", linewidth=1.4, alpha=0.9)
        ax.add_patch(rect)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("X (mm)", fontsize=9); ax.set_ylabel("Y (mm)", fontsize=9)
        ax.tick_params(labelsize=8)

    # Row 1: amplitude
    amp_v = (float(np.percentile(ref["amp"][np.isfinite(ref["amp"])], 1)),
             float(np.percentile(ref["amp"][np.isfinite(ref["amp"])], 99)))
    panel(axes[0, 0], ref["amp"], "gray", *amp_v,
          f"(a) Reference amplitude\n{ref_dir.name}", "amp (V)")
    panel(axes[0, 1], new["amp"], "gray", *amp_v,
          f"(b) New amplitude\n{new_dir.name}", "amp (V)")
    diff_amp = new["amp"] - ref["amp"]
    sa = float(np.nanstd(diff_amp))
    panel(axes[0, 2], diff_amp, "RdBu_r", -3*sa, 3*sa,
          "(c) Δ amplitude (new − ref)", "Δamp (V)")

    # Row 2: ToF
    tof_v = (float(np.percentile(ref["tof"][np.isfinite(ref["tof"])], 1)),
             float(np.percentile(ref["tof"][np.isfinite(ref["tof"])], 99)))
    panel(axes[1, 0], ref["tof"], "viridis", *tof_v,
          "(d) Reference ToF", "ToF (µs)")
    panel(axes[1, 1], new["tof"], "viridis", *tof_v,
          "(e) New ToF", "ToF (µs)")
    diff_tof = (new["tof"] - ref["tof"]) * 1000
    st = float(np.nanstd(diff_tof))
    panel(axes[1, 2], diff_tof, "RdBu_r", -3*st, 3*st,
          "(f) Δ ToF (new − ref)", "dToF (ns)")

    title_parts = [
        f"Tank alignment check — VERDICT: {verdict}",
        f"lateral shift = ({dx_mm:+.2f}, {dy_mm:+.2f}) mm,  |shift| = {shift_mag_mm:.2f} mm",
        f"ROI-mean dToF = {delta_tof_ns:+.1f} ns,  amp ratio = {amp_ratio:.3f},  "
        f"ToF tilt change = ({d_tof_slope_x:+.2f}, {d_tof_slope_y:+.2f}) ns/mm",
    ]
    fig.suptitle("\n".join(title_parts), fontsize=11, y=0.99)

    # Save outputs
    out_dir = Path(args.out) if args.out else (
        PROJECT / "reports" / "experiments" / f"alignment_check_{new_dir.name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "alignment_check.png", bbox_inches="tight")
    fig.savefig(out_dir / "alignment_check.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "reference":  ref_dir.name,
        "new":        new_dir.name,
        "shift_mm":   {"dx": dx_mm, "dy": dy_mm, "magnitude": shift_mag_mm},
        "tof_us":     {"ref_roi": tof_ref_roi, "new_roi": tof_new_roi,
                       "delta_ns": delta_tof_ns},
        "amp_v":      {"ref_roi": amp_ref_roi, "new_roi": amp_new_roi,
                       "ratio": amp_ratio, "delta": delta_amp_v},
        "tilt_ns_per_mm": {
            "tof_x_change": d_tof_slope_x,
            "tof_y_change": d_tof_slope_y,
        },
        "tilt_v_per_mm": {
            "amp_x_change": d_amp_slope_x,
            "amp_y_change": d_amp_slope_y,
        },
        "verdict": verdict,
        "notes": notes,
    }
    (out_dir / "alignment_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Tank alignment check\n",
        f"- Reference: `{ref_dir.name}`",
        f"- New:       `{new_dir.name}`",
        f"",
        f"## Verdict: **{verdict}**",
        f"",
    ] + [f"- {n}" for n in notes] + [
        "",
        "## Metrics",
        f"| Quantity | Value |",
        f"|---|---:|",
        f"| Lateral shift dx | {dx_mm:+.3f} mm |",
        f"| Lateral shift dy | {dy_mm:+.3f} mm |",
        f"| Shift magnitude  | {shift_mag_mm:.3f} mm |",
        f"| ROI-mean dToF    | {delta_tof_ns:+.1f} ns |",
        f"| ROI-mean amp ratio (new/ref) | {amp_ratio:.3f} |",
        f"| ToF tilt change (X, Y)       | ({d_tof_slope_x:+.2f}, {d_tof_slope_y:+.2f}) ns/mm |",
        f"| Amp tilt change (X, Y)       | ({d_amp_slope_x:+.4f}, {d_amp_slope_y:+.4f}) V/mm |",
        "",
        "## Decision tree",
        "- OK:    continue without correction",
        "- MINOR: apply registration shift to future scans, no physical action",
        "- MAJOR: re-align tank physically or re-do noise floor + ROI",
    ]
    (out_dir / "alignment_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved: {out_dir}")
    return 0 if verdict == "OK" else (1 if verdict == "MINOR" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
