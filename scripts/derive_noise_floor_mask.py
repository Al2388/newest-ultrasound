"""Derive a low-noise retained-pixel mask from a batch of repeat C-scans.

For each candidate feature (amplitude, ToF, energy), the per-pixel standard
deviation across the N repeats becomes the noise floor σ. A pixel survives
into the final analysis mask when (a) it sits inside the cell footprint and
(b) every feature's σ at that pixel is below a threshold derived from the
σ distribution itself.

The fixture region outside the footprint is saved as a separate reference
mask — it does not respond to SoC, so a drift in that band signals coupling
or alignment change rather than cell behaviour.

Defaults follow what we agreed for the first pass:
  threshold method = percentile (keep the lowest 70% of σ per feature)
  combination      = strict AND across all features

Usage
-----
  # Auto-find the most recent batch matching a name pattern
  python scripts/derive_noise_floor_mask.py --prefix noisefloor_v3.238

  # Or pass an explicit list of scan_*.npz paths
  python scripts/derive_noise_floor_mask.py --npz path1 path2 ... pathN
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
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"


# ---------------------------------------------------------------------------
# Scan discovery
# ---------------------------------------------------------------------------
def find_runs_by_prefix(prefix: str) -> list[Path]:
    """Return scan_*.npz paths for all completed C-scan runs whose folder
    name starts with `cscan_<prefix>`. If multiple batches share the prefix,
    keep only the most recent batch (identified by the timestamp embedded in
    the folder names of run_cscan_noise_floor_batch.py)."""
    candidates = []
    for d in CSCAN_ROOT.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(f"cscan_{prefix}"):
            continue
        npz = next(d.glob("scan_*.npz"), None)
        if npz is None:
            continue
        candidates.append(d)

    # Extract the batch timestamp — the substring just before "_rNN_" in the folder name.
    # batch script names runs as: cscan_<prefix>_<batch_ts>_r<NN>_<run_ts>
    batch_tag = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_r\d{2}_")
    by_batch: dict[str, list[Path]] = {}
    for d in candidates:
        m = batch_tag.search(d.name)
        if m:
            by_batch.setdefault(m.group(1), []).append(d)
    if not by_batch:
        return sorted(candidates)
    latest_batch_ts = max(by_batch.keys())
    return sorted(by_batch[latest_batch_ts])


def load_features(npz_path: Path) -> dict:
    d = np.load(npz_path)
    return {
        "amp": np.asarray(d["amplitude"], dtype=np.float64),
        "tof": np.asarray(d["tof"], dtype=np.float64),
        "eng": np.asarray(d["energy"], dtype=np.float64),
        "x_mm": np.asarray(d["x_mm"]),
        "y_mm": np.asarray(d["y_mm"]),
    }


# ---------------------------------------------------------------------------
# Footprint detection
# ---------------------------------------------------------------------------
def detect_cell_footprint(mean_amp: np.ndarray,
                          rel_threshold: float = 0.30) -> np.ndarray:
    """Coarse cell-footprint mask from mean amplitude.

    A pixel is considered "inside the cell" if its mean amplitude exceeds
    `rel_threshold` times the robust max (95th-percentile) of the map.
    Returns boolean array of the same shape.
    """
    finite = mean_amp[np.isfinite(mean_amp)]
    if finite.size == 0:
        return np.zeros_like(mean_amp, dtype=bool)
    robust_max = float(np.percentile(finite, 95))
    return np.isfinite(mean_amp) & (mean_amp >= rel_threshold * robust_max)


# ---------------------------------------------------------------------------
# Per-feature pass-mask
# ---------------------------------------------------------------------------
def per_feature_pass(sigma: np.ndarray,
                     footprint: np.ndarray,
                     method: str,
                     percentile_keep: float,
                     mad_k: float,
                     abs_threshold: float | None) -> tuple[np.ndarray, float]:
    """Return (pass_mask, threshold_used). threshold is in the units of sigma."""
    in_fp = footprint & np.isfinite(sigma)
    if not in_fp.any():
        return np.zeros_like(sigma, dtype=bool), float("nan")

    if method == "abs":
        if abs_threshold is None:
            raise ValueError("abs method requires --abs-threshold-* values")
        thresh = float(abs_threshold)
    elif method == "percentile":
        thresh = float(np.percentile(sigma[in_fp], percentile_keep))
    elif method == "mad":
        med = float(np.median(sigma[in_fp]))
        mad = float(np.median(np.abs(sigma[in_fp] - med)))
        thresh = med + mad_k * mad
    else:
        raise ValueError(f"unknown method: {method}")

    return (sigma <= thresh), thresh


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def imshow_with_overlay(ax, base, cmap, vmin, vmax, extent, title, ylabel=None):
    im = ax.imshow(base, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper", aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (mm)", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def plot_sigma_maps(mean_amp, sigma_amp, sigma_tof, sigma_eng, extent, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=160)
    imshow_with_overlay(axes[0], mean_amp, "turbo",
                        np.percentile(mean_amp[np.isfinite(mean_amp)], 1),
                        np.percentile(mean_amp[np.isfinite(mean_amp)], 99),
                        extent, "mean amplitude (V)", ylabel="Y (mm)")
    for ax, sigma, name, cmap in [
        (axes[1], sigma_amp, "σ amplitude (V)", "magma"),
        (axes[2], sigma_tof, "σ ToF (µs)", "magma"),
        (axes[3], sigma_eng, "σ energy", "magma"),
    ]:
        finite = sigma[np.isfinite(sigma)]
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
        im = imshow_with_overlay(ax, sigma, cmap, 0, max(vmax, 1e-9),
                                 extent, name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sigma_histograms(sigma_amp, sigma_tof, sigma_eng,
                          footprint, thresh_amp, thresh_tof, thresh_eng,
                          out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=160)
    for ax, sigma, name, thresh in [
        (axes[0], sigma_amp, "σ amplitude (V)", thresh_amp),
        (axes[1], sigma_tof, "σ ToF (µs)", thresh_tof),
        (axes[2], sigma_eng, "σ energy", thresh_eng),
    ]:
        s_in = sigma[footprint & np.isfinite(sigma)]
        if s_in.size == 0:
            ax.set_title(f"{name}: no footprint pixels")
            continue
        ax.hist(s_in, bins=80, color="tab:blue", alpha=0.7,
                edgecolor="white", linewidth=0.3)
        ax.axvline(thresh, color="tab:red", linewidth=1.5,
                   linestyle="--", label=f"threshold = {thresh:.3g}")
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel("pixels", fontsize=9)
        ax.set_title(f"footprint-only σ distribution\n"
                     f"median={np.median(s_in):.3g}, mean={np.mean(s_in):.3g}",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mask_overlay(mean_amp, mask, extent, out_path, title_extra=""):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    finite = mean_amp[np.isfinite(mean_amp)]
    vmin, vmax = (float(np.percentile(finite, 1)),
                  float(np.percentile(finite, 99))) if finite.size else (0, 1)
    ax.imshow(mean_amp, cmap="turbo", vmin=vmin, vmax=vmax,
              extent=extent, origin="upper", aspect="equal")
    # Overlay the rejected pixels with a translucent grey
    overlay = np.where(mask, np.nan, 1.0)
    cmap_overlay = mcolors.ListedColormap([[0.15, 0.15, 0.15, 0.55]])
    ax.imshow(overlay, cmap=cmap_overlay,
              extent=extent, origin="upper", aspect="equal", interpolation="nearest")
    n_in = int(mask.sum())
    n_total = int(np.isfinite(mean_amp).sum())
    pct = 100.0 * n_in / max(n_total, 1)
    ax.set_title(f"Final mask{title_extra}: {n_in:,} of {n_total:,} pixels "
                 f"({pct:.1f}% of footprint candidate)", fontsize=11)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None,
                    help="auto-find runs by folder prefix, latest batch only")
    ap.add_argument("--npz", nargs="+", default=None,
                    help="explicit list of scan_*.npz paths (overrides --prefix)")
    ap.add_argument("--method", choices=["percentile", "abs", "mad"],
                    default="percentile",
                    help="threshold method for σ per feature")
    ap.add_argument("--percentile-keep", type=float, default=70.0,
                    help="keep pixels whose σ is below this percentile of σ "
                         "within the footprint (default: 70)")
    ap.add_argument("--mad-k", type=float, default=1.5)
    ap.add_argument("--abs-amp", type=float, default=None,
                    help="absolute σ threshold for amplitude (V)")
    ap.add_argument("--abs-tof", type=float, default=None,
                    help="absolute σ threshold for ToF (µs)")
    ap.add_argument("--abs-eng", type=float, default=None,
                    help="absolute σ threshold for energy")
    ap.add_argument("--footprint-rel", type=float, default=0.30,
                    help="cell footprint cutoff = this × p95 of mean amplitude")
    ap.add_argument("--out", default=None,
                    help="output directory (default: inside the batch's reports dir)")
    args = ap.parse_args()

    # --- locate run NPZ files ---
    if args.npz:
        scan_npzs = [Path(p) for p in args.npz]
    elif args.prefix:
        run_dirs = find_runs_by_prefix(args.prefix)
        scan_npzs = [next(d.glob("scan_*.npz")) for d in run_dirs]
    else:
        raise SystemExit("provide --prefix or --npz")

    if not scan_npzs:
        raise SystemExit("no matching runs found")
    print(f"loading {len(scan_npzs)} runs:")
    for p in scan_npzs:
        print(f"  {p.parent.name}")

    # --- load and stack ---
    first = load_features(scan_npzs[0])
    nrows, ncols = first["amp"].shape
    amp = np.empty((len(scan_npzs), nrows, ncols), dtype=np.float64)
    tof = np.empty_like(amp)
    eng = np.empty_like(amp)
    amp[0] = first["amp"]; tof[0] = first["tof"]; eng[0] = first["eng"]
    for i, p in enumerate(scan_npzs[1:], start=1):
        d = load_features(p)
        if d["amp"].shape != (nrows, ncols):
            raise SystemExit(f"shape mismatch at {p}")
        amp[i] = d["amp"]; tof[i] = d["tof"]; eng[i] = d["eng"]
    x_mm = first["x_mm"]; y_mm = first["y_mm"]
    extent = [float(x_mm[0]), float(x_mm[-1]), float(y_mm[-1]), float(y_mm[0])]

    # --- per-pixel statistics ---
    mean_amp = np.nanmean(amp, axis=0)
    mean_tof = np.nanmean(tof, axis=0)
    mean_eng = np.nanmean(eng, axis=0)
    sigma_amp = np.nanstd(amp, axis=0, ddof=1)
    sigma_tof = np.nanstd(tof, axis=0, ddof=1)
    sigma_eng = np.nanstd(eng, axis=0, ddof=1)

    # --- cell footprint ---
    footprint = detect_cell_footprint(mean_amp, args.footprint_rel)
    fixture = (~footprint) & np.isfinite(mean_amp)

    # --- per-feature pass ---
    pass_amp, thr_amp = per_feature_pass(sigma_amp, footprint, args.method,
                                          args.percentile_keep, args.mad_k,
                                          args.abs_amp)
    pass_tof, thr_tof = per_feature_pass(sigma_tof, footprint, args.method,
                                          args.percentile_keep, args.mad_k,
                                          args.abs_tof)
    pass_eng, thr_eng = per_feature_pass(sigma_eng, footprint, args.method,
                                          args.percentile_keep, args.mad_k,
                                          args.abs_eng)

    # --- final mask ---
    mask = footprint & pass_amp & pass_tof & pass_eng

    # --- ROI-aggregated scalar noise floor + monotonic-drift check ---
    # For each scan i, compute mean of feature inside the mask. The SD of
    # those n values is the working noise floor: it captures the scan-to-scan
    # correlated shifts that move the whole ROI together (coupling, alignment,
    # thermal drift), which is the component that actually limits inversion
    # precision per plan §4.9.
    def _roi_aggregate(stack: np.ndarray) -> dict:
        # stack shape [n_scans, nrows, ncols]
        per_scan_mean = np.array(
            [float(np.nanmean(stack[i][mask])) for i in range(stack.shape[0])]
        )
        sigma_roi = float(np.std(per_scan_mean, ddof=1))
        # Linear fit of ROI-mean vs scan index → drift slope (per scan)
        x = np.arange(stack.shape[0], dtype=float)
        if np.all(np.isfinite(per_scan_mean)):
            res = sstats.linregress(x, per_scan_mean)
            slope, p_value, r_value = float(res.slope), float(res.pvalue), float(res.rvalue)
        else:
            slope, p_value, r_value = float("nan"), float("nan"), float("nan")
        return {
            "per_scan_mean": per_scan_mean.tolist(),
            "sigma_roi": sigma_roi,
            "drift_slope": slope,
            "drift_p": p_value,
            "drift_r": r_value,
        }

    roi_amp = _roi_aggregate(amp)
    roi_tof = _roi_aggregate(tof)
    roi_eng = _roi_aggregate(eng)

    # Reference mask: stable fixture pixels (low σ outside footprint).
    # Use the amp threshold for consistency.
    ref_pass = (sigma_amp <= thr_amp) & (sigma_tof <= thr_tof) & (sigma_eng <= thr_eng)
    reference_mask = fixture & ref_pass

    # --- output directory ---
    if args.out:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix_tag = args.prefix or "manual"
        out_dir = PROJECT / "reports" / "experiments" / f"mask_{prefix_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save arrays ---
    np.save(out_dir / "mean_amp.npy", mean_amp.astype(np.float32))
    np.save(out_dir / "mean_tof.npy", mean_tof.astype(np.float32))
    np.save(out_dir / "mean_eng.npy", mean_eng.astype(np.float32))
    np.save(out_dir / "sigma_amp.npy", sigma_amp.astype(np.float32))
    np.save(out_dir / "sigma_tof.npy", sigma_tof.astype(np.float32))
    np.save(out_dir / "sigma_eng.npy", sigma_eng.astype(np.float32))
    np.save(out_dir / "footprint_mask.npy", footprint)
    np.save(out_dir / "mask.npy", mask)
    np.save(out_dir / "reference_mask.npy", reference_mask)

    # --- plots ---
    plot_sigma_maps(mean_amp, sigma_amp, sigma_tof, sigma_eng, extent,
                    out_dir / "sigma_maps.png")
    plot_sigma_histograms(sigma_amp, sigma_tof, sigma_eng, footprint,
                          thr_amp, thr_tof, thr_eng,
                          out_dir / "sigma_histograms.png")
    plot_mask_overlay(mean_amp, mask, extent, out_dir / "mask_overlay.png")
    plot_mask_overlay(mean_amp, reference_mask, extent,
                      out_dir / "reference_overlay.png",
                      title_extra=" (fixture reference)")
    plot_mask_overlay(mean_amp, footprint, extent,
                      out_dir / "footprint_overlay.png",
                      title_extra=" (cell footprint candidate)")

    # --- drift plot: ROI-mean of each feature vs scan index ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=160)
    for ax, name, units, info in [
        (axes[0], "amplitude", "V",  roi_amp),
        (axes[1], "ToF",       "us", roi_tof),
        (axes[2], "energy",    "",   roi_eng),
    ]:
        y = np.array(info["per_scan_mean"], dtype=float)
        x = np.arange(y.size)
        ax.plot(x, y, "o-", color="tab:blue", linewidth=1.4, markersize=6)
        if np.isfinite(info["drift_slope"]):
            ax.plot(x, info["drift_slope"] * x + (y.mean() - info["drift_slope"] * x.mean()),
                    "--", color="tab:red", linewidth=0.9, alpha=0.7,
                    label=f"slope={info['drift_slope']:.3g}/scan, p={info['drift_p']:.2g}")
            ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("scan index")
        ax.set_ylabel(f"ROI-mean {name} ({units})" if units else f"ROI-mean {name}")
        ax.set_title(f"{name}: σ_ROI = {info['sigma_roi']:.4g}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.suptitle("ROI-aggregated feature vs scan index — monotonic-drift check",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "roi_drift_check.png", bbox_inches="tight")
    plt.close(fig)

    # --- markdown report ---
    n_total = int(np.isfinite(mean_amp).sum())
    n_footprint = int(footprint.sum())
    n_mask = int(mask.sum())
    n_ref = int(reference_mask.sum())

    def _roi_stats(sigma: np.ndarray, label: str) -> str:
        in_mask = mask & np.isfinite(sigma)
        if not in_mask.any():
            return f"  - {label}: no mask pixels"
        vals = sigma[in_mask]
        return (f"  - {label}: median = {np.median(vals):.4g}, "
                f"p95 = {np.percentile(vals, 95):.4g}, "
                f"max = {np.max(vals):.4g}")

    def _drift_verdict(info: dict, sigma_roi: float) -> str:
        if not np.isfinite(info["drift_slope"]):
            return "n/a"
        if info["drift_p"] < 0.05:
            return f"SIGNIFICANT drift (p={info['drift_p']:.3f})"
        return f"no significant drift (p={info['drift_p']:.2f})"

    report = [
        f"# Noise-floor mask\n",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runs used (n = {len(scan_npzs)}):",
        *[f"  - `{p.parent.name}`" for p in scan_npzs],
        f"",
        f"## Threshold settings",
        f"  - method: **{args.method}**",
        f"  - percentile-keep: {args.percentile_keep}",
        f"  - mad-k: {args.mad_k}",
        f"  - footprint cutoff: mean_amp ≥ {args.footprint_rel:.2f} × p95(mean_amp)",
        f"",
        f"## Derived thresholds (units of σ)",
        f"  - σ_amp ≤ **{thr_amp:.4g} V**",
        f"  - σ_tof ≤ **{thr_tof:.4g} µs**",
        f"  - σ_eng ≤ **{thr_eng:.4g}**",
        f"",
        f"## Pixel counts",
        f"  - total finite pixels: {n_total:,}",
        f"  - footprint candidate: {n_footprint:,} ({100*n_footprint/n_total:.1f}%)",
        f"  - **final mask:        {n_mask:,} ({100*n_mask/n_total:.1f}% of total, "
        f"{100*n_mask/max(n_footprint,1):.1f}% of footprint)**",
        f"  - fixture reference:   {n_ref:,}",
        f"",
        f"## ROI-aggregated scalar noise floor (PRIMARY — use this number downstream)",
        f"Computed per scan as mean of feature inside the mask, then SD across the {len(scan_npzs)} scans.",
        f"This captures the scan-to-scan correlated drift that per-pixel σ averages away.",
        f"",
        f"| Feature | per-scan ROI-mean | **σ_ROI** | drift slope/scan | drift p |",
        f"|---|---|---:|---:|---:|",
        f"| amplitude (V) | {[round(v,4) for v in roi_amp['per_scan_mean']]} | **{roi_amp['sigma_roi']:.4g}** | {roi_amp['drift_slope']:+.3g} | {roi_amp['drift_p']:.3f} |",
        f"| ToF (µs)      | {[round(v,4) for v in roi_tof['per_scan_mean']]} | **{roi_tof['sigma_roi']:.4g}** | {roi_tof['drift_slope']:+.3g} | {roi_tof['drift_p']:.3f} |",
        f"| energy        | {[round(v,2) for v in roi_eng['per_scan_mean']]} | **{roi_eng['sigma_roi']:.4g}** | {roi_eng['drift_slope']:+.3g} | {roi_eng['drift_p']:.3f} |",
        f"",
        f"Drift verdict (p < 0.05 = monotonic trend, likely incomplete equilibration):",
        f"  - amplitude: {_drift_verdict(roi_amp, roi_amp['sigma_roi'])}",
        f"  - ToF:       {_drift_verdict(roi_tof, roi_tof['sigma_roi'])}",
        f"  - energy:    {_drift_verdict(roi_eng, roi_eng['sigma_roi'])}",
        f"",
        f"## Per-pixel σ statistics within mask (DIAGNOSTIC — captures pixel-level jitter)",
        f"Used for the σ maps and ROI selection; **not** the noise floor used in significance tests.",
        _roi_stats(sigma_amp, "amplitude (V)"),
        _roi_stats(sigma_tof, "ToF (µs)"),
        _roi_stats(sigma_eng, "energy"),
        f"",
        f"## Files",
        f"  - `mask.npy` — main retained-pixel mask (boolean, shape {mask.shape})",
        f"  - `reference_mask.npy` — fixture / drift-reference pixels",
        f"  - `footprint_mask.npy` — coarse cell-only candidate before σ filtering",
        f"  - `mean_<feature>.npy` / `sigma_<feature>.npy` — per-pixel statistics",
        f"  - `sigma_maps.png`, `sigma_histograms.png`, `mask_overlay.png`",
        f"  - `roi_drift_check.png` — ROI-mean vs scan index with linear fit",
    ]
    (out_dir / "mask_report.md").write_text("\n".join(report), encoding="utf-8")

    # Summary CSV (one row per run)
    with (out_dir / "runs.csv").open("w", encoding="utf-8") as f:
        f.write("run_idx,scan_name,amp_p99,tof_p99,eng_p99\n")
        for i, p in enumerate(scan_npzs, start=1):
            d = load_features(p)
            f.write(f"{i},{p.parent.name},"
                    f"{float(np.nanpercentile(d['amp'],99)):.4f},"
                    f"{float(np.nanpercentile(d['tof'],99)):.4f},"
                    f"{float(np.nanpercentile(d['eng'],99)):.4f}\n")

    print(f"\n=== mask saved to: {out_dir} ===")
    print(f"  total mask pixels: {n_mask:,} / {n_total:,} ({100*n_mask/n_total:.1f}%)")
    print(f"  sigma_amp <= {thr_amp:.4g} V")
    print(f"  sigma_tof <= {thr_tof:.4g} us")
    print(f"  sigma_eng <= {thr_eng:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
