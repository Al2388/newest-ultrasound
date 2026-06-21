"""Compare two repeat-batches' noise floors and sweep mask threshold.

Loads two batches identified by folder-name prefix, computes per-pixel σ for
amplitude, ToF and energy in each, then produces:

  * sigma_maps_compare.png  — σ maps side-by-side (rows=features, cols=batches)
                              with shared per-row colour scales
  * sigma_histograms_compare.png — overlaid σ distributions, footprint only
  * threshold_sweep_<batch>.png  — for the NEW batch, mask overlay at several
                                   percentile-keep thresholds
  * comparison_table.md          — median / p95 / max per feature per batch,
                                   plus retained-pixel counts at each threshold
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"


def find_runs_by_prefix(prefix: str) -> list[Path]:
    """Find scan dirs by folder prefix; latest batch (by embedded timestamp) wins."""
    candidates = []
    for d in CSCAN_ROOT.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(f"cscan_{prefix}"):
            continue
        if next(d.glob("scan_*.npz"), None) is None:
            continue
        candidates.append(d)

    batch_tag = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_r\d{2}_")
    by_batch: dict[str, list[Path]] = {}
    for d in candidates:
        m = batch_tag.search(d.name)
        if m:
            by_batch.setdefault(m.group(1), []).append(d)
    if not by_batch:
        return sorted(candidates)
    return sorted(by_batch[max(by_batch.keys())])


def load_stack(scan_dirs: list[Path]) -> dict:
    """Stack feature arrays from a batch of scan dirs."""
    npzs = [next(d.glob("scan_*.npz")) for d in scan_dirs]
    first = np.load(npzs[0])
    nrows, ncols = first["amplitude"].shape
    amp = np.empty((len(npzs), nrows, ncols), dtype=np.float64)
    tof = np.empty_like(amp)
    eng = np.empty_like(amp)
    amp[0] = first["amplitude"]; tof[0] = first["tof"]; eng[0] = first["energy"]
    for i, p in enumerate(npzs[1:], start=1):
        d = np.load(p)
        amp[i] = d["amplitude"]; tof[i] = d["tof"]; eng[i] = d["energy"]
    return {
        "amp": amp, "tof": tof, "eng": eng,
        "mean_amp": np.nanmean(amp, axis=0),
        "sigma_amp": np.nanstd(amp, axis=0, ddof=1),
        "sigma_tof": np.nanstd(tof, axis=0, ddof=1),
        "sigma_eng": np.nanstd(eng, axis=0, ddof=1),
        "x_mm": first["x_mm"], "y_mm": first["y_mm"],
        "n": len(npzs), "scan_dirs": scan_dirs,
    }


def detect_footprint(mean_amp: np.ndarray, rel: float = 0.30) -> np.ndarray:
    finite = mean_amp[np.isfinite(mean_amp)]
    if finite.size == 0:
        return np.zeros_like(mean_amp, dtype=bool)
    return np.isfinite(mean_amp) & (mean_amp >= rel * float(np.percentile(finite, 95)))


def footprint_stats(sigma: np.ndarray, footprint: np.ndarray) -> dict:
    vals = sigma[footprint & np.isfinite(sigma)]
    if vals.size == 0:
        return {"median": np.nan, "p70": np.nan, "p95": np.nan, "max": np.nan, "n": 0}
    return {
        "median": float(np.median(vals)),
        "p70":    float(np.percentile(vals, 70)),
        "p95":    float(np.percentile(vals, 95)),
        "max":    float(np.max(vals)),
        "n":      int(vals.size),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_sigma_maps_compare(a: dict, b: dict, label_a: str, label_b: str,
                            footprint_a: np.ndarray, footprint_b: np.ndarray,
                            out_path: Path):
    extent_a = [float(a["x_mm"][0]), float(a["x_mm"][-1]),
                float(a["y_mm"][-1]), float(a["y_mm"][0])]
    extent_b = [float(b["x_mm"][0]), float(b["x_mm"][-1]),
                float(b["y_mm"][-1]), float(b["y_mm"][0])]

    feature_specs = [
        ("sigma_amp", "σ amplitude (V)",  "amp"),
        ("sigma_tof", "σ ToF (µs)",       "tof"),
        ("sigma_eng", "σ energy",          "eng"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), dpi=160,
                             gridspec_kw={"wspace": 0.18, "hspace": 0.30})

    for row, (key, name, _) in enumerate(feature_specs):
        sig_a, sig_b = a[key], b[key]
        # Shared colour scale per row from joint footprint statistics
        vals = np.concatenate([
            sig_a[footprint_a & np.isfinite(sig_a)].ravel(),
            sig_b[footprint_b & np.isfinite(sig_b)].ravel(),
        ])
        if vals.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.percentile(vals, 99))
            vmax = max(vmax, 1e-9)
        for col, (data, label, extent) in enumerate([(sig_a, label_a, extent_a),
                                                     (sig_b, label_b, extent_b)]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="magma", vmin=0, vmax=vmax,
                           extent=extent, origin="upper", aspect="equal")
            ax.tick_params(labelsize=7)
            ax.set_xlabel("X (mm)", fontsize=8)
            ax.set_ylabel("Y (mm)", fontsize=8)
            if row == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")
            fp = footprint_a if col == 0 else footprint_b
            stats = footprint_stats(data, fp)
            ax.text(0.02, 0.98,
                    f"{name}\nmed={stats['median']:.3g}\np95={stats['p95']:.3g}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5, color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"))
            if col == 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"σ comparison: {label_a} vs {label_b} "
                 f"(shared per-row colour scale, footprint-only stats)",
                 fontsize=11, y=0.995)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_histograms_compare(a: dict, b: dict, label_a: str, label_b: str,
                            footprint_a: np.ndarray, footprint_b: np.ndarray,
                            out_path: Path):
    feature_specs = [
        ("sigma_amp", "σ amplitude (V)"),
        ("sigma_tof", "σ ToF (µs)"),
        ("sigma_eng", "σ energy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=160)
    for ax, (key, name) in zip(axes, feature_specs):
        s_a = a[key][footprint_a & np.isfinite(a[key])]
        s_b = b[key][footprint_b & np.isfinite(b[key])]
        if s_a.size == 0 or s_b.size == 0:
            ax.set_title(f"{name}: empty")
            continue
        lo = min(float(np.percentile(s_a, 1)), float(np.percentile(s_b, 1)))
        hi = max(float(np.percentile(s_a, 99)), float(np.percentile(s_b, 99)))
        bins = np.linspace(lo, hi, 80)
        ax.hist(s_a, bins=bins, color="tab:blue", alpha=0.55,
                label=f"{label_a}  med={np.median(s_a):.3g}")
        ax.hist(s_b, bins=bins, color="tab:orange", alpha=0.55,
                label=f"{label_b}  med={np.median(s_b):.3g}")
        ax.axvline(np.median(s_a), color="tab:blue", linestyle="--", linewidth=1)
        ax.axvline(np.median(s_b), color="tab:orange", linestyle="--", linewidth=1)
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel("pixels", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("σ distributions inside the cell footprint", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(batch: dict, footprint: np.ndarray,
                         label: str, percentiles: list[float],
                         out_path: Path):
    """Show mask overlay at several percentile-keep thresholds."""
    extent = [float(batch["x_mm"][0]), float(batch["x_mm"][-1]),
              float(batch["y_mm"][-1]), float(batch["y_mm"][0])]
    n = len(percentiles)
    cols = n
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 4.0), dpi=160,
                             gridspec_kw={"wspace": 0.10})
    if cols == 1:
        axes = [axes]
    mean_amp = batch["mean_amp"]
    finite = mean_amp[np.isfinite(mean_amp)]
    vmin = float(np.percentile(finite, 1)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    overlay_cmap = mcolors.ListedColormap([[0.15, 0.15, 0.15, 0.55]])

    for ax, p in zip(axes, percentiles):
        # Per-feature percentile threshold within the footprint
        masks = []
        for key in ["sigma_amp", "sigma_tof", "sigma_eng"]:
            s = batch[key]
            in_fp = footprint & np.isfinite(s)
            if in_fp.any():
                thr = float(np.percentile(s[in_fp], p))
                masks.append(s <= thr)
            else:
                masks.append(np.zeros_like(s, dtype=bool))
        mask = footprint & masks[0] & masks[1] & masks[2]

        ax.imshow(mean_amp, cmap="turbo", vmin=vmin, vmax=vmax,
                  extent=extent, origin="upper", aspect="equal")
        overlay = np.where(mask, np.nan, 1.0)
        ax.imshow(overlay, cmap=overlay_cmap,
                  extent=extent, origin="upper", aspect="equal",
                  interpolation="nearest")
        n_kept = int(mask.sum())
        n_fp = int(footprint.sum())
        ax.set_title(f"percentile-keep = {p:.0f}\n"
                     f"{n_kept:,} px ({100*n_kept/max(n_fp,1):.0f}% of footprint)",
                     fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Y (mm)", fontsize=8)
        else:
            ax.set_yticklabels([])

    fig.suptitle(f"Mask threshold sweep on {label} (strict AND across amp/ToF/energy)",
                 fontsize=11, y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-a-prefix", required=True,
                    help="folder prefix for the first batch (control / baseline)")
    ap.add_argument("--batch-b-prefix", required=True,
                    help="folder prefix for the second batch (new / current)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--sweep-batch", choices=["a", "b"], default="b",
                    help="which batch to sweep the threshold on (default: b)")
    ap.add_argument("--percentiles", nargs="+", type=float,
                    default=[50.0, 60.0, 70.0, 80.0, 90.0])
    ap.add_argument("--footprint-rel", type=float, default=0.30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs_a = find_runs_by_prefix(args.batch_a_prefix)
    runs_b = find_runs_by_prefix(args.batch_b_prefix)
    if not runs_a or not runs_b:
        raise SystemExit(f"missing runs: a={len(runs_a)}, b={len(runs_b)}")
    label_a = args.label_a or args.batch_a_prefix
    label_b = args.label_b or args.batch_b_prefix
    print(f"batch A ({label_a}): {len(runs_a)} runs")
    for r in runs_a: print(f"  {r.name}")
    print(f"batch B ({label_b}): {len(runs_b)} runs")
    for r in runs_b: print(f"  {r.name}")

    a = load_stack(runs_a)
    b = load_stack(runs_b)

    footprint_a = detect_footprint(a["mean_amp"], args.footprint_rel)
    footprint_b = detect_footprint(b["mean_amp"], args.footprint_rel)

    # Output dir
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) if args.out else (
        PROJECT / "reports" / "experiments" / f"noise_floor_compare_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # σ maps
    plot_sigma_maps_compare(a, b, label_a, label_b,
                            footprint_a, footprint_b,
                            out_dir / "sigma_maps_compare.png")
    plot_histograms_compare(a, b, label_a, label_b,
                            footprint_a, footprint_b,
                            out_dir / "sigma_histograms_compare.png")

    # Threshold sweep on the chosen batch
    sweep = (b, footprint_b, label_b) if args.sweep_batch == "b" else (a, footprint_a, label_a)
    sweep_label = sweep[2].replace(" ", "_")
    plot_threshold_sweep(sweep[0], sweep[1], sweep[2], args.percentiles,
                         out_dir / f"threshold_sweep_{sweep_label}.png")

    # Table
    lines = [
        f"# Noise-floor comparison\n",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"",
        f"## Inputs",
        f"### A — {label_a}",
        *[f"  - `{r.name}`" for r in runs_a],
        f"### B — {label_b}",
        *[f"  - `{r.name}`" for r in runs_b],
        f"",
        f"## Footprint-only σ statistics",
        f"",
        f"| Feature | Batch | median | p70 | p95 | max | n_px |",
        f"|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, name in [("sigma_amp", "amplitude (V)"),
                       ("sigma_tof", "ToF (µs)"),
                       ("sigma_eng", "energy")]:
        sa = footprint_stats(a[key], footprint_a)
        sb = footprint_stats(b[key], footprint_b)
        lines.append(f"| {name} | A: {label_a} | {sa['median']:.4g} | {sa['p70']:.4g} | {sa['p95']:.4g} | {sa['max']:.4g} | {sa['n']:,} |")
        lines.append(f"| {name} | B: {label_b} | {sb['median']:.4g} | {sb['p70']:.4g} | {sb['p95']:.4g} | {sb['max']:.4g} | {sb['n']:,} |")

    # Mask-size sweep table
    lines += [
        f"",
        f"## Mask size at different percentile-keep (strict AND, sweep on {sweep[2]})",
        f"",
        f"| percentile-keep | retained pixels | % of footprint |",
        f"|---:|---:|---:|",
    ]
    sweep_batch, sweep_fp, _ = sweep
    n_fp = int(sweep_fp.sum())
    for p in args.percentiles:
        masks = []
        for key in ["sigma_amp", "sigma_tof", "sigma_eng"]:
            s = sweep_batch[key]
            in_fp = sweep_fp & np.isfinite(s)
            thr = float(np.percentile(s[in_fp], p)) if in_fp.any() else np.inf
            masks.append(s <= thr)
        m = sweep_fp & masks[0] & masks[1] & masks[2]
        n_m = int(m.sum())
        lines.append(f"| {p:.0f} | {n_m:,} | {100*n_m/max(n_fp,1):.1f}% |")

    (out_dir / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== saved to: {out_dir} ===")
    print("  sigma_maps_compare.png")
    print("  sigma_histograms_compare.png")
    print(f"  threshold_sweep_{sweep_label}.png")
    print("  comparison_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
