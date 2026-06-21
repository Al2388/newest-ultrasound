"""Two paper-grade figures comparing per-pixel noise between two batches.

Output:
  histograms.png/.pdf — 1×3 row of overlapping σ distributions per feature.
                        Shows that the per-pixel jitter distribution itself
                        shifted lower, not just the ROI-aggregated scalar.

  sigma_maps.png/.pdf — 3×2 grid (features × batches) of per-pixel σ maps
                        with shared per-row colour scale. Shows the
                        improvement is spatially uniform, not localised.

Both figures use the same OLD = amber / NEW = green palette as the
validation figure so they read as a three-panel set.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"
SOUND_SPEED_M_S = 989.0
MM_PER_US_ONE_WAY = SOUND_SPEED_M_S / 1000.0 / 2.0

EXPECTED = {"roi_w_mm": 80.0, "roi_h_mm": 72.0, "pitch_mm": 0.5,
            "speed_mm_s": 25.0, "accel_mm_s2": 800.0}

COL_A = "#d97706"     # amber (batch A)
COL_B = "#16a34a"     # green  (batch B)


def derive_centre_d_mm(scan_dir: Path) -> float | None:
    meta_p = next(scan_dir.glob("scan_*_meta.json"), None)
    if meta_p is None:
        return None
    meta = json.loads(meta_p.read_text())
    roi_w = float(meta["roi_w_mm"])
    line_files = sorted((scan_dir / "lines_raw").glob("line_*.npz"))
    if not line_files:
        return None
    centre_line = np.load(line_files[len(line_files) // 2])
    if "waveforms" not in centre_line.files:
        return None
    waveforms = centre_line["waveforms"]
    x_pulse = centre_line["x_mm"]
    fs_hz = float(centre_line["fs_hz"])
    gate_us = centre_line["gate_us"]
    valid = np.isfinite(x_pulse)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return None
    pulse_idx = int(valid_idx[np.argmin(np.abs(x_pulse[valid] - roi_w / 2.0))])
    wf = waveforms[pulse_idx].astype(np.float64)
    t_us = float(gate_us[0]) + np.arange(wf.size) * (1e6 / fs_hz)
    env = np.abs(hilbert(wf))
    i = int(np.argmax(env))
    if 0 < i < len(env) - 1:
        y0, y1, y2 = env[i - 1], env[i], env[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    tof_us = float(t_us[i]) + delta * (t_us[1] - t_us[0])
    return tof_us * MM_PER_US_ONE_WAY


def batch_median_d_mm(scan_dirs: list[Path]) -> float | None:
    ds = [derive_centre_d_mm(d) for d in scan_dirs]
    ds = [x for x in ds if x is not None]
    return float(np.median(ds)) if ds else None


# ---------- IO helpers (same as plot_noise_floor_validation_figure) ----------
def matches_params(meta: dict, tol: float = 1e-3) -> bool:
    for k, v in EXPECTED.items():
        if k not in meta or abs(float(meta[k]) - v) > tol:
            return False
    return True


def find_runs_by_prefix(prefix: str) -> list[Path]:
    candidates = []
    for d in CSCAN_ROOT.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(f"cscan_{prefix}"):
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
        expected_n = int(meta.get("nlines", 0))
        if expected_n <= 0:
            continue
        if len(list((d / "lines_raw").glob("line_*.npz"))) < expected_n:
            continue   # partial scan
        candidates.append(d)
    batch_tag = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_r\d{2}_")
    by_batch: dict[str, list[Path]] = {}
    for d in candidates:
        m = batch_tag.search(d.name)
        if m:
            by_batch.setdefault(m.group(1), []).append(d)
    if by_batch:
        return sorted(by_batch[max(by_batch.keys())])
    return sorted(candidates)


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
        "mean_amp": np.nanmean(amp, axis=0),
        "sigma_amp": np.nanstd(amp, axis=0, ddof=1),
        "sigma_tof": np.nanstd(tof, axis=0, ddof=1),
        "sigma_eng": np.nanstd(eng, axis=0, ddof=1),
    }


def detect_footprint(mean_amp: np.ndarray, rel: float = 0.30) -> np.ndarray:
    finite = mean_amp[np.isfinite(mean_amp)]
    if finite.size == 0:
        return np.zeros_like(mean_amp, dtype=bool)
    return np.isfinite(mean_amp) & (
        mean_amp >= rel * float(np.percentile(finite, 95)))


# ---------------------------------------------------------------- histograms
def plot_histograms(a: dict, b: dict, fp_a: np.ndarray, fp_b: np.ndarray,
                    label_a: str, label_b: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=160,
                             gridspec_kw={"wspace": 0.28})

    features = [
        ("sigma_amp", "σ amplitude (V)",    "V",  axes[0]),
        ("sigma_tof", "σ ToF (µs)",         "µs", axes[1]),
        ("sigma_eng", "σ energy",            "",   axes[2]),
    ]

    for key, name, units, ax in features:
        sa = a[key][fp_a & np.isfinite(a[key])]
        sb = b[key][fp_b & np.isfinite(b[key])]
        if sa.size == 0 or sb.size == 0:
            ax.set_title(f"{name} — empty")
            continue
        # Robust x range from both populations
        lo = float(np.percentile(np.concatenate([sa, sb]), 0.5))
        hi = float(np.percentile(np.concatenate([sa, sb]), 99))
        bins = np.linspace(max(lo, 0), hi, 70)

        ax.hist(sa, bins=bins, density=True, color=COL_A, alpha=0.55,
                edgecolor=COL_A, linewidth=0.4,
                label=f"{label_a}\nmed = {np.median(sa):.3g}{(' '+units) if units else ''}")
        ax.hist(sb, bins=bins, density=True, color=COL_B, alpha=0.55,
                edgecolor=COL_B, linewidth=0.4,
                label=f"{label_b}\nmed = {np.median(sb):.3g}{(' '+units) if units else ''}")

        # Median lines
        ax.axvline(np.median(sa), color=COL_A, linewidth=1.6, linestyle="--", alpha=0.9)
        ax.axvline(np.median(sb), color=COL_B, linewidth=1.6, linestyle="--", alpha=0.9)

        ratio = float(np.median(sa) / np.median(sb)) if np.median(sb) > 0 else float("nan")
        ax.set_title(f"{name}  —  median σ {ratio:.1f}× lower",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("density (footprint pixels)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8.5, loc="upper right",
                  framealpha=0.92, edgecolor="#bbb")
        ax.set_xlim(left=0)

    fig.suptitle("Per-pixel noise distribution: every feature shifts toward lower σ",
                 fontsize=12.5, y=1.02, fontweight="bold")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- σ maps
def plot_sigma_maps(a: dict, b: dict, fp_a: np.ndarray, fp_b: np.ndarray,
                    label_a: str, label_b: str, out_path: Path):
    extent_a = [float(a["x_mm"][0]), float(a["x_mm"][-1]),
                float(a["y_mm"][-1]), float(a["y_mm"][0])]
    extent_b = [float(b["x_mm"][0]), float(b["x_mm"][-1]),
                float(b["y_mm"][-1]), float(b["y_mm"][0])]

    feature_specs = [
        ("sigma_amp", "σ amplitude (V)", "V"),
        ("sigma_tof", "σ ToF (µs)",       "µs"),
        ("sigma_eng", "σ energy",          ""),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 13.5), dpi=160,
                             gridspec_kw={"wspace": 0.18, "hspace": 0.34,
                                          "left": 0.10, "right": 0.88,
                                          "top": 0.93, "bottom": 0.05})

    for row, (key, name, units) in enumerate(feature_specs):
        sig_a, sig_b = a[key], b[key]
        # Shared per-row colour scale based on the JOINT footprint distribution
        joint = np.concatenate([sig_a[fp_a & np.isfinite(sig_a)].ravel(),
                                sig_b[fp_b & np.isfinite(sig_b)].ravel()])
        if joint.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.percentile(joint, 98))
            vmax = max(vmax, 1e-9)

        im = None
        for col, (data, fp, label, extent, line_color) in enumerate([
            (sig_a, fp_a, label_a, extent_a, COL_A),
            (sig_b, fp_b, label_b, extent_b, COL_B),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="magma", vmin=0, vmax=vmax,
                           extent=extent, origin="upper", aspect="equal")
            ax.tick_params(labelsize=8)
            ax.set_xlabel("X (mm)", fontsize=9)
            ax.set_ylabel("Y (mm)", fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(line_color)
                spine.set_linewidth(2.0)
            if row == 0:
                ax.set_title(label, fontsize=11.5, fontweight="bold", color=line_color)
            # Stat annotation inside the panel
            vals = data[fp & np.isfinite(data)]
            if vals.size:
                ax.text(0.025, 0.975,
                        f"{name}\nmed = {np.median(vals):.3g}\np95 = {np.percentile(vals, 95):.3g}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=8.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=3,
                                 edgecolor="none"))

        # Per-row colourbar
        cbar = fig.colorbar(im, ax=axes[row, :].tolist(),
                            fraction=0.030, pad=0.02, shrink=0.92)
        cbar.set_label(name, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Per-pixel σ maps: the improvement is spatially uniform across the footprint",
                 fontsize=12.5, y=0.97, fontweight="bold")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-a-prefix", required=True)
    ap.add_argument("--batch-b-prefix", required=True)
    ap.add_argument("--label-a", default=None,
                    help="override auto-derived 'd = X.XX mm' label")
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs_a = find_runs_by_prefix(args.batch_a_prefix)
    runs_b = find_runs_by_prefix(args.batch_b_prefix)
    if not runs_a or not runs_b:
        raise SystemExit(f"missing runs: a={len(runs_a)} b={len(runs_b)}")
    d_a = batch_median_d_mm(runs_a)
    d_b = batch_median_d_mm(runs_b)
    label_a = args.label_a or (f"d = {d_a:.2f} mm" if d_a is not None else "batch A")
    label_b = args.label_b or (f"d = {d_b:.2f} mm" if d_b is not None else "batch B")
    print(f"A={label_a}: {len(runs_a)} runs")
    print(f"B={label_b}: {len(runs_b)} runs")
    a = load_stack(runs_a)
    b = load_stack(runs_b)
    fp_a = detect_footprint(a["mean_amp"])
    fp_b = detect_footprint(b["mean_amp"])

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) if args.out else (
        PROJECT / "reports" / "experiments" / f"noise_floor_paper_figs_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_histograms(a, b, fp_a, fp_b, label_a, label_b,
                    out_dir / "histograms")
    plot_sigma_maps(a, b, fp_a, fp_b, label_a, label_b,
                    out_dir / "sigma_maps")

    print(f"\nsaved figures in: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
