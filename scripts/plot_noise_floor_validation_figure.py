"""Paper figure: side-by-side drift overlay proving the new working
distance has a substantially lower noise floor than the previous one.

For each candidate feature (amplitude, ToF, energy), we plot the ROI-mean
of that feature across the 6 repeat scans, centred on each batch's own
mean so the two batches share a y-axis (deviation from typical value).

The figure shows three things at once:
  - σ_ROI (the scalar noise floor) — annotated as the legend
  - σ_ROI improvement factor — annotated in the title
  - drift slope and p-value — annotated below each line
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
from scipy import stats as sstats
from scipy.signal import hilbert


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"
SOUND_SPEED_M_S = 989.0
MM_PER_US_ONE_WAY = SOUND_SPEED_M_S / 1000.0 / 2.0

EXPECTED = {"roi_w_mm": 80.0, "roi_h_mm": 72.0, "pitch_mm": 0.5,
            "speed_mm_s": 25.0, "accel_mm_s2": 800.0}


def derive_centre_d_mm(scan_dir: Path) -> float | None:
    """Hilbert-envelope ToF of the centre pulse → one-way distance in mm."""
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
    tof = np.empty_like(amp)
    eng = np.empty_like(amp)
    amp[0] = first["amplitude"]; tof[0] = first["tof"]; eng[0] = first["energy"]
    for i, p in enumerate(npzs[1:], start=1):
        d = np.load(p)
        amp[i] = d["amplitude"]; tof[i] = d["tof"]; eng[i] = d["energy"]
    mean_amp = np.nanmean(amp, axis=0)
    sigma_amp = np.nanstd(amp, axis=0, ddof=1)
    sigma_tof = np.nanstd(tof, axis=0, ddof=1)
    sigma_eng = np.nanstd(eng, axis=0, ddof=1)
    return {"amp": amp, "tof": tof, "eng": eng,
            "mean_amp": mean_amp,
            "sigma_amp": sigma_amp, "sigma_tof": sigma_tof, "sigma_eng": sigma_eng}


def derive_mask(batch: dict, footprint_rel: float = 0.30,
                percentile_keep: float = 70.0) -> np.ndarray:
    finite = batch["mean_amp"][np.isfinite(batch["mean_amp"])]
    if finite.size == 0:
        return np.zeros_like(batch["mean_amp"], dtype=bool)
    fp = np.isfinite(batch["mean_amp"]) & (
        batch["mean_amp"] >= footprint_rel * float(np.percentile(finite, 95)))
    mask = fp.copy()
    for key in ("sigma_amp", "sigma_tof", "sigma_eng"):
        s = batch[key]
        in_fp = fp & np.isfinite(s)
        if in_fp.any():
            thr = float(np.percentile(s[in_fp], percentile_keep))
            mask &= (s <= thr)
    return mask


def roi_drift(stack: np.ndarray, mask: np.ndarray) -> dict:
    per_scan = np.array(
        [float(np.nanmean(stack[i][mask])) for i in range(stack.shape[0])]
    )
    sigma_roi = float(np.std(per_scan, ddof=1))
    x = np.arange(stack.shape[0], dtype=float)
    res = sstats.linregress(x, per_scan)
    return {"per_scan": per_scan,
            "sigma_roi": sigma_roi,
            "slope": float(res.slope),
            "p_value": float(res.pvalue)}


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

    # Auto-derive labels as "d = X.XX mm" from each batch's centre-pulse
    # envelope ToF, unless the user has overridden via --label-a/-b.
    d_a = batch_median_d_mm(runs_a)
    d_b = batch_median_d_mm(runs_b)
    label_a = args.label_a or (f"d = {d_a:.2f} mm" if d_a is not None else "batch A")
    label_b = args.label_b or (f"d = {d_b:.2f} mm" if d_b is not None else "batch B")
    print(f"A={label_a}: {len(runs_a)} runs")
    print(f"B={label_b}: {len(runs_b)} runs")

    a = load_stack(runs_a)
    b = load_stack(runs_b)
    mask_a = derive_mask(a)
    mask_b = derive_mask(b)
    print(f"mask A: {int(mask_a.sum())} px  mask B: {int(mask_b.sum())} px")

    drift_a = {k: roi_drift(a[k], mask_a) for k in ("amp", "tof", "eng")}
    drift_b = {k: roi_drift(b[k], mask_b) for k in ("amp", "tof", "eng")}

    # --- figure ---
    COL_A = "#d97706"   # amber (old)
    COL_B = "#16a34a"   # green (new)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=160,
                             gridspec_kw={"wspace": 0.30})

    for ax, (key, name, units) in zip(axes, [
        ("amp", "amplitude", "V"),
        ("tof", "ToF",       "µs"),
        ("eng", "energy",    ""),
    ]):
        da = drift_a[key]; db = drift_b[key]
        # Centre each batch on its own mean so the two y-scales are comparable
        ya = da["per_scan"] - da["per_scan"].mean()
        yb = db["per_scan"] - db["per_scan"].mean()
        x = np.arange(max(len(ya), len(yb)))

        # The shaded band shows ±σ_ROI for each batch
        ax.axhspan(-da["sigma_roi"], da["sigma_roi"],
                   color=COL_A, alpha=0.10, linewidth=0,
                   label=f"_{label_a} ±σ_ROI")
        ax.axhspan(-db["sigma_roi"], db["sigma_roi"],
                   color=COL_B, alpha=0.10, linewidth=0,
                   label=f"_{label_b} ±σ_ROI")

        ax.plot(x[:len(ya)], ya, "o-", color=COL_A, linewidth=1.6, markersize=6,
                label=f"{label_a}  σ_ROI = {da['sigma_roi']:.3g}{' '+units if units else ''}")
        ax.plot(x[:len(yb)], yb, "o-", color=COL_B, linewidth=1.6, markersize=6,
                label=f"{label_b}  σ_ROI = {db['sigma_roi']:.3g}{' '+units if units else ''}")

        ax.axhline(0, color="k", linewidth=0.4, alpha=0.6)

        # Improvement factor in title
        if db["sigma_roi"] > 0:
            factor = da["sigma_roi"] / db["sigma_roi"]
            title = f"{name}  —  σ_ROI improved {factor:.1f}×"
        else:
            title = f"{name}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("scan index")
        ax.set_ylabel(f"deviation from batch mean ({units})" if units
                      else "deviation from batch mean")
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks(x)
        ax.tick_params(labelsize=8)

        # Drift slope/p annotated in the corner
        ax.text(0.97, 0.04,
                f"drift (slope/scan, p)\n"
                f"  {label_a}: {da['slope']:+.3g} / p={da['p_value']:.2f}\n"
                f"  {label_b}: {db['slope']:+.3g} / p={db['p_value']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, family="monospace",
                bbox=dict(facecolor="white", edgecolor="#aaa",
                         linewidth=0.4, pad=3))

    fig.suptitle(
        "Noise-floor validation: ROI-aggregated feature vs scan index, "
        "both batches centred on their own mean",
        fontsize=12, y=1.01,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) if args.out else (
        PROJECT / "reports" / "experiments" / f"noise_floor_validation_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "validation.png", bbox_inches="tight")
    fig.savefig(out_dir / "validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out_dir/'validation.png'}")
    print(f"saved: {out_dir/'validation.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
