"""
Offline C-scan reprocessor — compare reconstruction algorithms.

Takes the per-line raw NPZ archive of a completed C-scan and rebuilds the
amplitude / ToF / energy maps using six different pixel-aggregation methods,
then renders a side-by-side comparison figure.

Why this exists
---------------
The production pipeline (row_from_pulses_nosmooth) does a bilinear scatter of
per-pulse pre-computed features onto the image grid. At high scan speeds the
pulses-per-pixel count drops (~23 at 40 mm/s vs ~80 at 10 mm/s), so the
incoherent feature average leaves visible per-pixel noise. This script tries
several alternatives — including coherent waveform averaging on the saved
gate windows — to quantify what's recoverable in post-processing.

Methods
-------
  M1_baseline      Bilinear scatter of per-pulse features (matches production).
  M2_mean          Per-column mean of features binned by physical x_mm.
  M3_median        Per-column median  — outlier-robust.
  M4_trimmed10     Per-column 10% trimmed mean — robust without losing all variance.
  M5_coherent      Align gate waveforms to their envelope peak inside each bin,
                   average coherently, then re-extract amplitude/ToF/energy from
                   the averaged waveform. True sqrt(N) SNR gain on amplitude.
  M6_gauss_kernel  Gaussian-weighted scatter (sigma matched to ~0.5 mm beam HW),
                   replacing the bilinear point-footprint assumption.

Output
------
  reports/experiments/cscan_methods_<scan_id>/
    comparison_<feature>.png   6-method panel for amplitude / tof / energy
    method_<i>_<feature>.npz   reconstructed map for each method
    noise_metrics.csv          row-to-row sigma in a flat region, per method
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
# I/O helpers
# ---------------------------------------------------------------------------

def latest_cscan_dir(base: Path) -> Path:
    """Return the most recent completed C-scan run folder under `base`."""
    runs = sorted(base.glob("cscan_scan_*"))
    for d in reversed(runs):
        man = d / "session_manifest.json"
        if not man.exists():
            continue
        text = man.read_text(encoding="utf-8")
        if '"state": "completed"' in text and (d / "lines_raw").exists():
            return d
    raise SystemExit(f"No completed C-scan with raw lines found under {base}")


def load_meta(scan_dir: Path) -> dict:
    """Load the scan-level meta JSON (gate, fs, roi, pitch, speed, cols)."""
    matches = list(scan_dir.glob("scan_*_meta.json"))
    if not matches:
        # Some runs only have session_manifest with config inside
        man = scan_dir / "session_manifest.json"
        import json
        cfg = json.loads(man.read_text())["config"]
        return {
            "roi_w_mm": cfg["roi_w"],
            "roi_h_mm": cfg["roi_h"],
            "pitch_mm": cfg["pitch"],
            "speed_mm_s": cfg["speed"],
            "ncols": cfg["cols"],
            "fs_hz": 20_000_000,
            "gate_us": [25.0, 50.0],
        }
    import json
    return json.loads(matches[0].read_text())


# ---------------------------------------------------------------------------
# Per-line aggregation methods
# ---------------------------------------------------------------------------

def _bin_indices(x_mm: np.ndarray, roi_w: float, ncols: int) -> np.ndarray:
    """Map each pulse's physical X to its column index in [0, ncols)."""
    idx = np.floor(x_mm / roi_w * ncols).astype(np.int64)
    return np.clip(idx, 0, ncols - 1)


def m1_baseline_bilinear(aa: np.ndarray, x_mm: np.ndarray,
                         roi_w: float, ncols: int) -> np.ndarray:
    """Bilinear scatter (closest match to production row_from_pulses_nosmooth)."""
    out  = np.full(ncols, np.nan, dtype=np.float64)
    valid = np.isfinite(aa)
    if not np.any(valid):
        return out.astype(np.float32)
    a = aa[valid].astype(np.float64)
    x = x_mm[valid].astype(np.float64) / roi_w * ncols
    i0 = np.floor(x).astype(np.int64)
    w1 = x - i0
    w0 = 1.0 - w1
    i1 = i0 + 1
    sumw = np.zeros(ncols)
    acc  = np.zeros(ncols)
    m = (i0 >= 0) & (i0 < ncols)
    np.add.at(sumw, i0[m], w0[m])
    np.add.at(acc,  i0[m], w0[m] * a[m])
    m = (i1 >= 0) & (i1 < ncols)
    np.add.at(sumw, i1[m], w1[m])
    np.add.at(acc,  i1[m], w1[m] * a[m])
    nz = sumw > 1e-12
    out[nz] = acc[nz] / sumw[nz]
    return out.astype(np.float32)


def m2_mean(aa: np.ndarray, x_mm: np.ndarray,
            roi_w: float, ncols: int) -> np.ndarray:
    """Simple per-column mean of all pulses binned by physical X."""
    out = np.full(ncols, np.nan, dtype=np.float32)
    valid = np.isfinite(aa)
    if not np.any(valid):
        return out
    idx = _bin_indices(x_mm[valid], roi_w, ncols)
    a   = aa[valid].astype(np.float64)
    sums   = np.bincount(idx, weights=a, minlength=ncols)
    counts = np.bincount(idx, minlength=ncols)
    nz = counts > 0
    out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    return out


def _per_bin_reduce(aa: np.ndarray, x_mm: np.ndarray,
                    roi_w: float, ncols: int, reducer) -> np.ndarray:
    """Group pulses by bin and apply a reducer (median, trimmed mean, ...)."""
    out = np.full(ncols, np.nan, dtype=np.float32)
    valid = np.isfinite(aa)
    if not np.any(valid):
        return out
    idx = _bin_indices(x_mm[valid], roi_w, ncols)
    a   = aa[valid]
    order = np.argsort(idx, kind="stable")
    idx_s = idx[order]
    a_s   = a[order]
    boundaries = np.flatnonzero(np.diff(idx_s)) + 1
    starts = np.concatenate(([0], boundaries))
    ends   = np.concatenate((boundaries, [a_s.size]))
    for s, e in zip(starts, ends):
        col = idx_s[s]
        out[col] = reducer(a_s[s:e])
    return out


def m3_median(aa, x_mm, roi_w, ncols):
    return _per_bin_reduce(aa, x_mm, roi_w, ncols, np.median)


def m4_trimmed10(aa, x_mm, roi_w, ncols):
    def trim(v):
        if v.size < 5:
            return float(np.mean(v))
        k = max(1, int(0.1 * v.size))
        v = np.sort(v)
        return float(np.mean(v[k:-k]))
    return _per_bin_reduce(aa, x_mm, roi_w, ncols, trim)


def m6_gauss_kernel(aa: np.ndarray, x_mm: np.ndarray,
                    roi_w: float, ncols: int,
                    sigma_mm: float = 0.5) -> np.ndarray:
    """
    Gaussian-weighted scatter — each pulse contributes to all columns within
    +/- 3 sigma with Gaussian weight matching the transducer beam profile.
    """
    out = np.full(ncols, np.nan, dtype=np.float32)
    valid = np.isfinite(aa)
    if not np.any(valid):
        return out
    a = aa[valid].astype(np.float64)
    x = x_mm[valid].astype(np.float64) / roi_w * ncols     # in column units
    sigma_cols = sigma_mm / roi_w * ncols
    half = int(np.ceil(3.0 * sigma_cols))

    # Offsets [-half, +half] columns from each pulse's nearest centre
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    # Gaussian weight kernel — same shape for every pulse
    sumw = np.zeros(ncols, dtype=np.float64)
    acc  = np.zeros(ncols, dtype=np.float64)
    i_centre = np.round(x).astype(np.int64)
    for off in offsets:
        cols = i_centre + off
        m = (cols >= 0) & (cols < ncols)
        if not np.any(m):
            continue
        # Real fractional distance from pulse to this column centre
        d = (cols[m] - x[m]).astype(np.float64)
        w = np.exp(-(d ** 2) / (2.0 * sigma_cols ** 2))
        np.add.at(sumw, cols[m], w)
        np.add.at(acc,  cols[m], w * a[m])
    nz = sumw > 1e-12
    out[nz] = (acc[nz] / sumw[nz]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Method 5: coherent waveform averaging
# ---------------------------------------------------------------------------

def _hilbert_env(x: np.ndarray) -> np.ndarray:
    """Hilbert envelope along axis=-1, matches scipy.signal.hilbert."""
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    H = np.zeros(n)
    if n % 2 == 0:
        H[0] = H[n // 2] = 1
        H[1 : n // 2] = 2
    else:
        H[0] = 1
        H[1 : (n + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * H, axis=-1))


def _parabolic_peak(env: np.ndarray) -> tuple[float, float]:
    """Sub-sample peak (index, value) via parabolic fit around argmax."""
    k = int(np.argmax(env))
    n = env.size
    if k == 0 or k == n - 1:
        return float(k), float(env[k])
    a = env[k - 1]; b = env[k]; c = env[k + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-12:
        return float(k), float(b)
    delta = 0.5 * (a - c) / denom
    peak  = b - 0.25 * (a - c) * delta
    return float(k + delta), float(peak)


def _fractional_shift(wf: np.ndarray, shift: float) -> np.ndarray:
    """Phase-ramp shift of one waveform (sub-sample)."""
    n = wf.size
    if abs(shift) < 1e-12:
        return wf
    k = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(wf) * np.exp(-2j * np.pi * k * shift), n=n)


def m5_coherent_per_line(
    wf: np.ndarray,            # [n_pulses, gate_samples]
    x_mm: np.ndarray,          # [n_pulses]
    roi_w: float, ncols: int,
    fs_hz: float, gate_start_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each column bin:
      1. Find each pulse's envelope peak (sub-sample).
      2. Shift all pulses to the bin's median peak position (small shifts only).
      3. Coherently average the shifted gate waveforms.
      4. Read amplitude from the averaged envelope's peak (sqrt(N) SNR gain).
         ToF: mean of original per-pulse positions (gate-relative, in us).
         Energy: mean of per-pulse energies (variance reduced by 1/N).

    Aligning to the per-bin median (rather than a global gate-centre target)
    keeps shifts in the +/-few-sample range, which avoids FFT wrap-around
    artefacts at the gate edges.
    """
    n_pulses, gate_len = wf.shape
    sample_us = 1e6 / fs_hz

    wf_dc = wf - wf.mean(axis=1, keepdims=True)
    env_all = _hilbert_env(wf_dc)
    # Per-pulse refined peak position (in samples, gate-relative)
    pos_all = np.empty(n_pulses, dtype=np.float64)
    for i in range(n_pulses):
        pos_all[i], _ = _parabolic_peak(env_all[i])
    # Per-pulse energy on the DC-removed waveform (matches production semantics)
    eng_all = np.sum(wf_dc ** 2, axis=1).astype(np.float64)

    out_amp = np.full(ncols, np.nan, dtype=np.float32)
    out_tof = np.full(ncols, np.nan, dtype=np.float32)
    out_eng = np.full(ncols, np.nan, dtype=np.float32)

    idx = _bin_indices(x_mm, roi_w, ncols)
    order = np.argsort(idx, kind="stable")
    idx_s = idx[order]
    boundaries = np.flatnonzero(np.diff(idx_s)) + 1
    starts = np.concatenate(([0], boundaries))
    ends   = np.concatenate((boundaries, [idx_s.size]))

    n  = gate_len
    k  = np.fft.rfftfreq(n)

    for s, e in zip(starts, ends):
        members = order[s:e]
        col = int(idx_s[s])
        bin_wf  = wf_dc[members]
        bin_pos = pos_all[members]
        bin_eng = eng_all[members]

        # Reference = median peak of the bin -> small intra-bin shifts.
        # Clamp residual shift to +/- 8 samples to guard against outlier peaks.
        ref_pos = float(np.median(bin_pos))
        shifts  = np.clip(ref_pos - bin_pos, -8.0, 8.0)

        spec    = np.fft.rfft(bin_wf, axis=-1)
        phasor  = np.exp(-2j * np.pi * np.outer(shifts, k))
        aligned = np.fft.irfft(spec * phasor, n=n, axis=-1)
        mean_wf = aligned.mean(axis=0)
        env     = _hilbert_env(mean_wf)
        _, peak_val = _parabolic_peak(env)

        out_amp[col] = peak_val
        out_tof[col] = float(np.mean(bin_pos)) * sample_us
        out_eng[col] = float(np.mean(bin_eng))
    return out_amp, out_tof, out_eng


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

METHODS_2D = [
    ("M1_baseline",    "Bilinear (current)"),
    ("M2_mean",        "Per-bin mean"),
    ("M3_median",      "Per-bin median"),
    ("M4_trimmed10",   "Trimmed mean 10%"),
    ("M5_coherent",    "Coherent WF avg"),
    ("M6_gauss",       "Gaussian beam kernel"),
]


def reprocess(scan_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(scan_dir)
    roi_w   = float(meta["roi_w_mm"])
    roi_h   = float(meta["roi_h_mm"])
    pitch   = float(meta["pitch_mm"])
    ncols   = int(meta["ncols"])
    fs_hz   = float(meta["fs_hz"])
    g0_us   = float(meta["gate_us"][0])

    line_files = sorted((scan_dir / "lines_raw").glob("line_*.npz"))
    nlines = len(line_files)
    print(f"Scan: {scan_dir.name}")
    print(f"  ROI={roi_w}x{roi_h} mm  pitch={pitch} mm  speed={meta.get('speed_mm_s','?')} mm/s")
    print(f"  nlines={nlines}  ncols={ncols}  fs={fs_hz/1e6:.1f} MHz  gate_us={meta['gate_us']}")

    # Allocate per-method, per-feature image arrays
    imgs = {name: {"amp": np.full((nlines, ncols), np.nan, dtype=np.float32),
                   "tof": np.full((nlines, ncols), np.nan, dtype=np.float32),
                   "eng": np.full((nlines, ncols), np.nan, dtype=np.float32)}
            for name, _ in METHODS_2D}

    for li, lf in enumerate(line_files):
        if li % 10 == 0:
            print(f"  line {li}/{nlines}")
        d = np.load(lf)
        amp   = d["amplitude"].astype(np.float32)
        tof   = d["tof_us"].astype(np.float32)
        eng   = d["energy"].astype(np.float32)
        x_mm  = d["x_mm"].astype(np.float32)
        wf    = d["waveforms"] if "waveforms" in d.files else None

        # M1 baseline — bilinear scatter on pre-computed features (per-feature)
        imgs["M1_baseline"]["amp"][li] = m1_baseline_bilinear(amp, x_mm, roi_w, ncols)
        imgs["M1_baseline"]["tof"][li] = m1_baseline_bilinear(tof, x_mm, roi_w, ncols)
        imgs["M1_baseline"]["eng"][li] = m1_baseline_bilinear(eng, x_mm, roi_w, ncols)

        # M2 plain bin-mean
        imgs["M2_mean"]["amp"][li] = m2_mean(amp, x_mm, roi_w, ncols)
        imgs["M2_mean"]["tof"][li] = m2_mean(tof, x_mm, roi_w, ncols)
        imgs["M2_mean"]["eng"][li] = m2_mean(eng, x_mm, roi_w, ncols)

        # M3 median
        imgs["M3_median"]["amp"][li] = m3_median(amp, x_mm, roi_w, ncols)
        imgs["M3_median"]["tof"][li] = m3_median(tof, x_mm, roi_w, ncols)
        imgs["M3_median"]["eng"][li] = m3_median(eng, x_mm, roi_w, ncols)

        # M4 trimmed mean
        imgs["M4_trimmed10"]["amp"][li] = m4_trimmed10(amp, x_mm, roi_w, ncols)
        imgs["M4_trimmed10"]["tof"][li] = m4_trimmed10(tof, x_mm, roi_w, ncols)
        imgs["M4_trimmed10"]["eng"][li] = m4_trimmed10(eng, x_mm, roi_w, ncols)

        # M6 Gaussian kernel scatter
        imgs["M6_gauss"]["amp"][li] = m6_gauss_kernel(amp, x_mm, roi_w, ncols)
        imgs["M6_gauss"]["tof"][li] = m6_gauss_kernel(tof, x_mm, roi_w, ncols)
        imgs["M6_gauss"]["eng"][li] = m6_gauss_kernel(eng, x_mm, roi_w, ncols)

        # M5 coherent waveform averaging
        if wf is not None:
            a5, t5, e5 = m5_coherent_per_line(wf, x_mm, roi_w, ncols, fs_hz, g0_us)
            imgs["M5_coherent"]["amp"][li] = a5
            # ToF and energy already gate-relative and per-pulse-scaled
            imgs["M5_coherent"]["tof"][li] = t5
            imgs["M5_coherent"]["eng"][li] = e5
        else:
            print(f"  WARN: line {li} has no waveforms; M5 skipped")

    # ---- save per-method NPZ ---------------------------------------------
    for name, _ in METHODS_2D:
        np.savez_compressed(
            out_dir / f"method_{name}.npz",
            amplitude=imgs[name]["amp"],
            tof=imgs[name]["tof"],
            energy=imgs[name]["eng"],
            x_mm=np.linspace(0, roi_w, ncols, dtype=np.float32),
            y_mm=np.linspace(0, roi_h, nlines, dtype=np.float32),
        )

    # ---- render comparison panel per feature -----------------------------
    feature_labels = {"amp": ("Amplitude", "V"),
                      "tof": ("ToF", "us"),
                      "eng": ("Energy", "V^2*samples")}
    for fkey, (flabel, funit) in feature_labels.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=110)
        # Common colour scale = global 5..95 percentile across methods so the
        # noise differences read visually, not the colour-stretch differences
        all_vals = np.concatenate([imgs[name][fkey][np.isfinite(imgs[name][fkey])].ravel()
                                   for name, _ in METHODS_2D])
        vmin, vmax = np.percentile(all_vals, [3, 97])
        cmap = matplotlib.colormaps.get_cmap("turbo").copy()
        cmap.set_bad("white")
        for ax, (name, label) in zip(axes.flat, METHODS_2D):
            im = ax.imshow(imgs[name][fkey], cmap=cmap, origin="upper",
                           aspect="equal",
                           extent=[0, roi_w, roi_h, 0],
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{label}")
            ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        fig.suptitle(f"{flabel}  ({funit}) — {scan_dir.name}\n"
                     f"speed {meta.get('speed_mm_s','?')} mm/s,  ~{int(11479/ncols)} pulses/pixel",
                     fontsize=11)
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.tight_layout(rect=[0, 0, 0.93, 0.95])
        out_png = out_dir / f"comparison_{fkey}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_png}")

    # ---- noise metric: row-to-row sigma in a homogeneous central strip ----
    csv_rows = [("method", "feature", "row_diff_sigma", "cv_pct")]
    for name, _ in METHODS_2D:
        for fkey in ("amp", "tof", "eng"):
            img = imgs[name][fkey]
            # Use a central rectangle, drop NaNs
            r0, r1 = int(0.35 * img.shape[0]), int(0.65 * img.shape[0])
            c0, c1 = int(0.35 * img.shape[1]), int(0.65 * img.shape[1])
            block = img[r0:r1, c0:c1]
            block = block[np.isfinite(block)]
            if block.size < 100:
                continue
            # High-pass: row-to-row difference removes the slow spatial structure,
            # what's left is the per-pixel noise floor
            diff = np.diff(img[r0:r1, c0:c1], axis=0)
            diff = diff[np.isfinite(diff)]
            row_sigma = float(np.nanstd(diff) / np.sqrt(2))   # sigma of one row from sigma of difference
            cv = 100.0 * float(np.nanstd(block) / abs(np.nanmean(block) + 1e-12))
            csv_rows.append((name, fkey, f"{row_sigma:.5g}", f"{cv:.3f}"))

    with open(out_dir / "noise_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)
    print(f"  wrote {out_dir / 'noise_metrics.csv'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=None,
                    help="Path to a cscan_scan_* directory. Default: latest completed.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output directory. Default: reports/experiments/cscan_methods_<scan_id>")
    args = ap.parse_args()

    base = Path("data/raw/cscan")
    scan = args.scan or latest_cscan_dir(base)
    scan_tag = re.sub(r"^cscan_scan_", "", scan.name)
    out = args.out or Path("reports/experiments") / f"cscan_methods_{scan_tag}"
    reprocess(scan, out)


if __name__ == "__main__":
    main()
