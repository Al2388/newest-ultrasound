"""
Diagnose the dominant noise source in a C-scan.

For one completed C-scan, separates the noise into three additive components:

  1. Within-bin (per-pulse) noise   -> intrinsic floor:  electronics + coupling
                                       fluctuations + transducer pulse-to-pulse
                                       repeatability. Measured INSIDE one
                                       physical pixel bin (~0.16 mm) where the
                                       material is genuinely uniform.

  2. Bin-to-bin (lateral) noise     -> per-column variability AFTER per-bin
                                       averaging. Removes (1); what's left is
                                       slow spatial drift + speckle that
                                       changes within ~mm scales.

  3. Row-to-row (vertical) noise    -> per-row variability AFTER per-bin
                                       averaging. Bidi mechanical offset and
                                       row alignment errors live here.

Also runs an LTR-vs-RTL row comparison to flag mechanical bidi bias as the
specific contributor to (3), and a Fourier spectrum of a vertical line cut
to look for periodic stripe artefacts.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_meta(scan_dir: Path) -> dict:
    import json
    matches = list(scan_dir.glob("scan_*_meta.json"))
    if matches:
        return json.loads(matches[0].read_text())
    cfg = json.loads((scan_dir / "session_manifest.json").read_text())["config"]
    return {"roi_w_mm": cfg["roi_w"], "roi_h_mm": cfg["roi_h"],
            "pitch_mm": cfg["pitch"], "speed_mm_s": cfg["speed"],
            "ncols": cfg["cols"]}


def diagnose(scan_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(scan_dir)
    roi_w = float(meta["roi_w_mm"]); roi_h = float(meta["roi_h_mm"])
    ncols = int(meta["ncols"]); pitch = float(meta["pitch_mm"])

    line_files = sorted((scan_dir / "lines_raw").glob("line_*.npz"))
    nlines = len(line_files)
    print(f"Scan: {scan_dir.name}  speed={meta.get('speed_mm_s','?')} mm/s  "
          f"nlines={nlines}  ncols={ncols}")

    # =========================================================================
    # 1) Within-bin (per-pulse) noise: pick a uniform interior region of the
    # cell, take all pulses falling in small ~0.16 mm bins from those rows,
    # measure their per-bin std. Average across bins.
    # =========================================================================
    # "Uniform interior" = middle ~30% in each dimension
    y0, y1 = int(0.35 * nlines), int(0.65 * nlines)
    x0_mm = 0.35 * roi_w
    x1_mm = 0.65 * roi_w
    col_w_mm = roi_w / ncols

    within_amp = []
    within_eng = []
    bin_mean_amp_per_row = []   # for downstream computations
    bin_mean_eng_per_row = []
    row_directions = []

    print("Scanning interior rows for within-bin variability...")
    for li in range(y0, y1):
        d = np.load(line_files[li])
        amp = d["amplitude"].astype(np.float64)
        eng = d["energy"].astype(np.float64)
        x   = d["x_mm"].astype(np.float64)
        direction = int(d["direction"])
        m = (x >= x0_mm) & (x <= x1_mm) & np.isfinite(amp) & np.isfinite(eng)
        amp = amp[m]; eng = eng[m]; x = x[m]
        # Bin into 0.16 mm columns
        idx = np.clip(np.floor(x / col_w_mm).astype(np.int64), 0, ncols - 1)
        # Group by bin, compute per-bin std (only bins with >= 8 pulses)
        order = np.argsort(idx, kind="stable")
        idx_s = idx[order]; amp_s = amp[order]; eng_s = eng[order]
        if idx_s.size == 0:
            continue
        bounds = np.flatnonzero(np.diff(idx_s)) + 1
        starts = np.concatenate(([0], bounds))
        ends   = np.concatenate((bounds, [idx_s.size]))
        row_amp_means = []
        row_eng_means = []
        for s, e in zip(starts, ends):
            if e - s >= 8:
                within_amp.append(np.std(amp_s[s:e], ddof=1))
                within_eng.append(np.std(eng_s[s:e], ddof=1))
                row_amp_means.append(np.mean(amp_s[s:e]))
                row_eng_means.append(np.mean(eng_s[s:e]))
        if row_amp_means:
            bin_mean_amp_per_row.append(np.array(row_amp_means))
            bin_mean_eng_per_row.append(np.array(row_eng_means))
            row_directions.append(direction)

    sigma_within_amp = float(np.mean(within_amp)) if within_amp else float("nan")
    sigma_within_eng = float(np.mean(within_eng)) if within_eng else float("nan")
    n_bins_sampled   = len(within_amp)

    # =========================================================================
    # 2) Bin-to-bin (within-row, after binning) sigma in the same region.
    # =========================================================================
    # Use the per-row sequences of bin-means; compute std of the
    # detrended (high-pass) profile.
    bb_amp = []
    bb_eng = []
    for ra, re_ in zip(bin_mean_amp_per_row, bin_mean_eng_per_row):
        if ra.size < 10:
            continue
        bb_amp.append(np.std(np.diff(ra), ddof=1) / np.sqrt(2))
        bb_eng.append(np.std(np.diff(re_), ddof=1) / np.sqrt(2))
    sigma_bb_amp = float(np.mean(bb_amp)) if bb_amp else float("nan")
    sigma_bb_eng = float(np.mean(bb_eng)) if bb_eng else float("nan")

    # =========================================================================
    # 3) Row-to-row (vertical) sigma — already computed for the M6 sweep,
    # here compute again on a fresh image built from per-row pulse means
    # to be self-contained.
    # =========================================================================
    # Build a quick image: each row's column-averaged amplitude in the
    # interior x band. Then diff vertically.
    row_means_amp_only = [np.mean(r) for r in bin_mean_amp_per_row if r.size > 5]
    row_means_eng_only = [np.mean(r) for r in bin_mean_eng_per_row if r.size > 5]
    sigma_rr_amp = (float(np.std(np.diff(row_means_amp_only), ddof=1) / np.sqrt(2))
                    if len(row_means_amp_only) > 3 else float("nan"))
    sigma_rr_eng = (float(np.std(np.diff(row_means_eng_only), ddof=1) / np.sqrt(2))
                    if len(row_means_eng_only) > 3 else float("nan"))

    # =========================================================================
    # 4) LTR vs RTL systematic difference (mechanical bidi bias)
    # =========================================================================
    ltr_means_amp = [m for m, dirn in zip(row_means_amp_only, row_directions) if dirn == 0]
    rtl_means_amp = [m for m, dirn in zip(row_means_amp_only, row_directions) if dirn == 1]
    bidi_bias_amp = float("nan")
    if ltr_means_amp and rtl_means_amp:
        bidi_bias_amp = float(np.mean(ltr_means_amp) - np.mean(rtl_means_amp))

    # =========================================================================
    # Report
    # =========================================================================
    print()
    print("=== Noise budget (amplitude, V) ===")
    print(f"  within-bin sigma (per-pulse floor)  = {sigma_within_amp:.4f}    "
          f"(N = {n_bins_sampled} bins)")
    print(f"  bin-to-bin sigma (lateral, after avg) = {sigma_bb_amp:.4f}")
    print(f"  row-to-row sigma (vertical)          = {sigma_rr_amp:.4f}")
    print(f"  LTR mean - RTL mean (bidi bias)      = {bidi_bias_amp:+.4f}")
    print()
    print("=== Noise budget (energy, V^2*samples) ===")
    print(f"  within-bin sigma  = {sigma_within_eng:.3f}")
    print(f"  bin-to-bin sigma  = {sigma_bb_eng:.3f}")
    print(f"  row-to-row sigma  = {sigma_rr_eng:.3f}")
    print()

    # If per-pulse sigma / sqrt(N_pulses_per_bin) ~ bin-to-bin sigma,
    # then per-pulse noise (coupling+electronic) fully explains the bin-to-bin
    # variability. If bin-to-bin >> floor, there is a slower spatial process
    # (e.g. drift, coupling layer thickness gradient).
    # Pull a typical pulses-per-bin estimate.
    d0 = np.load(line_files[0])
    n_pulses_total = d0["amplitude"].size
    avg_pulses_per_bin = n_pulses_total / ncols
    floor_after_avg = sigma_within_amp / np.sqrt(max(1, avg_pulses_per_bin))
    print(f"  avg pulses/bin                       = {avg_pulses_per_bin:.1f}")
    print(f"  predicted bin-to-bin sigma if pure   = {floor_after_avg:.4f}")
    print(f"  observed bin-to-bin sigma           = {sigma_bb_amp:.4f}")
    print(f"  ratio observed / predicted          = "
          f"{sigma_bb_amp / max(floor_after_avg, 1e-9):.1f}x")
    print()
    print("If ratio is near 1.0 -> bin-to-bin noise is dominated by per-pulse"
          " (electronic + coupling micro-variation).")
    print("If ratio >> 1.0       -> slower spatial process (coupling drift,"
          " surface curvature, scan-speed jitter) is the dominant lateral noise.")
    print()
    if not np.isnan(bidi_bias_amp):
        if abs(bidi_bias_amp) > 2 * sigma_rr_amp:
            print(f">>> LTR vs RTL bias ({bidi_bias_amp:+.4f}) is significant "
                  "vs vertical noise: probable MECHANICAL/bidi origin for striping.")
        else:
            print(f"    LTR vs RTL bias ({bidi_bias_amp:+.4f}) is small vs vertical "
                  "noise: striping likely from drift, not bidi offset.")

    # =========================================================================
    # Plots
    # =========================================================================
    # (a) Bar chart of the three sigmas + the per-pulse-floor / sqrt(N)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    cats   = ["Per-pulse\nsigma_within",
              "Expected bin-to-bin\n(sigma_within / sqrt(N))",
              "Observed bin-to-bin\n(lateral)",
              "Observed row-to-row\n(vertical)"]
    vals   = [sigma_within_amp, floor_after_avg, sigma_bb_amp, sigma_rr_amp]
    colors = ["#5cb8e6", "#9bd29b", "#ff8c66", "#d264d2"]
    ax.bar(cats, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Amplitude noise (V)")
    ax.set_title(f"Noise budget — {scan_dir.name}\n"
                 f"avg pulses/bin = {avg_pulses_per_bin:.0f}")
    fig.tight_layout()
    fig.savefig(out_dir / "noise_budget_amp.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'noise_budget_amp.png'}")

    # (b) Vertical profile through cell centre + 1-D FFT to look for periodic stripes
    # Build a quick per-row x-mean amplitude over the interior band.
    interior_means = []
    for li, lf in enumerate(line_files):
        d = np.load(lf)
        amp = d["amplitude"].astype(np.float64)
        x   = d["x_mm"].astype(np.float64)
        m = (x >= x0_mm) & (x <= x1_mm) & np.isfinite(amp)
        interior_means.append(float(np.mean(amp[m])) if np.any(m) else np.nan)
    interior_means = np.asarray(interior_means)

    # FFT of the row-mean trace, gives power vs spatial frequency in cycles/mm
    valid = np.isfinite(interior_means)
    y = interior_means[valid] - np.nanmean(interior_means[valid])
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(y.size, d=pitch)   # cycles per mm

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=110)
    axes[0].plot(np.arange(nlines) * pitch, interior_means, "-")
    axes[0].set_xlabel("Y (mm)"); axes[0].set_ylabel("Row-mean amplitude (V)")
    axes[0].set_title("Vertical profile through cell interior")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(freq[1:], spec[1:])    # drop DC bin
    axes[1].set_xlabel("Spatial frequency (cycles / mm)")
    axes[1].set_ylabel("|FFT| of row-mean amplitude")
    axes[1].set_title("Spectrum of vertical noise\n"
                      "(spike at 1/(2*pitch) -> alternating LTR/RTL bias)")
    axes[1].grid(True, alpha=0.3, which="both")
    nyq = 0.5 / pitch
    axes[1].axvline(nyq, color="gray", ls=":", label=f"row Nyquist = {nyq:.2f}")
    axes[1].legend()
    fig.suptitle(f"Vertical-noise diagnostic — {scan_dir.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "vertical_noise_diag.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'vertical_noise_diag.png'}")

    # Also save a numeric summary for later reference
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Scan: {scan_dir.name}\n")
        f.write(f"speed_mm_s: {meta.get('speed_mm_s','?')}\n")
        f.write(f"n_pulses_per_bin (avg): {avg_pulses_per_bin:.1f}\n\n")
        f.write(f"sigma_within_amp     = {sigma_within_amp:.5f}\n")
        f.write(f"sigma_within_amp/sqrt(N)= {floor_after_avg:.5f}\n")
        f.write(f"sigma_bb_amp         = {sigma_bb_amp:.5f}\n")
        f.write(f"sigma_rr_amp         = {sigma_rr_amp:.5f}\n")
        f.write(f"bidi_bias_amp        = {bidi_bias_amp:+.5f}\n\n")
        f.write(f"sigma_within_eng = {sigma_within_eng:.3f}\n")
        f.write(f"sigma_bb_eng     = {sigma_bb_eng:.3f}\n")
        f.write(f"sigma_rr_eng     = {sigma_rr_eng:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    scan = args.scan
    tag = re.sub(r"^cscan_scan_", "", scan.name)
    out = args.out or Path("reports/experiments") / f"noise_diag_{tag}"
    diagnose(scan, out)


if __name__ == "__main__":
    main()
