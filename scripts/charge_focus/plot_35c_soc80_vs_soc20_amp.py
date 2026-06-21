"""35C: amplitude difference between end-of-rest at SOC 80% and SOC 20%.

3-panel figure:
  (a) amp at SOC 20% (end-of-rest, step 5 last scan), ROI-cropped
  (b) amp at SOC 80% (end-of-rest, step 11 last scan), ROI-cropped
  (c) diff = amp_80 - amp_20, signed colormap, ROI-cropped

Quantitative summary printed + appended to companion markdown.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_orig_imshow = matplotlib.axes.Axes.imshow


def _patched_imshow(self, X, *args, **kwargs):
    extent = kwargs.get("extent")
    if extent is not None and len(extent) == 4 and extent[2] > extent[3]:
        kwargs["extent"] = [extent[0], extent[1], extent[3], extent[2]]
        kwargs.setdefault("origin", "lower")
    return _orig_imshow(self, X, *args, **kwargs)


matplotlib.axes.Axes.imshow = _patched_imshow

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"
OUT_PNG = OUT_ROOT / "soc80_vs_soc20_amplitude.png"
OUT_MD = OUT_ROOT / "soc80_vs_soc20_amplitude.md"

# subregion bands
TAB_DISTAL_X   = (14.0, 30.0)
INTERIOR_X     = (30.0, 50.0)
TAB_PROXIMAL_X = (50.0, 64.5)


def _xband(roi, x_mm, lo, hi):
    cl = int(np.searchsorted(x_mm, lo)); ch = int(np.searchsorted(x_mm, hi))
    m = np.zeros_like(roi, dtype=bool)
    m[:, cl:ch] = True
    return m & roi


def _roi_bounds(roi, x_mm, y_mm):
    rows = np.where(roi.any(axis=1))[0]; cols = np.where(roi.any(axis=0))[0]
    return (float(x_mm[cols.min()]), float(x_mm[cols.max()]),
            float(y_mm[rows.min()]), float(y_mm[rows.max()]))


def _masked(arr2d, roi):
    out = arr2d.astype(np.float32).copy()
    out[~roi] = np.nan
    return out


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    amp_stack = d["amplitude"]
    roi = d["roi_mask"]
    x_mm = d["x_mm"]; y_mm = d["y_mm"]

    # End-of-rest at each plateau (last scan of step 5 and step 11)
    idx_20 = meta[meta["step"] == 5].sort_values("timestamp").index[-1]
    idx_80 = meta[meta["step"] == 11].sort_values("timestamp").index[-1]
    print(f"SOC 20 end-of-rest: scan idx {idx_20}, ts {meta.iloc[idx_20]['timestamp']}, "
          f"T = {meta.iloc[idx_20]['line_T_mean_c']:.3f} C")
    print(f"SOC 80 end-of-rest: scan idx {idx_80}, ts {meta.iloc[idx_80]['timestamp']}, "
          f"T = {meta.iloc[idx_80]['line_T_mean_c']:.3f} C")

    amp_20 = amp_stack[idx_20] * 1000  # mV
    amp_80 = amp_stack[idx_80] * 1000
    diff = amp_80 - amp_20             # mV
    sigma_roi_mV = NOISE_FLOOR["amp_sigma_roi_mv"]

    # ROI / subregion statistics
    bands = {
        "ROI (all)":     roi,
        "tab-distal":    _xband(roi, x_mm, *TAB_DISTAL_X),
        "mid":           _xband(roi, x_mm, *INTERIOR_X),
        "tab-proximal":  _xband(roi, x_mm, *TAB_PROXIMAL_X),
    }
    stats = {}
    for name, m in bands.items():
        a20 = float(np.nanmean(amp_20[m]))
        a80 = float(np.nanmean(amp_80[m]))
        d_v = a80 - a20
        d_sigma = d_v / sigma_roi_mV
        stats[name] = {
            "n_pix": int(m.sum()),
            "amp_20": a20,
            "amp_80": a80,
            "delta": d_v,
            "delta_sigma_roi": d_sigma,
        }

    # ============ Figure ============
    x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi, x_mm, y_mm)
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]
    pad = 1.5

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)

    # ---- (a) and (b): SOC 20 and 80 amp maps ----
    a_only = np.concatenate([amp_20[roi], amp_80[roi]])
    vmin = float(np.percentile(a_only, 10))
    vmax = float(np.percentile(a_only, 90))
    cmap_amp = plt.colormaps["turbo"].copy(); cmap_amp.set_bad((1,1,1,0))
    for ax, mp, soc in zip(axes[:2], [amp_20, amp_80], [20, 80]):
        ax.imshow(_masked(mp, roi), extent=extent, aspect="equal",
                  cmap=cmap_amp, vmin=vmin, vmax=vmax, interpolation="nearest")
        for xb in (INTERIOR_X[0], INTERIOR_X[1]):
            ax.plot([xb, xb], [y_lo, y_hi], "--", color="white", lw=1.4, alpha=0.85)
        yt = y_hi + 0.4
        for cx, lbl in [(np.mean(TAB_DISTAL_X), "tab-distal"),
                        (np.mean(INTERIOR_X), "mid"),
                        (np.mean(TAB_PROXIMAL_X), "tab-proximal")]:
            ax.text(cx, yt, lbl, ha="center", va="bottom",
                    fontsize=9, color="black", fontweight="bold")
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.set_ylim(y_lo - pad - 0.2, y_hi + pad + 1.4)
        ax.set_title(f"amplitude  ·  SOC = {soc}%  (end-of-rest)\n"
                     f"ROI mean = {stats['ROI (all)']['amp_'+str(soc)]:.1f} mV", fontsize=10)
        ax.set_xlabel("X [mm]")
    axes[0].set_ylabel("Y [mm]")
    fig.colorbar(axes[1].images[0], ax=axes[:2], shrink=0.85,
                 location="right", label="amplitude [mV]")

    # ---- (c) diff = SOC80 - SOC20 ----
    diff_roi_vals = diff[roi]
    vmax_d = float(np.percentile(np.abs(diff_roi_vals), 95))
    vmax_d = max(vmax_d, 3 * sigma_roi_mV)
    ax = axes[2]
    cmap_diff = plt.colormaps["RdBu_r"].copy(); cmap_diff.set_bad((1,1,1,0))
    ax.imshow(_masked(diff, roi), extent=extent, aspect="equal",
              cmap=cmap_diff, vmin=-vmax_d, vmax=vmax_d, interpolation="nearest")
    for xb in (INTERIOR_X[0], INTERIOR_X[1]):
        ax.plot([xb, xb], [y_lo, y_hi], "--", color="black", lw=1.4, alpha=0.85)
    yt = y_hi + 0.4
    for cx, lbl in [(np.mean(TAB_DISTAL_X), "tab-distal"),
                    (np.mean(INTERIOR_X), "mid"),
                    (np.mean(TAB_PROXIMAL_X), "tab-proximal")]:
        ax.text(cx, yt, lbl, ha="center", va="bottom",
                fontsize=9, color="black", fontweight="bold")
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(y_lo - pad - 0.2, y_hi + pad + 1.4)
    ax.set_title(f"Δamplitude = SOC80 − SOC20  [mV]\n"
                 f"ROI mean Δ = {stats['ROI (all)']['delta']:+.2f} mV  "
                 f"({stats['ROI (all)']['delta_sigma_roi']:+.1f}σ_ROI)",
                 fontsize=10)
    ax.set_xlabel("X [mm]")
    fig.colorbar(ax.images[0], ax=ax, shrink=0.85, label="Δamp [mV]")

    fig.suptitle(
        f"35°C  amplitude change from SOC 20% → 80% (end-of-rest at each plateau)   "
        f"σ_ROI = {sigma_roi_mV:.2f} mV",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\nwrote {OUT_PNG}")

    # ============ text summary ============
    print(f"\n=== sub-region quantitative summary ===")
    print(f"{'band':<14s} {'n px':>7s} {'amp@20%':>10s} {'amp@80%':>10s} "
          f"{'Δ (mV)':>10s} {'Δ/σ_ROI':>10s}")
    for name, s in stats.items():
        print(f"{name:<14s} {s['n_pix']:>7d} {s['amp_20']:>10.2f} "
              f"{s['amp_80']:>10.2f} {s['delta']:>+10.2f} "
              f"{s['delta_sigma_roi']:>+10.1f}")

    md = ["# 35°C amplitude change from SOC 20% to SOC 80%\n\n",
          f"End-of-rest scans:\n",
          f"- SOC 20% (step 5, last scan): r{int(meta.iloc[idx_20]['run_idx']):03d}, "
          f"T = {float(meta.iloc[idx_20]['line_T_mean_c']):.3f} °C, "
          f"V = {float(meta.iloc[idx_20]['voltage_at_scan'])*1000:.1f} mV\n",
          f"- SOC 80% (step 11, last scan): r{int(meta.iloc[idx_80]['run_idx']):03d}, "
          f"T = {float(meta.iloc[idx_80]['line_T_mean_c']):.3f} °C, "
          f"V = {float(meta.iloc[idx_80]['voltage_at_scan'])*1000:.1f} mV\n\n",
          f"σ_ROI (amplitude) = {sigma_roi_mV:.3f} mV (from N=6 noisefloor_v3.238 repeats)\n\n",
          f"## Sub-region statistics\n\n",
          f"| band | n px | amp @ SOC 20% [mV] | amp @ SOC 80% [mV] | Δ [mV] | Δ / σ_ROI |\n",
          f"|:---|---:|---:|---:|---:|---:|\n"]
    for name, s in stats.items():
        md.append(f"| {name} | {s['n_pix']} | {s['amp_20']:.2f} | {s['amp_80']:.2f} | "
                  f"{s['delta']:+.2f} | {s['delta_sigma_roi']:+.1f} |\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
