"""Equilibrium-state evolution across the 5 rest plateaus.

This 22h run does not return to a previously-visited SOC (the 5 rest
plateaus sit at SOC = -18.3, 1.7, 21.7, 41.7, 61.7 — monotone), so a
strict reversibility test cannot be done. Instead this script characterises
how the equilibrium state evolves between plateaus:

  - equilibrium_vs_soc.png   ROI-mean of each modality vs SOC, 5 anchors
  - equilibrium_maps_<m>.png 5-panel side-by-side maps, one per plateau
  - consecutive_diff_<m>.png 4-panel maps: seg(i+1) − seg(i)
  - residual_vs_linear.png   ROI-mean residual after linear SOC detrend
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NOISE_FLOOR, OUT_ROOT, load, rest_segments, roi_bounds_mm  # noqa: E402

OUT = OUT_ROOT / "05_reversibility"
OUT.mkdir(parents=True, exist_ok=True)

UNITS_DISPLAY = {
    "tof": ("ns", 1e3, "µs"),
    "amplitude": ("mV", 1e3, "V"),
    "energy": ("", 1.0, ""),
}
SIGMA_ROI = {
    "tof": NOISE_FLOOR["tof_sigma_roi_ns"] * 1e-3,
    "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"] * 1e-3,
    "energy": NOISE_FLOOR["energy_sigma_roi"],
}


def _equilibrium_maps(stack, last_idx: np.ndarray, mod: str) -> np.ndarray:
    arr = getattr(stack, mod)[last_idx]
    return arr


def _plot_equilibrium_vs_soc(stack, last_idx: np.ndarray):
    meta = stack.meta
    soc = meta.soc_pct.values[last_idx]
    t_h = (pd.to_datetime(meta.time_utc.iloc[last_idx]) - pd.to_datetime(meta.time_utc.iloc[0])).dt.total_seconds().values / 3600

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, mod in zip(axes, ["tof", "amplitude", "energy"]):
        unit, scale, _ = UNITS_DISPLAY[mod]
        arr = getattr(stack, mod)[last_idx]
        roi_mean = np.array([np.nanmean(a[stack.roi_mask]) for a in arr])
        y = roi_mean * scale

        slope, intercept = np.polyfit(soc, y, 1)
        ax.plot(soc, y, "o-", color="#222", ms=8)
        soc_line = np.linspace(soc.min(), soc.max(), 100)
        ax.plot(soc_line, slope * soc_line + intercept, "--", color="tab:blue", alpha=0.7,
                label=f"linear fit: {slope:+.4f} {unit}/%SoC")
        for i, t in enumerate(t_h):
            ax.annotate(f"seg{i+1}\nt={t:.1f}h", (soc[i], y[i]), fontsize=8, xytext=(6, 6), textcoords="offset points")
        ax.set_xlabel("SOC [%]")
        ax.set_ylabel(f"ROI-mean {mod} [{unit if unit else UNITS_DISPLAY[mod][2]}]")
        ax.set_title(f"{mod}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Equilibrium ROI-mean vs SOC (end-of-rest scan per plateau)", fontsize=12)
    return fig


def _plot_residual_vs_linear(stack, last_idx: np.ndarray):
    meta = stack.meta
    soc = meta.soc_pct.values[last_idx]
    t_h = (pd.to_datetime(meta.time_utc.iloc[last_idx]) - pd.to_datetime(meta.time_utc.iloc[0])).dt.total_seconds().values / 3600

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, mod in zip(axes, ["tof", "amplitude", "energy"]):
        unit, scale, _ = UNITS_DISPLAY[mod]
        arr = getattr(stack, mod)[last_idx]
        roi_mean = np.array([np.nanmean(a[stack.roi_mask]) for a in arr])
        y = roi_mean * scale
        slope, intercept = np.polyfit(soc, y, 1)
        resid = y - (slope * soc + intercept)
        sigma_disp = SIGMA_ROI[mod] * scale

        ax.bar(np.arange(5) + 1, resid, color=["#2ca02c" if r >= 0 else "#d62728" for r in resid])
        ax.axhspan(-2 * sigma_disp, 2 * sigma_disp, color="gray", alpha=0.2, label="±2σ_ROI")
        for i, (r, t) in enumerate(zip(resid, t_h)):
            ax.annotate(f"{r:+.2f}\nt={t:.1f}h", (i + 1, r), fontsize=8, ha="center",
                        xytext=(0, 8 if r >= 0 else -16), textcoords="offset points")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f"seg{i+1}\n{soc[i]:+.1f}%" for i in range(5)])
        ax.set_ylabel(f"residual [{unit if unit else UNITS_DISPLAY[mod][2]}]")
        ax.set_title(f"{mod}  (σ_ROI={sigma_disp:.3g})")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Residual after linear SOC detrend  ·  non-linearity / drift relative to noise floor", fontsize=12)
    return fig


def _plot_equilibrium_maps(stack, last_idx: np.ndarray, mod: str):
    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]
    arr = getattr(stack, mod)[last_idx]
    soc = stack.meta.soc_pct.values[last_idx]

    vmin = float(np.nanpercentile(arr[:, stack.roi_mask], 2))
    vmax = float(np.nanpercentile(arr[:, stack.roi_mask], 98))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    for i, ax in enumerate(axes):
        im = ax.imshow(arr[i], extent=extent, aspect="equal", cmap="turbo", vmin=vmin, vmax=vmax)
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=0.8, alpha=0.7)
        ax.set_title(f"seg{i+1}  SOC={soc[i]:+.1f}%")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, location="right", label=mod)
    fig.suptitle(f"Equilibrium maps — {mod}  (end-of-rest of each plateau)", fontsize=12)
    return fig


def _plot_consecutive_diff(stack, last_idx: np.ndarray, mod: str):
    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]
    arr = getattr(stack, mod)[last_idx]
    soc = stack.meta.soc_pct.values[last_idx]
    unit, scale, _ = UNITS_DISPLAY[mod]

    diffs = [arr[i + 1] - arr[i] for i in range(4)]
    vmax = max(float(np.nanpercentile(np.abs(d), 99)) for d in diffs)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for i, ax in enumerate(axes):
        diff = diffs[i]
        im = ax.imshow(diff, extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        roi_mean = float(np.nanmean(diff[stack.roi_mask]))
        z = roi_mean / SIGMA_ROI[mod] if SIGMA_ROI[mod] else np.nan
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=0.8, alpha=0.7)
        ax.set_title(
            f"seg{i+2} − seg{i+1}\nΔSOC = {soc[i+1] - soc[i]:+.1f}%\nΔ_ROI = {roi_mean*scale:+.3g} {unit} ({z:+.1f}σ)"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, location="right", label=mod)
    fig.suptitle(f"Consecutive equilibrium differences — {mod}", fontsize=12)
    return fig


def main() -> None:
    stack = load()
    segs_meta = rest_segments(stack.meta)
    last_idx = np.array([np.where(np.isin(stack.meta.run_idx.values, s.run_idx.values))[0][-1] for s in segs_meta])

    fig = _plot_equilibrium_vs_soc(stack, last_idx)
    fig.savefig(OUT / "equilibrium_vs_soc.png", dpi=130)
    plt.close(fig)

    fig = _plot_residual_vs_linear(stack, last_idx)
    fig.savefig(OUT / "residual_vs_linear.png", dpi=130)
    plt.close(fig)

    summary = ["# Equilibrium-state evolution across 5 rest plateaus\n"]
    summary.append(f"5 rest plateaus at SOC = {', '.join(f'{stack.meta.soc_pct.iloc[i]:+.1f}%' for i in last_idx)}. SOC is monotone increasing → strict reversibility test not possible from this run.\n\n")
    summary.append("## Linear SOC sensitivity from equilibrium endpoints\n\n")
    summary.append("| modality | slope | residual max [σ_ROI] | non-linear? |\n|---|---:|---:|---|\n")

    soc = stack.meta.soc_pct.values[last_idx]
    for mod in ["tof", "amplitude", "energy"]:
        unit, scale, _ = UNITS_DISPLAY[mod]
        arr = getattr(stack, mod)[last_idx]
        roi_mean = np.array([np.nanmean(a[stack.roi_mask]) for a in arr])
        y = roi_mean * scale
        slope, intercept = np.polyfit(soc, y, 1)
        resid = y - (slope * soc + intercept)
        sigma_disp = SIGMA_ROI[mod] * scale
        z_max = float(np.max(np.abs(resid)) / sigma_disp)
        verdict = "YES" if z_max > 5 else ("marginal" if z_max > 2 else "no")
        summary.append(f"| {mod} | {slope:+.4f} {unit}/%SoC | {z_max:.1f} | {verdict} |\n")

        fig_eq = _plot_equilibrium_maps(stack, last_idx, mod)
        fig_eq.savefig(OUT / f"equilibrium_maps_{mod}.png", dpi=130)
        plt.close(fig_eq)

        fig_diff = _plot_consecutive_diff(stack, last_idx, mod)
        fig_diff.savefig(OUT / f"consecutive_diff_{mod}.png", dpi=130)
        plt.close(fig_diff)

    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
