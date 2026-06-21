"""Within-rest spatial diff maps — show how the cell map changes scan-by-scan
during a rest plateau.

For a chosen rest plateau (default SOC≈80%), produces two figure types per
modality:
  (A) cumulative diff:   scan[t] − scan[0]  for each timepoint t
  (B) consecutive diff:  scan[i+1] − scan[i]  (between-scan rate)

Run:  python analysis_10_within_rest_diff_maps.py [80|60|40|20|0]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
LONGRUN = PROJ / "scripts" / "longrun_analysis"
sys.path.insert(0, str(LONGRUN))

import matplotlib.axes  # noqa: E402

_orig_imshow = matplotlib.axes.Axes.imshow


def _patched_imshow(self, X, *args, **kwargs):
    extent = kwargs.get("extent")
    if extent is not None and len(extent) == 4 and extent[2] > extent[3]:
        kwargs["extent"] = [extent[0], extent[1], extent[3], extent[2]]
        kwargs.setdefault("origin", "lower")
    return _orig_imshow(self, X, *args, **kwargs)


matplotlib.axes.Axes.imshow = _patched_imshow

import common  # noqa: E402
from common import NOISE_FLOOR, Stack, rest_segments, roi_bounds_mm  # noqa: E402

NEW_OUT = PROJ / "reports" / "longrun_cycling_22h_charge_focus"
common.OUT_ROOT = NEW_OUT
common.CACHE_PATH = NEW_OUT / "_cache" / "stack.npz"
common.META_PATH = NEW_OUT / "_cache" / "meta.csv"
OUT = NEW_OUT / "10_within_rest_diff_maps"
OUT.mkdir(parents=True, exist_ok=True)

UNITS = {
    "amplitude": ("mV", 1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns", 1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

_orig_load = common.load


def filtered_load() -> Stack:
    s = _orig_load(rebuild=False)
    keep = s.meta.step_tag.isin({"charge", "rest"}).values
    idx = np.where(keep)[0]
    new_meta = s.meta.iloc[idx].copy().reset_index(drop=True)
    soc_min = float(new_meta.soc_pct.min())
    new_meta["soc_pct"] = new_meta["soc_pct"] - soc_min
    return Stack(
        amplitude=s.amplitude[idx], tof=s.tof[idx], energy=s.energy[idx],
        x_mm=s.x_mm, y_mm=s.y_mm, roi_mask=s.roi_mask, meta=new_meta,
    )


def _pick_segment(segs, target_soc: float):
    socs = np.array([s.soc_pct.iloc[0] for s in segs])
    i = int(np.argmin(np.abs(socs - target_soc)))
    return i, segs[i]


def _draw_cum_diff(stack: Stack, seg_idx: np.ndarray, t_min: np.ndarray, mod: str, save_to: Path, soc_label: float):
    arr = getattr(stack, mod)[seg_idx]
    scale = UNITS[mod][1]
    sigma = UNITS[mod][2]
    unit = UNITS[mod][0]
    cum = (arr - arr[0]) * scale
    n_t = cum.shape[0]
    n_cols = n_t - 1

    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]
    vmax = max(float(np.nanpercentile(np.abs(c[stack.roi_mask]), 99)) for c in cum[1:])
    vmax = max(vmax, 3 * sigma)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols + 1, 3.6), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]
    for i, ax in enumerate(axes, start=1):
        im = ax.imshow(cum[i], extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=0.6, alpha=0.7)
        roi_mean = float(np.nanmean(cum[i][stack.roi_mask]))
        z = roi_mean / sigma if sigma else np.nan
        ax.set_title(f"t = {t_min[i]:.0f} min\nΔROI = {roi_mean:+.2f} {unit}\n({z:+.1f}σ_ROI)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, location="right", label=f"{mod} [{unit}]")
    fig.suptitle(f"Cumulative diff scan[t] − scan[0]  ·  SOC≈{soc_label:.0f}%  ·  {mod}\n"
                 f"color scale = ±{vmax:.1f} {unit};  noise floor 2σ_ROI = {2*sigma:.2g} {unit}",
                 fontsize=11)
    fig.savefig(save_to, dpi=120)
    plt.close(fig)


def _draw_consec_diff(stack: Stack, seg_idx: np.ndarray, t_min: np.ndarray, mod: str, save_to: Path, soc_label: float):
    arr = getattr(stack, mod)[seg_idx]
    scale = UNITS[mod][1]
    sigma = UNITS[mod][2]
    unit = UNITS[mod][0]
    cd = (arr[1:] - arr[:-1]) * scale
    n_cols = cd.shape[0]

    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]
    vmax = max(float(np.nanpercentile(np.abs(c[stack.roi_mask]), 99)) for c in cd)
    vmax = max(vmax, 3 * sigma)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols + 1, 3.6), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(cd[i], extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=0.6, alpha=0.7)
        roi_mean = float(np.nanmean(cd[i][stack.roi_mask]))
        z = roi_mean / sigma if sigma else np.nan
        ax.set_title(f"{t_min[i]:.0f}→{t_min[i+1]:.0f} min\nΔROI = {roi_mean:+.2f} {unit}\n({z:+.1f}σ)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, location="right", label=f"Δ{mod} [{unit}]")
    fig.suptitle(f"Consecutive scan diff (rate of change)  ·  SOC≈{soc_label:.0f}%  ·  {mod}\n"
                 f"color scale = ±{vmax:.1f} {unit};  noise floor 2σ_ROI = {2*sigma:.2g} {unit}",
                 fontsize=11)
    fig.savefig(save_to, dpi=120)
    plt.close(fig)


def main():
    target_soc = 80.0
    if len(sys.argv) > 1:
        target_soc = float(sys.argv[1])

    stack = filtered_load()
    segs = rest_segments(stack.meta)
    i_seg, seg_meta = _pick_segment(segs, target_soc)
    soc = float(seg_meta.soc_pct.iloc[0])
    print(f"chose seg #{i_seg+1}, SOC ~ {soc:.1f}%, n_scans={len(seg_meta)}")

    seg_idx = np.where(np.isin(stack.meta.run_idx.values, seg_meta.run_idx.values))[0]
    t = pd.to_datetime(seg_meta.time_utc)
    t_min = (t - t.iloc[0]).dt.total_seconds().values / 60.0

    label = f"SOC{int(round(soc)):02d}"
    for mod in ["amplitude", "tof", "energy"]:
        _draw_cum_diff(stack, seg_idx, t_min, mod, OUT / f"{label}_cum_diff_{mod}.png", soc)
        _draw_consec_diff(stack, seg_idx, t_min, mod, OUT / f"{label}_consec_diff_{mod}.png", soc)
    print(f"wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
