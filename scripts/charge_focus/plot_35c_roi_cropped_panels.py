"""ROI-cropped versions of equilibrium_maps and consecutive_diff panels.

Same data as analyze_35c_charge_focus.py outputs of the same name, but the
view window is tightened to the canonical 50x40 mm ROI and non-ROI pixels
are masked transparent. Replaces the existing files in-place.
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

UNITS = {
    "amplitude": ("mV",   1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns",   1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]

# Sub-region boundaries (tabs are on the X-large side of the cell)
TAB_DISTAL_X   = (14.0, 30.0)
INTERIOR_X     = (30.0, 50.0)
TAB_PROXIMAL_X = (50.0, 64.5)


def _add_subregion_dividers(ax, y_lo, y_hi, line_color):
    """Draw 2 vertical dashed lines at X=30 and X=50 within the ROI."""
    for xb in (INTERIOR_X[0], INTERIOR_X[1]):
        ax.plot([xb, xb], [y_lo, y_hi], "--", color=line_color, lw=1.4, alpha=0.85)


def _label_subregions(ax, y_hi, txt_color):
    """Tiny labels just above the ROI."""
    y = y_hi + 0.4
    ax.text(np.mean(TAB_DISTAL_X),   y, "tab-distal",   ha="center", va="bottom",
            fontsize=8, color=txt_color, fontweight="bold")
    ax.text(np.mean(INTERIOR_X),     y, "mid",          ha="center", va="bottom",
            fontsize=8, color=txt_color, fontweight="bold")
    ax.text(np.mean(TAB_PROXIMAL_X), y, "tab-proximal", ha="center", va="bottom",
            fontsize=8, color=txt_color, fontweight="bold")


def _roi_bounds(roi, x_mm, y_mm):
    rows = np.where(roi.any(axis=1))[0]
    cols = np.where(roi.any(axis=0))[0]
    return (float(x_mm[cols.min()]), float(x_mm[cols.max()]),
            float(y_mm[rows.min()]), float(y_mm[rows.max()]))


def _masked(arr2d, roi):
    """Mask non-ROI pixels as NaN so they render transparent."""
    out = arr2d.astype(np.float32).copy()
    out[~roi] = np.nan
    return out


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    arrs = {"amplitude": d["amplitude"], "tof": d["tof"], "energy": d["energy"]}
    roi = d["roi_mask"]
    x_mm = d["x_mm"]; y_mm = d["y_mm"]
    x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi, x_mm, y_mm)
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]
    pad = 1.5  # mm of breathing room around ROI

    # end-of-rest indices per plateau
    end_idx = []
    for step in REST_STEPS:
        sel = meta[meta["step"] == step].sort_values("timestamp")
        end_idx.append(int(sel.index[-1]))
    end_idx = np.array(end_idx)

    # ============== Equilibrium maps (ROI only) ==============
    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, _ = UNITS[mod]
        arr_native = arrs[mod][end_idx]   # native units (V, us)
        # Tighten to 10-90 percentile of ROI pixels across all 4 plateaus
        # -> compresses bright/dark outliers, gives much more contrast in the
        #    band where SOC-dependent variation lives
        vmin = float(np.nanpercentile(arr_native[:, roi], 10))
        vmax = float(np.nanpercentile(arr_native[:, roi], 90))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), constrained_layout=True)
        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color=(1, 1, 1, 0))   # transparent for NaN
        for i, ax in enumerate(axes):
            im = ax.imshow(_masked(arr_native[i], roi),
                           extent=extent, aspect="equal",
                           cmap=cmap, vmin=vmin, vmax=vmax)
            _add_subregion_dividers(ax, y_lo, y_hi, line_color="white")
            _label_subregions(ax, y_hi, txt_color="black")
            ax.set_xlim(x_lo - pad, x_hi + pad)
            ax.set_ylim(y_lo - pad - 0.2, y_hi + pad + 1.4)   # extra headroom for labels
            ax.set_title(f"SOC = {SOC_LABELS[i]}%", fontsize=11)
            if i == 0:
                ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
            else:
                ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.85, location="right",
                     label=f"{mod}  [{'V' if mod=='amplitude' else 'us' if mod=='tof' else 'a.u.'}]")
        fig.suptitle(f"35°C equilibrium maps (ROI only) — {mod}  (end-of-rest at each plateau)",
                     fontsize=11)
        fig.savefig(OUT_ROOT / f"equilibrium_maps_{mod}.png", dpi=140)
        plt.close(fig)
        print(f"wrote equilibrium_maps_{mod}.png")

    # ============== Consecutive equilibrium diff (ROI only) ==============
    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, sigma = UNITS[mod]
        arr_native = arrs[mod][end_idx]
        diffs = [(arr_native[i + 1] - arr_native[i]) * scale for i in range(3)]
        # Use 90th percentile of |diff| inside ROI (was 99th) -> tighter range,
        # mid-magnitude pixels get most of the colormap. Floor at 2 sigma_ROI
        # so the band visually means "above noise".
        vmax_color = max(float(np.nanpercentile(np.abs(c[roi]), 90)) for c in diffs)
        vmax_color = max(vmax_color, 2 * sigma)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color=(1, 1, 1, 0))
        for i, ax in enumerate(axes):
            im = ax.imshow(_masked(diffs[i], roi),
                           extent=extent, aspect="equal",
                           cmap=cmap, vmin=-vmax_color, vmax=vmax_color)
            _add_subregion_dividers(ax, y_lo, y_hi, line_color="black")
            _label_subregions(ax, y_hi, txt_color="black")
            ax.set_xlim(x_lo - pad, x_hi + pad)
            ax.set_ylim(y_lo - pad - 0.2, y_hi + pad + 1.4)
            roi_mean = float(np.nanmean(diffs[i][roi]))
            z = roi_mean / sigma
            ax.set_title(f"SOC {SOC_LABELS[i]} → {SOC_LABELS[i + 1]}%\n"
                         f"ΔROI = {roi_mean:+.3g} {unit}  ({z:+.1f}σ_ROI)",
                         fontsize=10)
            if i == 0:
                ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
            else:
                ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.85, location="right",
                     label=f"Δ{mod} [{unit}]")
        fig.suptitle(f"35°C consecutive equilibrium differences (ROI only) — {mod}", fontsize=11)
        fig.savefig(OUT_ROOT / f"consecutive_diff_{mod}.png", dpi=140)
        plt.close(fig)
        print(f"wrote consecutive_diff_{mod}.png")


if __name__ == "__main__":
    main()
