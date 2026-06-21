"""Render the full-FOV (not cropped to ROI) first C-scan of the 35C batch (r001).

Three panels side-by-side, full 80x72 mm scan area.
ROI box shown as a thin reference, sub-region dividers as dashed lines inside it.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

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

R001_SESSION = PROJ / "data/raw/cscan/cscan_longrun_cycling_35c_2026-05-30_19-47-06_r001_2026-05-30_19-47-06"
ROI_MASK_PATH = PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp/roi_mask.npy"
OUT = PROJ / "reports/longrun_cycling_35c_charge_focus/batch_r001_full_fov.png"

UNITS = {
    "amplitude": ("V",   "turbo"),
    "tof":       ("us",  "turbo"),
    "energy":    ("a.u.","turbo"),
}

TAB_DISTAL_X   = (14.0, 30.0)
INTERIOR_X     = (30.0, 50.0)
TAB_PROXIMAL_X = (50.0, 64.5)


def _roi_bounds(roi, x_mm, y_mm):
    rows = np.where(roi.any(axis=1))[0]; cols = np.where(roi.any(axis=0))[0]
    return (float(x_mm[cols.min()]), float(x_mm[cols.max()]),
            float(y_mm[rows.min()]), float(y_mm[rows.max()]))


def main():
    npz = next(R001_SESSION.glob("scan_*.npz"))
    d = np.load(npz)
    roi = np.load(ROI_MASK_PATH)
    x_mm = d["x_mm"]; y_mm = d["y_mm"]
    T = float(np.nanmean(d["line_temperature_mean_c"]))
    t_unix0 = float(d["line_unix_start_s"][0])
    t_str = datetime.utcfromtimestamp(t_unix0).strftime("%Y-%m-%d %H:%M:%S UTC")

    x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi, x_mm, y_mm)
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]

    arrs = {"amplitude": d["amplitude"], "tof": d["tof"], "energy": d["energy"]}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    for ax, mod in zip(axes, ["amplitude", "tof", "energy"]):
        unit, cmap_name = UNITS[mod]
        arr = arrs[mod].astype(np.float32)
        # webapp-style colour scale: turbo cmap, 5-95 percentile of finite values,
        # NaN -> white, nearest interpolation (matches src/ultrasound_battery/
        # services/scanner.py:_save_plot)
        finite = arr[np.isfinite(arr)]
        vmin = float(np.percentile(finite, 5))
        vmax = float(np.percentile(finite, 95))

        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad("white")
        im = ax.imshow(arr, extent=extent, aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")

        # ROI rectangle (cell area) -- thin so it doesn't dominate the FOV view
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo],
                [y_lo, y_lo, y_hi, y_hi, y_lo],
                "-", color="red", lw=1.8, alpha=0.95)
        # sub-region dividers within ROI
        for xb in (INTERIOR_X[0], INTERIOR_X[1]):
            ax.plot([xb, xb], [y_lo, y_hi], "--", color="red", lw=1.0, alpha=0.7)
        # sub-region labels at top of ROI
        yt = y_hi + 0.6
        for cx, lbl in [(np.mean(TAB_DISTAL_X), "tab-distal"),
                        (np.mean(INTERIOR_X),   "mid"),
                        (np.mean(TAB_PROXIMAL_X), "tab-proximal")]:
            ax.text(cx, yt, lbl, ha="center", va="bottom",
                    fontsize=9, color="red", fontweight="bold")

        # full-FOV axis limits
        ax.set_xlim(x_mm.min(), x_mm.max())
        ax.set_ylim(y_mm.max(), y_mm.min())   # imshow patched flips internally
        # because patched imshow already flips Y, we want the *display* to show
        # Y increasing upward => set_ylim with lower-first
        ax.set_ylim(y_mm.min(), y_mm.max())

        ax.set_title(f"{mod}   (vmin={vmin:.3g}, vmax={vmax:.3g} {unit})", fontsize=11)
        ax.set_xlabel("X [mm]")
        if ax is axes[0]:
            ax.set_ylabel("Y [mm]")
        fig.colorbar(im, ax=ax, shrink=0.85, label=f"{mod} [{unit}]")

    fig.suptitle(
        f"35°C cycling batch -- FULL FOV first scan (r001)\n"
        f"{R001_SESSION.name}\n"
        f"start = {t_str}   cell T = {T:.3f}°C   FOV = {x_mm.min():.1f}-{x_mm.max():.1f} x "
        f"{y_mm.min():.1f}-{y_mm.max():.1f} mm",
        fontsize=11,
    )
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
