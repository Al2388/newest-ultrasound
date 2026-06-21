"""35C: C-scan of the FIRST scan in each rest plateau, side-by-side per modality.

4 columns (SOC 20/40/60/80), 1 row per modality (amplitude/tof/energy).
Output 3 separate PNGs (one per modality), shared colour scale within each.
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

OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"

REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]

UNITS = {
    "amplitude": ("mV",   1e3, "viridis"),
    "tof":       ("us",   1.0, "turbo"),
    "energy":    ("a.u.", 1.0, "viridis"),
}


def _roi_bounds(roi, x_mm, y_mm):
    rows = np.where(roi.any(axis=1))[0]
    cols = np.where(roi.any(axis=0))[0]
    return (float(x_mm[cols.min()]), float(x_mm[cols.max()]),
            float(y_mm[rows.min()]), float(y_mm[rows.max()]))


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    arrs = {"amplitude": d["amplitude"], "tof": d["tof"], "energy": d["energy"]}
    roi = d["roi_mask"]
    x_mm = d["x_mm"]; y_mm = d["y_mm"]

    # Pick the first scan (earliest timestamp) of each rest plateau
    first_idx = []
    first_meta = []
    for step in REST_STEPS:
        sel = meta[meta["step"] == step].sort_values("timestamp")
        first_idx.append(int(sel.index[0]))
        first_meta.append(sel.iloc[0])

    x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi, x_mm, y_mm)
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]

    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, cmap = UNITS[mod]
        arr3d = arrs[mod] * scale

        # Use shared colour scale across the 4 panels: ROI 2..98 percentile
        roi_vals = []
        for i in first_idx:
            roi_vals.append(arr3d[i][roi])
        all_roi = np.concatenate(roi_vals)
        vmin = float(np.nanpercentile(all_roi, 2))
        vmax = float(np.nanpercentile(all_roi, 98))

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), constrained_layout=True)
        for ax, idx, soc, m in zip(axes, first_idx, SOC_LABELS, first_meta):
            im = ax.imshow(arr3d[idx], extent=extent, aspect="equal",
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo],
                    [y_lo, y_lo, y_hi, y_hi, y_lo],
                    "-", color="red", lw=2.8, alpha=1.0)
            ts = pd.Timestamp(m["timestamp"]).strftime("%H:%M:%S")
            T = float(m["line_T_mean_c"])
            V = float(m["voltage_at_scan"]) * 1000
            ax.set_title(f"SOC {soc}%  rest start\n"
                         f"{ts}   T={T:.3f}°C   V={V:.1f} mV",
                         fontsize=10)
            ax.set_xlabel("X [mm]")
            if ax is axes[0]:
                ax.set_ylabel("Y [mm]")

        cbar = fig.colorbar(im, ax=axes, shrink=0.85, location="right",
                            label=f"{mod} [{unit}]")
        fig.suptitle(
            f"35°C  first C-scan of each rest plateau  ·  {mod}  "
            f"(shared colour scale = {vmin:.3g} – {vmax:.3g} {unit})",
            fontsize=11,
        )
        out = OUT_ROOT / f"first_rest_scan_{mod}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
