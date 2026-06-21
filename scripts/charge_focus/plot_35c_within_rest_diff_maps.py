"""35C: cumulative and consecutive C-scan diff galleries per SOC rest plateau.

For each of the 4 rest plateaus (SOC 20/40/60/80%) and each modality
(amplitude / tof / energy):
  cum_diff_<mod>.png    -> scan[t] - scan[t=0] panels (cumulative change)
  consec_diff_<mod>.png -> scan[i+1] - scan[i] panels (rate of change)

Output structure:
  within_rest_diffs/
    SOC20/{cum,consec}_diff_{amplitude,tof,energy}.png
    SOC40/...
    SOC60/...
    SOC80/...
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
DIFFS_ROOT = OUT_ROOT / "within_rest_diffs"

UNITS = {
    "amplitude": ("mV",   1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns",   1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]


def _roi_bounds(roi, x_mm, y_mm):
    rows = np.where(roi.any(axis=1))[0]
    cols = np.where(roi.any(axis=0))[0]
    return (float(x_mm[cols.min()]), float(x_mm[cols.max()]),
            float(y_mm[rows.min()]), float(y_mm[rows.max()]))


def plot_diff_panel(diffs, t_min, mod, unit, sigma, soc, roi, x_mm, y_mm,
                    title_prefix, out_path, time_label_fn):
    n_cols = len(diffs)
    x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi, x_mm, y_mm)
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]  # patched imshow flips

    # Symmetric color scale across all panels for fair comparison
    vmax = max(float(np.nanpercentile(np.abs(c[roi]), 99)) for c in diffs)
    vmax = max(vmax, 3 * sigma)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols + 1.2, 3.9),
                             constrained_layout=True)
    if n_cols == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(diffs[i], extent=extent, aspect="equal",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo],
                [y_lo, y_lo, y_hi, y_hi, y_lo],
                "k--", lw=0.6, alpha=0.7)
        roi_mean = float(np.nanmean(diffs[i][roi]))
        z = roi_mean / sigma
        ax.set_title(f"{time_label_fn(i)}\nΔROI = {roi_mean:+.2f} {unit}\n({z:+.1f}σ_ROI)",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, location="right",
                 label=f"Δ{mod} [{unit}]")
    fig.suptitle(
        f"{title_prefix}  ·  35°C SOC ≈ {soc}% rest  ·  {mod}\n"
        f"color scale = ±{vmax:.2g} {unit}    noise band: 2σ_ROI = {2*sigma:.2g} {unit}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    arrs = {"amplitude": d["amplitude"], "tof": d["tof"], "energy": d["energy"]}
    roi = d["roi_mask"]
    x_mm = d["x_mm"]; y_mm = d["y_mm"]

    DIFFS_ROOT.mkdir(exist_ok=True)

    for step, soc in zip(REST_STEPS, SOC_LABELS):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        idx = sel.index.values
        t_min = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0

        out_dir = DIFFS_ROOT / f"SOC{soc}"
        out_dir.mkdir(exist_ok=True)

        print(f"SOC {soc}%  step {step}  n={len(idx)}  span={t_min[-1]:.1f} min")

        for mod in ["amplitude", "tof", "energy"]:
            unit, scale, sigma = UNITS[mod]
            stack = arrs[mod][idx] * scale  # (n, H, W)

            # ----- cumulative: scan[t] - scan[0] -----
            cum = [stack[i] - stack[0] for i in range(1, len(stack))]
            plot_diff_panel(
                cum, t_min, mod, unit, sigma, soc, roi, x_mm, y_mm,
                title_prefix="Cumulative diff  scan[t] − scan[0]",
                out_path=out_dir / f"cum_diff_{mod}.png",
                time_label_fn=lambda i, tm=t_min: f"t = {tm[i+1]:.0f} min",
            )

            # ----- consecutive: scan[i+1] - scan[i] -----
            cd = [stack[i+1] - stack[i] for i in range(len(stack) - 1)]
            plot_diff_panel(
                cd, t_min, mod, unit, sigma, soc, roi, x_mm, y_mm,
                title_prefix="Consecutive diff  scan[i+1] − scan[i]",
                out_path=out_dir / f"consec_diff_{mod}.png",
                time_label_fn=lambda i, tm=t_min: f"{tm[i]:.0f}→{tm[i+1]:.0f} min",
            )
            print(f"   {mod:<10s}  cum+consec written")

    print(f"\nwrote 24 figures to {DIFFS_ROOT}/SOC{{20,40,60,80}}/")


if __name__ == "__main__":
    main()
