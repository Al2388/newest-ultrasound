"""Build a visual gallery + diff maps for SOC=60% rest scans (35C run).

Produces:
  raw_amp/, raw_tof/, raw_eng/  — 14 raw C-scan PNGs each, renamed by rest time
  cum_diff_<mod>.png            — scan[t] - scan[0] panels (cumulative change)
  consec_diff_<mod>.png         — scan[i+1] - scan[i] panels (rate of change)
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta

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
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

BATCH_DIR = PROJ / "reports" / "experiments" / "longrun_cycling_35c_2026-05-30_19-47-06"
CHECKPOINT = BATCH_DIR / "checkpoint.json"
ROI_MASK_PATH = PROJ / "reports" / "experiments" / "roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp" / "roi_mask.npy"
OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_2026-05-30_19-47-06_analysis" / "rest_at_SOC60"

REST_START_LOCAL = datetime(2026, 5, 31, 15, 7, 50)
REST_END_LOCAL = datetime(2026, 5, 31, 17, 7, 50)

UNITS = {
    "amplitude": ("mV", 1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns", 1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}
RAW_PNG = {"amplitude": "scan_amp.png", "tof": "scan_tof.png", "energy": "scan_eng.png"}


def _scan_ts_from_dir(name: str):
    m = re.search(r"_r(\d{3})_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$", name)
    if not m:
        return None, None
    run_idx, d, hh, mm, ss = m.groups()
    return int(run_idx), datetime.strptime(f"{d} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")


def _roi_bounds(roi_mask, x_mm, y_mm):
    rows = np.where(roi_mask.any(axis=1))[0]
    cols = np.where(roi_mask.any(axis=0))[0]
    return float(x_mm[cols.min()]), float(x_mm[cols.max()]), float(y_mm[rows.min()]), float(y_mm[rows.max()])


def main():
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    selected = []
    for c in cp["completed"]:
        sess_dir = PROJ / Path(c["session_dir"].replace("\\", "/"))
        run_idx, ts = _scan_ts_from_dir(sess_dir.name)
        if ts is None:
            continue
        if REST_START_LOCAL <= ts <= REST_END_LOCAL + timedelta(minutes=10):
            selected.append((run_idx, ts, sess_dir))
    selected.sort(key=lambda x: x[1])
    print(f"found {len(selected)} scans")

    # ============== 1. Raw PNG gallery (copy with sortable names) ==============
    for mod, fname in RAW_PNG.items():
        out_dir = OUT_ROOT / f"raw_{mod[:3]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for run_idx, ts, sess in selected:
            t_rel = (ts - selected[0][1]).total_seconds() / 60.0
            src = sess / fname
            if not src.exists():
                continue
            dst = out_dir / f"t{t_rel:05.1f}min_r{run_idx:03d}_{ts.strftime('%H-%M-%S')}_{mod[:3]}.png"
            shutil.copy2(src, dst)
        print(f"  wrote {len(list(out_dir.glob('*.png')))} raw {mod} PNGs to {out_dir.name}/")

    # ============== 2. Load NPZ stack ==============
    arrs = {"amplitude": [], "tof": [], "energy": []}
    x_mm = y_mm = None
    for _, _, sess in selected:
        npz = next(sess.glob("scan_*.npz"))
        d = np.load(npz)
        for k in arrs:
            arrs[k].append(d[k].astype(np.float32))
        if x_mm is None:
            x_mm = d["x_mm"].astype(np.float32)
            y_mm = d["y_mm"].astype(np.float32)
    for k in arrs:
        arrs[k] = np.stack(arrs[k], axis=0)
    roi_mask = np.load(ROI_MASK_PATH)

    x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi_mask, x_mm, y_mm)
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]  # patched imshow flips

    t_min = np.array([(s[1] - selected[0][1]).total_seconds() / 60.0 for s in selected])

    # ============== 3. Cumulative diff maps ==============
    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, sigma = UNITS[mod]
        cum = (arrs[mod] - arrs[mod][0]) * scale
        n_t = cum.shape[0]
        n_cols = n_t - 1
        vmax = max(float(np.nanpercentile(np.abs(c[roi_mask]), 99)) for c in cum[1:])
        vmax = max(vmax, 3 * sigma)

        fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols + 1, 3.8), constrained_layout=True)
        if n_cols == 1:
            axes = [axes]
        for i, ax in enumerate(axes, start=1):
            im = ax.imshow(cum[i], extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=0.6, alpha=0.7)
            roi_mean = float(np.nanmean(cum[i][roi_mask]))
            z = roi_mean / sigma
            ax.set_title(f"t = {t_min[i]:.0f} min\nΔROI = {roi_mean:+.2f} {unit}\n({z:+.1f}σ_ROI)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.85, location="right", label=f"{mod} [{unit}]")
        fig.suptitle(f"Cumulative diff scan[t] − scan[0]  ·  35°C SOC≈60% rest  ·  {mod}\n"
                     f"color scale = ±{vmax:.1f} {unit};  noise 2σ_ROI = {2*sigma:.2g} {unit}",
                     fontsize=11)
        fig.savefig(OUT_ROOT / f"cum_diff_{mod}.png", dpi=120)
        plt.close(fig)

    # ============== 4. Consecutive diff maps ==============
    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, sigma = UNITS[mod]
        cd = (arrs[mod][1:] - arrs[mod][:-1]) * scale
        n_cols = cd.shape[0]
        vmax = max(float(np.nanpercentile(np.abs(c[roi_mask]), 99)) for c in cd)
        vmax = max(vmax, 3 * sigma)

        fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols + 1, 3.8), constrained_layout=True)
        if n_cols == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(cd[i], extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=0.6, alpha=0.7)
            roi_mean = float(np.nanmean(cd[i][roi_mask]))
            z = roi_mean / sigma
            ax.set_title(f"{t_min[i]:.0f}→{t_min[i+1]:.0f} min\nΔROI = {roi_mean:+.2f} {unit}\n({z:+.1f}σ)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.85, location="right", label=f"Δ{mod} [{unit}]")
        fig.suptitle(f"Consecutive scan diff  ·  35°C SOC≈60% rest  ·  {mod}\n"
                     f"color scale = ±{vmax:.1f} {unit};  noise 2σ_ROI = {2*sigma:.2g} {unit}",
                     fontsize=11)
        fig.savefig(OUT_ROOT / f"consec_diff_{mod}.png", dpi=120)
        plt.close(fig)

    print(f"\nwrote diff maps to {OUT_ROOT}/")


if __name__ == "__main__":
    main()
