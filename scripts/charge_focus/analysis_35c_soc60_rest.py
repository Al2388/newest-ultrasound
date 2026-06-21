"""SOC=60% rest analysis for 35C run (within-rest evolution + reference check).

Identifies the 2-hour rest after charging to SOC=60% (Step 9 in
LFP860_35degrees.002.txt cycler, 2026-05-31 15:07:50 -> 17:07:50),
loads the C-scans whose acquisition timestamp falls in that window,
and tests whether the cell ROI is changing more than the off-cell
reference regions.

The question: is the cell internal state still changing during this
2h rest, AFTER controlling for system/thermal artifacts?
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

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

BATCH_DIR = PROJ / "reports" / "experiments" / "longrun_cycling_35c_2026-05-30_19-47-06"
CHECKPOINT = BATCH_DIR / "checkpoint.json"
CYCLER = PROJ / "data" / "raw" / "cycler" / "LFP860_35degrees.002.txt"
ROI_MASK_PATH = PROJ / "reports" / "experiments" / "roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp" / "roi_mask.npy"
RAW_ROOT = PROJ / "data" / "raw" / "cscan"
OUT = PROJ / "reports" / "longrun_cycling_35c_2026-05-30_19-47-06_analysis" / "rest_at_SOC60"
OUT.mkdir(parents=True, exist_ok=True)

UNITS = {
    "amplitude": ("mV", 1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns", 1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

REST_START_LOCAL = datetime(2026, 5, 31, 15, 7, 50)
REST_END_LOCAL = datetime(2026, 5, 31, 17, 7, 50)


def _scan_ts_from_dir(name: str) -> datetime | None:
    # cscan_longrun_cycling_35c_2026-05-30_19-47-06_r042_2026-05-31_03-12-34
    m = re.search(r"_r\d{3}_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$", name)
    if not m:
        return None
    d, hh, mm, ss = m.groups()
    return datetime.strptime(f"{d} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")


def _xy_band(roi_mask, x_mm, y_mm, x_lo, x_hi, y_lo, y_hi, exclude_roi=True):
    cl = int(np.searchsorted(x_mm, x_lo)); ch = int(np.searchsorted(x_mm, x_hi))
    rl = int(np.searchsorted(y_mm, y_lo)); rh = int(np.searchsorted(y_mm, y_hi))
    m = np.zeros_like(roi_mask, dtype=bool)
    m[rl:rh, cl:ch] = True
    if exclude_roi:
        m &= ~roi_mask
    return m


def main():
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    completed = cp["completed"]

    # Find scans whose acquisition timestamp (from session_dir name) falls in rest window
    selected = []
    for c in completed:
        sess_dir = PROJ / Path(c["session_dir"].replace("\\", "/"))
        ts = _scan_ts_from_dir(sess_dir.name)
        if ts is None:
            continue
        if REST_START_LOCAL <= ts <= REST_END_LOCAL + timedelta(minutes=10):
            selected.append((c["run_idx"], c["scan_id"], ts, sess_dir))

    print(f"Found {len(selected)} scans in SOC=60% rest window ({REST_START_LOCAL} -> {REST_END_LOCAL})")
    if not selected:
        print("no scans found, aborting")
        return

    selected.sort(key=lambda x: x[2])
    for r in selected:
        print(f"  r{r[0]:03d}  {r[2].strftime('%H:%M:%S')}  {r[3].name}")

    # Load NPZs, stack
    arrs = {"amplitude": [], "tof": [], "energy": []}
    line_T = []
    x_mm = y_mm = None
    for _, _, _, sess_dir in selected:
        npz = next(sess_dir.glob("scan_*.npz"))
        d = np.load(npz)
        arrs["amplitude"].append(d["amplitude"].astype(np.float32))
        arrs["tof"].append(d["tof"].astype(np.float32))
        arrs["energy"].append(d["energy"].astype(np.float32))
        line_T.append(float(np.nanmean(d["line_temperature_mean_c"])))
        if x_mm is None:
            x_mm = d["x_mm"].astype(np.float32)
            y_mm = d["y_mm"].astype(np.float32)
    for k in arrs:
        arrs[k] = np.stack(arrs[k], axis=0)

    roi_mask = np.load(ROI_MASK_PATH)

    # Reference regions outside ROI
    refs = {
        "far-left fixture (X<10)":  _xy_band(roi_mask, x_mm, y_mm, 0, 10, 0, 72),
        "far-right fixture (X>70)": _xy_band(roi_mask, x_mm, y_mm, 70, 80, 0, 72),
        "far-top (Y<12)":            _xy_band(roi_mask, x_mm, y_mm, 0, 80, 0, 12),
        "far-bottom (Y>58)":         _xy_band(roi_mask, x_mm, y_mm, 0, 80, 58, 72),
    }

    t_min = np.array([(s[2] - selected[0][2]).total_seconds() / 60.0 for s in selected])

    # Compute ROI mean series + reference mean series
    def band_series(arr, mask):
        flat = arr.reshape(arr.shape[0], -1)
        m = mask.reshape(-1)
        return np.nanmean(flat[:, m], axis=1)

    line_T_arr = np.array(line_T)
    line_T_drift_mC = (line_T_arr - line_T_arr[0]) * 1000

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), constrained_layout=True, sharex=True)
    summary = [f"# SOC=60% rest analysis — 35°C run\n",
               f"Rest window: {REST_START_LOCAL} -> {REST_END_LOCAL}\n",
               f"n scans: {len(selected)}, span: {t_min[-1]:.1f} min\n",
               f"TC08 line-mean T: {line_T_arr[0]:.3f} -> {line_T_arr[-1]:.3f} (ΔT = {line_T_drift_mC[-1]:+.1f} mC)\n\n"]

    # Panel 0: cell-side T
    ax = axes[0]
    ax.plot(t_min, line_T_drift_mC, "o-", color="tab:red", lw=1.5)
    ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
    ax.set_ylabel("Cell-side ΔT [mC]\n(TC08 in-scan mean)")
    ax.set_title(f"35°C SOC=60% rest, n={len(selected)} scans over {t_min[-1]:.0f} min  ·  TC08 drift {line_T_drift_mC[-1]:+.1f} mC")

    for r, mod in enumerate(["amplitude", "tof", "energy"], start=1):
        unit, scale, sigma = UNITS[mod]
        ax = axes[r]
        # Cell ROI
        s_roi = band_series(arrs[mod], roi_mask) * scale
        d_roi = s_roi - s_roi[0]
        ax.plot(t_min, d_roi, "o-", color="tab:blue", lw=2, ms=6, label="CELL ROI")
        summary.append(f"## {mod} (sigma_ROI = {sigma:.3g} {unit})\n\n")
        summary.append(f"| region | Δ@end [{unit}] | end-z [σ_ROI] | rate ratio late/early |\n|---|---:|---:|---:|\n")

        cd_roi = np.abs(np.diff(s_roi))
        rate_roi = cd_roi[-3:].mean() / max(cd_roi[:3].mean(), 1e-9) if len(cd_roi) >= 6 else np.nan
        summary.append(f"| CELL ROI | {d_roi[-1]:+.3g} | {d_roi[-1]/sigma:+.1f} | {rate_roi:.2f} |\n")

        for label, m in refs.items():
            if m.sum() == 0:
                continue
            s_ref = band_series(arrs[mod], m) * scale
            d_ref = s_ref - s_ref[0]
            ax.plot(t_min, d_ref, "--", lw=1, marker="o", ms=4, label=label, alpha=0.7)
            cd_ref = np.abs(np.diff(s_ref))
            rate_ref = cd_ref[-3:].mean() / max(cd_ref[:3].mean(), 1e-9) if len(cd_ref) >= 6 else np.nan
            summary.append(f"| {label} | {d_ref[-1]:+.3g} | {d_ref[-1]/sigma:+.1f} | {rate_ref:.2f} |\n")
        summary.append("\n")

        ax.fill_between(t_min, -2*sigma, 2*sigma, color="gray", alpha=0.15, label="ROI 2σ noise floor")
        ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
        ax.set_ylabel(f"Δ{mod} [{unit}]")
        ax.legend(loc="best", fontsize=7, ncol=2)
    axes[-1].set_xlabel("rest time [min]")
    fig.savefig(OUT / "soc60_rest_evolution_with_reference.png", dpi=130)
    plt.close(fig)

    # Reference-subtracted view: cell ROI minus mean of 4 reference bands
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    summary.append("\n## Reference-subtracted (cell-specific signal)\n\n")
    summary.append("| modality | cell Δend | mean-ref Δend | (cell − mean-ref) | (cell − mean-ref) / σ_ROI |\n|---|---:|---:|---:|---:|\n")
    for ax, mod in zip(axes2, ["amplitude", "tof", "energy"]):
        unit, scale, sigma = UNITS[mod]
        s_roi = band_series(arrs[mod], roi_mask) * scale
        d_roi = s_roi - s_roi[0]
        # Mean reference drift
        ref_drifts = []
        for label, m in refs.items():
            if m.sum() == 0: continue
            s_ref = band_series(arrs[mod], m) * scale
            ref_drifts.append(s_ref - s_ref[0])
        d_ref_mean = np.mean(ref_drifts, axis=0)
        d_cell_specific = d_roi - d_ref_mean
        ax.plot(t_min, d_roi, "o-", color="tab:gray", lw=1, ms=5, label="cell ROI (raw)")
        ax.plot(t_min, d_ref_mean, "o-", color="tab:olive", lw=1, ms=4, label="mean(4 references)")
        ax.plot(t_min, d_cell_specific, "o-", color="tab:red", lw=2, ms=6, label="cell − refs (specific)")
        ax.fill_between(t_min, -2*sigma, 2*sigma, color="gray", alpha=0.15, label="2σ_ROI noise floor")
        ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
        ax.set_xlabel("rest time [min]")
        ax.set_ylabel(f"Δ{mod} [{unit}]")
        ax.set_title(f"{mod}  (σ_ROI={sigma:.3g} {unit})")
        ax.legend(fontsize=7)
        summary.append(f"| {mod} | {d_roi[-1]:+.3g} | {d_ref_mean[-1]:+.3g} | {d_cell_specific[-1]:+.3g} | {d_cell_specific[-1]/sigma:+.1f} |\n")
    fig2.suptitle("Reference-subtracted within-rest evolution at SOC=60% (35°C)", fontsize=12)
    fig2.savefig(OUT / "soc60_rest_reference_subtracted.png", dpi=130)
    plt.close(fig2)

    summary.append("\n## Verdict\n\n")
    summary.append("- If (cell − refs) |Δend| > 2σ_ROI → cell-specific change is real, not shared system drift\n")
    summary.append("- If close to 0 → most/all of the cell ROI change is explained by shared thermal/coupling drift\n")
    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"\nwrote outputs to {OUT}")


if __name__ == "__main__":
    main()
