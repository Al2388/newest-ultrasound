"""Detect 'bad-looking' rows in every frame of every modality by checking row
ROI-mean against the local Y-neighbour median for the SAME scan. Anything
deviating > 1 sigma_ROI gets flagged for inspection.

Output: per-(scan, row) anomaly list + 6 example dump frames.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path('d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main')
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR

d = np.load(PROJ / 'reports/longrun_cycling_35c_charge_focus/_cache/stack.npz')
meta = pd.read_csv(PROJ / 'reports/longrun_cycling_35c_charge_focus/_cache/meta.csv')
roi = d['roi_mask']

SIGMAS = {
    "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"] / 1000.0,   # back to V
    "tof":       NOISE_FLOOR["tof_sigma_roi_ns"] / 1000.0,   # back to us
    "energy":    NOISE_FLOOR["energy_sigma_roi"],
}

bad = []
for mod in ['amplitude', 'tof', 'energy']:
    arr = d[mod]
    sigma = SIGMAS[mod]
    for s in range(arr.shape[0]):
        # per-row ROI mean inside roi mask
        row_means = []
        for r in range(arr.shape[1]):
            if roi[r].any():
                row_means.append(float(np.nanmean(arr[s, r, roi[r]])))
            else:
                row_means.append(np.nan)
        row_means = np.array(row_means)
        # local Y-window median (using ±3 rows, excluding self)
        for r in range(arr.shape[1]):
            if not np.isfinite(row_means[r]): continue
            window = []
            for dr in [-3, -2, -1, 1, 2, 3]:
                rr = r + dr
                if 0 <= rr < arr.shape[1] and np.isfinite(row_means[rr]):
                    window.append(row_means[rr])
            if len(window) < 3: continue
            med = np.median(window)
            std = np.std(window)
            local_anom = abs(row_means[r] - med)
            # flag if deviates > 3 sigma_ROI from local neighbours
            if local_anom > 3 * sigma:
                z_loc_std = local_anom / std if std > 0 else float("inf")
                z_sigma_roi = local_anom / sigma
                bad.append((mod, s, r, row_means[r], med, std,
                            z_loc_std, z_sigma_roi))

print(f"Found {len(bad)} suspicious (mod, scan, row) entries deviating > 3 sigma_ROI from Y-neighbour median\n")
# Group by modality
df = pd.DataFrame(bad, columns=["mod", "scan", "row", "val", "med", "local_std",
                                  "z_local", "z_sigma_roi"])
for mod, sub in df.groupby("mod"):
    print(f"--- {mod} ---  {len(sub)} flagged")
    show = sub.sort_values("z_sigma_roi", ascending=False).head(15)
    for _, r in show.iterrows():
        print(f"  scan {int(r['scan']):>3} row {int(r['row']):>3} "
              f"(Y={d['y_mm'][int(r['row'])]:>5.2f}mm)  "
              f"val={r['val']:.4f}  neighbour_med={r['med']:.4f}  "
              f"z(sigma_ROI)={r['z_sigma_roi']:+.1f}")

# Dump a few example frames
out_dir = PROJ / 'reports/longrun_cycling_35c_charge_focus/_bad_lines_check'
out_dir.mkdir(exist_ok=True)
if len(df):
    # 6 most-flagged unique scans across all modalities
    worst = (df.assign(absz=df['z_sigma_roi'].abs())
               .sort_values('absz', ascending=False)
               .drop_duplicates(['scan']).head(6))
    for _, r in worst.iterrows():
        s = int(r['scan']); mod = r['mod']
        arr = d[mod][s]
        vmin = float(np.nanpercentile(d[mod][:, roi], 5))
        vmax = float(np.nanpercentile(d[mod][:, roi], 95))
        x_mm = d['x_mm']; y_mm = d['y_mm']
        extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = plt.colormaps['turbo'].copy(); cm.set_bad('white')
        ax.imshow(arr, extent=extent, aspect='equal', cmap=cm,
                  vmin=vmin, vmax=vmax, interpolation='nearest', origin='upper')
        ax.set_title(f"scan {s} {mod}  -- z(sigma_ROI)={r['z_sigma_roi']:+.1f}")
        fig.savefig(out_dir / f"scan{s:03d}_{mod}_z{r['z_sigma_roi']:+.1f}.png", dpi=120)
        plt.close(fig)
    print(f"\nwrote {len(worst)} example frames to {out_dir.name}/")
