"""Diagnose source of 'missing lines' in animation GIF frames."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJ = Path('d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main')
d = np.load(PROJ / 'reports/longrun_cycling_35c_charge_focus/_cache/stack.npz')
meta = pd.read_csv(PROJ / 'reports/longrun_cycling_35c_charge_focus/_cache/meta.csv')
roi = d['roi_mask']

# 1) Render a frame and look at it as the GIF does
arr = d['amplitude']
vmin_g = float(np.nanpercentile(arr[:, roi], 5))
vmax_g = float(np.nanpercentile(arr[:, roi], 95))
print(f"GIF global vmin/vmax (across all 93 scans, ROI only): {vmin_g:.4f} / {vmax_g:.4f}")

# 2) For each scan, count rows that are >>vmax_g or <<vmin_g (would clip to extremes)
print()
print(f"=== rows that clip to vmin/vmax extremes inside ROI ===")
print(f"{'scan':>4} {'rows clipped low':>17} {'rows clipped high':>18} {'orig PNG missing rows?':>25}")
for s in range(arr.shape[0]):
    scan = arr[s]
    # Look only at ROI pixels per row
    n_low = 0; n_hi = 0
    for r in range(scan.shape[0]):
        if not roi[r].any(): continue
        vals = scan[r, roi[r]]
        if np.all(vals <= vmin_g): n_low += 1
        if np.all(vals >= vmax_g): n_hi += 1
    if n_low > 1 or n_hi > 1:
        sess = PROJ / meta.iloc[s]['session_dir'].replace('\\', '/')
        scan_png = sess / 'scan_amp.png'
        png_ok = "ok" if scan_png.exists() else "missing"
        print(f"{s:>4} {n_low:>17} {n_hi:>18}   ({png_ok})")

# 3) Compare patched cache rendering for the 5 originally-patched scans:
#    do the patched rows still look slightly different to eyes?
print()
print(f"=== patched (scan, row) cells -- check if value differs from local median ===")
patched = [(29, 3), (58, 66), (60, 82), (79, 102), (80, 76)]
for s, r in patched:
    vals = arr[s, r, roi[r]]
    neighbour_med = np.median([arr[s, r+dr, roi[r+dr]].mean()
                                for dr in [-2,-1,1,2] if roi[r+dr].any()])
    my_mean = float(np.mean(vals))
    print(f"  scan {s:>3d} row {r:>3d} (Y={float(d['y_mm'][r]):.2f}mm):  "
          f"patched_mean={my_mean:.4f},  neighbour_avg={neighbour_med:.4f},  "
          f"diff={(my_mean-neighbour_med):.4f}")
