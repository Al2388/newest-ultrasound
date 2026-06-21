"""Side-by-side: GIF-style frame (from patched cache) vs original session PNG
(from webapp at scan time, pre-patch). Pick one of the 5 originally-patched
scans so any 'missing line' artefact should be obvious in the source PNG and
absent in the cache frame.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import shutil

PROJ = Path('d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main')
d = np.load(PROJ / 'reports/longrun_cycling_35c_charge_focus/_cache/stack.npz')
meta = pd.read_csv(PROJ / 'reports/longrun_cycling_35c_charge_focus/_cache/meta.csv')
roi = d['roi_mask']

# scan 60 had row 82 patched
SCAN_IDX = 60
sess = PROJ / meta.iloc[SCAN_IDX]['session_dir'].replace('\\','/')
print(f"comparing scan {SCAN_IDX}: {sess.name}")

arr = d['amplitude'][SCAN_IDX]
x_mm = d['x_mm']; y_mm = d['y_mm']
extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]
vmin = float(np.nanpercentile(d['amplitude'][:, roi], 5))
vmax = float(np.nanpercentile(d['amplitude'][:, roi], 95))

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)

# left = patched cache rendering, same as GIF
ax = axes[0]
cmap = plt.colormaps['turbo'].copy(); cmap.set_bad('white')
ax.imshow(arr, extent=extent, aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax,
          interpolation='nearest', origin='upper')
ax.set_title(f"patched cache rendering (same as GIF) -- scan {SCAN_IDX} amp")
ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')

# right = the original session PNG that webapp wrote at scan time
ax = axes[1]
src_png = sess / 'scan_amp.png'
if src_png.exists():
    img = mpimg.imread(src_png)
    ax.imshow(img, aspect='auto')
    ax.set_title(f"ORIGINAL session PNG (pre-patch) -- {src_png.name}")
    ax.axis('off')
else:
    ax.text(0.5, 0.5, f"{src_png} not found", ha='center')

out = PROJ / 'reports/longrun_cycling_35c_charge_focus/_stripe_check_scan60.png'
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"wrote {out}")
