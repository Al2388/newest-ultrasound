"""One-shot: compute absolute round-trip ToF + apparent one-way path L."""
import numpy as np
import pandas as pd
import glob
from pathlib import Path

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
roi = np.load(PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp/roi_mask.npy")
gate_start_us = 25.0

# 1. 25 C reference
ref_sessions = sorted(glob.glob(str(PROJ / "data/raw/cscan/cscan_noisefloor_v3.238_2026-05-29_15-02-49_r0*")))
tof_roi_rel = []
for sess in ref_sessions:
    d = np.load(next(Path(sess).glob("scan_*.npz")))
    tof_roi_rel.append(float(np.median(d["tof"][roi])))
tof_rel_25c = float(np.mean(tof_roi_rel))
tof_abs_25c = gate_start_us + tof_rel_25c
c_25 = 989.0
L_apparent_25 = tof_abs_25c * c_25 / 2 / 1000  # mm one-way

print(f"=== 25 C cell-ROI (n={len(ref_sessions)}) ===")
print(f"  ToF in NPZ (relative to gate start):  {tof_rel_25c:.4f} us")
print(f"  + gate_start_us                    :  {gate_start_us:.1f}  us")
print(f"  = absolute round-trip ToF          :  {tof_abs_25c:.4f} us")
print(f"  c(25 C)=989 m/s, /2                :  one-way apparent path = {L_apparent_25:.3f} mm")

# 2. 35.9 C subset
meta = pd.read_csv(PROJ / "reports/longrun_cycling_35c_charge_focus/_cache/meta.csv")
tof_roi_36 = []
for _, row in meta.iterrows():
    sess = PROJ / row["session_dir"].replace("\\", "/")
    d = np.load(next(sess.glob("scan_*.npz")))
    tof_roi_36.append(float(np.nanmedian(d["tof"][roi])))
tof_rel_36 = float(np.mean(tof_roi_36))
tof_abs_36 = gate_start_us + tof_rel_36
c_36 = 936.0
L_apparent_36 = tof_abs_36 * c_36 / 2 / 1000

print(f"\n=== 35.9 C cell-ROI (n=93) ===")
print(f"  ToF in NPZ (relative):  {tof_rel_36:.4f} us")
print(f"  absolute round-trip  :  {tof_abs_36:.4f} us")
print(f"  c(36 C)=936 m/s / 2  :  one-way apparent path = {L_apparent_36:.3f} mm")
print(f"\nApparent one-way path change 25 -> 36 C: {(L_apparent_36 - L_apparent_25)*1000:+.1f} um")

# 3. Off-cell (pure oil paths)
def off_band(x, y, xlo, xhi, ylo, yhi):
    cl = int(np.searchsorted(x, xlo)); ch = int(np.searchsorted(x, xhi))
    rl = int(np.searchsorted(y, ylo)); rh = int(np.searchsorted(y, yhi))
    m = np.zeros_like(roi, dtype=bool)
    m[rl:rh, cl:ch] = True
    return m & ~roi

d0 = np.load(next(Path(ref_sessions[0]).glob("scan_*.npz")))
print(f"\n=== off-cell @ 25 C (pure oil round-trip to fixture) ===")
for name, args in [("far-left", (0, 10, 0, 72)), ("far-right", (70, 80, 0, 72)),
                   ("far-top", (0, 80, 0, 12)), ("far-bot", (0, 80, 58, 72))]:
    mask = off_band(d0["x_mm"], d0["y_mm"], *args)
    vals = [float(np.median(np.load(next(Path(s).glob("scan_*.npz")))["tof"][mask]))
            for s in ref_sessions]
    rel = float(np.mean(vals))
    abs_us = gate_start_us + rel
    L = abs_us * c_25 / 2 / 1000
    print(f"  {name:<10s} ToF abs = {abs_us:7.4f} us -> one-way = {L:6.3f} mm")
