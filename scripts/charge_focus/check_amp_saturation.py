"""Check for amplitude saturation in the 35C charge-focus batch.

The HS5 digitizer was set to 5 V range. Saturation = raw A-scan hitting +/- 5 V
rail, which propagates into the extracted amplitude feature as values at or
near 5.0 V.

Two checks:
  (1) C-scan summary amplitude: count and locate pixels >= saturation_limit
  (2) Raw waveform spot-check: pick a few sessions and look at peak |sample|
"""
from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CACHE = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"

# HS5 5 V range means raw samples saturate at +-5 V. The peak amplitude feature
# is computed from the envelope, so saturation shows up as values near 5 V.
SAT_LIMIT_V = 4.95   # conservative: anything >= this is at/near saturation
HARD_RAIL_V = 5.0


def main():
    d = np.load(CACHE / "stack.npz")
    meta = pd.read_csv(CACHE / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    amp = d["amplitude"]      # (93, 144, 500) V
    roi = d["roi_mask"]
    x_mm = d["x_mm"]; y_mm = d["y_mm"]

    finite = amp[np.isfinite(amp)]
    print(f"=== C-scan summary amplitude (V) ===")
    print(f"  93 scans x (144, 500) = {amp.size:,} pixels")
    print(f"  finite pixels:   {finite.size:,}")
    print(f"  global max:      {finite.max():.4f} V")
    print(f"  global p99.9:    {np.percentile(finite, 99.9):.4f} V")
    print(f"  global p99:      {np.percentile(finite, 99):.4f} V")
    print(f"  global p95:      {np.percentile(finite, 95):.4f} V")

    n_at_rail = int(np.sum(np.abs(amp) >= HARD_RAIL_V - 1e-6))
    n_near_sat = int(np.sum(np.abs(amp) >= SAT_LIMIT_V))
    print(f"\n=== saturation counts ===")
    print(f"  pixels >= {HARD_RAIL_V} V (hard rail):  {n_at_rail:,}  "
          f"({100*n_at_rail/amp.size:.4f}%)")
    print(f"  pixels >= {SAT_LIMIT_V} V (near sat):   {n_near_sat:,}  "
          f"({100*n_near_sat/amp.size:.4f}%)")

    # Locate them: ROI vs off-ROI
    in_roi = roi[np.newaxis, :, :] & (amp >= SAT_LIMIT_V)
    out_roi = (~roi[np.newaxis, :, :]) & (amp >= SAT_LIMIT_V)
    n_in = int(in_roi.sum())
    n_out = int(out_roi.sum())
    print(f"\n=== where are the near-saturation pixels? ===")
    print(f"  inside ROI:   {n_in:,}  ({100*n_in/max(n_near_sat,1):.1f}%)")
    print(f"  outside ROI:  {n_out:,}  ({100*n_out/max(n_near_sat,1):.1f}%)")

    if n_in > 0:
        # Which scans/rows/cols?
        scans_with_sat = sorted(set(np.where(np.any(np.any(in_roi, axis=2), axis=1))[0].tolist()))
        print(f"\n  ROI saturation scans: {scans_with_sat[:20]}")
        # row/col distribution
        any_per_scan = in_roi.sum(axis=(1, 2))
        print(f"  pixels per scan distribution (top 10):")
        worst = np.argsort(any_per_scan)[::-1][:10]
        for s in worst:
            if any_per_scan[s] == 0: break
            print(f"    scan {s:>3d}  n_saturated_ROI_px = {int(any_per_scan[s])}")

    if n_out > 0:
        print(f"\n  off-ROI saturation comes from fixture/tank-wall reflections")
        # Roughly where in the FOV
        ys, xs = np.where(np.any(out_roi, axis=0))
        if len(xs):
            print(f"  X range of off-ROI saturated px: [{x_mm[xs.min()]:.1f}, {x_mm[xs.max()]:.1f}] mm")
            print(f"  Y range:                          [{y_mm[ys.min()]:.1f}, {y_mm[ys.max()]:.1f}] mm")

    # Spot-check raw waveforms in 3 sessions to make sure raw ADC isn't clipping
    print(f"\n=== raw waveform spot-check (3 sessions) ===")
    for scan_idx in [0, 46, 92]:
        sess = PROJ / meta.iloc[scan_idx]["session_dir"].replace("\\", "/")
        line0 = sess / "lines_raw" / "line_0072.npz"  # centre line
        if not line0.exists():
            continue
        d_line = np.load(line0)
        wf = d_line["waveforms"]    # (17175, 500)
        max_abs = float(np.max(np.abs(wf)))
        n_clip = int(np.sum(np.abs(wf) >= HARD_RAIL_V - 0.01))
        pct_clip = 100 * n_clip / wf.size
        print(f"  scan {scan_idx:>3d}  centre line {line0.parent.name}:")
        print(f"    waveforms shape={wf.shape}, max |sample| = {max_abs:.4f} V, "
              f"clipped samples = {n_clip:,} ({pct_clip:.4f}%)")


if __name__ == "__main__":
    main()
