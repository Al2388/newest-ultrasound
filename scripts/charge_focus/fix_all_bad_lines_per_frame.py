"""Fourth-pass patch: per-frame per-row scan for any acquisition glitch.

Strategy:
  1. For every (scan, row) pair, compute the row's ROI-mean for each modality
     and its deviation from the Y-neighbour median.
  2. Build a frequency map: how many of the 93 scans have row r flagged?
     - If a row is flagged in >= 30% of scans -> it's a persistent physical
       feature (cell edge, tab area), DO NOT patch.
     - If it's flagged in < 30% -> it's a per-scan acquisition glitch, PATCH.
  3. Use a tight z-threshold (> 5 sigma_ROI relative to local Y-window median)
     since we already verified noise floor.

Inputs:  reports/longrun_cycling_35c_charge_focus/_cache/stack.npz
Output:  stack.npz overwritten, stack_pre_perframe_patch.npz backup,
         perframe_patch_log.md
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
STACK = CACHE_DIR / "stack.npz"
BACKUP = CACHE_DIR / "stack_pre_perframe_patch.npz"
LOG = CACHE_DIR.parent / "perframe_patch_log.md"

# z-threshold for "flagged" - tight to catch any visible stripe
Z_THRESH = 5.0
# A row that's flagged in >= this fraction of scans is treated as persistent
# physical feature (cell edge etc.) -- never patch
PERSISTENT_FRAC = 0.30

SIGMAS = {
    "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"] / 1000.0,
    "tof":       NOISE_FLOOR["tof_sigma_roi_ns"] / 1000.0,
    "energy":    NOISE_FLOOR["energy_sigma_roi"],
}


def row_roi_means(arr3d, roi):
    """Return (n_scans, n_rows) of per-row ROI mean."""
    n_s, n_r, _ = arr3d.shape
    out = np.full((n_s, n_r), np.nan)
    for r in range(n_r):
        if roi[r].any():
            out[:, r] = np.nanmean(arr3d[:, r, roi[r]], axis=1)
    return out


def local_median(row_means_per_scan, r, win=3):
    """Median of row r's neighbours in the same scan, within window."""
    n_r = row_means_per_scan.shape[0]
    idx = [rr for rr in range(max(0, r-win), min(n_r, r+win+1)) if rr != r]
    vals = row_means_per_scan[idx]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) >= 3 else np.nan


def main():
    print(f"loading {STACK}")
    d = np.load(STACK)
    keys = list(d.keys())
    data = {k: d[k].copy() for k in keys}
    if not BACKUP.exists():
        print(f"saving backup -> {BACKUP}")
        np.savez_compressed(BACKUP, **{k: d[k] for k in keys})

    roi = data["roi_mask"]
    arrs = {m: data[m] for m in ["amplitude", "tof", "energy"]}
    n_s = arrs["amplitude"].shape[0]
    n_r = arrs["amplitude"].shape[1]

    print(f"\ncomputing per-row ROI means for {n_s} scans x {n_r} rows ...")
    rmean = {m: row_roi_means(arrs[m], roi) for m in arrs}

    print(f"flagging per-(scan, row) anomalies (z > {Z_THRESH} sigma_ROI in any modality):")
    flagged = np.zeros((n_s, n_r), dtype=bool)
    z_stats = np.zeros((n_s, n_r))
    for s in range(n_s):
        for r in range(n_r):
            if not np.isfinite(rmean["amplitude"][s, r]):
                continue
            max_z = 0.0
            for mod in ["amplitude", "tof", "energy"]:
                local_neighbours = [rmean[mod][s, rr] for rr in range(max(0, r-3), min(n_r, r+4))
                                    if rr != r and np.isfinite(rmean[mod][s, rr])]
                if len(local_neighbours) < 3:
                    continue
                med = float(np.median(local_neighbours))
                z = abs(rmean[mod][s, r] - med) / SIGMAS[mod]
                max_z = max(max_z, z)
            z_stats[s, r] = max_z
            if max_z > Z_THRESH:
                flagged[s, r] = True

    # Per-row, what fraction of scans flag it?
    flag_count = flagged.sum(axis=0)   # shape (n_r,)
    persistent_rows = np.where(flag_count >= PERSISTENT_FRAC * n_s)[0]
    print(f"\npersistent physical-feature rows (flagged in >={PERSISTENT_FRAC*100:.0f}% of scans):")
    for r in persistent_rows:
        print(f"  row {r:>3d} (Y={float(data['y_mm'][r]):.2f}mm)  flagged in {flag_count[r]}/{n_s} scans")
    print(f"  -> {len(persistent_rows)} rows will NOT be patched (treated as physical features)")

    # Per-scan glitches: cells where flagged is True AND row is NOT persistent
    persistent_set = set(persistent_rows.tolist())
    to_patch = []
    for s in range(n_s):
        for r in range(n_r):
            if flagged[s, r] and r not in persistent_set:
                to_patch.append((s, r, float(z_stats[s, r])))

    print(f"\n{len(to_patch)} per-scan glitches to patch:")
    log_rows = []
    for s, r, z in sorted(to_patch, key=lambda t: -t[2])[:30]:
        print(f"  scan {s:>3d} row {r:>3d} (Y={float(data['y_mm'][r]):.2f}mm)  max z = {z:.1f}")
    if len(to_patch) > 30:
        print(f"  ... and {len(to_patch)-30} more")

    # Patch by Y-neighbour averaging in EACH modality
    n_filled = {m: 0 for m in arrs}
    for s, r, z in to_patch:
        for mod_name, arr in arrs.items():
            nbrs = []
            for dr in [-1, 1, -2, 2, -3, 3]:
                rr = r + dr
                if 0 <= rr < n_r:
                    nbrs.append(rr)
                if len(nbrs) >= 4:
                    break
            for c in range(arr.shape[2]):
                vals = []
                for rr in nbrs:
                    v = arr[s, rr, c]
                    if np.isfinite(v) and abs(v) > 1e-6:
                        vals.append(v)
                    if len(vals) >= 2:
                        break
                if vals:
                    arr[s, r, c] = float(np.mean(vals))
                    n_filled[mod_name] += 1
        log_rows.append({"scan_idx": s, "row_idx": r,
                         "Y_mm": float(data["y_mm"][r]), "max_z_sigma_roi": z})

    print(f"\nfilled cells: " + ", ".join(f"{m}={v}" for m, v in n_filled.items()))

    for m, arr in arrs.items():
        data[m] = arr
    print(f"\nwriting patched stack")
    np.savez_compressed(STACK, **data)

    md = ["# Per-frame per-row glitch patch (fourth pass)\n\n",
          f"For every (scan, row) pair, compared the row's ROI-mean to the\n",
          f"median of its 6 Y-neighbour rows in the same scan, separately for\n",
          f"each modality. Flagged if the largest deviation across the three\n",
          f"modalities exceeded **{Z_THRESH} sigma_ROI**.\n\n",
          f"Filter: rows flagged in >= {PERSISTENT_FRAC*100:.0f}% of all 93 scans\n",
          f"are treated as persistent physical features (cell edge bands at\n",
          f"Y in [16, 55] mm) and excluded from patching. This pass therefore\n",
          f"only fills *per-scan acquisition glitches*.\n\n",
          f"Backup: `_cache/stack_pre_perframe_patch.npz`.\n\n",
          f"## Persistent rows (NOT patched)\n\n",
          "| row | Y (mm) | flagged in N scans |\n|---:|---:|---:|\n"]
    for r in persistent_rows:
        md.append(f"| {r} | {float(data['y_mm'][r]):.2f} | {int(flag_count[r])} |\n")
    md.append(f"\n## Per-scan glitches patched ({len(log_rows)} cells)\n\n")
    md.append(f"| scan_idx | row | Y (mm) | max z (sigma_ROI) |\n|---:|---:|---:|---:|\n")
    for r in sorted(log_rows, key=lambda x: -x["max_z_sigma_roi"]):
        md.append(f"| {r['scan_idx']} | {r['row_idx']} | {r['Y_mm']:.2f} | {r['max_z_sigma_roi']:.1f} |\n")
    LOG.write_text("".join(md), encoding="utf-8")
    print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
