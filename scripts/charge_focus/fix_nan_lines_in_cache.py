"""Patch the 5 dropped scan-line NaN gaps in the cached 35C stack by
spatially interpolating from neighbour Y rows of the same scan.

Why this is OK:
  - raw NPZ shows these lines were truncated (only ~5-20% of the X-range
    was actually captured by the scanner), so the data is genuinely missing
  - they form thin (~1 row, 0.5 mm) gaps inside otherwise smooth fields
  - spatial interpolation from Y-row neighbours (also captured in the SAME scan,
    so same SOC/T/coupling state) is the correct cosmetic fix
  - we DO NOT modify the raw lines_raw NPZ files; only the cached stack.
    The raw record stays pristine; a NaN-aware analysis can still detect them.

Inputs:
  reports/longrun_cycling_35c_charge_focus/_cache/stack.npz
Outputs:
  stack.npz                 patched (overwritten)
  stack_pre_patch.npz       backup of the original
  nan_patch_log.md          which (scan, row, col-range) were patched
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/"
                 "reports/longrun_cycling_35c_charge_focus/_cache")
STACK = CACHE_DIR / "stack.npz"
BACKUP = CACHE_DIR / "stack_pre_patch.npz"
LOG = CACHE_DIR.parent / "nan_patch_log.md"


def find_nan_gaps(arr3d):
    """Return list of (scan_idx, row_idx) where any NaN appears in cols 0..nx."""
    gaps = []
    for s in range(arr3d.shape[0]):
        for r in range(arr3d.shape[1]):
            if np.any(np.isnan(arr3d[s, r, :])):
                gaps.append((s, r))
    return gaps


def fill_row_from_neighbours(arr3d, scan_idx, row_idx):
    """Fill NaN cols in arr3d[scan_idx, row_idx, :] using mean of row-1, row+1
    (and row-2, row+2 as fall-back for double-defect rows).
    Returns the list of column indices that were filled.
    """
    nrows = arr3d.shape[1]
    nan_mask = np.isnan(arr3d[scan_idx, row_idx, :])
    if not nan_mask.any():
        return []
    # Build neighbour candidates ordered by closeness
    candidates = []
    for dr in [-1, 1, -2, 2, -3, 3]:
        nb = row_idx + dr
        if 0 <= nb < nrows:
            candidates.append(nb)
    # For each NaN col, take mean of neighbour values that are finite
    filled = []
    for c in np.where(nan_mask)[0]:
        vals = []
        for nb in candidates:
            v = arr3d[scan_idx, nb, c]
            if np.isfinite(v):
                vals.append(v)
            if len(vals) >= 2:
                break
        if vals:
            arr3d[scan_idx, row_idx, c] = float(np.mean(vals))
            filled.append(int(c))
    return filled


def main():
    print(f"loading {STACK} ...")
    d = np.load(STACK)
    keys = list(d.keys())
    data = {k: d[k].copy() for k in keys}
    print(f"keys: {keys}")

    if not BACKUP.exists():
        print(f"saving original backup -> {BACKUP}")
        np.savez_compressed(BACKUP, **{k: d[k] for k in keys})
    else:
        print(f"backup already exists at {BACKUP}, leaving alone")

    log_rows = []
    for mod in ["amplitude", "tof", "energy"]:
        arr = data[mod]
        gaps = find_nan_gaps(arr)
        print(f"\n{mod}: {len(gaps)} (scan, row) NaN gaps")
        for s, r in gaps:
            n_nan = int(np.isnan(arr[s, r, :]).sum())
            filled = fill_row_from_neighbours(arr, s, r)
            n_left = int(np.isnan(arr[s, r, :]).sum())
            log_rows.append({
                "modality": mod, "scan_idx": s, "row_idx": r,
                "n_nan_before": n_nan, "n_filled": len(filled), "n_nan_after": n_left,
            })
            print(f"  scan {s:>3d}  row {r:>3d}  (Y={float(data['y_mm'][r]):.2f} mm)  "
                  f"NaN before={n_nan}, filled={len(filled)}, NaN after={n_left}")
        data[mod] = arr

    # Save patched stack
    print(f"\nwriting patched stack -> {STACK}")
    np.savez_compressed(STACK, **data)

    # Log
    log = ["# NaN-gap patch log (35C charge-focus stack)\n\n",
           "5 individual scan-lines were truncated during acquisition (raw\n",
           "lines_raw shows ~5-20% of the line captured before stop). For\n",
           "downstream visualization we filled the NaN columns from Y-neighbour\n",
           "rows of the **same scan** (same SOC/T/coupling state). The raw NPZ\n",
           "files under `data/raw/cscan/.../lines_raw/` are unchanged.\n\n",
           "Backup of unpatched cache: `_cache/stack_pre_patch.npz`.\n\n",
           "## Patched cells\n\n",
           "| modality | scan_idx | row | NaN before | filled | NaN after |\n",
           "|:---|---:|---:|---:|---:|---:|\n"]
    for r in log_rows:
        log.append(f"| {r['modality']} | {r['scan_idx']} | {r['row_idx']} | "
                   f"{r['n_nan_before']} | {r['n_filled']} | {r['n_nan_after']} |\n")
    LOG.write_text("".join(log), encoding="utf-8")
    print(f"wrote {LOG}")

    # Verify
    print(f"\nverification:")
    d2 = np.load(STACK)
    for mod in ["amplitude", "tof", "energy"]:
        remaining = int(np.isnan(d2[mod]).sum())
        print(f"  {mod}: {remaining} NaN remaining in (93, 144, 500) stack")


if __name__ == "__main__":
    main()
