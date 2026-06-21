"""Second-pass patch: also fill rows where amp == 0 (and tof == 0) across the
ROI cols. These are dropout rows that survived the NaN-only patch because the
values are explicitly 0 rather than NaN.

Conservative: only patch rows where BOTH amplitude and tof are <= 1e-6 across
>= 50% of the ROI columns (clear acquisition dropout signature; real cell
features never have both amp and tof simultaneously zero).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/"
                 "reports/longrun_cycling_35c_charge_focus/_cache")
STACK = CACHE_DIR / "stack.npz"
BACKUP = CACHE_DIR / "stack_pre_zero_patch.npz"
LOG = CACHE_DIR.parent / "zero_patch_log.md"


def find_zero_dropout_rows(amp, tof, roi):
    """Return list of (scan_idx, row_idx) where amp and tof are simultaneously
    ~zero across most of the ROI columns of that row."""
    out = []
    for s in range(amp.shape[0]):
        for r in range(amp.shape[1]):
            roi_cols = np.where(roi[r])[0]
            if len(roi_cols) < 5:
                continue
            amp_zero = (np.abs(amp[s, r, roi_cols]) < 1e-6).sum()
            tof_zero = (np.abs(tof[s, r, roi_cols]) < 1e-6).sum()
            if amp_zero >= 0.5 * len(roi_cols) and tof_zero >= 0.5 * len(roi_cols):
                out.append((s, r))
    return out


def fill_row_from_neighbours(arr3d, s, r):
    nrows = arr3d.shape[1]
    candidates = [r + dr for dr in [-1, 1, -2, 2, -3, 3]
                  if 0 <= r + dr < nrows]
    filled = []
    for c in range(arr3d.shape[2]):
        if np.abs(arr3d[s, r, c]) >= 1e-6:
            continue   # not a zero, leave alone
        vals = []
        for nb in candidates:
            v = arr3d[s, nb, c]
            if np.isfinite(v) and np.abs(v) > 1e-6:
                vals.append(v)
            if len(vals) >= 2:
                break
        if vals:
            arr3d[s, r, c] = float(np.mean(vals))
            filled.append(c)
    return filled


def main():
    print(f"loading {STACK}")
    d = np.load(STACK)
    keys = list(d.keys())
    data = {k: d[k].copy() for k in keys}

    if not BACKUP.exists():
        print(f"saving backup -> {BACKUP}")
        np.savez_compressed(BACKUP, **{k: d[k] for k in keys})

    roi = data["roi_mask"]
    amp = data["amplitude"]; tof = data["tof"]; eng = data["energy"]

    dropouts = find_zero_dropout_rows(amp, tof, roi)
    print(f"\nfound {len(dropouts)} (scan, row) ZERO-DROPOUT pairs:")
    log_rows = []
    for s, r in dropouts:
        print(f"  scan {s:>3d}  row {r:>3d}  (Y={float(data['y_mm'][r]):.2f} mm)")
        for mod, arr in [("amplitude", amp), ("tof", tof), ("energy", eng)]:
            n_zero_before = int((np.abs(arr[s, r, :]) < 1e-6).sum())
            filled = fill_row_from_neighbours(arr, s, r)
            n_zero_after = int((np.abs(arr[s, r, :]) < 1e-6).sum())
            log_rows.append({
                "modality": mod, "scan_idx": s, "row_idx": r,
                "Y_mm": float(data["y_mm"][r]),
                "zeros_before": n_zero_before,
                "filled": len(filled),
                "zeros_after": n_zero_after,
            })
        data["amplitude"] = amp; data["tof"] = tof; data["energy"] = eng

    print(f"\nwriting patched stack")
    np.savez_compressed(STACK, **data)

    log = ["# Zero-line patch log (second pass)\n\n",
           "5 NaN-row dropouts were already patched (see `nan_patch_log.md`).\n",
           "This second pass catches dropouts that were saved as **explicit 0**\n",
           "instead of NaN -- visible as thin BLACK stripes in GIF/PNG renders.\n\n",
           "Detection criterion: both amplitude AND tof <= 1e-6 across >=50% of\n",
           "the ROI columns of a row.\n\n",
           "Fill: same Y-neighbour averaging as `nan_patch_log.md`.\n\n",
           "Backup: `_cache/stack_pre_zero_patch.npz`.\n\n",
           "## Patched cells\n\n",
           "| modality | scan_idx | row | Y (mm) | zeros before | filled | zeros after |\n",
           "|:---|---:|---:|---:|---:|---:|---:|\n"]
    for r in log_rows:
        log.append(f"| {r['modality']} | {r['scan_idx']} | {r['row_idx']} | "
                   f"{r['Y_mm']:.2f} | {r['zeros_before']} | {r['filled']} | "
                   f"{r['zeros_after']} |\n")
    LOG.write_text("".join(log), encoding="utf-8")
    print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
