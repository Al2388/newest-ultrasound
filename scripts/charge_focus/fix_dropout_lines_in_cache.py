"""Third-pass patch: catches rows with near-zero (noise-level) values that
the NaN-only and zero-only patches missed.

Criterion: a row is a 'dropout' if its absolute mean amp inside ROI is below
10% of the per-scan median row-mean amp. This catches lines where the
digitizer captured but signal was effectively absent (acquisition glitch
landing values at ~1e-5 V instead of ~1 V).

Patches the row in all three modalities together (amp/tof/energy) since the
dropout is per-line, not per-modality.

Inputs:  reports/longrun_cycling_35c_charge_focus/_cache/stack.npz
Output:  stack.npz overwritten;
         stack_pre_dropout_patch.npz backup;
         dropout_patch_log.md
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/"
                 "reports/longrun_cycling_35c_charge_focus/_cache")
STACK = CACHE_DIR / "stack.npz"
BACKUP = CACHE_DIR / "stack_pre_dropout_patch.npz"
LOG = CACHE_DIR.parent / "dropout_patch_log.md"


def find_dropout_rows(amp, roi, frac=0.10):
    """Return list of (scan_idx, row_idx) where row ROI-mean |amp| < frac * per-scan median."""
    out = []
    for s in range(amp.shape[0]):
        row_abs_means = []
        for r in range(amp.shape[1]):
            if roi[r].any():
                row_abs_means.append(float(np.mean(np.abs(amp[s, r, roi[r]]))))
            else:
                row_abs_means.append(np.nan)
        row_abs_means = np.array(row_abs_means)
        finite = np.isfinite(row_abs_means) & (row_abs_means > 0)
        if not finite.any():
            continue
        med = float(np.median(row_abs_means[finite]))
        threshold = frac * med
        for r in range(amp.shape[1]):
            if not np.isfinite(row_abs_means[r]):
                continue
            if row_abs_means[r] < threshold and roi[r].any():
                out.append((s, r, row_abs_means[r], med))
    return out


def fill_row_from_neighbours(arr3d, s, r, valid_mask):
    """Fill arr3d[s, r, :] cols where valid_mask is False, using Y-neighbour mean."""
    nrows = arr3d.shape[1]
    candidates = [r + dr for dr in [-1, 1, -2, 2, -3, 3] if 0 <= r + dr < nrows]
    filled = []
    for c in np.where(~valid_mask)[0]:
        vals = []
        for nb in candidates:
            v = arr3d[s, nb, c]
            if np.isfinite(v) and abs(v) > 1e-6:
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

    dropouts = find_dropout_rows(amp, roi, frac=0.10)
    print(f"\nfound {len(dropouts)} dropout (scan, row) pairs:")
    log_rows = []
    for s, r, row_mean, scan_med in dropouts:
        Y = float(data["y_mm"][r])
        print(f"  scan {s:>3d} row {r:>3d} Y={Y:.2f}mm  "
              f"row|amp|mean={row_mean:.4g}  scan median={scan_med:.4g}  "
              f"ratio={row_mean/scan_med:.3f}")
        # For each modality, treat the entire row of this scan as bad and fill
        valid_mask = np.zeros(amp.shape[2], dtype=bool)   # nothing valid -> fill all
        for mod_name, arr in [("amplitude", amp), ("tof", tof), ("energy", eng)]:
            filled = fill_row_from_neighbours(arr, s, r, valid_mask)
            log_rows.append({"modality": mod_name, "scan_idx": s, "row_idx": r,
                             "Y_mm": Y, "filled_cols": len(filled)})

    print(f"\nwriting patched stack")
    data["amplitude"] = amp; data["tof"] = tof; data["energy"] = eng
    np.savez_compressed(STACK, **data)

    log = ["# Dropout-line patch log (third pass)\n\n",
           "Catches rows whose absolute mean |amp| inside ROI is below 10% of\n",
           "the per-scan median row-mean -- typical signature of an acquisition\n",
           "glitch that captured values around the ADC zero (~1e-5 V) instead\n",
           "of the nominal ~1 V signal. These rows look like a thin BLACK\n",
           "horizontal stripe in turbo-cmap renders.\n\n",
           "Fill: Y-neighbour averaging (same as `nan_patch_log.md`).\n\n",
           "Backup: `_cache/stack_pre_dropout_patch.npz`.\n\n",
           "## Patched cells\n\n",
           "| modality | scan_idx | row | Y (mm) | cols filled |\n",
           "|:---|---:|---:|---:|---:|\n"]
    for r in log_rows:
        log.append(f"| {r['modality']} | {r['scan_idx']} | {r['row_idx']} | "
                   f"{r['Y_mm']:.2f} | {r['filled_cols']} |\n")
    LOG.write_text("".join(log), encoding="utf-8")
    print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
