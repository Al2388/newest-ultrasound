"""Pragmatic patch: catches rows that render visually as 'dark stripes' in
turbo-cmap C-scan figures.

Criterion (per scan, per row):
  row_mean_amp_inside_roi  <  threshold_frac * scan_typical_row_mean_amp

Where scan_typical = median of ROI-mean across that scan's non-edge rows.

The threshold is intentionally loose (0.7 = 70%) to catch any visible dark
stripe. Persistent rows (always-low cell-edge bands at Y=16, Y=55) are excluded
via the persistent-row filter.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

CACHE_DIR = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/"
                 "reports/longrun_cycling_35c_charge_focus/_cache")
STACK = CACHE_DIR / "stack.npz"
BACKUP = CACHE_DIR / "stack_pre_dark_patch.npz"
LOG = CACHE_DIR.parent / "dark_line_patch_log.md"

THRESHOLD_FRAC = 0.70   # row mean must be >= 70% of scan median to pass
PERSISTENT_FRAC = 0.25  # rows flagged in >= 25% of scans are physical features


def per_scan_row_means(amp, roi):
    n_s, n_r, _ = amp.shape
    out = np.full((n_s, n_r), np.nan)
    for r in range(n_r):
        if roi[r].any():
            out[:, r] = np.nanmean(amp[:, r, roi[r]], axis=1)
    return out


def main():
    print(f"loading {STACK}")
    d = np.load(STACK)
    keys = list(d.keys())
    data = {k: d[k].copy() for k in keys}
    if not BACKUP.exists():
        np.savez_compressed(BACKUP, **{k: d[k] for k in keys})
        print(f"saved backup -> {BACKUP}")

    roi = data["roi_mask"]
    amp = data["amplitude"]; tof = data["tof"]; eng = data["energy"]
    n_s, n_r, _ = amp.shape

    rmean_amp = per_scan_row_means(amp, roi)

    # First pass: flag rows where row_mean / scan_median < THRESHOLD_FRAC
    flagged = np.zeros((n_s, n_r), dtype=bool)
    for s in range(n_s):
        scan_med = np.nanmedian(rmean_amp[s, np.isfinite(rmean_amp[s])])
        if not np.isfinite(scan_med):
            continue
        thr = THRESHOLD_FRAC * scan_med
        for r in range(n_r):
            if np.isfinite(rmean_amp[s, r]) and rmean_amp[s, r] < thr:
                flagged[s, r] = True

    flag_count = flagged.sum(axis=0)
    persistent = set(np.where(flag_count >= PERSISTENT_FRAC * n_s)[0].tolist())
    if persistent:
        print(f"\npersistent rows (flagged in >= {PERSISTENT_FRAC*100:.0f}% scans, NOT patched):")
        for r in sorted(persistent):
            print(f"  row {r} (Y={float(data['y_mm'][r]):.2f}mm) flagged in {flag_count[r]}/{n_s}")

    to_patch = [(s, r) for s in range(n_s) for r in range(n_r)
                if flagged[s, r] and r not in persistent]

    print(f"\nflagged {len(to_patch)} per-scan dark-line cells:")
    for s, r in sorted(to_patch, key=lambda x: (x[0], x[1])):
        Y = float(data["y_mm"][r])
        scan_med = np.nanmedian(rmean_amp[s, np.isfinite(rmean_amp[s])])
        ratio = rmean_amp[s, r] / scan_med
        print(f"  scan {s:>3d} row {r:>3d} Y={Y:>5.2f}mm  "
              f"row_amp={rmean_amp[s,r]:.4f}  scan_med={scan_med:.4f}  ratio={ratio:.2f}")

    # Patch
    log_rows = []
    n_filled = {"amplitude": 0, "tof": 0, "energy": 0}
    for s, r in to_patch:
        nbrs = []
        for dr in [-1, 1, -2, 2, -3, 3]:
            rr = r + dr
            if 0 <= rr < n_r and rr not in persistent:
                nbrs.append(rr)
            if len(nbrs) >= 4:
                break
        for mod_name, arr in [("amplitude", amp), ("tof", tof), ("energy", eng)]:
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
        log_rows.append({
            "scan_idx": s, "row_idx": r, "Y_mm": float(data["y_mm"][r])
        })

    print(f"\nfilled cells: " + ", ".join(f"{m}={v}" for m, v in n_filled.items()))

    data["amplitude"] = amp; data["tof"] = tof; data["energy"] = eng
    np.savez_compressed(STACK, **data)
    print(f"wrote patched stack -> {STACK}")

    md = [f"# Dark-line pragmatic patch\n\n",
          f"Criterion (per scan, per row): `row_mean_amp_inside_roi < "
          f"{THRESHOLD_FRAC*100:.0f}% * scan_median_row_mean_amp`.\n\n",
          f"Catches any line that renders visibly darker than typical inside\n",
          f"the same scan. Rows that are below threshold in >= "
          f"{PERSISTENT_FRAC*100:.0f}% of scans are treated as persistent\n",
          f"physical features (cell edges) and excluded.\n\n",
          f"Backup: `_cache/stack_pre_dark_patch.npz`.\n\n",
          f"## Patched cells ({len(log_rows)})\n\n",
          f"| scan | row | Y (mm) |\n|---:|---:|---:|\n"]
    for r in log_rows:
        md.append(f"| {r['scan_idx']} | {r['row_idx']} | {r['Y_mm']:.2f} |\n")
    LOG.write_text("".join(md), encoding="utf-8")
    print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
