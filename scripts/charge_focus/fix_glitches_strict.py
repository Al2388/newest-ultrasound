"""Strict per-frame glitch patch: only fixes lines where ALL THREE modalities
simultaneously go anomalous (true acquisition dropout signature), not just
amplitude alone.

Detection (per scan, per row):
  z_amp = |row_mean_amp - Y_neighbour_median(amp)| / sigma_ROI(amp)
  z_tof = |row_mean_tof - Y_neighbour_median(tof)| / sigma_ROI(tof)
  z_eng = |row_mean_eng - Y_neighbour_median(eng)| / sigma_ROI(eng)
  flag if MIN(z_amp, z_tof, z_eng) > 8.0

Reasoning: a real cell physical edge feature shows up in amp + maybe energy,
but tof rarely deviates strongly there because tof depends on geometry not
amplitude. A genuine acquisition dropout takes all three out together because
the digitizer captured the wrong samples for the whole line.

Plus row must NOT be persistent (flagged in >= 20% of scans).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
STACK = CACHE_DIR / "stack.npz"
BACKUP = CACHE_DIR / "stack_pre_strict_patch.npz"
LOG = CACHE_DIR.parent / "strict_glitch_patch_log.md"

Z_THRESH_MIN_ALL_MODS = 8.0   # MIN across 3 modalities must exceed this
PERSISTENT_FRAC = 0.20

SIGMAS = {
    "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"] / 1000.0,
    "tof":       NOISE_FLOOR["tof_sigma_roi_ns"] / 1000.0,
    "energy":    NOISE_FLOOR["energy_sigma_roi"],
}


def row_roi_means(arr3d, roi):
    n_s, n_r, _ = arr3d.shape
    out = np.full((n_s, n_r), np.nan)
    for r in range(n_r):
        if roi[r].any():
            out[:, r] = np.nanmean(arr3d[:, r, roi[r]], axis=1)
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
    arrs = {m: data[m] for m in ["amplitude", "tof", "energy"]}
    n_s, n_r, _ = arrs["amplitude"].shape

    print(f"\ncomputing per-row ROI means ...")
    rmean = {m: row_roi_means(arrs[m], roi) for m in arrs}

    print(f"flagging rows where MIN z across 3 modalities > {Z_THRESH_MIN_ALL_MODS} sigma_ROI:")
    flagged = np.zeros((n_s, n_r), dtype=bool)
    z_stats = {m: np.zeros((n_s, n_r)) for m in arrs}
    for s in range(n_s):
        for r in range(n_r):
            if not np.isfinite(rmean["amplitude"][s, r]):
                continue
            zs = {}
            for mod in arrs:
                nbrs = [rmean[mod][s, rr] for rr in range(max(0, r-3), min(n_r, r+4))
                        if rr != r and np.isfinite(rmean[mod][s, rr])]
                if len(nbrs) < 3:
                    zs[mod] = 0.0
                    continue
                med = float(np.median(nbrs))
                zs[mod] = abs(rmean[mod][s, r] - med) / SIGMAS[mod]
                z_stats[mod][s, r] = zs[mod]
            min_z = min(zs.values())
            if min_z > Z_THRESH_MIN_ALL_MODS:
                flagged[s, r] = True

    flag_count = flagged.sum(axis=0)
    persistent_rows = set(np.where(flag_count >= PERSISTENT_FRAC * n_s)[0].tolist())
    if persistent_rows:
        print(f"\npersistent rows (flagged in >= {PERSISTENT_FRAC*100:.0f}% of scans, NOT patched):")
        for r in sorted(persistent_rows):
            print(f"  row {r} (Y={float(data['y_mm'][r]):.2f}mm) flagged in {flag_count[r]}/{n_s} scans")

    to_patch = [(s, r) for s in range(n_s) for r in range(n_r)
                if flagged[s, r] and r not in persistent_rows]

    print(f"\n{len(to_patch)} (scan, row) cells to patch:")
    log_rows = []
    for s, r in to_patch:
        za = z_stats["amplitude"][s, r]
        zt = z_stats["tof"][s, r]
        ze = z_stats["energy"][s, r]
        Y = float(data["y_mm"][r])
        print(f"  scan {s:>3d} row {r:>3d} Y={Y:.2f}mm  z_amp={za:.1f}  z_tof={zt:.1f}  z_eng={ze:.1f}")
        log_rows.append({"scan_idx": s, "row_idx": r, "Y_mm": Y,
                          "z_amp": za, "z_tof": zt, "z_eng": ze})

    # Patch
    n_filled = {m: 0 for m in arrs}
    for s, r in to_patch:
        nbrs = []
        for dr in [-1, 1, -2, 2, -3, 3]:
            rr = r + dr
            if 0 <= rr < n_r and rr not in persistent_rows:
                nbrs.append(rr)
            if len(nbrs) >= 4:
                break
        for mod_name, arr in arrs.items():
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

    print(f"\nfilled cells: " + ", ".join(f"{m}={v}" for m, v in n_filled.items()))

    for m, arr in arrs.items():
        data[m] = arr
    np.savez_compressed(STACK, **data)
    print(f"wrote patched stack -> {STACK}")

    md = [f"# Strict per-frame glitch patch\n\n",
          f"Criterion: a row is patched only if it shows simultaneous z > "
          f"{Z_THRESH_MIN_ALL_MODS} sigma_ROI deviation from Y-neighbour median\n",
          f"in ALL THREE modalities (amp + tof + energy). This requires a true\n",
          f"acquisition dropout signature (all sensors lost), not a single-channel\n",
          f"physical feature.\n\n",
          f"Persistent rows (flagged in >= {PERSISTENT_FRAC*100:.0f}% of scans): excluded.\n\n",
          f"Backup: `_cache/stack_pre_strict_patch.npz`.\n\n",
          f"## Patched cells ({len(log_rows)})\n\n",
          f"| scan | row | Y (mm) | z_amp | z_tof | z_eng |\n",
          f"|---:|---:|---:|---:|---:|---:|\n"]
    for r in sorted(log_rows, key=lambda x: -max(x['z_amp'], x['z_tof'], x['z_eng'])):
        md.append(f"| {r['scan_idx']} | {r['row_idx']} | {r['Y_mm']:.2f} | "
                  f"{r['z_amp']:.1f} | {r['z_tof']:.1f} | {r['z_eng']:.1f} |\n")
    LOG.write_text("".join(md), encoding="utf-8")
    print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
