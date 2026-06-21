"""Add the 6-repeat noise-floor source data to D:/35c_dataset_summary/.

Adds a self-contained `noise_floor/` subfolder that lets the user re-derive
sigma_ROI from scratch (the canonical noise-floor measurement on which
analysis_inputs/roi_summary.json is based).
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
import zipfile
from pathlib import Path

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
DEST = Path("D:/35c_dataset_summary")
ZIP  = Path("D:/35c_dataset_summary.zip")

NF_DIR = PROJ / "reports/experiments/baseline_repeat_6scan_2026-05-28_startpoint"
ROI_DIR = PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp"
CSCAN_ROOT = PROJ / "data/raw/cscan"


def main():
    if not DEST.exists():
        print(f"Summary dataset missing: {DEST}")
        sys.exit(1)

    out = DEST / "noise_floor"
    if out.exists():
        print(f"Removing previous {out} ...")
        shutil.rmtree(out)
    out.mkdir()

    t0 = time.time()
    n_files, n_bytes = 0, 0

    # ---- 1) the per-pixel noise-floor NPZ (key derived artifact) ----
    src = NF_DIR / "noise_floor_diffs.npz"
    shutil.copy2(src, out / "noise_floor_diffs.npz")
    n_files += 1; n_bytes += src.stat().st_size

    # ---- 2) text reports ----
    for name in ["NOISE_FLOOR_REPORT.md",
                 "noise_floor_summary.csv",
                 "noise_floor_span_summary.csv"]:
        src = NF_DIR / name
        if src.exists():
            shutil.copy2(src, out / name)
            n_files += 1; n_bytes += src.stat().st_size

    # ---- 3) canonical figure PDFs (mean maps + sigma maps) ----
    figs_dir = out / "figures"
    figs_dir.mkdir()
    for name in ["figA_mean_maps.pdf", "figA_mean_maps.png",
                 "figB_sigma_maps.pdf", "figB_sigma_maps.png"]:
        src = ROI_DIR / name
        if src.exists():
            shutil.copy2(src, figs_dir / name)
            n_files += 1; n_bytes += src.stat().st_size

    # ---- 4) the 6 source scan summary NPZs (re-derive from scratch) ----
    src_scans_dir = out / "source_scans"
    src_scans_dir.mkdir()
    src_sessions = sorted(CSCAN_ROOT.glob("cscan_baseline_repeat_6scan_startpoint_*"))
    for sess in src_sessions:
        rel = sess.name
        dst = src_scans_dir / rel
        dst.mkdir()
        for f in sess.iterdir():
            if f.is_file() and not f.name.endswith(".png"):
                shutil.copy2(f, dst / f.name)
                n_files += 1; n_bytes += f.stat().st_size
    print(f"copied {len(src_sessions)} source scan sessions")

    # ---- 5) noise_floor README explaining contents ----
    nf_readme = """# Noise-floor measurement (6 repeats at startpoint, ~25 degC)

This subfolder contains the data used to derive `sigma_ROI` -- the canonical
per-modality noise floor referenced throughout the analysis.

## What was measured

6 back-to-back C-scans of the *same physical state*:
- Voltage: 3.232 V (LFP cell at startpoint)
- Temperature: ~24.6-24.9 degC (TC08, stable within ~330 mC across all 6)
- ~9 min between scans
- Source: `data/raw/cscan/cscan_baseline_repeat_6scan_startpoint_2026-05-28_*`

Run 4 (`scan_2026-05-28_18-52-24`) is the **baseline**; the other 5 are
compared against it.

## Files

```
noise_floor_diffs.npz     - key derived artifact, per-pixel:
                            amplitude_diff_stack    shape (5,144,500) - run_i minus baseline
                            amplitude_noise_floor_min/max/span   shape (144,500)
                            tof_diff_stack          shape (5,144,500)
                            tof_noise_floor_*       shape (144,500)
                            energy_diff_stack       shape (5,144,500)
                            energy_noise_floor_*    shape (144,500)
NOISE_FLOOR_REPORT.md     - per-scan summary text
noise_floor_summary.csv   - per-scan ROI/global stats (mean, std, p95, range)
noise_floor_span_summary.csv - median/p95/max span per modality
figures/figA_mean_maps    - 6-scan-mean amplitude / tof / energy maps (canonical)
figures/figB_sigma_maps   - per-pixel sigma_pixel maps from the 6 repeats (canonical)
source_scans/             - the 6 raw scan_*.npz (summary level, lines_raw excluded)
                            so the calculation can be redone from scratch
```

## How sigma_ROI was computed

For each modality:
1. Stack the 6 scan_*.npz outputs -> (6, 144, 500) per modality
2. Per-pixel sigma = stdev across the 6 scans                    -> (144, 500) pixel-sigma
3. Apply the canonical 50 x 40 mm ROI mask (24,648 pixels)       -> ROI-only pixel-sigma
4. `sigma_ROI` = standard deviation of the 6 per-scan ROI MEANS
   (not the median pixel-sigma, not the mean -- the cross-scan SD of the
   spatially-averaged ROI mean)

That definition gives:
| modality | sigma_ROI |
|:---|---:|
| amplitude | 2.219 mV (0.002219 V) |
| ToF       | 4.700 ns (0.004700 us) |
| energy    | 0.0814 a.u. |

Same numbers appear in `../analysis_inputs/roi_summary.json` under
`roi_aggregated_sigma`. The two files agree by construction; this folder
keeps the per-pixel sigma maps and the source NPZs so the value is
reproducible.

## Quick re-derivation

```python
import numpy as np
from pathlib import Path
roi = np.load('../analysis_inputs/roi_mask.npy')      # (144, 500) bool
stacks = {'amplitude': [], 'tof': [], 'energy': []}
for sess in sorted(Path('source_scans').iterdir()):
    d = np.load(next(sess.glob('scan_*.npz')))
    for k in stacks: stacks[k].append(d[k])
for k, arr in stacks.items():
    stack = np.stack(arr)                              # (6, 144, 500)
    roi_means = np.array([np.nanmean(s[roi]) for s in stack])
    sigma_roi = float(np.std(roi_means))
    print(f'{k:10s} sigma_ROI = {sigma_roi:.6g}')
```
"""
    (out / "README.md").write_text(nf_readme, encoding="utf-8")
    n_files += 1

    print(f"\nadded {n_files} files, {n_bytes/1e6:.1f} MB in {time.time()-t0:.1f}s")

    # ---- rebuild MANIFEST + zip ----
    print("\nrebuilding MANIFEST and zip ...")
    manifest_rows = []
    for f in DEST.rglob("*"):
        if not f.is_file() or f.name == "MANIFEST.csv":
            continue
        manifest_rows.append({"path": f.relative_to(DEST).as_posix(),
                              "size_bytes": f.stat().st_size})
    with (DEST / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "size_bytes"])
        w.writeheader(); w.writerows(manifest_rows)

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in DEST.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(DEST))

    total_dir = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    zip_size = ZIP.stat().st_size
    print(f"  folder: {DEST}            ({total_dir/1e6:.1f} MB, {len(manifest_rows)} files)")
    print(f"  zip:    {ZIP}     ({zip_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
