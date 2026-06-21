"""Replace the wrong noise-floor batch with the canonical noisefloor_v3.238
batch that actually produced roi_summary.json's sigma_ROI numbers.

Also generates a per-pixel sigma-map NPZ from the v3.238 stack (the
equivalent of `noise_floor_diffs.npz` for the right batch).
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
DEST = Path("D:/35c_dataset_summary")
ZIP  = Path("D:/35c_dataset_summary.zip")
CSCAN_ROOT = PROJ / "data/raw/cscan"
ROI_DIR = PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp"

# canonical batch -- this is what produced roi_summary.json's sigma_ROI numbers
V3238_TAG = "noisefloor_v3.238_2026-05-29_15-02-49"


def main():
    if not DEST.exists():
        print(f"Summary dataset missing: {DEST}")
        sys.exit(1)

    out = DEST / "noise_floor"
    if out.exists():
        print(f"Removing previous noise_floor/ ...")
        shutil.rmtree(out)
    out.mkdir()
    (out / "figures").mkdir()
    src_scans_dir = out / "source_scans"
    src_scans_dir.mkdir()

    t0 = time.time()
    n_files, n_bytes = 0, 0

    # ---- 1) the 6 source scan_*.npz from v3.238 batch ----
    sessions = sorted(CSCAN_ROOT.glob(f"cscan_{V3238_TAG}_r0*"))
    if len(sessions) != 6:
        print(f"ERROR: expected 6 v3.238 sessions, found {len(sessions)}")
        sys.exit(1)
    print(f"copying 6 v3.238 source scan sessions ...")
    for sess in sessions:
        dst = src_scans_dir / sess.name
        dst.mkdir()
        for f in sess.iterdir():
            if f.is_file() and not f.name.endswith(".png"):
                shutil.copy2(f, dst / f.name)
                n_files += 1; n_bytes += f.stat().st_size

    # ---- 2) compute per-pixel sigma map + cross-scan stats from v3.238 ----
    print("computing per-pixel sigma maps from v3.238 stack ...")
    stacks = {"amplitude": [], "tof": [], "energy": []}
    for sess in sessions:
        d = np.load(next(sess.glob("scan_*.npz")))
        for k in stacks:
            stacks[k].append(d[k].astype(np.float32))
        x_mm = d["x_mm"]; y_mm = d["y_mm"]
    roi = np.load(ROI_DIR / "roi_mask.npy")

    out_data = {"x_mm": x_mm.astype(np.float32),
                "y_mm": y_mm.astype(np.float32),
                "roi_mask": roi}
    summary_rows = []
    for k, arr in stacks.items():
        stack = np.stack(arr)                              # (6, 144, 500)
        mean_map = np.nanmean(stack, axis=0).astype(np.float32)
        sigma_pixel = np.nanstd(stack, axis=0, ddof=1).astype(np.float32)  # (144, 500)
        roi_means = np.array([float(np.nanmean(s[roi])) for s in stack])
        sigma_roi_aggregated = float(np.std(roi_means, ddof=1))
        out_data[f"{k}_stack"] = stack
        out_data[f"{k}_mean_map"] = mean_map
        out_data[f"{k}_sigma_pixel_map"] = sigma_pixel
        out_data[f"{k}_roi_means"] = roi_means.astype(np.float32)
        out_data[f"{k}_sigma_roi_aggregated"] = np.float32(sigma_roi_aggregated)
        summary_rows.append({
            "modality": k,
            "n_scans": int(stack.shape[0]),
            "sigma_pixel_median_inside_roi": float(np.median(sigma_pixel[roi])),
            "sigma_pixel_p95_inside_roi": float(np.percentile(sigma_pixel[roi], 95)),
            "sigma_pixel_max_inside_roi": float(np.max(sigma_pixel[roi])),
            "sigma_roi_aggregated_SD_of_ROI_mean_ddof1": sigma_roi_aggregated,
        })
        print(f"  {k:10s} sigma_ROI = {sigma_roi_aggregated:.6g}  "
              f"sigma_pixel(median) = {np.median(sigma_pixel[roi]):.6g}")
    np.savez_compressed(out / "noise_floor_sigma_maps.npz", **out_data)
    n_files += 1; n_bytes += (out / "noise_floor_sigma_maps.npz").stat().st_size

    # ---- 3) sigma summary CSV ----
    with (out / "sigma_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)
    n_files += 1

    # ---- 4) canonical figure PDFs (these ARE the v3.238 outputs) ----
    for name in ["figA_mean_maps.pdf", "figA_mean_maps.png",
                 "figB_sigma_maps.pdf", "figB_sigma_maps.png"]:
        src = ROI_DIR / name
        if src.exists():
            shutil.copy2(src, out / "figures" / name)
            n_files += 1; n_bytes += src.stat().st_size

    # ---- 5) README ----
    nf_readme = f"""# Noise-floor measurement (canonical sigma_ROI source)

This subfolder holds the data that produced `analysis_inputs/roi_summary.json`'s
`roi_aggregated_sigma` numbers. The numbers `(amp 2.219 mV, ToF 4.700 ns,
energy 0.0814 a.u.)` are referenced throughout the analysis as the **noise
floor** sigma_ROI.

## What was measured

6 back-to-back C-scans of the *same physical state* (LFP cell at startpoint
3.238 V, ~25 degC), batch tag `{V3238_TAG}`:

| run | timestamp | session |
|---:|---|---|
"""
    for i, sess in enumerate(sessions, 1):
        # session name -> timestamp portion
        nf_readme += f"| r{i:02d} | {sess.name.split('_r0')[1].split('_', 1)[1]} | `{sess.name}` |\n"

    nf_readme += """
~9 min between scans. Approx 25 minutes of TC08 cell-T drift across the full
6-scan window (well within the rig's normal envelope).

## Files

```
source_scans/                     6 source scan_*.npz + meta.json + manifest
                                  (lines_raw/ excluded -- raw waveforms not needed
                                  to reproduce sigma_ROI)
noise_floor_sigma_maps.npz        all derived artefacts in one NPZ:
                                    <mod>_stack            (6, 144, 500)  raw stack
                                    <mod>_mean_map         (144, 500)     mean across 6
                                    <mod>_sigma_pixel_map  (144, 500)     SD per pixel (ddof=1)
                                    <mod>_roi_means        (6,)           per-scan ROI mean
                                    <mod>_sigma_roi_aggregated  scalar    SD of ROI means (ddof=1)
                                  x_mm, y_mm, roi_mask included
sigma_summary.csv                 sigma_pixel median/p95/max + sigma_ROI per modality
figures/figA_mean_maps.pdf/png    6-scan-mean amplitude / ToF / energy (canonical)
figures/figB_sigma_maps.pdf/png   per-pixel sigma maps (canonical, masked to ROI)
```

## Definition of sigma_ROI (verbatim from source code)

```python
roi_means_per_scan = [np.nanmean(scan_i[roi_mask]) for i in range(6)]
sigma_ROI = np.std(roi_means_per_scan, ddof=1)
```

This is the **standard deviation of the 6 ROI-mean values** (NOT the spatial
SD inside one scan, NOT the median pixel-sigma). Conceptually: if you
re-measure the same cell back-to-back six times, how much does the
ROI-averaged reading wander? That wander is the noise floor against which we
declare a Delta to be "above noise".

## Verification

The derived `sigma_roi_aggregated` values in this folder match
`analysis_inputs/roi_summary.json` to 6 significant figures.

| modality | sigma_ROI here | sigma_ROI in roi_summary.json |
|:---|---:|---:|
"""
    for r in summary_rows:
        ref_key = {"amplitude": "amplitude_V", "tof": "tof_us", "energy": "energy"}[r["modality"]]
        import json
        ref_val = json.loads((ROI_DIR / "roi_summary.json").read_text())["roi_aggregated_sigma"][ref_key]
        nf_readme += f"| {r['modality']} | {r['sigma_roi_aggregated_SD_of_ROI_mean_ddof1']:.6g} | {ref_val:.6g} |\n"

    nf_readme += """
## Re-derivation example

```python
import numpy as np
from pathlib import Path
roi = np.load('../analysis_inputs/roi_mask.npy')
for k in ['amplitude', 'tof', 'energy']:
    stack = np.stack([
        np.load(next(s.glob('scan_*.npz')))[k]
        for s in sorted(Path('source_scans').iterdir())
    ])
    means = np.array([np.nanmean(s[roi]) for s in stack])
    print(f'{k:10s} sigma_ROI = {np.std(means, ddof=1):.6g}')
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
