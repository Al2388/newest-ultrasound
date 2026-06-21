"""Assemble a self-contained 35C charge-focus dataset folder.

Output structure:
  D:/35c_dataset/
    README.md                   - dataset description + provenance
    MANIFEST.csv                - file index (path, size, sha256)
    ultrasound/                 - 93 scan sessions (incl. lines_raw raw waveforms)
    cycler/                     - Maccor cycler text export
    temperature/                - watchdog log (safety thermocouple) + per-line T
    analysis_inputs/            - meta.csv, ROI mask, noise floor JSON
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
DEST = Path("D:/35c_dataset")

META_CSV = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache/meta.csv"
CYCLER   = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
WATCHDOG = PROJ / "data/raw/temperature/watchdog_2026-05-30_19-46-40"
ROI_MASK = PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp/roi_mask.npy"
ROI_JSON = PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp/roi_summary.json"
BATCH_CHECKPOINT = PROJ / "reports/experiments/longrun_cycling_35c_2026-05-30_19-47-06/checkpoint.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_session(src: Path, dst: Path) -> tuple[int, int]:
    """Copy one scan session dir; return (n_files, total_bytes)."""
    n, b = 0, 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
        b += f.stat().st_size
    return n, b


def main():
    if DEST.exists():
        print(f"DEST already exists: {DEST}")
        print("Refusing to overwrite. Remove or rename it first if you want a fresh package.")
        sys.exit(1)
    DEST.mkdir(parents=True)
    (DEST / "ultrasound").mkdir()
    (DEST / "cycler").mkdir()
    (DEST / "temperature").mkdir()
    (DEST / "analysis_inputs").mkdir()

    meta = pd.read_csv(META_CSV)
    print(f"loaded {len(meta)} sessions from meta.csv\n")

    # ============ ultrasound sessions ============
    t0 = time.time()
    total_bytes = 0
    total_files = 0
    print("=== copying 93 scan sessions ===")
    for i, row in meta.iterrows():
        src = PROJ / row["session_dir"].replace("\\", "/")
        dst = DEST / "ultrasound" / src.name
        n, b = _copy_session(src, dst)
        total_files += n; total_bytes += b
        elapsed = time.time() - t0
        rate = total_bytes / max(elapsed, 0.1) / 1e6
        print(f"  [{i+1:>2}/93] {src.name}  ({n} files, {b/1e6:.0f} MB)  "
              f"total {total_bytes/1e9:.1f} GB  rate {rate:.0f} MB/s")
    print(f"\nultrasound: {total_files} files, {total_bytes/1e9:.2f} GB, "
          f"elapsed {time.time()-t0:.0f}s")

    # ============ cycler ============
    print("\n=== copying cycler ===")
    shutil.copy2(CYCLER, DEST / "cycler" / CYCLER.name)
    print(f"  {CYCLER.name}")

    # ============ temperature ============
    print("\n=== copying watchdog logs ===")
    for f in WATCHDOG.iterdir():
        if f.is_file():
            shutil.copy2(f, DEST / "temperature" / f.name)
            print(f"  watchdog/{f.name}")

    # Consolidated per-line T extracted from NPZ (matches per-line analysis)
    print("\n=== consolidating per-line temperature record ===")
    rows = []
    for _, row in meta.iterrows():
        sess = PROJ / row["session_dir"].replace("\\", "/")
        npz = next(sess.glob("scan_*.npz"))
        d = np.load(npz)
        for u, t in zip(d["line_unix_center_s"], d["line_temperature_mean_c"]):
            rows.append({"unix_s": float(u), "cell_c": float(t),
                         "scan_id": row["scan_id"], "step": int(row["step"])})
    per_line = pd.DataFrame(rows).sort_values("unix_s").reset_index(drop=True)
    per_line["iso"] = pd.to_datetime(per_line["unix_s"], unit="s", utc=True).dt.tz_convert(None)
    per_line.to_csv(DEST / "temperature" / "per_line_temperature.csv", index=False)
    print(f"  per_line_temperature.csv  ({len(per_line)} rows, "
          f"{per_line['cell_c'].std()*1000:.1f} mC sigma over full window)")

    # ============ analysis inputs ============
    print("\n=== copying analysis inputs ===")
    shutil.copy2(META_CSV, DEST / "analysis_inputs" / "meta.csv")
    shutil.copy2(ROI_MASK, DEST / "analysis_inputs" / "roi_mask.npy")
    shutil.copy2(ROI_JSON, DEST / "analysis_inputs" / "roi_summary.json")
    shutil.copy2(BATCH_CHECKPOINT, DEST / "analysis_inputs" / "batch_checkpoint.json")
    print("  meta.csv, roi_mask.npy, roi_summary.json, batch_checkpoint.json")

    # ============ MANIFEST.csv ============
    print("\n=== building MANIFEST.csv (with sha256 for top-level files) ===")
    manifest_rows = []
    for f in DEST.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(DEST).as_posix()
        size = f.stat().st_size
        # only sha256 the small top-level inputs (NPZ files are too many)
        do_hash = ("ultrasound" not in rel) or rel.endswith("_meta.json")
        sha = _sha256(f) if do_hash and size < 50_000_000 else ""
        manifest_rows.append({"path": rel, "size_bytes": size, "sha256": sha})
    with (DEST / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        w.writeheader(); w.writerows(manifest_rows)
    print(f"  {len(manifest_rows)} entries")

    # ============ README.md ============
    readme = f"""# 35 degC LFP cycling C-scan dataset (charge-focus subset)

Self-contained snapshot of the 35 degC long-run LFP860 cycling experiment,
restricted to the SOC 20% -> 80% charge-focus window (Steps 5-11: 4 rest
plateaus + 3 constant-current charge segments).

- **Cell**: LFP860 pouch, nominal 860 mAh
- **Temperature setpoint**: cell ~ 35.6 degC (water bath, Marlin S35 chiller)
- **Time window**: 2026-05-31 07:07:49 -> 21:07:51 (local), 14.0 h
- **Scans**: 93 C-scans, ~9 min apart, 80 x 72 mm FOV at 0.5 mm pitch
- **Cycler protocol**: Maccor schedule on LFP860 channel,
   2 h rest at each SOC plateau between CC charge segments
- **Source batch**: `longrun_cycling_35c_2026-05-30_19-47-06`
  (charge-focus subset of the full 170-scan run)

## Folder layout

```
ultrasound/         93 scan sessions
  cscan_..._rNNN_<ts>/
    scan_<ts>.npz           - C-scan amplitude/tof/energy + per-line metadata
    scan_<ts>_meta.json
    lines_raw/              - 144 per-line raw A-scan NPZ (raw waveforms)
    session_manifest.json
    scan_amp.png            - quick-look C-scan PNGs
    scan_tof.png
    scan_eng.png
cycler/
  LFP860_35degrees.002.txt  - Maccor text export (tab-sep, skiprows=6)
temperature/
  safety_temperature.csv    - watchdog cell TC at 10 s resolution (cell_c column)
  safety_events.log         - watchdog alarms (none triggered in this window)
  per_line_temperature.csv  - consolidated cell TC at ~3.5 s resolution,
                              extracted from each NPZ's line_temperature_mean_c
                              (recommended source for cell-T statistics)
analysis_inputs/
  meta.csv                  - per-scan metadata (step, SOC, V, I, T)
  roi_mask.npy              - canonical 50x40 mm cell ROI (24648 px)
  roi_summary.json          - ROI geometry + sigma_ROI noise floor (N=6 repeats)
  batch_checkpoint.json     - original batch run checkpoint
MANIFEST.csv                - all file paths + sizes + sha256 (smaller files)
README.md                   - this file
```

## Key reproducibility numbers

- **sigma_ROI** (from N=6 startpoint repeats, source `roi_summary.json`):
  - amplitude: 2.22 mV
  - ToF:       4.70 ns
  - energy:    0.0814 a.u.
- **Canonical ROI**: 50.02 x 39.78 mm, center (39.59, 36.00) mm,
  bounds X in [14.59, 64.45], Y in [16.11, 55.38] mm, 24648 pixels.
- **Cell temperature stability over the 14 h window**
  (from `per_line_temperature.csv`, ~3.5 s resolution):
  span = 354 mC, sigma_T = 66 mC, mean = 35.876 degC.
- **Tab orientation**: both tabs on the X-large edge of the ROI.

## Per-plateau time windows (local time)

| Step | Tag    | SOC | Start             | End               |
|---:|---|---:|---|---|
|  5 | rest   | 20% | 2026-05-31 07:07:49 | 2026-05-31 09:07:49 |
|  6 | charge |  -  | 2026-05-31 09:07:49 | 2026-05-31 11:07:49 |
|  7 | rest   | 40% | 2026-05-31 11:07:49 | 2026-05-31 13:07:49 |
|  8 | charge |  -  | 2026-05-31 13:07:49 | 2026-05-31 15:07:50 |
|  9 | rest   | 60% | 2026-05-31 15:07:50 | 2026-05-31 17:07:50 |
| 10 | charge |  -  | 2026-05-31 17:07:50 | 2026-05-31 19:07:51 |
| 11 | rest   | 80% | 2026-05-31 19:07:51 | 2026-05-31 21:07:51 |

## How to load one scan

```python
import numpy as np
from pathlib import Path
sess = Path('ultrasound') / next(Path('ultrasound').iterdir()).name
d = np.load(next(sess.glob('scan_*.npz')))
# C-scan maps (shape 144 x ~160):
amp = d['amplitude']          # mV
tof = d['tof']                # us
eng = d['energy']             # a.u.
# Per-line metadata (length 144):
t   = d['line_temperature_mean_c']     # cell temperature per line, degC
ux  = d['line_unix_center_s']          # line center time, unix s
# Raw waveforms (one NPZ per line, 144 lines):
raw = np.load(next(sess.glob('lines_raw/line_0000.npz')))
# raw includes the digitised A-scans for every (x, y) point on that line
```

## Provenance

Source machine: see project root `d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main`.
Dataset packaged by `scripts/charge_focus/package_35c_dataset.py` on
{time.strftime("%Y-%m-%d %H:%M:%S")}.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    print(f"\n  README.md written")

    # ============ final summary ============
    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"\n=== DONE ===")
    print(f"  destination: {DEST}")
    print(f"  total size:  {total/1e9:.2f} GB")
    print(f"  elapsed:     {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
