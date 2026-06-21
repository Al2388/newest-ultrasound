"""Summary-only 35C dataset (no raw A-scan waveforms).

Reuses D:/35c_dataset/ but excludes the lines_raw/ subdirectories.
Suitable for sharing via email/Box/chat upload (target < 100 MB).

Output: D:/35c_dataset_summary/  + D:/35c_dataset_summary.zip
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
import zipfile
from pathlib import Path

SRC  = Path("D:/35c_dataset")
DEST = Path("D:/35c_dataset_summary")
ZIP  = Path("D:/35c_dataset_summary.zip")


def main():
    if not SRC.exists():
        print(f"Source missing: {SRC}")
        sys.exit(1)
    if DEST.exists():
        print(f"Removing previous {DEST} ...")
        shutil.rmtree(DEST)
    if ZIP.exists():
        ZIP.unlink()
    DEST.mkdir(parents=True)

    t0 = time.time()
    n_files = 0; n_bytes = 0
    for src in SRC.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(SRC)
        # skip raw waveforms
        if "lines_raw" in rel.parts:
            continue
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        n_files += 1; n_bytes += src.stat().st_size

    print(f"copied {n_files} files, {n_bytes/1e6:.1f} MB in {time.time()-t0:.1f}s")

    # Update README to flag this is summary-only
    readme = DEST / "README.md"
    text = readme.read_text(encoding="utf-8")
    note = (
        "## ⚠ Summary-only subset\n\n"
        "This package excludes the raw A-scan waveforms (`ultrasound/*/lines_raw/`,\n"
        "~45 GB) from the full dataset. Each `scan_*.npz` here contains the\n"
        "**extracted** amplitude / ToF / energy 2D maps plus all per-line metadata\n"
        "(temperature, timestamps, pulse counts) - sufficient for all C-scan-level\n"
        "analysis (cell-ROI statistics, spatial sub-region comparisons, equilibrium\n"
        "vs SOC, hysteresis, PCA on the 2D maps, etc).\n\n"
        "To re-derive features from the raw 500-sample A-scans (e.g. new ToF\n"
        "algorithm, waveform-domain PCA), get the full dataset (`D:/35c_dataset/`\n"
        "on the source machine).\n\n"
        "---\n\n"
    )
    readme.write_text(note + text, encoding="utf-8")

    # Rebuild MANIFEST (the original one referenced lines_raw paths)
    rows = []
    for f in DEST.rglob("*"):
        if not f.is_file() or f.name == "MANIFEST.csv":
            continue
        rows.append({"path": f.relative_to(DEST).as_posix(),
                     "size_bytes": f.stat().st_size})
    with (DEST / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "size_bytes"])
        w.writeheader(); w.writerows(rows)
    print(f"MANIFEST.csv: {len(rows)} entries")

    # Zip it
    print("zipping ...")
    t1 = time.time()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in DEST.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(DEST))
    zip_size = ZIP.stat().st_size
    print(f"  {ZIP.name} = {zip_size/1e6:.1f} MB  (zipped in {time.time()-t1:.1f}s)")

    # Final summary
    total_dir = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print()
    print(f"=== summary ===")
    print(f"  folder: {DEST}            ({total_dir/1e6:.1f} MB, {n_files} files)")
    print(f"  zip:    {ZIP}     ({zip_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
