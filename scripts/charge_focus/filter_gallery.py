"""Build a charge+rest only gallery with shifted SOC in filenames.

Mirrors the 22h gallery but only includes charge + rest scans, with SOC
re-zeroed so the first scan is at SOC=0%.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "reports" / "experiments" / "longrun_cycling_3p238start_2026-05-29_18-54-23"
CHECKPOINT = BATCH_DIR / "checkpoint.json"
TAGGED_CSV = BATCH_DIR / "overview" / "scans_tagged.csv"
OUT_DIR = ROOT / "reports" / "longrun_cycling_22h_charge_focus" / "gallery"

KEEP_TAGS = {"charge", "rest"}
MODALITIES = {"tof": "scan_tof.png", "amp": "scan_amp.png", "eng": "scan_eng.png"}


def main() -> None:
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    scan_to = {c["scan_id"]: (c["run_idx"], Path(c["session_dir"].replace("\\", "/"))) for c in cp["completed"]}

    rows = []
    with TAGGED_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["step_tag"] in KEEP_TAGS:
                rows.append(row)

    soc_min = min(float(r["soc_pct"]) for r in rows)

    for sub in MODALITIES:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    copied = 0
    index_rows = []
    for row in rows:
        scan_id = row["scan_id"]
        if scan_id not in scan_to:
            continue
        run_idx, sess_dir = scan_to[scan_id]
        src_dir = ROOT / sess_dir
        if not src_dir.exists():
            continue
        soc = float(row["soc_pct"]) - soc_min
        soc_str = f"{soc:05.1f}".replace(".", "p")
        step_tag = row["step_tag"]
        stem = f"r{run_idx:03d}_soc{soc_str}_{step_tag}_{scan_id}"
        for sub, fname in MODALITIES.items():
            src = src_dir / fname
            if not src.exists():
                continue
            dst = OUT_DIR / sub / f"{stem}.png"
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
        index_rows.append({
            "run_idx": run_idx,
            "scan_id": scan_id,
            "time_utc": row["time_utc"],
            "voltage_v": row["voltage_at_scan"],
            "soc_shifted_pct": f"{soc:.2f}",
            "soc_original_pct": row["soc_pct"],
            "temp_c": row["temp_at_scan"],
            "step_tag": step_tag,
            "stem": stem,
        })

    with (OUT_DIR / "index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    (OUT_DIR / "README.txt").write_text(
        "Long-run cycling 22h — charge + rest only, SOC re-zeroed\n"
        f"Source batch: {BATCH_DIR.name}\n"
        f"Step tags kept: {sorted(KEEP_TAGS)}\n"
        f"SOC shift applied: {-soc_min:+.2f}%  (first kept scan -> SOC=0%)\n"
        f"Scans organized: {len(index_rows)}\n"
        f"PNGs copied: {copied}\n",
        encoding="utf-8",
    )
    print(f"copied {copied} PNGs across {len(MODALITIES)} modalities, kept {len(index_rows)} scans")


if __name__ == "__main__":
    main()
