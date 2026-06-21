"""
Match PicoLog temperature samples onto C-scan rows.

Each C-scan row has a Unix wall-clock time window saved as
line_unix_start_s/line_unix_end_s in the scan NPZ. PicoLog samples that fall
inside a row window are averaged; if no sample falls inside, the script falls
back to interpolation at the row center only when the row is inside the PicoLog
time range.

Example:
  python scripts/match_cscan_picolog_temperature.py \
      data/raw/cscan/.../scan_YYYY-MM-DD_HH-MM-SS.npz \
      data/raw/temperature/run.csv \
      --picolog-start "2026-05-25T17:29:52+01:00"
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np


def _parse_iso_to_unix(value: str) -> float:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def read_picolog_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read PicoLog elapsed HH:MM:SS + temperature CSV."""
    elapsed_s: list[float] = []
    temp_c: list[float] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or not row[0] or not row[1]:
                continue
            try:
                hh, mm, ss = row[0].split(":")
                t_s = int(hh) * 3600 + int(mm) * 60 + float(ss)
                temp = float(row[1])
            except ValueError:
                continue
            elapsed_s.append(t_s)
            temp_c.append(temp)
    return np.asarray(elapsed_s, dtype=np.float64), np.asarray(temp_c, dtype=np.float64)


def match_rows(scan_npz: Path, picolog_csv: Path, picolog_start_unix_s: float) -> list[dict]:
    scan = np.load(scan_npz)
    required = ["line_unix_start_s", "line_unix_end_s", "line_unix_center_s"]
    missing = [name for name in required if name not in scan.files]
    if missing:
        raise KeyError(
            f"{scan_npz} is missing {missing}; run a new C-scan with line timing enabled."
        )

    temp_elapsed_s, temp_c = read_picolog_csv(picolog_csv)
    if temp_elapsed_s.size == 0:
        raise ValueError(f"No PicoLog temperature rows found in {picolog_csv}")
    temp_unix_s = picolog_start_unix_s + temp_elapsed_s

    starts = scan["line_unix_start_s"].astype(np.float64)
    ends = scan["line_unix_end_s"].astype(np.float64)
    centers = scan["line_unix_center_s"].astype(np.float64)
    y_mm = scan["y_mm"].astype(np.float64) if "y_mm" in scan.files else np.arange(starts.size)

    rows: list[dict] = []
    for i, (start, end, center) in enumerate(zip(starts, ends, centers)):
        if not np.isfinite(start) or not np.isfinite(end):
            rows.append(
                dict(
                    line=i,
                    y_mm=float(y_mm[i]),
                    line_unix_start_s="",
                    line_unix_end_s="",
                    temperature_c="",
                    n_picolog_samples=0,
                    match_mode="unscanned",
                )
            )
            continue

        inside = (temp_unix_s >= start) & (temp_unix_s <= end)
        if np.any(inside):
            temperature_c = float(np.mean(temp_c[inside]))
            n_samples = int(np.sum(inside))
            mode = "window_average"
        elif center < temp_unix_s[0] or center > temp_unix_s[-1]:
            temperature_c = ""
            n_samples = 0
            mode = "outside_range"
        else:
            temperature_c = float(np.interp(center, temp_unix_s, temp_c))
            n_samples = 0
            mode = "center_interp"

        rows.append(
            dict(
                line=i,
                y_mm=float(y_mm[i]),
                line_unix_start_s=float(start),
                line_unix_end_s=float(end),
                temperature_c=temperature_c,
                n_picolog_samples=n_samples,
                match_mode=mode,
            )
        )
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "line",
        "y_mm",
        "line_unix_start_s",
        "line_unix_end_s",
        "temperature_c",
        "n_picolog_samples",
        "match_mode",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_npz", type=Path)
    parser.add_argument("picolog_csv", type=Path)
    parser.add_argument(
        "--picolog-start",
        required=True,
        help="PicoLog logging start time as ISO datetime, e.g. 2026-05-25T17:29:52+01:00",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or args.scan_npz.with_name("line_temperature.csv")
    rows = match_rows(args.scan_npz, args.picolog_csv, _parse_iso_to_unix(args.picolog_start))
    write_csv(rows, out)
    averaged = sum(1 for row in rows if row["match_mode"] == "window_average")
    print(f"Wrote {out} ({averaged}/{len(rows)} rows used PicoLog window averages)")


if __name__ == "__main__":
    main()
