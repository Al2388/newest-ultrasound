"""
Match C-scan rows to Maccor cycler context and PicoLog temperature.

Inputs:
  - C-scan scan_*.npz with line_unix_start_s/line_unix_end_s/line_unix_center_s
  - Maccor tab-delimited cycler export with DPT Time
  - PicoLog CSV with elapsed HH:MM:SS temperature rows

For each C-scan line, cycler values are matched at the line center time, while
all PicoLog samples inside the line time window are averaged. If no PicoLog
sample falls inside a line window, temperature is interpolated at line center
only when the line center is within the PicoLog time range.

The three logs do not need to start together, but each stream needs an absolute
clock:
  - C-scan line Unix times are stored by the scanner.
  - Maccor cycler absolute times come from DPT Time.
  - PicoLog CSVs only contain elapsed HH:MM:SS, so --picolog-start is required.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def parse_iso_to_unix(value: str) -> float:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def nearest_indices(source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(source_t, target_t), 0, len(source_t) - 1)
    left = np.clip(idx - 1, 0, len(source_t) - 1)
    return np.where(
        np.abs(source_t[left] - target_t) <= np.abs(source_t[idx] - target_t),
        left,
        idx,
    )


def read_picolog_csv(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or not row[0] or not row[1]:
                continue
            try:
                hh, mm, ss = row[0].strip('"').split(":")
                elapsed_s = int(hh) * 3600 + int(mm) * 60 + float(ss)
                temperature_c = float(row[1])
            except ValueError:
                continue
            rows.append((elapsed_s, temperature_c))
    if not rows:
        raise ValueError(f"No PicoLog rows found in {path}")
    df = pd.DataFrame(rows, columns=["temp_elapsed_s", "temperature_c"])
    df["temp_elapsed_s"] -= float(df["temp_elapsed_s"].iloc[0])
    return df


def read_maccor_txt(path: Path, q_nominal_ah: float, tz_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skiprows=6, engine="python").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    required = ["DPT Time", "Step", "Capacity", "Energy", "Current", "Voltage", "MD"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing cycler columns: {missing}")

    for col in ["Rec", "Cycle P", "Cycle C", "Step", "Capacity", "Energy", "Current", "Voltage"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    local_tz = ZoneInfo(tz_name)
    dt_local = pd.to_datetime(df["DPT Time"], dayfirst=True, errors="coerce")
    unix_s = np.asarray(
        [
            x.to_pydatetime().replace(tzinfo=local_tz).astimezone(ZoneInfo("UTC")).timestamp()
            if pd.notna(x)
            else np.nan
            for x in dt_local
        ],
        dtype=np.float64,
    )
    df["cycler_unix_s"] = unix_s
    finite = np.isfinite(unix_s)
    if not np.any(finite):
        raise ValueError(f"No valid DPT Time rows found in {path}")
    df["cycler_elapsed_s"] = unix_s - unix_s[finite][0]

    mode = df["MD"].astype(str).str.strip()
    signed_current = np.zeros(len(df), dtype=np.float64)
    signed_current[mode == "C"] = pd.to_numeric(df.loc[mode == "C", "Current"], errors="coerce")
    signed_current[mode == "D"] = -pd.to_numeric(df.loc[mode == "D", "Current"], errors="coerce")
    df["signed_current_a"] = np.nan_to_num(signed_current, nan=0.0)

    dt_h = np.diff(df["cycler_unix_s"].to_numpy(dtype=np.float64)) / 3600.0
    i_mid = 0.5 * (
        df["signed_current_a"].to_numpy(dtype=np.float64)[:-1]
        + df["signed_current_a"].to_numpy(dtype=np.float64)[1:]
    )
    valid_dt = np.isfinite(dt_h) & (dt_h >= 0.0) & (dt_h < 0.1)
    q = np.zeros(len(df), dtype=np.float64)
    q[1:] = np.cumsum(np.where(valid_dt, i_mid * dt_h, 0.0))
    q_zero = float(np.nanmin(q))
    df["relative_q_ah"] = q
    df["soc_pct"] = (q - q_zero) / q_nominal_ah * 100.0
    df["soc_pct_clipped"] = df["soc_pct"].clip(0.0, 100.0)
    df["branch"] = np.select(
        [mode == "C", mode == "D", mode == "R"],
        ["charge", "discharge", "rest"],
        default="other",
    )
    return df.sort_values("cycler_unix_s").reset_index(drop=True)


def load_cscan_rows(scan_npz: Path) -> pd.DataFrame:
    scan = np.load(scan_npz)
    required = ["line_unix_start_s", "line_unix_end_s", "line_unix_center_s"]
    missing = [name for name in required if name not in scan.files]
    if missing:
        raise KeyError(
            f"{scan_npz} is missing {missing}. New C-scans have these fields; "
            "older scans need a timing backfill or manual offset estimate."
        )

    n = len(scan["line_unix_center_s"])
    y_mm = scan["y_mm"] if "y_mm" in scan.files else np.arange(n, dtype=np.float64)
    out = pd.DataFrame(
        {
            "line": np.arange(n, dtype=np.int32),
            "y_mm": y_mm.astype(np.float64),
            "line_unix_start_s": scan["line_unix_start_s"].astype(np.float64),
            "line_unix_end_s": scan["line_unix_end_s"].astype(np.float64),
            "line_unix_center_s": scan["line_unix_center_s"].astype(np.float64),
        }
    )
    if "line_pulse_count" in scan.files:
        out["line_pulse_count"] = scan["line_pulse_count"].astype(np.int32)
    return out


def add_temperature(rows: pd.DataFrame, temp: pd.DataFrame, temp_start_unix_s: float) -> pd.DataFrame:
    out = rows.copy()
    temp_unix_s = temp_start_unix_s + temp["temp_elapsed_s"].to_numpy(dtype=np.float64)
    temp_c = temp["temperature_c"].to_numpy(dtype=np.float64)

    values = []
    counts = []
    modes = []
    for start, end, center in zip(
        out["line_unix_start_s"].to_numpy(dtype=np.float64),
        out["line_unix_end_s"].to_numpy(dtype=np.float64),
        out["line_unix_center_s"].to_numpy(dtype=np.float64),
    ):
        inside = (temp_unix_s >= start) & (temp_unix_s <= end)
        if np.any(inside):
            values.append(float(np.mean(temp_c[inside])))
            counts.append(int(np.sum(inside)))
            modes.append("window_average")
        elif center < temp_unix_s[0] or center > temp_unix_s[-1]:
            values.append(np.nan)
            counts.append(0)
            modes.append("outside_range")
        else:
            values.append(float(np.interp(center, temp_unix_s, temp_c)))
            counts.append(0)
            modes.append("center_interp")
    out["temperature_c"] = values
    out["n_picolog_samples"] = counts
    out["temperature_match_mode"] = modes
    return out


def add_cycler(rows: pd.DataFrame, cycler: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    center = out["line_unix_center_s"].to_numpy(dtype=np.float64)
    cy_t = cycler["cycler_unix_s"].to_numpy(dtype=np.float64)
    nearest = nearest_indices(cy_t, center)
    out["cycler_index"] = nearest
    out["cycler_delta_s"] = center - cy_t[nearest]
    in_range = (center >= cy_t[0]) & (center <= cy_t[-1])
    out["cycler_match_mode"] = np.where(in_range, "center_interp", "outside_range")

    for col in ["Step", "MD", "branch"]:
        values = cycler.iloc[nearest][col].to_numpy()
        out[col] = np.where(in_range, values, "")

    numeric_cols = [
        "cycler_elapsed_s",
        "Capacity",
        "Energy",
        "Current",
        "Voltage",
        "signed_current_a",
        "relative_q_ah",
        "soc_pct",
        "soc_pct_clipped",
    ]
    for col in numeric_cols:
        values = cycler[col].to_numpy(dtype=np.float64)
        matched = np.full(len(out), np.nan, dtype=np.float64)
        matched[in_range] = np.interp(center[in_range], cy_t, values)
        out[col] = matched
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_npz", type=Path)
    parser.add_argument("--cycler", type=Path, required=True, help="Maccor .txt export")
    parser.add_argument("--temp", type=Path, required=True, help="PicoLog CSV export")
    parser.add_argument(
        "--picolog-start",
        required=True,
        help="PicoLog start ISO datetime, e.g. 2026-05-25T17:20:00+01:00.",
    )
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--q-nominal-ah", type=float, default=0.86)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_cscan_rows(args.scan_npz)
    temp = read_picolog_csv(args.temp)
    cycler = read_maccor_txt(args.cycler, args.q_nominal_ah, args.timezone)

    temp_start = parse_iso_to_unix(args.picolog_start)
    out = add_temperature(rows, temp, temp_start)
    out = add_cycler(out, cycler)

    out_path = args.out or args.scan_npz.with_name("line_context.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    averaged = int((out["temperature_match_mode"] == "window_average").sum())
    print(
        f"Wrote {out_path} "
        f"({averaged}/{len(out)} rows used PicoLog window averages; "
        f"{int((out['temperature_match_mode'] == 'outside_range').sum())} outside PicoLog range; "
        f"{int((out['cycler_match_mode'] == 'outside_range').sum())} outside cycler range; "
        f"max |cycler_delta_s|={out['cycler_delta_s'].abs().max():.2f}s)"
    )


if __name__ == "__main__":
    main()
