"""One-off: bundle 35C charge-phase data into a single HDF5 for sharing."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("D:/35c_charge_start_data")
CSCAN_DIR = ROOT / "cscan"
CYCLER_FILE = ROOT / "cycler" / "LFP860_35degrees.002.txt"
TEMP_FILE = ROOT / "temperature" / "temperature_log_charge_window.csv"
OUT_FILE = ROOT / "35c_charge_phase.h5"

# Charge phase UTC window (local BST/UTC+1 -> 05:07:49 -> 21:18:08 = UTC 04:07:49 -> 20:18:08)
# Use a slightly wider window so r062 (baseline) and r170 (mid-scan at end) fit cleanly.
CHARGE_START_UTC = datetime(2026, 5, 31, 3, 30, 0, tzinfo=timezone.utc).timestamp()
CHARGE_END_UTC = datetime(2026, 5, 31, 20, 50, 0, tzinfo=timezone.utc).timestamp()


def parse_scan_ts(name: str) -> float:
    """Extract the trailing scan start timestamp (last YYYY-MM-DD_HH-MM-SS) as Unix UTC."""
    m = re.findall(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name)
    ts_local = datetime.strptime(m[-1], "%Y-%m-%d_%H-%M-%S")
    # Logs are BST (UTC+1) during this period
    return ts_local.timestamp() - 3600  # treat as BST -> subtract 1 hr for UTC


def load_scans():
    scan_dirs = sorted(p for p in CSCAN_DIR.iterdir() if p.is_dir())
    print(f"Loading {len(scan_dirs)} scans...")

    tof_stack, amp_stack, eng_stack = [], [], []
    line_ts_stack, line_temp_stack = [], []
    x_mm = y_mm = None
    timestamps, run_indices, scan_ids = [], [], []
    keys_seen = None

    for d in scan_dirs:
        npz_path = next(d.glob("scan_*.npz"))
        meta_path = next(d.glob("scan_*_meta.json"))
        meta = json.loads(meta_path.read_text())
        z = np.load(npz_path, allow_pickle=True)
        if keys_seen is None:
            keys_seen = list(z.keys())
            print(f"  npz keys: {keys_seen}")

        amp_stack.append(z["amplitude"].astype(np.float32))
        tof_stack.append(z["tof"].astype(np.float32))
        eng_stack.append(z["energy"].astype(np.float32))
        line_ts_stack.append(z["line_unix_center_s"].astype(np.float64))
        line_temp_stack.append(z["line_temperature_mean_c"].astype(np.float32))

        if x_mm is None:
            x_mm = z["x_mm"].astype(np.float32)
            y_mm = z["y_mm"].astype(np.float32)

        timestamps.append(parse_scan_ts(d.name))
        m = re.search(r"_r(\d{3})_", d.name)
        run_indices.append(int(m.group(1)) if m else -1)
        scan_ids.append(d.name)

    return {
        "tof": np.stack(tof_stack),
        "amp": np.stack(amp_stack),
        "energy": np.stack(eng_stack),
        "line_unix_center_s": np.stack(line_ts_stack),
        "line_temperature_mean_c": np.stack(line_temp_stack),
        "x_mm": x_mm,
        "y_mm": y_mm,
        "timestamps_utc": np.array(timestamps, dtype=np.float64),
        "run_index": np.array(run_indices, dtype=np.int32),
        "scan_id": np.array(scan_ids, dtype="S80"),
    }, meta


def load_cycler():
    """Parse Maccor txt, return charge-window slice as dict of arrays."""
    print(f"Loading cycler {CYCLER_FILE.name}...")
    # Header line is line 7 (1-indexed), data starts line 8
    with open(CYCLER_FILE, "r") as f:
        lines = f.readlines()
    # Find header row that starts with "Rec"
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Rec"))
    cols = [c.strip() for c in lines[header_idx].rstrip("\n").split("\t") if c.strip()]
    df = pd.read_csv(
        CYCLER_FILE, sep="\t", skiprows=header_idx + 1, header=None,
        names=cols + ["_extra"], engine="python", on_bad_lines="skip",
    )
    df = df[cols]
    # Parse timestamp (local BST) -> UTC
    df["ts_utc"] = pd.to_datetime(df["DPT Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["ts_utc"])
    df["ts_utc"] = df["ts_utc"].astype("int64") / 1e9 - 3600  # BST -> UTC
    # Coerce Rec (string with commas like "27,793") to int
    df["Rec"] = df["Rec"].astype(str).str.replace(",", "").astype(int)

    mask = (df["ts_utc"] >= CHARGE_START_UTC) & (df["ts_utc"] <= CHARGE_END_UTC)
    df = df.loc[mask].reset_index(drop=True)
    print(f"  charge window rows: {len(df)}")

    # Numeric coercion
    out = {}
    out["ts_utc"] = df["ts_utc"].to_numpy(dtype=np.float64)
    for src, dst, dtype in [
        ("Step", "step", np.int16),
        ("Capacity", "capacity_ah", np.float32),
        ("Energy", "energy_wh", np.float32),
        ("Current", "current_a", np.float32),
        ("Voltage", "voltage_v", np.float32),
        ("Cycle P", "cycle_p", np.int32),
        ("Cycle C", "cycle_c", np.int32),
    ]:
        out[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=dtype)
    out["md"] = df["MD"].astype(str).to_numpy(dtype="S2")
    return out


def load_temp():
    print(f"Loading temperature {TEMP_FILE.name}...")
    df = pd.read_csv(TEMP_FILE)
    print(f"  rows: {len(df)}")
    return {
        "ts_utc": df["unix_s"].to_numpy(dtype=np.float64),
        "cell_c": df["ch1_c"].to_numpy(dtype=np.float32),
    }


def main():
    scans, sample_meta = load_scans()
    cycler = load_cycler()
    temp = load_temp()

    print(f"\nWriting {OUT_FILE}...")
    if OUT_FILE.exists():
        OUT_FILE.unlink()

    with h5py.File(OUT_FILE, "w") as f:
        # Top-level attrs
        f.attrs["description"] = (
            "35C cell C-scan campaign, charging phase only. "
            "Bundle of ToF/amplitude/energy maps + aligned cycler + cell thermocouple."
        )
        f.attrs["run_prefix"] = "longrun_cycling_35c_2026-05-30_19-47-06"
        f.attrs["scan_count"] = scans["tof"].shape[0]
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["charge_window_utc_start"] = CHARGE_START_UTC
        f.attrs["charge_window_utc_end"] = CHARGE_END_UTC
        f.attrs["timestamp_note"] = "All ts_utc fields are Unix seconds (UTC). Cycler/scan timestamps were originally local BST (UTC+1)."

        # /cscan
        g = f.create_group("cscan")
        opts = dict(compression="gzip", compression_opts=4, shuffle=True)
        g.create_dataset("tof", data=scans["tof"], **opts).attrs["units"] = "seconds (or as in original npz)"
        g.create_dataset("amp", data=scans["amp"], **opts).attrs["units"] = "linear amplitude"
        g.create_dataset("energy", data=scans["energy"], **opts).attrs["units"] = "linear (sum of squared amplitude)"
        g.create_dataset("x_mm", data=scans["x_mm"]).attrs["desc"] = "column (x) axis, mm"
        g.create_dataset("y_mm", data=scans["y_mm"]).attrs["desc"] = "row (y) axis, mm"
        g.create_dataset("line_unix_center_s", data=scans["line_unix_center_s"], **opts).attrs["desc"] = "per-row scan time (Unix s, local-time-as-recorded — see timestamp_note)"
        g.create_dataset("line_temperature_mean_c", data=scans["line_temperature_mean_c"], **opts).attrs["desc"] = "TC-08 mean cell temp during each row"
        g.create_dataset("timestamps_utc", data=scans["timestamps_utc"])
        g.create_dataset("run_index", data=scans["run_index"])
        g.create_dataset("scan_id", data=scans["scan_id"])
        # ROI info from sample meta
        for k in ("roi_w", "roi_h", "pitch", "cols"):
            if k in sample_meta:
                g.attrs[k] = sample_meta[k]
        g.attrs["axes"] = "(n_scans, y_rows, x_cols)"

        # /cycler
        g = f.create_group("cycler")
        for k, v in cycler.items():
            g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        g.attrs["source_file"] = "LFP860_35degrees.002.txt"
        g.attrs["procedure"] = "LFP860_25_SOC.000 (SOC sweep: 5x CC charge steps with rests)"

        # /temperature
        g = f.create_group("temperature")
        for k, v in temp.items():
            g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        g.attrs["source"] = "Pico TC-08 ch1, K-type thermocouple on cell underside"
        g.attrs["sample_rate_hz"] = 1.0

    size_mb = OUT_FILE.stat().st_size / 1024**2
    print(f"\nDone. {OUT_FILE} = {size_mb:.1f} MB")
    print(f"  cscan: {scans['tof'].shape}")
    print(f"  cycler: {len(cycler['ts_utc'])} samples")
    print(f"  temp:   {len(temp['ts_utc'])} samples")


if __name__ == "__main__":
    main()
