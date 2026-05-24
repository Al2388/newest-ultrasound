"""
Extract the complete full-charge and full-discharge segments from one
experiment, merging all three streams (cycler + A-scan + temperature) onto a
common elapsed-time axis and adding a Coulomb-counted SOC column.

Procedure assumed (matches the lab protocol):
    step 2  initial partial discharge to ~0% SOC   (excluded)
    step 4  full charge 0 -> 100%                  --> FULL CHARGE
    step 5  CV topping tail                         (part of full charge)
    step 7  full discharge 100% -> 0%              --> FULL DISCHARGE
    step 9  partial re-charge to keep stable        (excluded)

Stream alignment
----------------
The cycler, A-scan logger and temperature logger are all started together, so
elapsed-seconds-from-start is used as the shared clock (cycler is the master).
The A-scan snapshots (1 Hz) define the output sampling; cycler voltage/current/
SOC and temperature are linearly interpolated onto each snapshot time.

SOC convention
--------------
SOC is referenced to each segment's own full capacity (max Coulomb count), so
charge spans 0->100% and discharge spans 100->0% cleanly. The reference
capacities are printed and stored in the manifest.

Usage
-----
  python extract_charge_discharge.py EXP_DIR
  python extract_charge_discharge.py data/experiments/18-5
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --- segment definition: which Maccor steps make up each segment ------------
SEGMENTS = {
    "full_charge":    [4, 5],   # CC charge + CV topping tail
    "full_discharge": [7],      # CC discharge
}
SEGMENT_DIRECTION = {"full_charge": "charge", "full_discharge": "discharge"}


# =============================================================================
# Readers
# =============================================================================
def read_cycler(txt_path: str) -> dict:
    """Parse a Maccor tab-delimited export. Returns elapsed-s + per-row arrays."""
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # Header row is the one starting with 'Rec'
    hdr_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Rec\t"))
    cols = [c.strip() for c in lines[hdr_idx].rstrip("\n").split("\t")]
    rows = []
    for ln in lines[hdr_idx + 1:]:
        parts = ln.rstrip("\n").split("\t")
        # Rec column may use comma thousands separators (e.g. "1,000")
        if len(parts) < len(cols) or not parts[0].strip().replace(",", "").isdigit():
            continue
        rows.append(parts)

    def col(name):
        return [r[cols.index(name)] for r in rows]

    # Wall-clock per row from DPT Time -> elapsed seconds from first row
    dpt = col("DPT Time")
    t0 = datetime.strptime(dpt[0].strip(), "%d/%m/%Y %H:%M:%S")
    elapsed = np.array(
        [(datetime.strptime(d.strip(), "%d/%m/%Y %H:%M:%S") - t0).total_seconds()
         for d in dpt],
        dtype=np.float64,
    )
    return {
        "elapsed_s": elapsed,
        "step":      np.array([int(s) for s in col("Step")]),
        "capacity":  np.array([float(x) for x in col("Capacity")]),
        "voltage":   np.array([float(x) for x in col("Voltage")]),
        "current":   np.array([float(x) for x in col("Current")]),
        "md":        np.array([m.strip() for m in col("MD")]),
        "t0":        t0,
    }


def read_temperature(csv_path: str) -> dict:
    """PicoLog CSV: HH:MM:SS elapsed + Channel 1 Ave (C)."""
    t_s, temp = [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or row[0] == "" or row[1] == "":
                continue
            try:
                hh, mm, ss = row[0].split(":")
                t_s.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
                temp.append(float(row[1]))
            except ValueError:
                continue
    return {"elapsed_s": np.asarray(t_s, float), "temp_c": np.asarray(temp, float)}


def read_ascan(h5_path: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        ts = f["timestamps"][:].astype(np.float64)
        return {
            "elapsed_s": ts - ts[0],
            "tof_us":    f["tof_us"][:].astype(np.float64),
            "amplitude": f["amplitude"][:].astype(np.float64),
            "energy":    f["energy"][:].astype(np.float64),
        }


# =============================================================================
# Extraction
# =============================================================================
def _find_one(pattern: str) -> str:
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern}")
    return max(hits, key=os.path.getsize)  # largest match (handles stray files)


def extract(exp_dir: str) -> dict:
    exp = Path(exp_dir)
    cyc_path  = _find_one(str(exp / "cycler" / "*.txt"))
    temp_path = _find_one(str(exp / "temperature" / "*.csv"))
    h5_path   = _find_one(str(exp / "ascan" / "*.h5"))

    cyc  = read_cycler(cyc_path)
    temp = read_temperature(temp_path)
    asc  = read_ascan(h5_path)

    out_dir = exp / "segments"
    out_dir.mkdir(exist_ok=True)

    manifest = {
        "experiment":  exp.name,
        "sources":     {"cycler": os.path.basename(cyc_path),
                        "temperature": os.path.basename(temp_path),
                        "ascan": os.path.basename(h5_path)},
        "segments":    {},
    }

    for seg_name, steps in SEGMENTS.items():
        mask = np.isin(cyc["step"], steps)
        if not mask.any():
            print(f"  [skip] {seg_name}: steps {steps} not present")
            continue

        seg_t   = cyc["elapsed_s"][mask]
        seg_cap = cyc["capacity"][mask]
        seg_v   = cyc["voltage"][mask]
        seg_i   = cyc["current"][mask]
        seg_step = cyc["step"][mask]

        # Continuous capacity across concatenated steps (Capacity resets per step)
        cap_cont = seg_cap.copy()
        offset = 0.0
        for st in steps:
            m = seg_step == st
            if not m.any():
                continue
            cap_cont[m] = seg_cap[m] + offset
            offset = cap_cont[m].max()
        q_ref = cap_cont.max()

        if SEGMENT_DIRECTION[seg_name] == "charge":
            soc_cyc = 100.0 * cap_cont / q_ref
        else:
            soc_cyc = 100.0 * (1.0 - cap_cont / q_ref)

        t_lo, t_hi = seg_t.min(), seg_t.max()

        # A-scan snapshots inside the window define the output rows
        a_mask = (asc["elapsed_s"] >= t_lo) & (asc["elapsed_s"] <= t_hi)
        a_t = asc["elapsed_s"][a_mask]
        if a_t.size == 0:
            print(f"  [skip] {seg_name}: no A-scan snapshots in window")
            continue

        # Interpolate cycler + temperature onto snapshot times
        order = np.argsort(seg_t)
        soc  = np.interp(a_t, seg_t[order], soc_cyc[order])
        volt = np.interp(a_t, seg_t[order], seg_v[order])
        curr = np.interp(a_t, seg_t[order], seg_i[order])
        torder = np.argsort(temp["elapsed_s"])
        temp_c = np.interp(a_t, temp["elapsed_s"][torder], temp["temp_c"][torder])

        # Write CSV
        csv_path = out_dir / f"{seg_name}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["elapsed_s", "elapsed_h", "soc_pct", "voltage_v",
                        "current_a", "tof_us", "amplitude_v", "energy", "temp_c"])
            for k in range(a_t.size):
                w.writerow([f"{a_t[k]:.1f}", f"{a_t[k]/3600:.5f}", f"{soc[k]:.3f}",
                            f"{volt[k]:.4f}", f"{curr[k]:.5f}",
                            f"{asc['tof_us'][a_mask][k]:.5f}",
                            f"{asc['amplitude'][a_mask][k]:.4f}",
                            f"{asc['energy'][a_mask][k]:.3f}",
                            f"{temp_c[k]:.4f}"])

        manifest["segments"][seg_name] = {
            "maccor_steps":   steps,
            "direction":      SEGMENT_DIRECTION[seg_name],
            "q_ref_ah":       round(float(q_ref), 5),
            "window_h":       [round(float(t_lo / 3600), 4), round(float(t_hi / 3600), 4)],
            "duration_h":     round(float((t_hi - t_lo) / 3600), 4),
            "n_ascan_rows":   int(a_t.size),
            "soc_span_pct":   [round(float(soc.min()), 2), round(float(soc.max()), 2)],
            "tof_span_us":    round(float(asc["tof_us"][a_mask].max()
                                          - asc["tof_us"][a_mask].min()), 4),
            "temp_span_c":    [round(float(temp_c.min()), 3), round(float(temp_c.max()), 3)],
            "csv":            str(csv_path).replace("\\", "/"),
        }
        print(f"  {seg_name:15s} steps={steps}  {a_t.size:5d} rows  "
              f"SOC {soc.min():5.1f}->{soc.max():5.1f}%  "
              f"Qref={q_ref:.3f}Ah  {(t_hi-t_lo)/3600:.2f}h")

    with open(out_dir / "segments_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dir", help="experiment folder (contains ascan/ cycler/ temperature/)")
    args = p.parse_args()
    print(f"=== {args.exp_dir} ===")
    m = extract(args.exp_dir)
    print(f"  manifest: {Path(args.exp_dir) / 'segments' / 'segments_manifest.json'}")


if __name__ == "__main__":
    main()
