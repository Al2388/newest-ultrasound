"""Build 35°C charge-focus tagged scan table + cache, restricted to
SOC 20% → 80% (Steps 5–11 = 4 rest plateaus + 3 charge segments).

The 35°C run cycler is data/raw/cycler/LFP860_35degrees.002.txt
- Step 5  (rest):   2026-05-31 07:07:49 → 09:07:49   SOC≈20%
- Step 6  (charge): 2026-05-31 09:07:49 → 11:07:49   20→40
- Step 7  (rest):   2026-05-31 11:07:49 → 13:07:49   SOC≈40%
- Step 8  (charge): 2026-05-31 13:07:49 → 15:07:50   40→60
- Step 9  (rest):   2026-05-31 15:07:50 → 17:07:50   SOC≈60%
- Step 10 (charge): 2026-05-31 17:07:50 → 19:07:51   60→80
- Step 11 (rest):   2026-05-31 19:07:51 → 21:07:51   SOC≈80%
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))

BATCH_DIR = PROJ / "reports" / "experiments" / "longrun_cycling_35c_2026-05-30_19-47-06"
CHECKPOINT = BATCH_DIR / "checkpoint.json"
CYCLER = PROJ / "data" / "raw" / "cycler" / "LFP860_35degrees.002.txt"
ROI_MASK_PATH = PROJ / "reports" / "experiments" / "roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp" / "roi_mask.npy"

OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTS = [
    # (step, kind, start, end, soc_label)
    (5,  "rest",   datetime(2026, 5, 31, 7, 7, 49),  datetime(2026, 5, 31, 9, 7, 49),  20),
    (6,  "charge", datetime(2026, 5, 31, 9, 7, 49),  datetime(2026, 5, 31, 11, 7, 49), None),  # 20→40
    (7,  "rest",   datetime(2026, 5, 31, 11, 7, 49), datetime(2026, 5, 31, 13, 7, 49), 40),
    (8,  "charge", datetime(2026, 5, 31, 13, 7, 49), datetime(2026, 5, 31, 15, 7, 50), None),  # 40→60
    (9,  "rest",   datetime(2026, 5, 31, 15, 7, 50), datetime(2026, 5, 31, 17, 7, 50), 60),
    (10, "charge", datetime(2026, 5, 31, 17, 7, 50), datetime(2026, 5, 31, 19, 7, 51), None),  # 60→80
    (11, "rest",   datetime(2026, 5, 31, 19, 7, 51), datetime(2026, 5, 31, 21, 7, 51), 80),
]
WINDOW_START = SEGMENTS[0][2]
WINDOW_END = SEGMENTS[-1][3]


def _scan_info(name: str):
    m = re.search(r"_r(\d{3})_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$", name)
    if not m:
        return None, None
    return int(m.group(1)), datetime.strptime(
        f"{m.group(2)} {m.group(3)}:{m.group(4)}:{m.group(5)}", "%Y-%m-%d %H:%M:%S"
    )


def _tag_segment(ts: datetime):
    for step, kind, t0, t1, soc in SEGMENTS:
        if t0 <= ts <= t1 + timedelta(minutes=2):  # small tolerance
            return step, kind, soc
    return None, None, None


def _load_cycler():
    df = pd.read_csv(CYCLER, sep="\t", skiprows=6, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DPT Time"] = pd.to_datetime(df["DPT Time"].astype(str), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for c in ["Step", "Capacity", "Voltage", "Current"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _interp_cycler(cycler_df, ts: datetime):
    """Get nearest cycler row to scan timestamp."""
    diffs = (cycler_df["DPT Time"] - ts).abs()
    idx = diffs.idxmin()
    return cycler_df.loc[idx]


def main():
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    cycler = _load_cycler()

    # SOC reference: q at start of Step 4 = SOC 0% reference (after the initial discharge to floor)
    step4_start = cycler[cycler["Step"] == 4]["DPT Time"].iloc[0]
    q_at_soc0 = float(cycler[cycler["DPT Time"] == step4_start]["Capacity"].iloc[0])
    # Q resets to 0 at each step (Maccor convention), so we need cumulative tracking
    # Actually look at the cycler: cumulative charge passed = sum of Q increments per step
    # Simpler: tag SOC by step_label (since each rest is at known SOC level)
    print(f"SOC=0% reference timestamp: {step4_start}")

    selected = []
    for c in cp["completed"]:
        sess_dir = PROJ / Path(c["session_dir"].replace("\\", "/"))
        run_idx, ts = _scan_info(sess_dir.name)
        if ts is None or not (WINDOW_START <= ts <= WINDOW_END + timedelta(minutes=10)):
            continue
        step, kind, soc_lbl = _tag_segment(ts)
        if step is None:
            continue
        # Pull cycler V at this scan
        cyc_row = _interp_cycler(cycler, ts)
        selected.append({
            "run_idx": run_idx,
            "scan_id": c["scan_id"],
            "timestamp": ts.isoformat(),
            "session_dir": str(sess_dir.relative_to(PROJ)),
            "step": step,
            "step_tag": kind,
            "soc_plateau_label": soc_lbl if kind == "rest" else None,
            "voltage_at_scan": float(cyc_row["Voltage"]),
            "current_at_scan": float(cyc_row["Current"]),
            "step_time_relative_min": (ts - dict([(s[0], s[2]) for s in SEGMENTS])[step]).total_seconds() / 60.0,
        })

    df = pd.DataFrame(selected).sort_values("timestamp").reset_index(drop=True)
    print(f"\nselected {len(df)} scans in window [{WINDOW_START} -> {WINDOW_END}]")
    print(df.groupby(["step", "step_tag"]).size())

    # Load NPZ stack
    amp, tof, energy = [], [], []
    line_T_per_scan = []
    x_mm = y_mm = None
    for _, row in df.iterrows():
        sess = PROJ / row["session_dir"]
        npz = next(sess.glob("scan_*.npz"))
        d = np.load(npz)
        amp.append(d["amplitude"].astype(np.float32))
        tof.append(d["tof"].astype(np.float32))
        energy.append(d["energy"].astype(np.float32))
        line_T_per_scan.append(float(np.nanmean(d["line_temperature_mean_c"])))
        if x_mm is None:
            x_mm = d["x_mm"].astype(np.float32)
            y_mm = d["y_mm"].astype(np.float32)

    df["line_T_mean_c"] = line_T_per_scan
    df["amp_mean_roi"] = [float(np.nanmean(a[np.load(ROI_MASK_PATH)])) for a in amp]
    df["tof_mean_roi"] = [float(np.nanmean(a[np.load(ROI_MASK_PATH)])) for a in tof]
    df["energy_mean_roi"] = [float(np.nanmean(a[np.load(ROI_MASK_PATH)])) for a in energy]

    roi_mask = np.load(ROI_MASK_PATH)
    np.savez_compressed(
        CACHE_DIR / "stack.npz",
        amplitude=np.stack(amp, axis=0),
        tof=np.stack(tof, axis=0),
        energy=np.stack(energy, axis=0),
        x_mm=x_mm,
        y_mm=y_mm,
        roi_mask=roi_mask,
    )
    df.to_csv(CACHE_DIR / "meta.csv", index=False)
    print(f"\nwrote cache: {CACHE_DIR}/stack.npz + meta.csv")

    summary = [
        "# 35°C charge-focus subset (SOC 20→80 with rests)\n\n",
        f"Source batch: longrun_cycling_35c_2026-05-30_19-47-06\n",
        f"Window: {WINDOW_START} → {WINDOW_END}  ({(WINDOW_END-WINDOW_START).total_seconds()/3600:.1f} h)\n",
        f"Total scans: {len(df)}\n\n",
        "## Segment breakdown\n\n",
        "| step | tag    | SOC plateau | n scans | duration |\n|---|---|---|---:|---|\n",
    ]
    for step, kind, t0, t1, soc in SEGMENTS:
        n = int((df["step"] == step).sum())
        summary.append(f"| {step} | {kind} | {soc if soc else f'{kind}'} | {n} | {(t1-t0).total_seconds()/60:.0f} min |\n")
    summary.append(f"\n## Temperature stability\n")
    summary.append(f"- TC08 line-mean T range: {min(line_T_per_scan):.3f} → {max(line_T_per_scan):.3f} °C\n")
    summary.append(f"- T span: {(max(line_T_per_scan)-min(line_T_per_scan))*1000:.1f} mC\n")
    summary.append(f"- T std: {np.std(line_T_per_scan)*1000:.1f} mC\n")
    (OUT_ROOT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"\nwrote README to {OUT_ROOT}/README.md")


if __name__ == "__main__":
    main()
