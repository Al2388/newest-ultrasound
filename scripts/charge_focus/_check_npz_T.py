"""Per-line cell-temperature stats per rest plateau, joined via meta.step.

Avoids any timezone confusion by using meta.step (already computed from local
time during cache build) to label which scans belong to which plateau, then
concatenating each scan's per-line line_temperature_mean_c array.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
meta = pd.read_csv(PROJ / "reports/longrun_cycling_35c_charge_focus/_cache/meta.csv")
meta["timestamp"] = pd.to_datetime(meta["timestamp"])
meta = meta.sort_values("timestamp").reset_index(drop=True)

# Build per-line T record per scan
per_scan = []
for _, row in meta.iterrows():
    sess = PROJ / row["session_dir"].replace("\\", "/")
    npz = next(sess.glob("*.npz"))
    d = np.load(npz)
    per_scan.append({
        "step": row["step"],
        "step_tag": row["step_tag"],
        "scan_t0": float(d["line_unix_center_s"][0]),
        "t_per_line": d["line_temperature_mean_c"].astype(np.float32),
        "u_per_line": d["line_unix_center_s"].astype(np.float64),
    })

# Per-plateau stats
print(f"{'SOC':>4} | {'N scans':>7} | {'N lines':>7} | {'dT_end (mC)':>12} | {'span (mC)':>9} | {'sigma (mC)':>10}")
for step, soc in [(5, 20), (7, 40), (9, 60), (11, 80)]:
    Ts = np.concatenate([s["t_per_line"] for s in per_scan if s["step"] == step])
    Us = np.concatenate([s["u_per_line"] for s in per_scan if s["step"] == step])
    order = np.argsort(Us); Ts = Ts[order]; Us = Us[order]
    n_scans = sum(1 for s in per_scan if s["step"] == step)
    # dT_end uses first 30s and last 30s averages, robust to single-sample noise
    head = Ts[:10]; tail = Ts[-10:]
    dT_end = (tail.mean() - head.mean()) * 1000
    span = (Ts.max() - Ts.min()) * 1000
    std = float(np.std(Ts)) * 1000
    print(f"{soc:>3}% | {n_scans:>7} | {len(Ts):>7} | {dT_end:>+12.1f} | {span:>9.1f} | {std:>10.1f}")

# Full window across all 93 scans
Ts_all = np.concatenate([s["t_per_line"] for s in per_scan])
print(f"\nfull 14h: {len(Ts_all)} per-line T samples, "
      f"span {(Ts_all.max()-Ts_all.min())*1000:.1f} mC, sigma {np.std(Ts_all)*1000:.1f} mC")
