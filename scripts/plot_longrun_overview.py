"""Long-run overview: 147-scan ROI-aggregated features + temperature trace.

For each scan in the longrun batch, compute the ROI-mean of amplitude / ToF /
energy inside the canonical 50×40 mm mask. Plot as a time series across the
22-hour run, overlaid with the TC-08 cell-temperature trace on a twin axis.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"

# Canonical ROI (from memory)
ROI_W_MM = 50.02
ROI_H_MM = 39.78
ROI_CX_MM = 39.59
ROI_CY_MM = 36.00


def build_roi_mask(x_mm, y_mm, w_mm, h_mm, cx_mm, cy_mm):
    """Boolean ROI mask from the canonical bounds."""
    dx = float(x_mm[1] - x_mm[0])
    dy = float(y_mm[1] - y_mm[0])
    col_lo = max(0, int(round((cx_mm - w_mm/2 - float(x_mm[0])) / dx)))
    col_hi = min(len(x_mm), int(round((cx_mm + w_mm/2 - float(x_mm[0])) / dx)))
    row_lo = max(0, int(round((cy_mm - h_mm/2 - float(y_mm[0])) / dy)))
    row_hi = min(len(y_mm), int(round((cy_mm + h_mm/2 - float(y_mm[0])) / dy)))
    mask = np.zeros((len(y_mm), len(x_mm)), dtype=bool)
    mask[row_lo:row_hi, col_lo:col_hi] = True
    return mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True,
                    help="reports/experiments/longrun_cycling_*")
    ap.add_argument("--temp-csv", default=None,
                    help="TC-08 CSV (default: auto-find by start time)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    cp = json.loads((batch_dir / "checkpoint.json").read_text())
    completed = cp["completed"]
    print(f"loaded checkpoint: {len(completed)} completed scans")

    # ----- iterate scans, compute ROI-mean -----
    rows = []
    roi_mask = None
    for entry in completed:
        scan_dir_str = entry.get("session_dir") or ""
        scan_dir = Path(scan_dir_str)
        if not scan_dir.is_absolute():
            scan_dir = PROJECT / scan_dir
        if not scan_dir.exists():
            # fallback by scan_name
            cand = list(CSCAN_ROOT.glob(f"cscan_{entry.get('scan_id','')}*"))
            if cand:
                scan_dir = cand[0]
            else:
                continue
        npz_p = next(scan_dir.glob("scan_*.npz"), None)
        meta_p = next(scan_dir.glob("scan_*_meta.json"), None)
        if npz_p is None or meta_p is None:
            continue
        d = np.load(npz_p)
        amp = np.asarray(d["amplitude"], dtype=np.float64)
        tof = np.asarray(d["tof"], dtype=np.float64)
        eng = np.asarray(d["energy"], dtype=np.float64)
        x_mm = np.asarray(d["x_mm"])
        y_mm = np.asarray(d["y_mm"])
        if roi_mask is None or roi_mask.shape != amp.shape:
            roi_mask = build_roi_mask(x_mm, y_mm, ROI_W_MM, ROI_H_MM,
                                       ROI_CX_MM, ROI_CY_MM)
            print(f"ROI mask: {int(roi_mask.sum())} pixels")
        meta = json.loads(meta_p.read_text())
        t_iso = meta.get("timestamp_iso", "")
        try:
            t_dt = pd.Timestamp(t_iso)
            if t_dt.tzinfo is None:
                t_dt = t_dt.tz_localize("UTC")
        except Exception:
            continue
        rows.append({
            "scan_id":   meta.get("scan_id"),
            "time_utc":  t_dt,
            "amp_mean":  float(np.nanmean(amp[roi_mask])),
            "tof_mean":  float(np.nanmean(tof[roi_mask])),
            "eng_mean":  float(np.nanmean(eng[roi_mask])),
        })

    df = pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)
    print(f"feature rows: {len(df)}")

    # ----- TC-08 trace -----
    temp_csv = args.temp_csv
    if temp_csv is None:
        # auto-find: most recent tc08_longrun_cycling_*
        cands = sorted((PROJECT / "data/raw/temperature").glob("tc08_longrun_cycling_*"))
        if cands:
            temp_csv = cands[-1] / "temperature_log.csv"
    if temp_csv and Path(temp_csv).exists():
        temp_df = pd.read_csv(temp_csv)
        temp_df["time_utc"] = pd.to_datetime(temp_df["timestamp_iso"])
        print(f"temp samples: {len(temp_df)}, "
              f"{temp_df['ch1_c'].min():.2f}–{temp_df['ch1_c'].max():.2f}°C")
    else:
        temp_df = None
        print("no temperature CSV found")

    # ----- plot -----
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.5), dpi=160, sharex=True,
                             gridspec_kw={"hspace": 0.18, "top": 0.94,
                                          "bottom": 0.07, "left": 0.07,
                                          "right": 0.93})
    features = [
        ("amp_mean", "amplitude (V)",   "tab:blue"),
        ("tof_mean", "ToF (µs)",         "tab:red"),
        ("eng_mean", "energy",            "tab:purple"),
    ]
    t = df["time_utc"]
    for ax, (col, ylabel, color) in zip(axes, features):
        ax.plot(t, df[col], "-", color=color, linewidth=1.5,
                marker="o", markersize=3.5, label="ROI-mean")
        ax.set_ylabel(ylabel, color=color, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=8)

        # temperature overlay
        if temp_df is not None:
            ax2 = ax.twinx()
            ax2.plot(temp_df["time_utc"], temp_df["ch1_c"],
                     "-", color="gray", linewidth=0.8, alpha=0.65,
                     label="cell temp")
            ax2.set_ylabel("cell temp (°C)", color="gray", fontsize=10)
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)

    axes[-1].set_xlabel("time (UTC)", fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m-%d"))
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.suptitle(
        f"22-hour continuous C-scan + cycling overview  —  "
        f"{len(df)} scans  ·  ROI-mean over {int(roi_mask.sum()):,} px "
        f"({ROI_W_MM:.1f}×{ROI_H_MM:.1f} mm)",
        fontsize=12, y=0.98)

    out_dir = Path(args.out) if args.out else batch_dir / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "overview.png", bbox_inches="tight")
    fig.savefig(out_dir / "overview.pdf", bbox_inches="tight")
    plt.close(fig)

    # save the underlying CSV so future analysis can re-use without re-loading
    df.to_csv(out_dir / "roi_mean_timeseries.csv", index=False)
    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
