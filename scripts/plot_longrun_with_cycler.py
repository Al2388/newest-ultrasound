"""Longrun overview with full Maccor cycler overlay.

Joins three streams by UTC timestamp:
  - Maccor cycler (voltage, current, capacity, step) — sampled ~1 Hz
  - TC-08 thermocouple (cell temperature) — sampled ~1 Hz
  - C-scan ROI-mean features (amp, ToF, energy) — every ~9 min

Plots a 5-row time-series:
  (1) cycler voltage + cumulative capacity (twin axis)
  (2) ROI-mean amplitude + temperature
  (3) ROI-mean ToF + temperature
  (4) ROI-mean energy + temperature
  (5) cell temperature alone (full resolution)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"

ROI_W_MM = 50.02
ROI_H_MM = 39.78
ROI_CX_MM = 39.59
ROI_CY_MM = 36.00


def build_roi_mask(x_mm, y_mm):
    dx = float(x_mm[1] - x_mm[0])
    dy = float(y_mm[1] - y_mm[0])
    col_lo = max(0, int(round((ROI_CX_MM - ROI_W_MM/2 - float(x_mm[0])) / dx)))
    col_hi = min(len(x_mm), int(round((ROI_CX_MM + ROI_W_MM/2 - float(x_mm[0])) / dx)))
    row_lo = max(0, int(round((ROI_CY_MM - ROI_H_MM/2 - float(y_mm[0])) / dy)))
    row_hi = min(len(y_mm), int(round((ROI_CY_MM + ROI_H_MM/2 - float(y_mm[0])) / dy)))
    m = np.zeros((len(y_mm), len(x_mm)), dtype=bool)
    m[row_lo:row_hi, col_lo:col_hi] = True
    return m


# ---------------------------------------------------------------------------
# Maccor parser
# ---------------------------------------------------------------------------
def parse_maccor(path: Path) -> pd.DataFrame:
    """Maccor tab-delimited TXT → DataFrame with timestamps + voltage + current + capacity."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # Find the header row that starts with "Rec\t"
    hdr_idx = None
    for i, line in enumerate(lines[:30]):
        if line.startswith("Rec\t"):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise SystemExit(f"no Maccor header in {path}")
    cols = [c.strip() for c in lines[hdr_idx].rstrip("\n").split("\t")]

    records = []
    for line in lines[hdr_idx + 1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < len(cols):
            continue
        records.append([p.strip() for p in parts])
    df = pd.DataFrame(records, columns=cols)

    # Cast numerics
    for c in ["Capacity", "Energy", "Current", "Voltage"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # "DPT Time" is dd/mm/yyyy HH:MM:SS
    df["time_utc"] = pd.to_datetime(df["DPT Time"], format="%d/%m/%Y %H:%M:%S",
                                     errors="coerce", utc=True)
    df = df.dropna(subset=["time_utc"]).reset_index(drop=True)
    df["MD"] = df["MD"].astype(str)
    return df


def cumulative_capacity_ah(df: pd.DataFrame) -> np.ndarray:
    """Net signed capacity (Ah) — positive = charge, negative = discharge.
    Maccor's per-step Capacity field resets each step, so we cumsum the per-step
    increments using ΔQ = current × Δt (Ah), with sign from MD (C=+, D=-, R=0)."""
    t = df["time_utc"].astype("int64") / 1e9   # seconds
    dt = t.diff().fillna(0.0)
    # signed current per row
    sign = np.where(df["MD"] == "C", 1.0,
            np.where(df["MD"] == "D", -1.0, 0.0))
    i_signed = df["Current"].fillna(0.0) * sign
    dq_ah = i_signed * dt / 3600.0   # Δt is in seconds → Ah
    return dq_ah.cumsum().to_numpy()


# ---------------------------------------------------------------------------
# C-scan ROI-mean loader
# ---------------------------------------------------------------------------
def load_cscan_features(batch_dir: Path) -> pd.DataFrame:
    overview_csv = batch_dir / "overview" / "roi_mean_timeseries.csv"
    if overview_csv.exists():
        df = pd.read_csv(overview_csv)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df.sort_values("time_utc").reset_index(drop=True)

    # Otherwise recompute from scratch
    cp = json.loads((batch_dir / "checkpoint.json").read_text())
    rows = []
    roi_mask = None
    for entry in cp["completed"]:
        scan_dir = Path(entry["session_dir"])
        if not scan_dir.is_absolute():
            scan_dir = PROJECT / scan_dir
        npz_p = next(scan_dir.glob("scan_*.npz"), None)
        meta_p = next(scan_dir.glob("scan_*_meta.json"), None)
        if npz_p is None or meta_p is None:
            continue
        d = np.load(npz_p)
        if roi_mask is None or roi_mask.shape != d["amplitude"].shape:
            roi_mask = build_roi_mask(d["x_mm"], d["y_mm"])
        meta = json.loads(meta_p.read_text())
        rows.append({
            "scan_id":  meta["scan_id"],
            "time_utc": pd.Timestamp(meta["timestamp_iso"], tz="UTC"),
            "amp_mean": float(np.nanmean(d["amplitude"][roi_mask])),
            "tof_mean": float(np.nanmean(d["tof"][roi_mask])),
            "eng_mean": float(np.nanmean(d["energy"][roi_mask])),
        })
    return pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# TC-08 loader
# ---------------------------------------------------------------------------
def load_temperature(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["time_utc"] = pd.to_datetime(df["timestamp_iso"], utc=True)
    return df.sort_values("time_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--maccor",    required=True)
    ap.add_argument("--temp-csv",  default=None)
    ap.add_argument("--out",       default=None)
    ap.add_argument("--nominal-ah", type=float, default=0.860,
                    help="cell nominal capacity (Ah) for SoC computation")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    cyc = parse_maccor(Path(args.maccor))
    cyc["cum_q_ah"] = cumulative_capacity_ah(cyc)
    print(f"Maccor: {len(cyc)} rows, "
          f"{cyc['time_utc'].iloc[0]} -> {cyc['time_utc'].iloc[-1]}")
    print(f"  Voltage: {cyc['Voltage'].min():.3f}–{cyc['Voltage'].max():.3f} V")
    print(f"  Current: {cyc['Current'].min():.3f}–{cyc['Current'].max():.3f} A")
    print(f"  Net charge passed: {cyc['cum_q_ah'].iloc[-1]*1000:+.1f} mAh "
          f"(= {cyc['cum_q_ah'].iloc[-1]*1000/args.nominal_ah:.1f}% of nominal {args.nominal_ah*1000:.0f} mAh)")

    feat = load_cscan_features(batch_dir)
    print(f"C-scan: {len(feat)} feature rows")

    temp_csv = args.temp_csv
    if temp_csv is None:
        cands = sorted((PROJECT / "data/raw/temperature").glob("tc08_longrun_cycling_*"))
        if cands:
            temp_csv = cands[-1] / "temperature_log.csv"
    temp_df = load_temperature(Path(temp_csv)) if temp_csv and Path(temp_csv).exists() else None
    if temp_df is not None:
        print(f"TC-08: {len(temp_df)} rows, {temp_df['ch1_c'].min():.2f}–{temp_df['ch1_c'].max():.2f}°C")

    # ----- combined 5-row plot -----
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), dpi=160, sharex=True,
                             gridspec_kw={"hspace": 0.18, "top": 0.96,
                                          "bottom": 0.05, "left": 0.07,
                                          "right": 0.93,
                                          "height_ratios": [1.2, 1, 1, 1, 0.8]})

    # (1) Voltage + cum charge
    ax = axes[0]
    ax.plot(cyc["time_utc"], cyc["Voltage"], "-", color="tab:blue", linewidth=1.1,
            label="cell V")
    ax.set_ylabel("voltage (V)", color="tab:blue", fontsize=10)
    ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax2 = ax.twinx()
    ax2.plot(cyc["time_utc"], cyc["cum_q_ah"] * 1000, "-",
             color="tab:green", linewidth=1.3, alpha=0.85, label="net charge")
    ax2.set_ylabel("net charge passed (mAh)", color="tab:green", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="tab:green", labelsize=8)
    ax.set_title("Cycler — voltage (blue) + net cumulative charge (green, charge − discharge)",
                 fontsize=10, loc="left")

    # (2-4) ROI-mean features + temperature
    feat_panels = [
        ("amp_mean", "(ROI-mean) amplitude (V)", "tab:blue",   axes[1]),
        ("tof_mean", "(ROI-mean) ToF (µs)",       "tab:red",    axes[2]),
        ("eng_mean", "(ROI-mean) energy",          "tab:purple", axes[3]),
    ]
    for col, ylabel, color, ax in feat_panels:
        ax.plot(feat["time_utc"], feat[col], "o-", color=color, linewidth=1.4,
                markersize=4, label="ROI-mean")
        ax.set_ylabel(ylabel, color=color, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        if temp_df is not None:
            ax2 = ax.twinx()
            ax2.plot(temp_df["time_utc"], temp_df["ch1_c"], "-", color="gray",
                     linewidth=0.7, alpha=0.6)
            ax2.set_ylabel("temp (°C)", color="gray", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)

    # (5) Temperature alone (full res)
    ax = axes[4]
    if temp_df is not None:
        ax.plot(temp_df["time_utc"], temp_df["ch1_c"], "-", color="gray", linewidth=1.0)
        ax.set_ylabel("cell temp (°C)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.set_title(f"Cell temperature  —  range {temp_df['ch1_c'].min():.2f}–"
                     f"{temp_df['ch1_c'].max():.2f}°C", fontsize=10, loc="left")
    ax.set_xlabel("time (UTC)", fontsize=10)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m-%d"))

    fig.suptitle(
        f"22-hour C-scan + Maccor cycling + temperature  —  {len(feat)} scans on "
        f"{int(np.sum(build_roi_mask(np.linspace(0,80,500), np.linspace(0,72,144)))):,} px ROI",
        fontsize=12, y=0.99)

    out_dir = Path(args.out) if args.out else (batch_dir / "overview")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "longrun_with_cycler.png", bbox_inches="tight")
    fig.savefig(out_dir / "longrun_with_cycler.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- also: ToF vs Voltage scatter (the headline science plot) ----
    # Interpolate cycler voltage and net charge to each scan timestamp
    feat["voltage_at_scan"] = np.interp(
        feat["time_utc"].astype("int64") / 1e9,
        cyc["time_utc"].astype("int64") / 1e9,
        cyc["Voltage"].to_numpy(),
    )
    feat["q_at_scan_mah"] = np.interp(
        feat["time_utc"].astype("int64") / 1e9,
        cyc["time_utc"].astype("int64") / 1e9,
        cyc["cum_q_ah"].to_numpy() * 1000,
    )
    feat["soc_pct"] = 100.0 * feat["q_at_scan_mah"] / (args.nominal_ah * 1000)
    if temp_df is not None:
        feat["temp_at_scan"] = np.interp(
            feat["time_utc"].astype("int64") / 1e9,
            temp_df["time_utc"].astype("int64") / 1e9,
            temp_df["ch1_c"].to_numpy(),
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160,
                             gridspec_kw={"wspace": 0.28, "top": 0.86,
                                          "bottom": 0.13, "left": 0.06,
                                          "right": 0.97})
    color_by = feat["temp_at_scan"] if "temp_at_scan" in feat.columns else feat["soc_pct"]
    cbar_label = "cell temp (°C)" if "temp_at_scan" in feat.columns else "SoC (%)"
    for ax, col, ylabel in zip(axes,
                                ("amp_mean", "tof_mean", "eng_mean"),
                                ("ROI-mean amplitude (V)", "ROI-mean ToF (µs)", "ROI-mean energy")):
        sc = ax.scatter(feat["soc_pct"], feat[col], c=color_by, cmap="viridis",
                        s=24, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(f"SoC (% of {args.nominal_ah*1000:.0f} mAh nominal)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=8)
        plt.colorbar(sc, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    fig.suptitle(f"ROI-mean feature vs SoC (color = {cbar_label}) — 147 scans across 22 h",
                 fontsize=12, y=0.98)
    fig.savefig(out_dir / "feature_vs_soc.png", bbox_inches="tight")
    fig.savefig(out_dir / "feature_vs_soc.pdf", bbox_inches="tight")
    plt.close(fig)

    feat.to_csv(out_dir / "scans_with_cycler.csv", index=False)
    print(f"saved: {out_dir}")
    print(f"  feature_vs_soc.png / .pdf   ← headline scatter")
    print(f"  longrun_with_cycler.png / .pdf  ← 5-row time series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
