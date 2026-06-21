"""For each rest plateau, summarise: SoC (raw + corrected to 0% = discharged
to 2.5V), voltage, temperature, n_scans, duration. Output a table + plot."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def group_plateaus(rest: pd.DataFrame, max_gap_s: float = 1200.0):
    rest = rest.sort_values("time_utc").reset_index(drop=True).copy()
    t = rest["time_utc"].astype("int64") / 1e9
    plateau_id = (t.diff().fillna(0) > max_gap_s).cumsum()
    return [g.reset_index(drop=True) for _, g in rest.groupby(plateau_id)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    feat = pd.read_csv(batch_dir / "overview" / "scans_tagged.csv")
    feat["time_utc"] = pd.to_datetime(feat["time_utc"], utc=True)
    rest = feat[feat["step_tag"] == "rest"].copy()
    plateaus = group_plateaus(rest)

    # Discharge plateau (lowest SoC) = the cell's true 0% SoC reference
    plateau_socs = np.array([p["soc_pct"].mean() for p in plateaus])
    discharge_idx = int(np.argmin(plateau_socs))
    offset = -float(plateau_socs[discharge_idx])
    print(f"Discharge plateau = plateau {discharge_idx} (raw SoC = {plateau_socs[discharge_idx]:+.2f}%)")
    print(f"Offset to apply: +{offset:.2f}%  -> true SoC = raw SoC + offset\n")

    # Build table
    rows = []
    for i, p in enumerate(plateaus):
        first = p.iloc[0]
        last = p.iloc[-1]
        dur_min = (last["time_utc"] - first["time_utc"]).total_seconds() / 60.0
        rows.append({
            "plateau":       i,
            "n_scans":       len(p),
            "duration_min":  dur_min,
            "soc_raw_pct":   p["soc_pct"].mean(),
            "soc_true_pct":  p["soc_pct"].mean() + offset,
            "voltage_mean":  p["voltage_at_scan"].mean(),
            "voltage_last":  last["voltage_at_scan"],
            "temp_mean":     p["temp_at_scan"].mean() if "temp_at_scan" in p.columns else np.nan,
            "temp_last":     last["temp_at_scan"] if "temp_at_scan" in p.columns else np.nan,
            "tof_mean_us":   p["tof_mean"].mean(),
            "tof_last_us":   last["tof_mean"],
            "amp_last_v":    last["amp_mean"],
            "eng_last":      last["eng_mean"],
            "time_start":    first["time_utc"],
            "time_end":      last["time_utc"],
            "last_scan_id":  last["scan_id"],
        })
    df = pd.DataFrame(rows)

    # Print human-readable table
    print(f"{'plateau':>7} {'n':>4} {'dur(min)':>9} {'SoC raw':>9} {'SoC true':>10} "
          f"{'V last':>8} {'T last':>8} {'ToF last (us)':>15} {'last scan_id':>30}")
    print("-" * 105)
    for r in rows:
        print(f"{r['plateau']:>7} {r['n_scans']:>4} {r['duration_min']:>9.1f} "
              f"{r['soc_raw_pct']:>+9.2f} {r['soc_true_pct']:>+10.2f} "
              f"{r['voltage_last']:>8.3f} {r['temp_last']:>8.2f} "
              f"{r['tof_last_us']:>15.4f} {r['last_scan_id']:>30}")

    out_dir = Path(args.out) if args.out else batch_dir / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "rest_plateaus_summary.csv", index=False)

    # ----- plot: each plateau visualized -----
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=160, sharex=False,
                             gridspec_kw={"hspace": 0.30, "top": 0.93,
                                          "bottom": 0.10, "left": 0.10,
                                          "right": 0.95,
                                          "height_ratios": [1.0, 1.0]})

    # Top: SoC plateaus over time (showing flatness during rest)
    ax = axes[0]
    for _, r in df.iterrows():
        sub = rest[(rest["time_utc"] >= r["time_start"]) & (rest["time_utc"] <= r["time_end"])]
        ax.plot(sub["time_utc"], sub["soc_pct"] + offset, "o-",
                linewidth=1.4, markersize=5, label=f"plateau {int(r['plateau'])}")
        # Annotate with last-scan SoC
        ax.text(r["time_end"], r["soc_true_pct"] + 1.5,
                f"{r['soc_true_pct']:+.1f}%",
                fontsize=9, ha="right", va="bottom", fontweight="bold")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("true SoC (% of 860 mAh, 0% = 2.5V discharge)")
    ax.set_title("Rest plateaus over time — each dot is one C-scan", fontsize=11, loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=8)

    # Bottom: ToF at last-rest-scan vs true SoC
    ax = axes[1]
    ax.plot(df["soc_true_pct"], df["tof_last_us"] * 1000, "o-",
            color="#dc2626", linewidth=1.5, markersize=10,
            markerfacecolor="white", markeredgewidth=1.5)
    for _, r in df.iterrows():
        ax.text(r["soc_true_pct"], r["tof_last_us"] * 1000 + 0.5,
                f"V={r['voltage_last']:.3f}V\nT={r['temp_last']:.2f}°C",
                fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("true SoC (%)")
    ax.set_ylabel("ToF (last scan in plateau) — ns from sync")
    ax.set_title("ToF at quasi-equilibrium vs SoC — 5 plateaus", fontsize=11, loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=8)

    fig.suptitle(f"Rest plateau summary  —  batch `{batch_dir.name}`\n"
                 f"raw SoC was offset by {offset:+.2f}% (cell discharged to "
                 f"2.5V = true 0% SoC)",
                 fontsize=11, y=0.98)

    fig.savefig(out_dir / "rest_plateaus_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "rest_plateaus_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
