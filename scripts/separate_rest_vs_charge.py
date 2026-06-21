"""Tag each C-scan as rest / charge / discharge / transition by intersecting
its 9-minute acquisition window with the Maccor cycler MD codes.

Outputs:
  - scans_tagged.csv         : feature rows with `step_tag` column
  - timeseries_tagged.png    : ROI-mean time series with scans colour-coded
  - feature_vs_soc_rest.png  : ToF/amp/energy vs SoC, REST scans only +
                                rest-line fit overlay vs all-scans cloud
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

# Acquisition window for each scan (from batch elapsed_s ~ 540 s)
SCAN_DURATION_S = 540

TAG_COLORS = {
    "rest":        "#16a34a",   # green
    "charge":      "#dc2626",   # red
    "discharge":   "#2563eb",   # blue
    "transition":  "#6b7280",   # gray
}


# ---------------------------------------------------------------------------
def parse_maccor(path: Path) -> pd.DataFrame:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    hdr_idx = next(i for i, l in enumerate(lines[:30]) if l.startswith("Rec\t"))
    cols = [c.strip() for c in lines[hdr_idx].split("\t")]
    rows = [[p.strip() for p in l.split("\t")] for l in lines[hdr_idx + 1:]
            if len(l.split("\t")) >= len(cols)]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("Capacity", "Energy", "Current", "Voltage"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time_utc"] = pd.to_datetime(df["DPT Time"], format="%d/%m/%Y %H:%M:%S",
                                     errors="coerce", utc=True)
    df = df.dropna(subset=["time_utc"]).reset_index(drop=True)
    df["MD"] = df["MD"].astype(str)
    # cumulative signed charge (Ah)
    sign = np.where(df["MD"] == "C", 1.0,
            np.where(df["MD"] == "D", -1.0, 0.0))
    dt = df["time_utc"].astype("int64").diff().fillna(0) / 1e9
    df["cum_q_ah"] = (df["Current"].fillna(0.0) * sign * dt / 3600).cumsum()
    return df


def tag_scan(maccor: pd.DataFrame, scan_t0: pd.Timestamp,
             scan_dur_s: float = SCAN_DURATION_S) -> str:
    """Return rest / charge / discharge / transition based on Maccor MD codes
    in [scan_t0, scan_t0 + dur]."""
    t1 = scan_t0 + pd.Timedelta(seconds=scan_dur_s)
    sub = maccor[(maccor["time_utc"] >= scan_t0) & (maccor["time_utc"] <= t1)]
    if sub.empty:
        return "transition"
    md = set(sub["MD"].unique())
    if md == {"R"}:
        return "rest"
    if md == {"C"}:
        return "charge"
    if md == {"D"}:
        return "discharge"
    return "transition"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--maccor",    required=True)
    ap.add_argument("--nominal-ah", type=float, default=0.860)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    maccor = parse_maccor(Path(args.maccor))
    print(f"Maccor: {len(maccor)} rows, MD codes: {sorted(maccor['MD'].unique())}")

    # ROI-mean timeseries already on disk
    feat = pd.read_csv(batch_dir / "overview" / "scans_with_cycler.csv")
    feat["time_utc"] = pd.to_datetime(feat["time_utc"], utc=True)

    # ---- tag each scan ----
    feat["step_tag"] = feat["time_utc"].apply(
        lambda t: tag_scan(maccor, t)
    )
    tag_counts = feat["step_tag"].value_counts()
    print("\nscan-window tag distribution:")
    for tag, n in tag_counts.items():
        print(f"  {tag:12s}: {n}")

    out_dir = Path(args.out) if args.out else batch_dir / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_dir / "scans_tagged.csv", index=False)

    # ----------------------------------------------------------- timeseries
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), dpi=160, sharex=True,
                             gridspec_kw={"hspace": 0.18, "top": 0.95,
                                          "bottom": 0.06, "left": 0.07,
                                          "right": 0.96,
                                          "height_ratios": [1.1, 1, 1, 1]})
    # (1) Voltage + cum charge
    ax = axes[0]
    ax.plot(maccor["time_utc"], maccor["Voltage"], "-", color="tab:blue",
            linewidth=1.0, label="V")
    ax.set_ylabel("voltage (V)", color="tab:blue", fontsize=10)
    ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax2 = ax.twinx()
    ax2.plot(maccor["time_utc"], maccor["cum_q_ah"] * 1000, "-",
             color="tab:green", linewidth=1.2, alpha=0.85)
    ax2.set_ylabel("net charge (mAh)", color="tab:green", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="tab:green", labelsize=8)
    ax.set_title("Cycler protocol — voltage (blue) + net charge (green)",
                 fontsize=10, loc="left")

    # (2-4) features with scans colour-coded by tag
    for ax, (col, ylabel) in zip(axes[1:],
                                  [("amp_mean", "ROI-mean amplitude (V)"),
                                   ("tof_mean", "ROI-mean ToF (µs)"),
                                   ("eng_mean", "ROI-mean energy")]):
        for tag, sub in feat.groupby("step_tag"):
            ax.plot(sub["time_utc"], sub[col], "o",
                    color=TAG_COLORS.get(tag, "#000"),
                    markersize=5, alpha=0.9, markeredgecolor="black",
                    markeredgewidth=0.3, label=tag)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=8)
        if ax is axes[1]:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("time (UTC)", fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m-%d"))
    fig.suptitle(f"Scans tagged by cycler step  —  "
                 f"rest={tag_counts.get('rest',0)}, charge={tag_counts.get('charge',0)}, "
                 f"discharge={tag_counts.get('discharge',0)}, "
                 f"transition={tag_counts.get('transition',0)}",
                 fontsize=12, y=0.98)
    fig.savefig(out_dir / "timeseries_tagged.png", bbox_inches="tight")
    fig.savefig(out_dir / "timeseries_tagged.pdf", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------- feature vs SoC (rest only)
    rest = feat[feat["step_tag"] == "rest"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160,
                             gridspec_kw={"wspace": 0.28, "top": 0.86,
                                          "bottom": 0.13, "left": 0.06,
                                          "right": 0.97})
    for ax, col, ylabel, unit in zip(
        axes,
        ("amp_mean", "tof_mean", "eng_mean"),
        ("ROI-mean amplitude", "ROI-mean ToF", "ROI-mean energy"),
        ("V", "µs", "")):
        # all scans, faint
        ax.scatter(feat["soc_pct"], feat[col], c="#cbd5e1", s=14,
                   edgecolor="none", label="all scans (n=%d)" % len(feat),
                   zorder=1)
        # rest scans, colored by temperature
        sc = ax.scatter(rest["soc_pct"], rest[col],
                        c=rest["temp_at_scan"], cmap="viridis",
                        s=35, edgecolor="black", linewidth=0.4,
                        label="rest only (n=%d)" % len(rest), zorder=3)
        # linear fit on rest-only
        if len(rest) >= 3:
            xs = rest["soc_pct"].to_numpy()
            ys = rest[col].to_numpy()
            slope, intercept = np.polyfit(xs, ys, 1)
            xx = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xx, slope * xx + intercept, "--", color="black",
                    linewidth=1.2, alpha=0.7,
                    label=f"linear: {slope:+.4g} {unit}/%SoC")
        ax.set_xlabel(f"SoC (% of {args.nominal_ah*1000:.0f} mAh nominal)",
                      fontsize=10)
        ax.set_ylabel(f"{ylabel} ({unit})" if unit else ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="best")
        plt.colorbar(sc, ax=ax, label="cell temp (°C)", fraction=0.046, pad=0.04)
    fig.suptitle("ROI-mean features vs SoC — rest scans (color = temp) vs all scans (grey)",
                 fontsize=12, y=0.97)
    fig.savefig(out_dir / "feature_vs_soc_rest.png", bbox_inches="tight")
    fig.savefig(out_dir / "feature_vs_soc_rest.pdf", bbox_inches="tight")
    plt.close(fig)

    # report
    rep = [
        "# Rest vs charge separation report\n",
        f"Batch: `{batch_dir.name}`",
        f"Maccor file: `{Path(args.maccor).name}`",
        f"Scan duration window: {SCAN_DURATION_S} s",
        "",
        "## Tag distribution",
        "",
    ]
    for tag, n in tag_counts.items():
        rep.append(f"- {tag}: {n}")
    rep += [
        "",
        f"## Rest-only linear fits (n={len(rest)})",
        "",
        f"| Feature | slope per %SoC | intercept | dynamic range over rest |",
        f"|---|---:|---:|---:|",
    ]
    for col, name, unit in [("amp_mean", "amplitude", "V"),
                              ("tof_mean", "ToF", "µs"),
                              ("eng_mean", "energy", "")]:
        if len(rest) >= 3:
            xs = rest["soc_pct"].to_numpy()
            ys = rest[col].to_numpy()
            slope, intercept = np.polyfit(xs, ys, 1)
            dyn = ys.max() - ys.min()
            rep.append(f"| {name} | {slope:+.4g} {unit}/%SoC | {intercept:+.4g} {unit} | {dyn:.4g} {unit} |")
    (out_dir / "rest_vs_charge_report.md").write_text("\n".join(rep), encoding="utf-8")

    print(f"\nsaved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
