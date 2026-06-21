"""Compare cell-temperature control between the 30 C and 36 C legs.

35 C / 36 C leg:  watchdog_2026-05-30_19-46-40  (full run, batch started shortly after)
30 C leg:         watchdog_2026-05-31_21-26-33  (use only samples after batch launch 22:23:14)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")

LEG_36 = PROJECT / "data/raw/temperature/watchdog_2026-05-30_19-46-40/safety_temperature.csv"
LEG_30 = PROJECT / "data/raw/temperature/watchdog_2026-05-31_21-26-33/safety_temperature.csv"

# When the batch actually started in the 30 C log -> drop cooldown.
# The first ~1.3 h after batch start contained a PID overshoot (22:46-23:42),
# so we trim an extra 90 min for a fair "steady-state vs steady-state" comparison.
LEG_30_BATCH_START   = datetime(2026, 5, 31, 22, 23, 14)
LEG_30_STEADY_START  = datetime(2026, 5, 31, 23, 53, 14)   # batch_start + 1.5 h


def load(path: Path, start: datetime | None = None) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["cell_c"])
    df["t"] = pd.to_datetime(df["iso"])
    if start is not None:
        df = df[df["t"] >= start].copy()
    df["elapsed_h"] = (df["t"] - df["t"].iloc[0]).dt.total_seconds() / 3600.0
    return df


def stats(df: pd.DataFrame) -> dict:
    a = df["cell_c"].values
    span_h = float(df["elapsed_h"].iloc[-1])
    return {
        "N": len(a),
        "span_h": span_h,
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "ptp": float(a.ptp()),
        "first": float(a[0]),
        "last": float(a[-1]),
        "drift_per_h": float((a[-1] - a[0]) / span_h) if span_h > 0 else float("nan"),
    }


def main() -> None:
    df36 = load(LEG_36)
    df30 = load(LEG_30, LEG_30_STEADY_START)
    s36, s30 = stats(df36), stats(df30)

    print("=" * 70)
    print(f"{'metric':22}  {'36 C leg':>14}  {'30 C leg':>14}")
    print("-" * 70)
    print(f"{'N samples':22}  {s36['N']:>14d}  {s30['N']:>14d}")
    print(f"{'span [h]':22}  {s36['span_h']:>14.2f}  {s30['span_h']:>14.2f}")
    print(f"{'mean cell [C]':22}  {s36['mean']:>14.3f}  {s30['mean']:>14.3f}")
    print(f"{'median cell [C]':22}  {s36['median']:>14.3f}  {s30['median']:>14.3f}")
    print(f"{'sigma cell [C]':22}  {s36['std']:>14.4f}  {s30['std']:>14.4f}")
    print(f"{'min [C]':22}  {s36['min']:>14.3f}  {s30['min']:>14.3f}")
    print(f"{'max [C]':22}  {s36['max']:>14.3f}  {s30['max']:>14.3f}")
    print(f"{'peak-to-peak [C]':22}  {s36['ptp']:>14.3f}  {s30['ptp']:>14.3f}")
    print(f"{'drift  [mC / h]':22}  {s36['drift_per_h']*1000:>14.1f}  {s30['drift_per_h']*1000:>14.1f}")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))

    # 1) time series (each on its own elapsed-h axis)
    ax = axes[0, 0]
    ax.plot(df36["elapsed_h"], df36["cell_c"], color="tab:red", lw=0.6,
            label=f"36 C leg (set 35)  σ={s36['std']:.3f}")
    ax.plot(df30["elapsed_h"], df30["cell_c"], color="tab:blue", lw=0.6,
            label=f"30 C leg (set 29)  σ={s30['std']:.3f}")
    ax.set_xlabel("elapsed [h]")
    ax.set_ylabel("cell temperature [°C]")
    ax.set_title("Cell temperature vs elapsed time")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2) deviation from each leg's own mean (zoomed)
    ax = axes[0, 1]
    ax.plot(df36["elapsed_h"], (df36["cell_c"] - s36["mean"]) * 1000,
            color="tab:red", lw=0.5, label="36 C")
    ax.plot(df30["elapsed_h"], (df30["cell_c"] - s30["mean"]) * 1000,
            color="tab:blue", lw=0.5, label="30 C")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("elapsed [h]")
    ax.set_ylabel("Δ from leg-mean [m°C]")
    ax.set_title("Deviation from each leg's mean")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) histogram
    ax = axes[1, 0]
    bins36 = np.linspace(s36["mean"] - 0.5, s36["mean"] + 0.5, 60)
    bins30 = np.linspace(s30["mean"] - 0.5, s30["mean"] + 0.5, 60)
    ax.hist(df36["cell_c"], bins=bins36, color="tab:red", alpha=0.5,
            density=True, label=f"36 C  μ={s36['mean']:.2f}")
    ax.hist(df30["cell_c"], bins=bins30, color="tab:blue", alpha=0.5,
            density=True, label=f"30 C  μ={s30['mean']:.2f}")
    ax.set_xlabel("cell temperature [°C]")
    ax.set_ylabel("density")
    ax.set_title("Distribution (both centered near own mean)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) numeric summary table
    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ("samples",       f"{s36['N']:d}",                          f"{s30['N']:d}"),
        ("duration [h]",  f"{s36['span_h']:.2f}",                   f"{s30['span_h']:.2f}"),
        ("mean [°C]",     f"{s36['mean']:.3f}",                     f"{s30['mean']:.3f}"),
        ("σ [°C]",        f"{s36['std']:.4f}",                      f"{s30['std']:.4f}"),
        ("peak-pk [°C]",  f"{s36['ptp']:.3f}",                      f"{s30['ptp']:.3f}"),
        ("drift [mC/h]",  f"{s36['drift_per_h']*1000:+.1f}",        f"{s30['drift_per_h']*1000:+.1f}"),
        ("min / max",     f"{s36['min']:.2f} / {s36['max']:.2f}",   f"{s30['min']:.2f} / {s30['max']:.2f}"),
    ]
    tbl = ax.table(cellText=[[k, a, b] for k, a, b in rows],
                   colLabels=["metric", "36 C leg", "30 C leg"],
                   loc="center", cellLoc="center", colWidths=[0.4, 0.3, 0.3])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    ax.set_title("Summary")

    fig.suptitle(
        f"Cell-temperature control: 36 C leg vs 30 C leg "
        f"(30 C: post-overshoot, t > {LEG_30_STEADY_START.strftime('%H:%M')})",
        fontsize=11, y=1.01)
    fig.tight_layout()
    out = PROJECT / "reports/experiments/temp_control_30c_vs_36c_steady.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nfigure -> {out}")

    # Also compute 9-min rolling SD comparison
    df36s = df36.set_index("t").sort_index()
    df30s = df30.set_index("t").sort_index()
    r36 = df36s["cell_c"].rolling("9min").std().dropna()
    r30 = df30s["cell_c"].rolling("9min").std().dropna()
    print("\n9-min rolling SD (per-scan smearing proxy):")
    print(f"  36 C steady: mean={r36.mean():.3f}  median={r36.median():.3f}  p95={r36.quantile(.95):.3f}")
    print(f"  30 C steady: mean={r30.mean():.3f}  median={r30.median():.3f}  p95={r30.quantile(.95):.3f}")


if __name__ == "__main__":
    main()
