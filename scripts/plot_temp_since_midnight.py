"""Plot cell temperature from midnight (2026-05-31 00:00) to now,
using both the watchdog CSV (10 s) and the TC-08 service log (1 Hz)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
WATCHDOG_CSV = PROJECT / "data/raw/temperature/watchdog_2026-05-30_19-46-40/safety_temperature.csv"
TC08_CSV = PROJECT / "data/raw/temperature/tc08_longrun_cycling_2026-05-29_18-37-03/temperature_log.csv"

T0 = datetime(2026, 5, 31, 0, 0, 0)
T1 = datetime.now()


def load_watchdog() -> pd.DataFrame:
    df = pd.read_csv(WATCHDOG_CSV)
    df["t"] = pd.to_datetime(df["iso"])
    df = df[(df["t"] >= T0) & (df["t"] <= T1)].copy()
    df = df.dropna(subset=["cell_c"])
    return df


def load_tc08() -> pd.DataFrame:
    df = pd.read_csv(TC08_CSV)
    # try common time columns
    for col in ("iso", "timestamp", "timestamp_iso", "datetime"):
        if col in df.columns:
            df["t"] = pd.to_datetime(df[col])
            break
    else:
        df["t"] = pd.to_datetime(df.iloc[:, 1])
    # temperature column
    for col in ("temperature_mean_c", "temperature_c", "cell_c", "T_c", "temp_c"):
        if col in df.columns:
            df = df.rename(columns={col: "cell_c"})
            break
    df = df[(df["t"] >= T0) & (df["t"] <= T1)].copy()
    df = df.dropna(subset=["cell_c"])
    return df


def main() -> None:
    wd = load_watchdog()
    print(f"watchdog rows midnight..now: {len(wd)}")
    try:
        tc = load_tc08()
        print(f"tc08    rows midnight..now: {len(tc)}")
    except Exception as exc:
        print(f"could not load tc08: {exc}")
        tc = None

    src = tc if tc is not None and len(tc) > len(wd) else wd
    label = "TC-08 (1 Hz)" if src is tc else "watchdog (10 s)"

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(src["t"], src["cell_c"], color="tab:blue", lw=0.8, label=label)
    ax.axhline(35.0, color="k", lw=0.6, ls="--", alpha=0.5, label="target 35.0 °C")
    ax.axhline(38.0, color="tab:orange", lw=0.6, ls=":", alpha=0.7, label="WARN 38 °C")
    ax.axhline(45.0, color="tab:red", lw=0.6, ls=":", alpha=0.7, label="TRIP 45 °C")
    ax.set_ylabel("Cell temperature [°C]")
    ax.set_title(f"Cell temperature 00:00 → {T1.strftime('%H:%M:%S')} on {T1.date()}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # tight y range so 35 ± 0.5 detail is visible
    lo = float(src["cell_c"].min()) - 0.15
    hi = float(src["cell_c"].max()) + 0.15
    ax.set_ylim(lo, hi)

    # Drift residual
    ax2 = axes[1]
    resid = src["cell_c"] - 35.0
    ax2.plot(src["t"], resid * 1000, color="tab:gray", lw=0.6)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("Δ from 35 °C [m°C]")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.set_xlabel("local time")

    fig.tight_layout()
    out = PROJECT / "reports/experiments/temp_since_midnight_2026-05-31.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"figure -> {out}")

    # summary stats
    arr = src["cell_c"].values
    print(f"\nN={len(arr)}  mean={arr.mean():.3f}  median={np.median(arr):.3f}")
    print(f"min={arr.min():.3f}  max={arr.max():.3f}  std={arr.std():.4f}  ptp={arr.ptp():.3f}")
    print(f"first={arr[0]:.3f} at {src['t'].iloc[0]}")
    print(f"last ={arr[-1]:.3f} at {src['t'].iloc[-1]}")


if __name__ == "__main__":
    main()
