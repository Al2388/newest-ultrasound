"""C-scan time-series animation across the 35C charge-focus subset.

Mirrors `scripts/longrun_analysis/analysis_06_animation.py` (the 25C one):
  - 4-panel layout: C-scan + SOC trajectory + V trajectory + T trajectory
  - Per-frame moving dot on each trajectory
  - Title with run idx, time, SOC, V, T, step tag
  - One animation per modality (tof / amplitude / energy)

SOC is pulled from the Maccor cycler file (interpolated at each scan timestamp,
sign-aware via MD column). T from per-scan `line_T_mean_c`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

_orig_imshow = matplotlib.axes.Axes.imshow


def _patched_imshow(self, X, *args, **kwargs):
    extent = kwargs.get("extent")
    if extent is not None and len(extent) == 4 and extent[2] > extent[3]:
        kwargs["extent"] = [extent[0], extent[1], extent[3], extent[2]]
        kwargs.setdefault("origin", "lower")
    return _orig_imshow(self, X, *args, **kwargs)


matplotlib.axes.Axes.imshow = _patched_imshow

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"
OUT = OUT_ROOT / "06_animation"
OUT.mkdir(parents=True, exist_ok=True)

CYCLER = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
NOMINAL_AH = 0.860

UNITS = {"tof": "us", "amplitude": "V", "energy": "(a.u.)"}
TAG_COLOR = {"charge": "tab:red", "rest": "#444"}


def _build_cumulative_soc():
    """Return DataFrame(time, SOC_pct) from the cycler, coulomb-counted with
    sign from MD column."""
    df = pd.read_csv(CYCLER, sep="\t", skiprows=6, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DPT Time"] = pd.to_datetime(df["DPT Time"].astype(str),
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for c in ["Step", "Capacity", "Voltage", "Current"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MD"] = df["MD"].astype(str).str.strip()
    df = df.dropna(subset=["DPT Time"]).reset_index(drop=True)

    mode_sign = {"C": +1.0, "D": -1.0, "R": 0.0, "P": 0.0}
    dq = np.zeros(len(df))
    for _, sub in df.groupby("Step", sort=False):
        idx = sub.index.values
        q_step = sub["Capacity"].values
        d_abs = np.clip(np.diff(q_step, prepend=q_step[0]), 0, None)
        md_majority = sub["MD"].mode().iloc[0]
        sign = mode_sign.get(md_majority, 0.0)
        dq[idx] = d_abs * sign
    Q_cum = np.cumsum(dq)
    soc = (Q_cum / NOMINAL_AH) * 100.0
    soc -= soc.min()       # zero at minimum
    return pd.DataFrame({"time": df["DPT Time"], "SOC_pct": soc})


def _interp_soc_at(soc_df, ts):
    """Linear interpolation of SOC at a given timestamp."""
    t_ref = soc_df["time"].values.astype("datetime64[s]").astype(np.int64)
    ts_int = np.datetime64(ts, "s").astype(np.int64)
    return float(np.interp(ts_int, t_ref, soc_df["SOC_pct"].values))


def main():
    print("loading cache + cycler ...")
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    roi = d["roi_mask"]
    x_mm = d["x_mm"]; y_mm = d["y_mm"]
    arrs = {"tof": d["tof"], "amplitude": d["amplitude"], "energy": d["energy"]}

    soc_df = _build_cumulative_soc()
    # per-scan SOC (interpolate cycler at each scan timestamp)
    soc_per_scan = np.array([_interp_soc_at(soc_df, ts) for ts in meta["timestamp"]])
    print(f"SOC range per scan: {soc_per_scan.min():.1f} -> {soc_per_scan.max():.1f}%")

    t_h = (meta["timestamp"] - meta["timestamp"].iloc[0]).dt.total_seconds().values / 3600.0
    V_arr = meta["voltage_at_scan"].values
    T_arr = meta["line_T_mean_c"].values
    tags = meta["step_tag"].values
    run_idx = meta["run_idx"].values
    display_run_idx = run_idx - int(run_idx.min())

    rows = np.where(roi.any(axis=1))[0]; cols = np.where(roi.any(axis=0))[0]
    x_lo, x_hi = float(x_mm[cols.min()]), float(x_mm[cols.max()])
    y_lo, y_hi = float(y_mm[rows.min()]), float(y_mm[rows.max()])
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]

    for mod, arr in arrs.items():
        print(f"\nbuilding animation: {mod}")
        vmin = float(np.nanpercentile(arr[:, roi], 5))
        vmax = float(np.nanpercentile(arr[:, roi], 95))

        fig = plt.figure(figsize=(13, 7.5), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1.0])
        ax_img = fig.add_subplot(gs[:, 0])
        ax_soc = fig.add_subplot(gs[0, 1])
        ax_v = fig.add_subplot(gs[1, 1])
        ax_t = fig.add_subplot(gs[2, 1])

        im = ax_img.imshow(arr[0], extent=extent, aspect="equal",
                           cmap="turbo", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
        ax_img.plot([x_lo, x_hi, x_hi, x_lo, x_lo],
                    [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=1.0, alpha=0.7)
        ax_img.set_xlabel("X [mm]"); ax_img.set_ylabel("Y [mm]")
        fig.colorbar(im, ax=ax_img, shrink=0.85, label=UNITS[mod])

        title = ax_img.set_title("", fontsize=11)

        ax_soc.plot(t_h, soc_per_scan, "-", color="#222", lw=1)
        soc_dot, = ax_soc.plot([], [], "o", color="tab:orange", ms=9)
        ax_soc.set_ylabel("SOC [%]"); ax_soc.grid(alpha=0.3)

        ax_v.plot(t_h, V_arr * 1000, "-", color="tab:red", lw=1)
        v_dot, = ax_v.plot([], [], "o", color="tab:orange", ms=9)
        ax_v.set_ylabel("V [mV]"); ax_v.grid(alpha=0.3)

        ax_t.plot(t_h, T_arr, "-", color="tab:purple", lw=1)
        t_dot, = ax_t.plot([], [], "o", color="tab:orange", ms=9)
        ax_t.set_ylabel("T [degC]"); ax_t.set_xlabel("time [h]")
        ax_t.grid(alpha=0.3)

        def update(i):
            im.set_data(arr[i])
            tag = tags[i]
            col = TAG_COLOR.get(tag, "black")
            title.set_text(
                f"{mod}  |  r{int(display_run_idx[i]):03d}  |  t={t_h[i]:6.2f} h\n"
                f"SOC={soc_per_scan[i]:+5.1f}%   V={V_arr[i]*1000:.1f} mV   "
                f"T={T_arr[i]:.3f} degC   [{tag}]"
            )
            title.set_color(col)
            soc_dot.set_data([t_h[i]], [soc_per_scan[i]])
            v_dot.set_data([t_h[i]], [V_arr[i] * 1000])
            t_dot.set_data([t_h[i]], [T_arr[i]])
            return im, soc_dot, v_dot, t_dot, title

        anim = FuncAnimation(fig, update, frames=arr.shape[0], interval=120, blit=False)
        out_path = OUT / f"longrun_35c_{mod}.gif"
        anim.save(out_path, fps=8, dpi=100, writer="pillow")
        plt.close(fig)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
