"""Combined 3-modality animation for the 35C charge-focus subset.

One GIF, top row = amplitude / energy / tof C-scans synchronised, bottom row =
SOC + V + T trajectories with a moving dot.
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

OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"
OUT_DIR = OUT_ROOT / "06_animation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "longrun_35c_combined.gif"

CYCLER = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
NOMINAL_AH = 0.860

UNITS = {"amplitude": "V", "tof": "us", "energy": "(a.u.)"}
TAG_COLOR = {"charge": "tab:red", "rest": "#444"}


def _build_cumulative_soc():
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
        q = sub["Capacity"].values
        d_abs = np.clip(np.diff(q, prepend=q[0]), 0, None)
        md_majority = sub["MD"].mode().iloc[0]
        dq[idx] = d_abs * mode_sign.get(md_majority, 0.0)
    cum = np.cumsum(dq)
    soc = (cum / NOMINAL_AH) * 100
    soc -= soc.min()
    return pd.DataFrame({"time": df["DPT Time"], "SOC_pct": soc})


def main():
    print("loading patched cache + cycler ...")
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    roi = d["roi_mask"]
    x_mm = d["x_mm"]; y_mm = d["y_mm"]
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]
    arrs = {"amplitude": d["amplitude"], "tof": d["tof"], "energy": d["energy"]}

    rows = np.where(roi.any(axis=1))[0]; cols = np.where(roi.any(axis=0))[0]
    x_lo, x_hi = float(x_mm[cols.min()]), float(x_mm[cols.max()])
    y_lo, y_hi = float(y_mm[rows.min()]), float(y_mm[rows.max()])

    soc_df = _build_cumulative_soc()
    t_ref = soc_df["time"].values.astype("datetime64[s]").astype(np.int64)
    ts_int = meta["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    soc_per_scan = np.interp(ts_int, t_ref, soc_df["SOC_pct"].values)
    print(f"  SOC range per scan: {soc_per_scan.min():.1f} -> {soc_per_scan.max():.1f}%")

    t_h = (meta["timestamp"] - meta["timestamp"].iloc[0]).dt.total_seconds().values / 3600
    V_arr = meta["voltage_at_scan"].values
    T_arr = meta["line_T_mean_c"].values
    tags = meta["step_tag"].values
    run_idx = meta["run_idx"].values
    display_run_idx = run_idx - int(run_idx.min())

    # Per-modality global vmin/vmax (ROI-only, 5-95 percentile -- webapp style)
    vrange = {m: (float(np.nanpercentile(arrs[m][:, roi], 5)),
                  float(np.nanpercentile(arrs[m][:, roi], 95))) for m in arrs}

    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1])
    ax_img = {m: fig.add_subplot(gs[0, i]) for i, m in enumerate(["amplitude", "energy", "tof"])}
    ax_soc = fig.add_subplot(gs[1, 0])
    ax_v   = fig.add_subplot(gs[1, 1])
    ax_t   = fig.add_subplot(gs[1, 2])

    ims = {}
    for m, ax in ax_img.items():
        vmin, vmax = vrange[m]
        cmap = plt.colormaps["turbo"].copy(); cmap.set_bad("white")
        im = ax.imshow(arrs[m][0], extent=extent, aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo],
                [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=1, alpha=0.7)
        ax.set_title(m, fontsize=11)
        ax.set_xlabel("X [mm]")
        if m == "amplitude":
            ax.set_ylabel("Y [mm]")
        fig.colorbar(im, ax=ax, shrink=0.85, label=UNITS[m])
        ims[m] = im

    suptitle = fig.suptitle("", fontsize=12)

    # SOC / V / T panels
    ax_soc.plot(t_h, soc_per_scan, "-", color="#222", lw=1)
    soc_dot, = ax_soc.plot([], [], "o", color="tab:orange", ms=10)
    ax_soc.set_xlabel("time [h]"); ax_soc.set_ylabel("SOC [%]")
    ax_soc.grid(alpha=0.3)

    ax_v.plot(t_h, V_arr * 1000, "-", color="tab:red", lw=1)
    v_dot, = ax_v.plot([], [], "o", color="tab:orange", ms=10)
    ax_v.set_xlabel("time [h]"); ax_v.set_ylabel("V [mV]")
    ax_v.grid(alpha=0.3)

    ax_t.plot(t_h, T_arr, "-", color="tab:purple", lw=1)
    t_dot, = ax_t.plot([], [], "o", color="tab:orange", ms=10)
    ax_t.set_xlabel("time [h]"); ax_t.set_ylabel("T [°C]")
    ax_t.grid(alpha=0.3)

    def update(i):
        for m, im in ims.items():
            im.set_data(arrs[m][i])
        tag = tags[i]
        col = TAG_COLOR.get(tag, "black")
        suptitle.set_text(
            f"35°C charge-focus  ·  r{int(display_run_idx[i]):03d}  ·  t = {t_h[i]:5.2f} h   "
            f"SOC = {soc_per_scan[i]:5.1f}%   V = {V_arr[i]*1000:.1f} mV   "
            f"T = {T_arr[i]:.3f} °C   [{tag}]"
        )
        suptitle.set_color(col)
        soc_dot.set_data([t_h[i]], [soc_per_scan[i]])
        v_dot.set_data([t_h[i]], [V_arr[i] * 1000])
        t_dot.set_data([t_h[i]], [T_arr[i]])
        return tuple(ims.values()) + (soc_dot, v_dot, t_dot, suptitle)

    print("rendering combined GIF ...")
    anim = FuncAnimation(fig, update, frames=arrs["amplitude"].shape[0],
                         interval=120, blit=False)
    anim.save(OUT_PATH, fps=8, dpi=90, writer="pillow")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
