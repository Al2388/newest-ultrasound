"""Full 35C campaign (Steps 2-11) animation, baseline-subtracted.

Differences from plot_35c_animation_combined_with_soc0.py:
  - Prepends Step 2 (initial discharge) so the GIF starts at pre-discharge state.
  - Image panels show (frame - baseline), where baseline = mean of last 5 frames
    of Step 3 (0% rest, fully discharged equilibrium).
  - Diverging colormap (RdBu_r) centered at 0, symmetric vmin/vmax from p99
    of |diff| inside ROI.
  - Bottom time-series panels (SOC, V, T) still raw.
  - Output: 06_animation/longrun_35c_full_baseline_subtracted.gif
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
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

BATCH_DIR = PROJ / "reports/experiments/longrun_cycling_35c_2026-05-30_19-47-06"
CHECKPOINT = BATCH_DIR / "checkpoint.json"
CYCLER = PROJ / "data/raw/cycler/LFP860_35degrees.002.txt"
ROI_MASK_PATH = PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp/roi_mask.npy"

OUT_ROOT = PROJ / "reports/longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache_full"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = OUT_ROOT / "06_animation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "longrun_35c_full_baseline_subtracted.gif"

NOMINAL_AH = 0.860

UNITS = {"amplitude": "V", "tof": "us", "energy": "(a.u.)"}
TAG_COLOR = {"charge": "tab:red", "rest": "#444", "discharge": "tab:blue"}

# Extended: include Step 2 (initial discharge to 0%).
SEGMENTS = [
    # Step 2 start pulled back to 19:47:00 to include r001 (2026-05-30 19:47:06,
    # the very first scan of the campaign, captured ~1.5 min BEFORE the cycler
    # actually started the discharge step).
    (2,  "discharge", datetime(2026, 5, 30, 19, 47, 0),  datetime(2026, 5, 31, 3, 7, 48),  None),
    (3,  "rest",      datetime(2026, 5, 31, 3, 7, 48),   datetime(2026, 5, 31, 5, 7, 48),  0),
    (4,  "charge",    datetime(2026, 5, 31, 5, 7, 49),   datetime(2026, 5, 31, 7, 7, 49),  None),
    (5,  "rest",      datetime(2026, 5, 31, 7, 7, 49),   datetime(2026, 5, 31, 9, 7, 49),  20),
    (6,  "charge",    datetime(2026, 5, 31, 9, 7, 49),   datetime(2026, 5, 31, 11, 7, 49), None),
    (7,  "rest",      datetime(2026, 5, 31, 11, 7, 49),  datetime(2026, 5, 31, 13, 7, 49), 40),
    (8,  "charge",    datetime(2026, 5, 31, 13, 7, 49),  datetime(2026, 5, 31, 15, 7, 50), None),
    (9,  "rest",      datetime(2026, 5, 31, 15, 7, 50),  datetime(2026, 5, 31, 17, 7, 50), 60),
    (10, "charge",    datetime(2026, 5, 31, 17, 7, 50),  datetime(2026, 5, 31, 19, 7, 51), None),
    (11, "rest",      datetime(2026, 5, 31, 19, 7, 51),  datetime(2026, 5, 31, 21, 7, 51), 80),
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
        if t0 <= ts <= t1 + timedelta(minutes=2):
            return step, kind, soc
    return None, None, None


def _load_cycler() -> pd.DataFrame:
    df = pd.read_csv(CYCLER, sep="\t", skiprows=6, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DPT Time"] = pd.to_datetime(df["DPT Time"].astype(str), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for c in ["Step", "Capacity", "Voltage", "Current"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MD"] = df["MD"].astype(str).str.strip()
    df = df.dropna(subset=["DPT Time"]).reset_index(drop=True)
    return df


def _interp_cycler(cycler_df: pd.DataFrame, ts: datetime) -> pd.Series:
    diffs = (cycler_df["DPT Time"] - ts).abs()
    idx = diffs.idxmin()
    return cycler_df.loc[idx]


def _build_cumulative_soc(cycler_df: pd.DataFrame) -> pd.DataFrame:
    mode_sign = {"C": +1.0, "D": -1.0, "R": 0.0, "P": 0.0}
    dq = np.zeros(len(cycler_df))
    for _, sub in cycler_df.groupby("Step", sort=False):
        idx = sub.index.values
        q = sub["Capacity"].values
        d_abs = np.clip(np.diff(q, prepend=q[0]), 0, None)
        md_majority = sub["MD"].mode().iloc[0]
        dq[idx] = d_abs * mode_sign.get(md_majority, 0.0)
    cum = np.cumsum(dq)
    soc = (cum / NOMINAL_AH) * 100
    soc -= soc.min()
    return pd.DataFrame({"time": cycler_df["DPT Time"], "SOC_pct": soc})


def _build_cache(rebuild: bool = False):
    stack_path = CACHE_DIR / "stack.npz"
    meta_path = CACHE_DIR / "meta.csv"
    if stack_path.exists() and meta_path.exists() and not rebuild:
        print(f"reusing cache: {stack_path}")
        d = np.load(stack_path)
        meta = pd.read_csv(meta_path)
        meta["timestamp"] = pd.to_datetime(meta["timestamp"])
        roi = d["roi_mask"]; x_mm = d["x_mm"]; y_mm = d["y_mm"]
        arrs = {"amplitude": d["amplitude"], "tof": d["tof"], "energy": d["energy"]}
        return arrs, meta, roi, x_mm, y_mm

    print("building extended cache (Steps 2-11) ...")
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    cycler = _load_cycler()

    selected = []
    for c in cp["completed"]:
        sess_dir = PROJ / Path(c["session_dir"].replace("\\", "/"))
        run_idx, ts = _scan_info(sess_dir.name)
        if ts is None or not (WINDOW_START <= ts <= WINDOW_END + timedelta(minutes=10)):
            continue
        step, kind, soc_lbl = _tag_segment(ts)
        if step is None:
            continue
        cyc_row = _interp_cycler(cycler, ts)
        seg_start = next(s[2] for s in SEGMENTS if s[0] == step)
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
            "step_time_relative_min": (ts - seg_start).total_seconds() / 60.0,
        })

    df = pd.DataFrame(selected).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"selected {len(df)} scans in window [{WINDOW_START} -> {WINDOW_END}]")
    print(df.groupby(["step", "step_tag"]).size())

    amp, tof, energy = [], [], []
    line_T_per_scan = []
    x_mm = y_mm = None
    roi_mask = np.load(ROI_MASK_PATH)
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
    df["amp_mean_roi"] = [float(np.nanmean(a[roi_mask])) for a in amp]
    df["tof_mean_roi"] = [float(np.nanmean(a[roi_mask])) for a in tof]
    df["energy_mean_roi"] = [float(np.nanmean(a[roi_mask])) for a in energy]

    arrs = {
        "amplitude": np.stack(amp, axis=0),
        "tof": np.stack(tof, axis=0),
        "energy": np.stack(energy, axis=0),
    }
    np.savez_compressed(
        stack_path,
        amplitude=arrs["amplitude"], tof=arrs["tof"], energy=arrs["energy"],
        x_mm=x_mm, y_mm=y_mm, roi_mask=roi_mask,
    )
    df.to_csv(meta_path, index=False)
    print(f"wrote {stack_path} + {meta_path}")
    return arrs, df, roi_mask, x_mm, y_mm


def main():
    arrs, meta, roi, x_mm, y_mm = _build_cache(rebuild=False)

    # Baseline = the very first scan of the campaign (r001, pre-discharge OCV state)
    baseline_idx = np.array([0])
    first_run = int(meta["run_idx"].values[0])
    first_ts = meta["timestamp"].values[0]
    print(f"baseline frame: run_idx r{first_run:03d} at {first_ts} (pre-discharge first scan)")

    baselines = {m: np.nanmean(arrs[m][baseline_idx], axis=0) for m in arrs}

    # Compute diffs
    diffs = {m: arrs[m] - baselines[m] for m in arrs}

    # Symmetric vmin/vmax from p99 of |diff| inside ROI
    vrange = {}
    for m in diffs:
        roi_mask_3d = np.broadcast_to(roi, diffs[m].shape)
        roi_vals = np.abs(diffs[m][roi_mask_3d])
        roi_vals = roi_vals[np.isfinite(roi_vals)]
        p99 = float(np.nanpercentile(roi_vals, 99))
        vrange[m] = (-p99, p99)
        print(f"  {m}: |diff| p99 inside ROI = {p99:.4g} {UNITS[m]}  -> vmin/vmax +/-{p99:.4g}")

    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]
    rows = np.where(roi.any(axis=1))[0]; cols = np.where(roi.any(axis=0))[0]
    x_lo, x_hi = float(x_mm[cols.min()]), float(x_mm[cols.max()])
    y_lo, y_hi = float(y_mm[rows.min()]), float(y_mm[rows.max()])

    cycler = _load_cycler()
    soc_df = _build_cumulative_soc(cycler)
    t_ref = soc_df["time"].values.astype("datetime64[s]").astype(np.int64)
    ts_int = meta["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    soc_per_scan = np.interp(ts_int, t_ref, soc_df["SOC_pct"].values)
    print(f"  SOC range per scan: {soc_per_scan.min():.1f} -> {soc_per_scan.max():.1f}%")

    t_h = (meta["timestamp"] - meta["timestamp"].iloc[0]).dt.total_seconds().values / 3600
    V_arr = meta["voltage_at_scan"].values
    T_arr = meta["line_T_mean_c"].values
    tags = meta["step_tag"].values
    run_idx = meta["run_idx"].values

    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1])
    ax_img = {m: fig.add_subplot(gs[0, i]) for i, m in enumerate(["amplitude", "tof", "energy"])}
    ax_soc = fig.add_subplot(gs[1, 0])
    ax_v   = fig.add_subplot(gs[1, 1])
    ax_t   = fig.add_subplot(gs[1, 2])

    ims = {}
    for m, ax in ax_img.items():
        vmin, vmax = vrange[m]
        cmap = plt.colormaps["RdBu_r"].copy(); cmap.set_bad("white")
        im = ax.imshow(diffs[m][0], extent=extent, aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo],
                [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=1, alpha=0.7)
        ax.set_title(f"Δ{m}  (frame − baseline)", fontsize=11)
        ax.set_xlabel("X [mm]")
        if m == "amplitude":
            ax.set_ylabel("Y [mm]")
        fig.colorbar(im, ax=ax, shrink=0.85, label=f"Δ{m} [{UNITS[m]}]")
        ims[m] = im

    suptitle = fig.suptitle("", fontsize=12)

    # mark baseline frames on the SOC panel as a grey band
    ax_soc.plot(t_h, soc_per_scan, "-", color="#222", lw=1)
    # mark baseline frame(s) with a vertical line on SOC panel
    for bi in baseline_idx:
        ax_soc.axvline(t_h[bi], color="orange", lw=2, alpha=0.6,
                       label="baseline frame" if bi == baseline_idx[0] else None)
    soc_dot, = ax_soc.plot([], [], "o", color="tab:orange", ms=10)
    ax_soc.set_xlabel("time [h]"); ax_soc.set_ylabel("SOC [%]")
    ax_soc.grid(alpha=0.3); ax_soc.legend(fontsize=8, loc="best")

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
            im.set_data(diffs[m][i])
        tag = tags[i]
        col = TAG_COLOR.get(tag, "black")
        suptitle.set_text(
            f"35°C full campaign (baseline = r{first_run:03d} pre-discharge)  ·  "
            f"r{int(run_idx[i]):03d}  ·  t={t_h[i]:5.2f} h   "
            f"SOC={soc_per_scan[i]:5.1f}%   V={V_arr[i]*1000:.1f} mV   "
            f"T={T_arr[i]:.3f} °C   [{tag}]"
        )
        suptitle.set_color(col)
        soc_dot.set_data([t_h[i]], [soc_per_scan[i]])
        v_dot.set_data([t_h[i]], [V_arr[i] * 1000])
        t_dot.set_data([t_h[i]], [T_arr[i]])
        return tuple(ims.values()) + (soc_dot, v_dot, t_dot, suptitle)

    print("rendering baseline-subtracted full-campaign GIF ...")
    anim = FuncAnimation(fig, update, frames=arrs["amplitude"].shape[0], interval=120, blit=False)
    anim.save(OUT_PATH, fps=8, dpi=90, writer="pillow")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
