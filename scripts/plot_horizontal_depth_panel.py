"""Paper-style horizontal layout: row of amplitude C-scans + envelope overlay.

Top:    N amplitude C-scans laid out horizontally, shared colour scale,
        each labelled only with derived distance d and envelope ToF.
Bottom: a single voltage-vs-time plot of the Hilbert envelope of each
        scan's centre pulse, colour-coded by depth.

The filter / dedup logic mirrors `build_depth_series_pdf.py`. We restate it
here so this script is self-contained.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.signal import hilbert, savgol_filter


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"
SOUND_SPEED_M_S = 989.0
MM_PER_US_ONE_WAY = SOUND_SPEED_M_S / 1000.0 / 2.0

EXPECTED_PARAMS = {
    "roi_w_mm":   80.0,
    "roi_h_mm":   72.0,
    "pitch_mm":   0.5,
    "speed_mm_s": 25.0,
    "accel_mm_s2": 800.0,
}


def matches_params(meta: dict, tol: float = 1e-3) -> bool:
    for k, v in EXPECTED_PARAMS.items():
        if k not in meta or abs(float(meta[k]) - v) > tol:
            return False
    return True


def envelope_tof_us(wf: np.ndarray, t_us: np.ndarray) -> tuple[float, np.ndarray]:
    env = np.abs(hilbert(wf))
    i = int(np.argmax(env))
    if 0 < i < len(env) - 1:
        y0, y1, y2 = env[i - 1], env[i], env[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return float(t_us[i]) + delta * (t_us[1] - t_us[0]), env


def load_scan(scan_dir: Path) -> dict | None:
    """Same completeness check as build_depth_series_pdf: scan_*.npz exists,
    parameters match, line count equals expected nlines, and centre-line npz
    has waveforms."""
    final_npz = next(scan_dir.glob("scan_*.npz"), None)
    meta_json = next(scan_dir.glob("scan_*_meta.json"), None)
    if final_npz is None or meta_json is None:
        return None
    try:
        meta = json.loads(meta_json.read_text())
    except json.JSONDecodeError:
        return None
    if not matches_params(meta):
        return None

    expected_nlines = int(meta.get("nlines", 0))
    if expected_nlines <= 0:
        return None
    lines_dir = scan_dir / "lines_raw"
    line_files = sorted(lines_dir.glob("line_*.npz"))
    if len(line_files) < expected_nlines:
        return None

    d = np.load(final_npz)
    amp_map = np.asarray(d["amplitude"], dtype=np.float64)
    roi_w = float(meta["roi_w_mm"])
    roi_h = float(meta["roi_h_mm"])

    centre_line = np.load(line_files[len(line_files) // 2])
    if "waveforms" not in centre_line.files:
        return None
    waveforms = centre_line["waveforms"]
    x_pulse = centre_line["x_mm"]
    fs_hz = float(centre_line["fs_hz"])
    gate_us = centre_line["gate_us"]

    valid = np.isfinite(x_pulse)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return None
    pulse_idx = int(valid_idx[np.argmin(np.abs(x_pulse[valid] - roi_w / 2.0))])
    wf = waveforms[pulse_idx].astype(np.float64)
    t_us = float(gate_us[0]) + np.arange(wf.size) * (1e6 / fs_hz)
    tof_us, env = envelope_tof_us(wf, t_us)

    # Smooth the envelope for display — at 20 MHz / 2.5 MHz centre, a window
    # of ~0.45 µs (9 samples) suppresses sample-rate jitter without flattening
    # the wave-packet structure.
    if env.size >= 11:
        env_smooth = savgol_filter(env, window_length=11, polyorder=3)
        env_smooth = np.clip(env_smooth, 0, None)  # no negative envelope
    else:
        env_smooth = env

    return {
        "scan_name": scan_dir.name,
        "scan_dir": scan_dir,
        "mtime": scan_dir.stat().st_mtime,
        "amp_map": amp_map,
        "roi_w": roi_w, "roi_h": roi_h,
        "wf_t_us": t_us, "wf_v": wf,
        "wf_env": env, "wf_env_smooth": env_smooth,
        "tof_us": tof_us,
        "one_way_mm": tof_us * MM_PER_US_ONE_WAY,
        "vpp": float(wf.max() - wf.min()),
    }


def dedup_by_depth(scans: list[dict], tol_mm: float) -> list[dict]:
    if tol_mm <= 0 or not scans:
        return scans
    scans = sorted(scans, key=lambda s: s["one_way_mm"])
    groups: list[list[dict]] = [[scans[0]]]
    for s in scans[1:]:
        if s["one_way_mm"] - groups[-1][-1]["one_way_mm"] <= tol_mm:
            groups[-1].append(s)
        else:
            groups.append([s])
    # keep the most recently modified representative per group
    return [max(g, key=lambda s: s["mtime"]) for g in groups]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-min-mm", type=float, required=True)
    ap.add_argument("--depth-max-mm", type=float, required=True)
    ap.add_argument("--dedup-tol-mm", type=float, default=0.1)
    ap.add_argument("--days-back", type=int, default=2)
    ap.add_argument("--label", default="horizontal")
    ap.add_argument("--panel-w-in", type=float, default=2.6,
                    help="width of each C-scan panel (inches)")
    ap.add_argument("--show-envelope-tof-marker", action="store_true",
                    help="draw a small dot at each envelope peak (ToF)")
    args = ap.parse_args()

    cutoff_t = datetime.now().timestamp() - args.days_back * 86400
    candidates = [p for p in CSCAN_ROOT.iterdir()
                  if p.is_dir() and p.stat().st_mtime >= cutoff_t]
    loaded = []
    for c in candidates:
        s = load_scan(c)
        if s is None:
            continue
        loaded.append(s)
    print(f"complete scans matching params, last {args.days_back}d: {len(loaded)}")
    loaded = [s for s in loaded
              if args.depth_min_mm <= s["one_way_mm"] <= args.depth_max_mm]
    print(f"after depth filter [{args.depth_min_mm}, {args.depth_max_mm}]: {len(loaded)}")
    loaded = dedup_by_depth(loaded, args.dedup_tol_mm)
    loaded.sort(key=lambda s: s["one_way_mm"])
    print(f"after dedup (tol = {args.dedup_tol_mm} mm): {len(loaded)}")

    if not loaded:
        return 1

    # Output dir
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = PROJECT / "reports" / "experiments" / f"depth_panel_{ts}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- shared colour scales ---
    all_amp = np.concatenate([s["amp_map"][np.isfinite(s["amp_map"])].ravel()
                              for s in loaded])
    vmin = float(np.percentile(all_amp, 1))
    vmax = float(np.percentile(all_amp, 99))

    n = len(loaded)
    depths = np.array([s["one_way_mm"] for s in loaded])
    depth_norm = mcolors.Normalize(vmin=float(depths.min()), vmax=float(depths.max()))
    depth_cmap = mcm.viridis

    # --- figure layout: top row of C-scans + colourbar; bottom envelope panel
    fig_w = args.panel_w_in * n + 1.4   # extra for colourbar
    fig_h = args.panel_w_in * 0.95 + 4.6  # square-ish C-scans + envelope panel
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    gs = fig.add_gridspec(
        nrows=2, ncols=n + 1,
        width_ratios=[1.0] * n + [0.06],
        height_ratios=[args.panel_w_in * 0.95, 4.0],
        wspace=0.10, hspace=0.55,
        left=0.05, right=0.97, top=0.86, bottom=0.10,
    )

    # Top row: C-scans
    im = None
    for col, s in enumerate(loaded):
        ax = fig.add_subplot(gs[0, col])
        extent = [0, s["roi_w"], s["roi_h"], 0]
        im = ax.imshow(s["amp_map"], cmap="turbo", vmin=vmin, vmax=vmax,
                       extent=extent, origin="upper", aspect="equal")
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("Y (mm)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("X (mm)", fontsize=8)

        # Colour chip (left) + d/ToF text (right) above each panel.
        # Both placed in axes-fraction coords with clip_on=False so they
        # render in the gap between the gridspec rows.
        chip_color = depth_cmap(depth_norm(s["one_way_mm"]))
        chip = Rectangle((0.03, 1.08), 0.11, 0.18,
                         transform=ax.transAxes,
                         facecolor=chip_color, edgecolor="#333",
                         linewidth=0.5, clip_on=False)
        ax.add_patch(chip)
        ax.text(0.60, 1.18,
                f"d = {s['one_way_mm']:.2f} mm\nToF = {s['tof_us']:.2f} µs",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9.5, clip_on=False)

    # Colour bar (right of top row)
    cbar_ax = fig.add_subplot(gs[0, -1])
    fig.colorbar(im, cax=cbar_ax, label="amplitude (V)")
    cbar_ax.tick_params(labelsize=7)

    # Bottom: ridge-style envelope display — each curve has its own baseline
    # so overlapping wave-packets stay readable. Order is shallowest depth at
    # top, deepest at bottom, so the eye sweeps downward = increasing d.
    ax_env = fig.add_subplot(gs[1, :])
    max_peak = max(float(s["wf_env_smooth"].max()) for s in loaded)
    step = max_peak * 0.45   # 55% overlap between adjacent ridges
    for i, s in enumerate(loaded):
        color = depth_cmap(depth_norm(s["one_way_mm"]))
        baseline = (len(loaded) - 1 - i) * step
        env = s["wf_env_smooth"]
        ax_env.fill_between(s["wf_t_us"], baseline, baseline + env,
                            color=color, alpha=0.45, linewidth=0,
                            zorder=len(loaded) - i)
        ax_env.plot(s["wf_t_us"], baseline + env, color=color,
                    linewidth=1.4, zorder=len(loaded) - i + 0.1)
        # Inline tag on the right (just inside the axis)
        ax_env.text(40.0 - 0.15, baseline + 0.15,
                    f"d = {s['one_way_mm']:.2f} mm",
                    fontsize=9, va="bottom", ha="right", color="#111",
                    bbox=dict(facecolor=color, edgecolor="#333",
                             linewidth=0.4, pad=2, alpha=0.85),
                    zorder=len(loaded) + 5)
        if args.show_envelope_tof_marker:
            ax_env.plot([s["tof_us"]], [baseline + env.max()],
                        "o", color=color, markersize=4,
                        zorder=len(loaded) - i + 0.2)

    ax_env.set_xlim(30.0, 40.0)
    ax_env.set_ylim(-0.05 * max_peak,
                    (len(loaded) - 1) * step + max_peak * 1.10)
    ax_env.set_yticks([])
    ax_env.set_xlabel("time from sync (µs)")
    ax_env.set_ylabel("envelope amplitude (V) — stacked")
    ax_env.grid(True, axis="x", alpha=0.3, linewidth=0.4)
    ax_env.set_title("Centre-pulse Hilbert envelope (Sav–Gol smoothed), ridge plot — "
                     "shallowest d top, deepest bottom",
                     fontsize=10)
    ax_env.tick_params(labelsize=8)

    out_png = out_dir / "panel.png"
    out_pdf = out_dir / "panel.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out_png}")
    print(f"saved: {out_pdf}")

    # tiny summary csv
    with (out_dir / "summary.csv").open("w", encoding="utf-8") as f:
        f.write("one_way_mm,tof_us,vpp_v,scan_name\n")
        for s in loaded:
            f.write(f"{s['one_way_mm']:.3f},{s['tof_us']:.3f},"
                    f"{s['vpp']:.3f},{s['scan_name']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
