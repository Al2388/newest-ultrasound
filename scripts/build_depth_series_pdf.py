"""Collect all matching-parameter, completed C-scans from the last N days,
sort by ToF-derived one-way distance, and emit a long comparison PDF.

Parameter filter (the operator's current standard config):
    roi_w_mm = 80, roi_h_mm = 72, pitch_mm = 0.5,
    speed_mm_s = 25, accel_mm_s2 = 800

A scan is "completed" iff the final aggregated NPZ scan_<ts>.npz exists in
the run folder — that file is only written after the worker exits cleanly.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CSCAN_ROOT = PROJECT / "data/raw/cscan"
SOUND_SPEED_M_S = 989.0
MM_PER_US_ONE_WAY = SOUND_SPEED_M_S / 1000.0 / 2.0

# Parameter filter — drop any scan whose meta deviates from these.
EXPECTED = {
    "roi_w_mm":   80.0,
    "roi_h_mm":   72.0,
    "pitch_mm":   0.5,
    "speed_mm_s": 25.0,
    "accel_mm_s2": 800.0,
}
FLOAT_TOL = 1e-3

DAYS_BACK = 2

# When >0, collapse scans within this depth tolerance to a single
# representative (the most recently modified one) — strips out the
# repeatability/noise-floor clusters where many scans share the same
# physical setup.
DEDUP_TOL_MM = 0.0


def matches_params(meta: dict) -> bool:
    for k, v in EXPECTED.items():
        if k not in meta:
            return False
        if abs(float(meta[k]) - v) > FLOAT_TOL:
            return False
    return True


def envelope_tof_us(wf: np.ndarray, t_us: np.ndarray) -> tuple[float, float]:
    env = np.abs(hilbert(wf))
    i = int(np.argmax(env))
    if 0 < i < len(env) - 1:
        y0, y1, y2 = env[i - 1], env[i], env[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    dt = float(t_us[1] - t_us[0])
    return float(t_us[i]) + delta * dt, float(env[i])


def load_scan(scan_dir: Path) -> dict | None:
    """Return None if the scan is incomplete or missing the centre line."""
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

    # Drop partial scans: the worker writes scan_*.npz even after a stop, but
    # only the lines acquired before the stop have line_*.npz files. Compare
    # to the expected line count from the meta.
    expected_nlines = int(meta.get("nlines", 0))
    if expected_nlines <= 0:
        return None
    lines_dir = scan_dir / "lines_raw"
    line_files = sorted(lines_dir.glob("line_*.npz"))
    if not line_files:
        return None
    if len(line_files) < expected_nlines:
        # partial / aborted scan — skip
        return None

    try:
        d = np.load(final_npz)
        amp_map = np.asarray(d["amplitude"], dtype=np.float64)
        tof_map = np.asarray(d["tof"], dtype=np.float64) if "tof" in d.files else None
    except Exception:
        return None
    try:
        centre_line = np.load(line_files[len(line_files) // 2])
        waveforms = centre_line["waveforms"]
        x_pulse = centre_line["x_mm"]
        fs_hz = float(centre_line["fs_hz"])
        gate_us = centre_line["gate_us"]
    except (KeyError, OSError):
        return None

    roi_w = float(meta["roi_w_mm"])
    roi_h = float(meta["roi_h_mm"])
    valid = np.isfinite(x_pulse)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return None
    pulse_idx = int(valid_idx[np.argmin(np.abs(x_pulse[valid] - roi_w / 2.0))])
    wf = waveforms[pulse_idx].astype(np.float64)
    t_us = float(gate_us[0]) + np.arange(wf.size) * (1e6 / fs_hz)
    tof, env_peak = envelope_tof_us(wf, t_us)

    return {
        "scan_dir":   scan_dir,
        "scan_name":  scan_dir.name,
        "mtime":      datetime.fromtimestamp(scan_dir.stat().st_mtime),
        "amp_map":    amp_map,
        "tof_map":    tof_map,
        "roi_w":      roi_w,
        "roi_h":      roi_h,
        "wf_t_us":    t_us,
        "wf_v":       wf,
        "tof_us":     tof,
        "one_way_mm": tof * MM_PER_US_ONE_WAY,
        "env_peak_v": env_peak,
        "vpp":        float(wf.max() - wf.min()),
        "meta":       meta,
    }


def dedup_by_depth(scans: list[dict], tol_mm: float) -> tuple[list[dict], list[int]]:
    """Collapse adjacent scans within tol_mm of each other to the most
    recently modified representative. Returns (kept, group_sizes)."""
    if tol_mm <= 0 or not scans:
        return scans, [1] * len(scans)
    scans = sorted(scans, key=lambda s: s["one_way_mm"])
    groups: list[list[dict]] = [[scans[0]]]
    for s in scans[1:]:
        if s["one_way_mm"] - groups[-1][-1]["one_way_mm"] <= tol_mm:
            groups[-1].append(s)
        else:
            groups.append([s])
    kept = [max(g, key=lambda s: s["mtime"]) for g in groups]
    return kept, [len(g) for g in groups]


def main() -> int:
    import argparse
    import re as _re
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup-tol-mm", type=float, default=DEDUP_TOL_MM,
                    help="collapse scans within this depth tolerance to their most recent representative")
    ap.add_argument("--days-back", type=int, default=DAYS_BACK)
    ap.add_argument("--depth-min-mm", type=float, default=None,
                    help="keep only scans with one_way_mm >= this")
    ap.add_argument("--depth-max-mm", type=float, default=None,
                    help="keep only scans with one_way_mm <= this")
    ap.add_argument("--name-pattern", default=None,
                    help="regex on scan_dir.name; only matches are kept "
                         "(e.g. 'gauge-height-\\d+mm' for curated scans only)")
    ap.add_argument("--label", default="",
                    help="suffix added to the output folder name")
    args = ap.parse_args()

    cutoff = datetime.now() - timedelta(days=args.days_back)
    candidates = [p for p in CSCAN_ROOT.iterdir()
                  if p.is_dir() and datetime.fromtimestamp(p.stat().st_mtime) >= cutoff]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    print(f"candidates modified in last {args.days_back} days: {len(candidates)}")

    loaded = []
    for c in candidates:
        s = load_scan(c)
        if s is None:
            continue
        loaded.append(s)

    print(f"matched parameter filter and complete: {len(loaded)}")
    if not loaded:
        print("nothing to plot")
        return 1

    loaded.sort(key=lambda s: s["one_way_mm"])

    # Optional filters: depth band and name regex
    if args.name_pattern:
        rgx = _re.compile(args.name_pattern)
        before = len(loaded)
        loaded = [s for s in loaded if rgx.search(s["scan_name"])]
        print(f"after name filter /{args.name_pattern}/: {len(loaded)} (was {before})")
    if args.depth_min_mm is not None:
        before = len(loaded)
        loaded = [s for s in loaded if s["one_way_mm"] >= args.depth_min_mm]
        print(f"after depth_min={args.depth_min_mm}: {len(loaded)} (was {before})")
    if args.depth_max_mm is not None:
        before = len(loaded)
        loaded = [s for s in loaded if s["one_way_mm"] <= args.depth_max_mm]
        print(f"after depth_max={args.depth_max_mm}: {len(loaded)} (was {before})")
    if not loaded:
        print("filters left no scans")
        return 1

    group_sizes = [1] * len(loaded)
    if args.dedup_tol_mm > 0:
        before = len(loaded)
        loaded, group_sizes = dedup_by_depth(loaded, args.dedup_tol_mm)
        print(f"after dedup (tol = {args.dedup_tol_mm} mm): {len(loaded)} groups "
              f"(was {before}, dropped {before - len(loaded)})")

    # ---- output folder ----
    ts_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix_parts = []
    if args.label:
        suffix_parts.append(f"_{args.label}")
    if args.dedup_tol_mm > 0:
        suffix_parts.append(f"_dedup{args.dedup_tol_mm:.2f}mm")
    suffix = "".join(suffix_parts)
    out_dir = PROJECT / "reports" / "experiments" / f"depth_series_{ts_now}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_depth_dir = out_dir / "by_depth_amp"
    by_depth_dir.mkdir(exist_ok=True)

    # ---- summary csv ----
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["one_way_mm", "tof_us", "env_peak_v", "vpp_v",
                    "scan_name", "modified", "amp_min", "amp_p99"])
        for s in loaded:
            w.writerow([
                f"{s['one_way_mm']:.3f}", f"{s['tof_us']:.3f}",
                f"{s['env_peak_v']:.3f}", f"{s['vpp']:.3f}",
                s["scan_name"], s["mtime"].isoformat(timespec="seconds"),
                f"{np.nanmin(s['amp_map']):.3f}",
                f"{np.nanpercentile(s['amp_map'], 99):.3f}",
            ])
    print(f"summary: {csv_path}")

    # ---- copy amp PNGs renamed by depth (so user can browse by depth) ----
    for i, s in enumerate(loaded):
        src = next(s["scan_dir"].glob("scan_amp.png"), None)
        if src is None:
            continue
        group_tag = f"_n{group_sizes[i]}" if group_sizes[i] > 1 else ""
        dst = by_depth_dir / f"d_{s['one_way_mm']:06.3f}mm{group_tag}__{s['scan_name']}.png"
        shutil.copy2(src, dst)

    # ---- shared scales ----
    all_amp = np.concatenate([s["amp_map"][np.isfinite(s["amp_map"])].ravel()
                              for s in loaded])
    vmin = float(np.percentile(all_amp, 1))
    vmax = float(np.percentile(all_amp, 99))

    have_tof = all(s["tof_map"] is not None for s in loaded)
    if have_tof:
        all_tof = np.concatenate([s["tof_map"][np.isfinite(s["tof_map"])].ravel()
                                  for s in loaded])
        # Use a slightly tighter percentile for ToF since outliers blow the scale.
        tof_vmin = float(np.percentile(all_tof, 2))
        tof_vmax = float(np.percentile(all_tof, 98))

    wf_xlim = (min(float(s["wf_t_us"][0]) for s in loaded),
               max(float(s["wf_t_us"][-1]) for s in loaded))
    wf_ymin = min(float(s["wf_v"].min()) for s in loaded)
    wf_ymax = max(float(s["wf_v"].max()) for s in loaded)
    pad = 0.08 * (wf_ymax - wf_ymin)
    wf_ylim = (wf_ymin - pad, wf_ymax + pad)

    # ---- figure layout ----
    n = len(loaded)
    cols = 3 if have_tof else 2
    # Per-row height: 2 in. Plus header band 1.4 in. Cap if huge.
    row_h = 2.0
    fig_h = row_h * n + 1.4
    fig_w = 13.5 if have_tof else 11.0
    print(f"figure: {fig_w} x {fig_h} in")

    width_ratios = [1.0, 1.0, 1.6] if have_tof else [1.0, 1.6]
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    gs = fig.add_gridspec(
        nrows=n, ncols=cols,
        width_ratios=width_ratios,
        wspace=0.22, hspace=0.42,
        left=0.08, right=0.91, top=1.0 - 0.9 / fig_h, bottom=0.6 / fig_h,
    )

    amp_im = tof_im = None
    for r, s in enumerate(loaded):
        ax_amp = fig.add_subplot(gs[r, 0])
        amp_im = ax_amp.imshow(s["amp_map"], cmap="turbo", vmin=vmin, vmax=vmax,
                               extent=[0, s["roi_w"], s["roi_h"], 0],
                               aspect="equal", origin="upper")
        ax_amp.set_ylabel(f"d = {s['one_way_mm']:.2f} mm\nToF {s['tof_us']:.2f} µs",
                          fontsize=9, labelpad=4)
        group_tag = f"  [×{group_sizes[r]} repeats]" if group_sizes[r] > 1 else ""
        ax_amp.set_title(
            f"{s['scan_name']}{group_tag}   ·   "
            f"V$_{{pp}}$={s['vpp']:.2f} V  env={s['env_peak_v']:.2f} V  "
            f"({s['mtime'].strftime('%m-%d %H:%M')})",
            fontsize=8.5, loc="left",
        )
        ax_amp.tick_params(labelsize=7.5)
        if r == n - 1:
            ax_amp.set_xlabel("X (mm)", fontsize=9)
        else:
            ax_amp.set_xticklabels([])

        if have_tof:
            ax_tof = fig.add_subplot(gs[r, 1])
            tof_im = ax_tof.imshow(s["tof_map"], cmap="viridis",
                                   vmin=tof_vmin, vmax=tof_vmax,
                                   extent=[0, s["roi_w"], s["roi_h"], 0],
                                   aspect="equal", origin="upper")
            ax_tof.tick_params(labelsize=7.5)
            ax_tof.set_yticklabels([])
            if r == n - 1:
                ax_tof.set_xlabel("X (mm)", fontsize=9)
            else:
                ax_tof.set_xticklabels([])
            if r == 0:
                ax_tof.set_title("ToF map", fontsize=9, loc="left")

        ax_wf = fig.add_subplot(gs[r, cols - 1])
        ax_wf.plot(s["wf_t_us"], s["wf_v"], color="tab:blue", linewidth=0.8)
        ax_wf.axhline(0, color="k", linewidth=0.3, alpha=0.4)
        ax_wf.axvline(s["tof_us"], color="tab:red", linestyle="--",
                      linewidth=0.6, alpha=0.7)
        ax_wf.set_xlim(*wf_xlim)
        ax_wf.set_ylim(*wf_ylim)
        ax_wf.grid(True, alpha=0.25, linewidth=0.4)
        ax_wf.set_ylabel("V", fontsize=9)
        ax_wf.tick_params(labelsize=7.5)
        if r == n - 1:
            ax_wf.set_xlabel("time from sync (µs)", fontsize=9)
        else:
            ax_wf.set_xticklabels([])

    # ---- shared colourbars ----
    cbar_amp = fig.add_axes([0.92, 0.06, 0.011, 0.85])
    fig.colorbar(amp_im, cax=cbar_amp, label="amplitude (V)")
    cbar_amp.tick_params(labelsize=7.5)
    if have_tof:
        cbar_tof = fig.add_axes([0.955, 0.06, 0.011, 0.85])
        fig.colorbar(tof_im, cax=cbar_tof, label="ToF (µs)")
        cbar_tof.tick_params(labelsize=7.5)

    title_lines = [
        f"Depth-sorted C-scan comparison  ·  {n} scans  ·  last {DAYS_BACK} days",
        f"params: ROI 80×72 mm, pitch 0.5 mm, speed 25 mm/s, accel 800 mm/s², gate 25–50 µs"
        f"   ·   depth via centre-pulse envelope ToF (c = {SOUND_SPEED_M_S:.0f} m/s, silicone oil 20 cSt)"
    ]
    fig.suptitle("\n".join(title_lines), fontsize=11, y=1.0 - 0.15 / fig_h)

    out_png = out_dir / "comparison.png"
    out_pdf = out_dir / "comparison.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    print(f"by-depth PNG copies: {by_depth_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
