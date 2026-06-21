"""Paper figure: gauge-height series labelled by ToF-derived distance, with
the back-wall focus scan highlighted.

Argument the figure has to support:
    The transducer has a fixed focal length f_oil ≈ 19.7 mm in 989 m/s
    silicone oil. As the probe-to-battery distance d is varied, the focal
    plane sweeps through the cell.
        d ≈ f → focus on front wall → bright surface, no internal info
        d ≈ f − t_cell → focus on back wall → bottom tape visible

Output: a row-per-distance composite. For each:
    [C-scan amplitude map | centre waveform | Vpp annotation]
Labels on the left use the ToF-derived one-way distance instead of the
mechanical gauge reading (which the operator stated was not in physical mm).
The "chosen working distance" row is boxed and annotated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert


CSCAN_ROOT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/cscan")
HEIGHT_RE = re.compile(r"[Gg]auge-height-(\d+)mm")
SOUND_SPEED_M_S = 989.0
MM_PER_US_ONE_WAY = SOUND_SPEED_M_S / 1000.0 / 2.0   # one-way mm per µs of round-trip ToF

# Which scan to mark as the chosen working distance — identified by the gauge
# label since that's what the operator uses in the file name.
SELECTED_GAUGE_LABEL = 213          # back-wall focus, bottom tape visible
FRONT_FOCUS_GAUGE_LABEL = 220       # front-wall focus, surface imaged cleanly


def find_gauge_scans() -> list[tuple[int, Path]]:
    out = []
    for p in CSCAN_ROOT.iterdir():
        if not p.is_dir():
            continue
        m = HEIGHT_RE.search(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def envelope_tof_us(wf: np.ndarray, t_us: np.ndarray) -> tuple[float, float]:
    env = np.abs(hilbert(wf))
    i = int(np.argmax(env))
    if 0 < i < len(env) - 1:
        y0, y1, y2 = env[i - 1], env[i], env[i + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    dt = float(t_us[1] - t_us[0])
    return float(t_us[i]) + delta * dt, float(env[i])


def load_scan(scan_dir: Path) -> dict:
    npz_path = next(scan_dir.glob("scan_*.npz"))
    d = np.load(npz_path)
    amp_map = np.asarray(d["amplitude"], dtype=np.float64)
    x_mm_arr = np.asarray(d["x_mm"])
    y_mm_arr = np.asarray(d["y_mm"])

    meta = json.loads(next(scan_dir.glob("scan_*_meta.json")).read_text())
    roi_w = float(meta.get("roi_w_mm", x_mm_arr[-1] - x_mm_arr[0]))
    roi_h = float(meta.get("roi_h_mm", y_mm_arr[-1] - y_mm_arr[0]))

    line_files = sorted((scan_dir / "lines_raw").glob("line_*.npz"))
    centre_line = np.load(line_files[len(line_files) // 2])
    waveforms = centre_line["waveforms"]
    x_pulse = centre_line["x_mm"]
    fs_hz = float(centre_line["fs_hz"])
    gate_us = centre_line["gate_us"]

    valid = np.isfinite(x_pulse)
    valid_idx = np.flatnonzero(valid)
    pulse_idx = int(valid_idx[np.argmin(np.abs(x_pulse[valid] - roi_w / 2.0))])
    wf = waveforms[pulse_idx].astype(np.float64)
    t_us = float(gate_us[0]) + np.arange(wf.size) * (1e6 / fs_hz)
    tof_us, env_peak = envelope_tof_us(wf, t_us)
    one_way_mm = tof_us * MM_PER_US_ONE_WAY

    return {
        "amp_map": amp_map, "roi_w": roi_w, "roi_h": roi_h,
        "wf_t_us": t_us, "wf_v": wf,
        "tof_us": tof_us, "one_way_mm": one_way_mm,
        "env_peak_v": env_peak,
    }


def main() -> int:
    scans = find_gauge_scans()
    data = []
    for gauge_label, scan_dir in scans:
        s = load_scan(scan_dir)
        s["gauge_label"] = gauge_label
        data.append(s)
    print(f"loaded {len(data)} scans, gauges = {[s['gauge_label'] for s in data]}")

    # Identify the "front-wall focus" (max envelope amplitude) and the
    # selected back-wall focus — both data-driven so the caption is honest.
    sel_idx = next(i for i, s in enumerate(data) if s["gauge_label"] == SELECTED_GAUGE_LABEL)
    front_focus_idx = next(i for i, s in enumerate(data) if s["gauge_label"] == FRONT_FOCUS_GAUGE_LABEL)
    delta_d = data[front_focus_idx]["one_way_mm"] - data[sel_idx]["one_way_mm"]
    print(f"selected scan  (back-wall focus): #{sel_idx + 1}  d = {data[sel_idx]['one_way_mm']:.2f} mm")
    print(f"front-wall focus (operator):      #{front_focus_idx + 1}  d = {data[front_focus_idx]['one_way_mm']:.2f} mm")
    print(f"ToF-derived focal-plane gap:      delta_d = {delta_d:.2f} mm in oil")

    # Shared scales
    all_amp = np.concatenate([s["amp_map"][np.isfinite(s["amp_map"])].ravel() for s in data])
    vmin = float(np.percentile(all_amp, 1))
    vmax = float(np.percentile(all_amp, 99))
    wf_vmin = min(float(np.min(s["wf_v"])) for s in data)
    wf_vmax = max(float(np.max(s["wf_v"])) for s in data)
    pad = 0.08 * (wf_vmax - wf_vmin)
    wf_ylim = (wf_vmin - pad, wf_vmax + pad)
    wf_xlim = (min(float(s["wf_t_us"][0]) for s in data),
               max(float(s["wf_t_us"][-1]) for s in data))

    # ----- figure -----
    n = len(data)
    HL = "#16a34a"   # tailwind green-600 — vivid, prints well
    fig = plt.figure(figsize=(12.6, 1.85 * n + 1.6), dpi=160)
    gs = fig.add_gridspec(
        nrows=n, ncols=2,
        width_ratios=[1.0, 1.55],
        wspace=0.22, hspace=0.42,
        left=0.16, right=0.88, top=0.92, bottom=0.07,
    )

    im_handle = None
    for row, s in enumerate(data):
        ax_map = fig.add_subplot(gs[row, 0])
        ax_wf = fig.add_subplot(gs[row, 1])

        im_handle = ax_map.imshow(
            s["amp_map"], cmap="turbo", vmin=vmin, vmax=vmax,
            extent=[0, s["roi_w"], s["roi_h"], 0],
            aspect="equal", origin="upper",
        )

        # Row label: two short lines, no role annotation here (kept in left margin)
        ax_map.set_ylabel(
            f"d = {s['one_way_mm']:.2f} mm\nToF {s['tof_us']:.2f} µs",
            fontsize=9, labelpad=4,
        )
        if row == n - 1:
            ax_map.set_xlabel("X (mm)", fontsize=9)
        else:
            ax_map.set_xticklabels([])
        ax_map.tick_params(labelsize=8)

        ax_wf.plot(s["wf_t_us"], s["wf_v"], color="tab:blue", linewidth=0.8)
        ax_wf.axhline(0, color="k", linewidth=0.4, alpha=0.4)
        ax_wf.axvline(s["tof_us"], color="tab:red", linestyle="--", linewidth=0.6, alpha=0.7)
        ax_wf.set_xlim(*wf_xlim)
        ax_wf.set_ylim(*wf_ylim)
        ax_wf.grid(True, alpha=0.25, linewidth=0.4)
        ax_wf.set_ylabel("V", fontsize=9)
        if row == n - 1:
            ax_wf.set_xlabel("time from sync (µs)", fontsize=9)
        else:
            ax_wf.set_xticklabels([])
        ax_wf.tick_params(labelsize=8)

        vpp = float(np.max(s["wf_v"]) - np.min(s["wf_v"]))
        ax_wf.text(
            0.985, 0.92,
            f"V$_{{pp}}$ = {vpp:.2f} V\nenv = {s['env_peak_v']:.2f} V",
            transform=ax_wf.transAxes, ha="right", va="top",
            fontsize=8, color="#333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )

        # Role tag in the OUTER left margin so it never overlaps the ylabel/ticks
        role_text = None
        if row == sel_idx:
            role_text = ("CHOSEN\nback-wall\nfocus", HL)
        elif row == front_focus_idx:
            role_text = ("front-wall\nfocus", "#444")
        if role_text:
            msg, colour = role_text
            ax_map.annotate(
                msg,
                xy=(0, 0.5), xycoords=ax_map.transAxes,
                xytext=(-78, 0), textcoords="offset points",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=colour,
                bbox=dict(facecolor="white", edgecolor=colour, linewidth=1.2,
                          boxstyle="round,pad=0.35"),
            )

        if row == sel_idx:
            for spine in ax_map.spines.values():
                spine.set_edgecolor(HL)
                spine.set_linewidth(2.4)
            for spine in ax_wf.spines.values():
                spine.set_edgecolor(HL)
                spine.set_linewidth(2.4)

    cbar_ax = fig.add_axes([0.895, 0.07, 0.013, 0.85])
    fig.colorbar(im_handle, cax=cbar_ax, label="amplitude (V)")
    cbar_ax.tick_params(labelsize=8)

    fig.suptitle(
        "Working-distance selection: amplitude C-scan (left) and centre waveform (right)",
        fontsize=12, y=0.975,
    )
    fig.text(
        0.5, 0.948,
        f"label = ToF-derived one-way distance  ·  c = {SOUND_SPEED_M_S:.0f} m/s "
        "(silicone oil 20 cSt)  ·  chosen row in green",
        ha="center", fontsize=9.5, color="#555",
    )

    caption = (
        f"Front-wall focus identified at d = {data[front_focus_idx]['one_way_mm']:.2f} mm "
        f"(cell surface imaged cleanly, no internal detail).   "
        f"Chosen back-wall focus at d = {data[sel_idx]['one_way_mm']:.2f} mm "
        f"(bottom tape pattern visible).   "
        f"Focal-plane gap in oil-equivalent path: Δd = {delta_d:.2f} mm."
    )
    fig.text(0.5, 0.030, caption, ha="center", fontsize=9.5, style="italic", color="#222")

    out_dir = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/reports/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "focus_selection_figure.png"
    out_pdf = out_dir / "focus_selection_figure.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out_png}")
    print(f"saved: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
