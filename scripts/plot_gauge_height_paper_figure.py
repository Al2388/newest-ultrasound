"""Paper-quality composite: amplitude C-scan + centre waveform, per gauge height.

Layout: one row per height, two columns:
    [ amplitude map (shared colour scale) | centre pulse waveform (shared axes) ]

The shared scales mean colour and amplitude can be compared directly across
heights — that's the key reason this script re-renders from NPZ instead of
stitching the auto-scaled per-scan PNGs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSCAN_ROOT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/cscan")
HEIGHT_RE = re.compile(r"[Gg]auge-height-(\d+)mm")


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


def load_scan(scan_dir: Path) -> dict:
    npz_path = next(scan_dir.glob("scan_*.npz"))
    d = np.load(npz_path)
    amp_map = np.asarray(d["amplitude"], dtype=np.float64)
    x_mm = np.asarray(d["x_mm"])
    y_mm = np.asarray(d["y_mm"])

    meta = json.loads(next(scan_dir.glob("scan_*_meta.json")).read_text())
    roi_w = float(meta.get("roi_w_mm", x_mm[-1] - x_mm[0]))
    roi_h = float(meta.get("roi_h_mm", y_mm[-1] - y_mm[0]))

    # Centre pulse waveform from the middle scan line.
    line_files = sorted((scan_dir / "lines_raw").glob("line_*.npz"))
    centre_line = np.load(line_files[len(line_files) // 2])
    waveforms = centre_line["waveforms"]
    x_pulse = centre_line["x_mm"]
    fs_hz = float(centre_line["fs_hz"])
    gate_us = centre_line["gate_us"]

    valid = np.isfinite(x_pulse)
    valid_idx = np.flatnonzero(valid)
    centre_x = roi_w / 2.0
    pulse_idx = int(valid_idx[np.argmin(np.abs(x_pulse[valid] - centre_x))])
    wf = waveforms[pulse_idx].astype(np.float64)
    t_us = float(gate_us[0]) + np.arange(wf.size) * (1e6 / fs_hz)

    return {
        "amp_map": amp_map,
        "roi_w": roi_w, "roi_h": roi_h,
        "wf_t_us": t_us, "wf_v": wf,
    }


def main() -> int:
    scans = find_gauge_scans()
    if not scans:
        raise SystemExit("no gauge-height-* scans found")

    data = [(h, load_scan(p)) for h, p in scans]
    print(f"loaded {len(data)} heights: {[h for h, _ in data]}")

    # ----- shared scales -----
    all_amp = np.concatenate([s["amp_map"][np.isfinite(s["amp_map"])].ravel()
                              for _, s in data])
    vmin = float(np.percentile(all_amp, 1))
    vmax = float(np.percentile(all_amp, 99))

    wf_vmin = min(float(np.min(s["wf_v"])) for _, s in data)
    wf_vmax = max(float(np.max(s["wf_v"])) for _, s in data)
    pad = 0.08 * (wf_vmax - wf_vmin)
    wf_ylim = (wf_vmin - pad, wf_vmax + pad)

    t_min = min(float(s["wf_t_us"][0]) for _, s in data)
    t_max = max(float(s["wf_t_us"][-1]) for _, s in data)
    wf_xlim = (t_min, t_max)

    print(f"shared amp colour scale: [{vmin:.2f}, {vmax:.2f}] V")
    print(f"shared waveform y: {wf_ylim}, x: {wf_xlim}")

    # ----- figure -----
    n = len(data)
    fig = plt.figure(figsize=(11.0, 1.8 * n + 0.6), dpi=160)
    # 2 columns: C-scan map (square-ish) | waveform (wider). gridspec keeps
    # both rows aligned and leaves room on the right for a shared colourbar.
    gs = fig.add_gridspec(
        nrows=n, ncols=2,
        width_ratios=[1.0, 1.45],
        wspace=0.18, hspace=0.30,
        left=0.07, right=0.91, top=0.95, bottom=0.06,
    )

    im_handle = None
    for row, (h, s) in enumerate(data):
        ax_map = fig.add_subplot(gs[row, 0])
        ax_wf = fig.add_subplot(gs[row, 1])

        im_handle = ax_map.imshow(
            s["amp_map"], cmap="turbo", vmin=vmin, vmax=vmax,
            extent=[0, s["roi_w"], s["roi_h"], 0],
            aspect="equal", origin="upper",
        )
        ax_map.set_ylabel(f"{h} mm\nY (mm)", fontsize=9)
        if row == n - 1:
            ax_map.set_xlabel("X (mm)", fontsize=9)
        else:
            ax_map.set_xticklabels([])
        ax_map.tick_params(labelsize=8)

        ax_wf.plot(s["wf_t_us"], s["wf_v"], color="tab:blue", linewidth=0.8)
        ax_wf.axhline(0, color="k", linewidth=0.4, alpha=0.4)
        ax_wf.set_xlim(*wf_xlim)
        ax_wf.set_ylim(*wf_ylim)
        ax_wf.grid(True, alpha=0.25, linewidth=0.4)
        ax_wf.set_ylabel("V", fontsize=9)
        if row == n - 1:
            ax_wf.set_xlabel("time (µs)", fontsize=9)
        else:
            ax_wf.set_xticklabels([])
        ax_wf.tick_params(labelsize=8)

        # Vpp annotation inside the waveform panel so reviewers can read it off.
        vpp = float(np.max(s["wf_v"]) - np.min(s["wf_v"]))
        ax_wf.text(
            0.985, 0.92, f"V$_{{pp}}$ = {vpp:.2f} V",
            transform=ax_wf.transAxes, ha="right", va="top",
            fontsize=8, color="#333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )

    # Shared colourbar on the right
    cbar_ax = fig.add_axes([0.925, 0.06, 0.015, 0.89])
    fig.colorbar(im_handle, cax=cbar_ax, label="amplitude (V)")
    cbar_ax.tick_params(labelsize=8)

    fig.suptitle(
        "Amplitude C-scan (left) and centre-point waveform (right) "
        "vs gauge-height",
        fontsize=12, y=0.985,
    )

    out_dir = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/reports/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "gauge_height_paper_figure.png"
    out_pdf = out_dir / "gauge_height_paper_figure.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
