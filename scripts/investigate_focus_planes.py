"""Carefully compare candidate front-wall and back-wall focus scans.

Shows the C-scan amplitude maps + centre waveforms side by side for the
extreme scans of the working-distance series so we can see what each one
actually images.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert


CSCAN_ROOT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/cscan")
HEIGHT_RE = re.compile(r"[Gg]auge-height-(\d+)mm")
SOUND_SPEED_M_S = 989.0
MM_PER_US_ONE_WAY = SOUND_SPEED_M_S / 1000.0 / 2.0


def find_gauge_scans():
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


def envelope_tof(wf, t_us):
    env = np.abs(hilbert(wf))
    i = int(np.argmax(env))
    if 0 < i < len(env) - 1:
        y0, y1, y2 = env[i - 1], env[i], env[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    dt = float(t_us[1] - t_us[0])
    return float(t_us[i]) + delta * dt, env


def load_scan(scan_dir):
    npz_path = next(scan_dir.glob("scan_*.npz"))
    d = np.load(npz_path)
    amp_map = np.asarray(d["amplitude"], dtype=np.float64)

    meta = json.loads(next(scan_dir.glob("scan_*_meta.json")).read_text())
    roi_w = float(meta["roi_w_mm"])
    roi_h = float(meta["roi_h_mm"])

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
    tof, env = envelope_tof(wf, t_us)
    return {
        "amp_map": amp_map, "roi_w": roi_w, "roi_h": roi_h,
        "wf_t_us": t_us, "wf_v": wf, "wf_env": env,
        "tof_us": tof, "one_way_mm": tof * MM_PER_US_ONE_WAY,
    }


def main():
    scans = find_gauge_scans()
    rows = [(h, load_scan(p)) for h, p in scans]

    # Pick three scans to compare in detail:
    #   - scan 2  (gauge 213, d ≈ 16.65 mm) – chosen back-wall focus
    #   - scan 5  (gauge 216, d ≈ 18.85 mm) – previous (incorrect) "front" guess
    #   - scan 9  (gauge 220, d ≈ 21.88 mm) – operator-identified front-wall focus
    pick_labels = [213, 216, 220]
    annotations = {
        213: "back-wall focus (chosen):\nbottom tape visible",
        216: "max envelope amplitude\n(NOT front-wall focus on its own)",
        220: "front-wall focus (operator):\nbattery surface only",
    }
    picks = [(g, s) for g, s in rows if g in pick_labels]

    # Shared scales
    all_amp = np.concatenate([s["amp_map"][np.isfinite(s["amp_map"])].ravel()
                              for _, s in rows])
    vmin = float(np.percentile(all_amp, 1))
    vmax = float(np.percentile(all_amp, 99))
    wf_xlim = (min(s["wf_t_us"][0] for _, s in rows),
               max(s["wf_t_us"][-1] for _, s in rows))
    wf_ylim_lo = min(s["wf_v"].min() for _, s in rows)
    wf_ylim_hi = max(s["wf_v"].max() for _, s in rows)
    pad = 0.08 * (wf_ylim_hi - wf_ylim_lo)
    wf_ylim = (wf_ylim_lo - pad, wf_ylim_hi + pad)

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), dpi=160,
                             gridspec_kw={"width_ratios": [1, 1.4],
                                          "wspace": 0.20, "hspace": 0.32,
                                          "left": 0.08, "right": 0.92,
                                          "top": 0.93, "bottom": 0.06})

    HL_BACK = "#16a34a"      # green
    HL_FRONT = "#2563eb"     # blue
    HL_NEUTRAL = "#6b7280"   # gray

    im = None
    for r, (g, s) in enumerate(picks):
        colour = HL_BACK if g == 213 else (HL_FRONT if g == 220 else HL_NEUTRAL)
        ax_map = axes[r][0]
        ax_wf = axes[r][1]

        im = ax_map.imshow(s["amp_map"], cmap="turbo", vmin=vmin, vmax=vmax,
                           extent=[0, s["roi_w"], s["roi_h"], 0],
                           aspect="equal", origin="upper")
        ax_map.set_ylabel("Y (mm)", fontsize=9)
        ax_map.set_xlabel("X (mm)", fontsize=9)
        ax_map.tick_params(labelsize=8)
        ax_map.set_title(
            f"d = {s['one_way_mm']:.2f} mm   ToF = {s['tof_us']:.2f} µs",
            fontsize=10, color=colour, fontweight="bold", loc="left",
        )

        ax_wf.plot(s["wf_t_us"], s["wf_v"], color="tab:blue", linewidth=0.9, label="raw")
        ax_wf.plot(s["wf_t_us"], s["wf_env"], color="tab:red", linewidth=1.1, alpha=0.85, label="envelope")
        ax_wf.plot(s["wf_t_us"], -s["wf_env"], color="tab:red", linewidth=1.1, alpha=0.85)
        ax_wf.axvline(s["tof_us"], color="k", linestyle="--", linewidth=0.7, alpha=0.6,
                      label=f"envelope peak (ToF)")
        ax_wf.axhline(0, color="k", linewidth=0.3, alpha=0.4)
        ax_wf.set_xlim(*wf_xlim)
        ax_wf.set_ylim(*wf_ylim)
        ax_wf.grid(True, alpha=0.25, linewidth=0.4)
        ax_wf.set_xlabel("time from sync (µs)", fontsize=9)
        ax_wf.set_ylabel("V", fontsize=9)
        ax_wf.tick_params(labelsize=8)
        ax_wf.set_title(annotations[g], fontsize=9, color=colour, loc="left")
        if r == 0:
            ax_wf.legend(loc="lower right", fontsize=7)

        for spine in ax_map.spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(2.0)
        for spine in ax_wf.spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(2.0)

    cbar_ax = fig.add_axes([0.945, 0.07, 0.012, 0.84])
    fig.colorbar(im, cax=cbar_ax, label="amplitude (V)")
    cbar_ax.tick_params(labelsize=8)

    fig.suptitle("What does each working distance actually image?",
                 fontsize=12, y=0.975)

    # Print a summary table
    print(f"{'gauge':>5} {'ToF (µs)':>10} {'d (mm)':>9} {'env peak ToF':>14}")
    print("-" * 44)
    for g, s in rows:
        print(f"{g:>5} {s['tof_us']:>10.3f} {s['one_way_mm']:>9.3f} {s['wf_env'].max():>14.3f}")

    out = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/reports/experiments/focus_plane_investigation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
