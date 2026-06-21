"""Plot the centre-point waveform of a C-scan as time vs amplitude.

Used to compare waveform shape between scans taken at different battery
heights — the centre pixel is a stable reference point across scans.

Usage
-----
    python scripts/plot_center_waveform.py
        # defaults to the most recent run in data/raw/cscan

    python scripts/plot_center_waveform.py path/to/cscan_dir
        # plot specific scan
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSCAN_ROOT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/cscan")


def latest_scan_dir() -> Path:
    candidates = [p for p in CSCAN_ROOT.iterdir() if p.is_dir() and (p / "lines_raw").exists()]
    if not candidates:
        raise SystemExit(f"no scan with lines_raw/ under {CSCAN_ROOT}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_center(scan_dir: Path) -> Path:
    lines_dir = scan_dir / "lines_raw"
    line_files = sorted(lines_dir.glob("line_*.npz"))
    if not line_files:
        raise SystemExit(f"no line_*.npz files in {lines_dir}")

    # Centre line = midpoint of the actual scanned lines
    centre_line_idx = len(line_files) // 2
    line_path = line_files[centre_line_idx]
    data = np.load(line_path)

    if "waveforms" not in data.files:
        raise SystemExit(f"{line_path.name} has no 'waveforms' — was save_waveforms=False?")

    waveforms = data["waveforms"]     # [n_pulses, gate_samples]
    x_mm = data["x_mm"]               # [n_pulses]
    y_mm = float(data["y_mm"])
    fs_hz = float(data["fs_hz"])
    gate_us = data["gate_us"]         # [gate_start, gate_end]

    # Centre pulse along this line = pulse with x closest to ROI midpoint.
    # Read ROI width from the sibling meta json so this works for non-default scans.
    roi_w_mm = 80.0
    meta_files = list(scan_dir.glob("scan_*_meta.json"))
    if meta_files:
        import json
        with meta_files[0].open() as f:
            roi_w_mm = float(json.load(f).get("roi_w_mm", roi_w_mm))
    centre_x = roi_w_mm / 2.0
    # x_mm has NaN entries for pulses outside the modelled motion window
    # (accel/decel ramps). Skip them so we land on a real centre pulse.
    valid = np.isfinite(x_mm)
    if not np.any(valid):
        raise SystemExit("no pulses with finite x_mm in this line")
    valid_idx = np.flatnonzero(valid)
    pulse_idx = int(valid_idx[np.argmin(np.abs(x_mm[valid] - centre_x))])

    wf = waveforms[pulse_idx].astype(np.float64)
    dt_us = 1e6 / fs_hz
    t_us = float(gate_us[0]) + np.arange(wf.size) * dt_us

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(t_us, wf, color="tab:blue", linewidth=0.9)
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("amplitude (V)")
    ax.set_title(
        f"{scan_dir.name}\n"
        f"centre pulse @ y={y_mm:.2f} mm, x={float(x_mm[pulse_idx]):.2f} mm "
        f"(line {centre_line_idx}/{len(line_files)-1}, pulse {pulse_idx}/{waveforms.shape[0]-1})"
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    out = scan_dir / "center_waveform.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    if len(sys.argv) > 1:
        scan_dir = Path(sys.argv[1])
        if not scan_dir.is_absolute():
            scan_dir = (CSCAN_ROOT / scan_dir).resolve()
    else:
        scan_dir = latest_scan_dir()

    print(f"scan: {scan_dir}")
    out = plot_center(scan_dir)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
