"""Derive probe-to-battery distance from centre-waveform ToF.

Uses the envelope peak (Hilbert) of the centre pulse as the round-trip ToF,
then converts to one-way distance at silicone oil 20 cSt (c = 989 m/s).

Reports a table and a plot of gauge-height vs derived distance, plus a linear
fit so any constant electronic/cable delay shows up as the intercept.
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
SOUND_SPEED_M_S = 989.0          # silicone oil 20 cSt, room temperature
MM_PER_US = SOUND_SPEED_M_S / 1000.0   # = 0.989 mm/µs (one-way)


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
    """Return (envelope-peak ToF in µs, peak envelope amplitude V).

    Sub-sample refinement via 3-point parabolic interpolation around argmax,
    so we don't quantise to the 50 ns sample period.
    """
    env = np.abs(hilbert(wf))
    i = int(np.argmax(env))
    if 0 < i < len(env) - 1:
        y0, y1, y2 = env[i - 1], env[i], env[i + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    dt = float(t_us[1] - t_us[0])
    t_peak = float(t_us[i]) + delta * dt
    a_peak = float(env[i])
    return t_peak, a_peak


def centre_waveform(scan_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (waveform_v, time_us) for the geometric centre pulse."""
    meta = json.loads(next(scan_dir.glob("scan_*_meta.json")).read_text())
    roi_w = float(meta["roi_w_mm"])

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
    return wf, t_us


def main() -> int:
    scans = find_gauge_scans()
    rows = []
    for h, d in scans:
        wf, t_us = centre_waveform(d)
        tof, amp = envelope_tof_us(wf, t_us)
        one_way_mm = 0.5 * tof * MM_PER_US
        rows.append({
            "gauge_mm": h,
            "tof_us": tof,
            "one_way_mm": one_way_mm,
            "env_amp_v": amp,
        })

    # Table
    print(f"{'gauge':>7} {'ToF (us)':>10} {'one-way (mm)':>14} {'env Vpk':>10}")
    print("-" * 44)
    for r in rows:
        print(f"{r['gauge_mm']:>7} {r['tof_us']:>10.3f} {r['one_way_mm']:>14.3f} {r['env_amp_v']:>10.3f}")

    # Linear fit: gauge = a + b * one_way_distance
    # If gauge and distance both move with the printer Z axis equally, b should
    # be 1.0 and a should equal (printer Z when probe touches battery).
    x = np.array([r["one_way_mm"] for r in rows])
    y = np.array([r["gauge_mm"] for r in rows], dtype=float)
    # Exclude near-field heights where the envelope doesn't track the front-wall
    # reflection — packet shape goes weird below ~215 mm gauge. Keep clean ones.
    keep = np.array([r["env_amp_v"] for r in rows]) > 1.5
    if keep.sum() >= 2:
        b, a = np.polyfit(x[keep], y[keep], 1)
        residuals = y[keep] - (a + b * x[keep])
        rms_residual = float(np.sqrt(np.mean(residuals ** 2)))
        print()
        print(f"linear fit (clean envelopes only, n={int(keep.sum())}):")
        print(f"  gauge = {a:.3f} + {b:.4f} × one_way_mm")
        print(f"  slope b = {b:.4f}  (expected 1.0 if printer Z == probe gap)")
        print(f"  intercept a = {a:.3f} mm  (printer Z when one_way distance is 0)")
        print(f"  RMS residual = {rms_residual*1000:.0f} µm")
    else:
        b = a = rms_residual = float("nan")

    # ----- plot -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160)

    ax = axes[0]
    ax.plot([r["one_way_mm"] for r in rows], [r["gauge_mm"] for r in rows],
            "o", color="tab:blue", markersize=6, label="all heights")
    ax.plot([r["one_way_mm"] for i, r in enumerate(rows) if not keep[i]],
            [r["gauge_mm"] for i, r in enumerate(rows) if not keep[i]],
            "x", color="red", markersize=10, label="near-field (excluded)")
    if np.isfinite(b):
        xx = np.array([x.min() - 0.5, x.max() + 0.5])
        ax.plot(xx, a + b * xx, "-", color="tab:gray",
                label=f"fit: gauge = {a:.2f} + {b:.3f}·d")
    ax.set_xlabel("one-way distance from ToF (mm)\nc = 989 m/s")
    ax.set_ylabel("gauge-height label (mm)")
    ax.set_title("Derived distance vs gauge reading")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    gauges = [r["gauge_mm"] for r in rows]
    tofs = [r["tof_us"] for r in rows]
    ax.plot(gauges, tofs, "o-", color="tab:red", markersize=6)
    ax.set_xlabel("gauge-height (mm)")
    ax.set_ylabel("envelope ToF (µs)")
    ax.set_title("ToF vs gauge — slope should be 2/c = 2.02 µs/mm")
    ax.grid(True, alpha=0.3)
    # Annotate the empirical slope from the clean linear region
    if keep.sum() >= 2:
        gauges_clean = np.array(gauges)[keep]
        tofs_clean = np.array(tofs)[keep]
        slope_tof, intercept_tof = np.polyfit(gauges_clean, tofs_clean, 1)
        ax.text(0.04, 0.94,
                f"empirical slope: {slope_tof:.3f} µs/mm\n"
                f"theory (2/c):    {2/MM_PER_US:.3f} µs/mm",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#999", alpha=0.85))

    fig.tight_layout()
    out_dir = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/reports/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "height_from_tof.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
