"""Grid-plot the amplitude C-scan from every gauge-height-* run.

Re-renders from the NPZ feature maps with a SHARED colorbar so the colours
mean the same thing across heights — the existing per-scan PNGs auto-scale
their own colourbar and aren't directly comparable.

Usage
-----
    python scripts/plot_gauge_height_amplitude_grid.py
        # auto-find all cscan_gauge-height-*mm folders, sort by height
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSCAN_ROOT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/cscan")
HEIGHT_RE = re.compile(r"[Gg]auge-height-(\d+)mm")


def find_gauge_scans() -> list[tuple[int, Path]]:
    """Return [(height_mm, scan_dir), ...] sorted by height ascending."""
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


def load_amplitude(scan_dir: Path) -> tuple[np.ndarray, float, float]:
    """Return (amplitude 2-D array, roi_w_mm, roi_h_mm)."""
    npz_path = next(scan_dir.glob("scan_*.npz"), None)
    if npz_path is None:
        raise SystemExit(f"no scan_*.npz in {scan_dir}")
    d = np.load(npz_path)
    amp = np.asarray(d["amplitude"], dtype=np.float64)
    x_mm = np.asarray(d["x_mm"])
    y_mm = np.asarray(d["y_mm"])
    return amp, float(x_mm[-1] - x_mm[0]), float(y_mm[-1] - y_mm[0])


def main() -> int:
    scans = find_gauge_scans()
    if not scans:
        raise SystemExit("no gauge-height-* scans found")

    print(f"found {len(scans)} gauge-height scans:")
    arrays = []
    for h, d in scans:
        amp, w, hgt = load_amplitude(d)
        arrays.append((h, d, amp, w, hgt))
        print(f"  {h} mm  shape={amp.shape}  Vpp=[{np.nanmin(amp):.2f}, {np.nanmax(amp):.2f}]")

    # Shared colour scale = robust 1-99 percentile across all panels,
    # so a single hot/dead panel doesn't blow out the range for the rest.
    all_finite = np.concatenate([a[np.isfinite(a)].ravel() for _, _, a, _, _ in arrays])
    vmin = float(np.percentile(all_finite, 1))
    vmax = float(np.percentile(all_finite, 99))
    print(f"shared colour range (1-99 pct): [{vmin:.2f}, {vmax:.2f}] V")

    n = len(arrays)
    ncols = 3 if n > 2 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.4 * ncols, 4.3 * nrows),
                             dpi=140, squeeze=False)
    axes_flat = axes.ravel()

    im = None
    for ax, (h, d, amp, w, hgt) in zip(axes_flat, arrays):
        im = ax.imshow(amp, cmap="turbo", vmin=vmin, vmax=vmax,
                       extent=[0, w, hgt, 0], aspect="equal", origin="upper")
        ax.set_title(f"gauge-height = {h} mm", fontsize=11)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Amplitude C-scan vs gauge-height (shared colour scale)",
                 fontsize=13, y=1.00)
    fig.subplots_adjust(right=0.92, wspace=0.25, hspace=0.30)
    cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
    fig.colorbar(im, cax=cbar_ax, label="amplitude (V)")

    out_dir = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/reports/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gauge_height_amplitude_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
