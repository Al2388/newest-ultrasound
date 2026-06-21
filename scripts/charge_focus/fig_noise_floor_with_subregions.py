"""Re-render the noise-floor figure (baseline + 3 sigma maps) with sub-region
dividers and labels drawn inside the ROI: tab-distal / mid / tab-proximal.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
SRC_PATTERN = str(PROJ / "data/raw/cscan/cscan_noisefloor_v3.238_2026-05-29_15-02-49_r0*")
ROI_JSON_PATH = (PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp"
                 / "roi_summary.json")

OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "fig_noise_floor_subregions.pdf"
OUT_PNG = OUT_DIR / "fig_noise_floor_subregions.png"

# Sub-region X bands (mm) inside the ROI
ROI_X_MM = (14.6, 64.5)
ROI_Y_MM = (16.1, 55.4)
DIVIDERS_MM = [30.0, 50.0]
SUB_LABELS = [
    ((14.6 + 30.0) / 2, "tab-distal"),
    ((30.0 + 50.0) / 2, "mid"),
    ((50.0 + 64.5) / 2, "tab-prox"),
]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      9,
    "axes.linewidth":      0.6,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "savefig.dpi":        300,
    "pdf.fonttype":        42,
})


def add_roi_and_dividers(ax, color="white"):
    """White ROI rectangle + vertical sub-region dividers + top labels (all white)."""
    # ROI rectangle
    rect = mpatches.Rectangle(
        (ROI_X_MM[0], ROI_Y_MM[0]),
        ROI_X_MM[1] - ROI_X_MM[0],
        ROI_Y_MM[1] - ROI_Y_MM[0],
        fill=False, edgecolor=color, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(ROI_X_MM[0] + 0.8, ROI_Y_MM[0] + 1.6, "ROI",
            color=color, fontsize=8, fontweight="bold",
            va="top", ha="left")

    # Vertical dashed dividers inside the ROI
    for xb in DIVIDERS_MM:
        ax.plot([xb, xb], [ROI_Y_MM[0], ROI_Y_MM[1]],
                "--", color=color, lw=1.2, alpha=0.95)

    # Sub-region labels just above the ROI -- white text, no background box
    y_lbl = ROI_Y_MM[0] - 1.0
    for cx, lbl in SUB_LABELS:
        ax.text(cx, y_lbl, lbl, ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold")


def build_roi(x_mm, y_mm):
    x_idx = np.where((x_mm >= ROI_X_MM[0]) & (x_mm <= ROI_X_MM[1]))[0]
    y_idx = np.where((y_mm >= ROI_Y_MM[0]) & (y_mm <= ROI_Y_MM[1]))[0]
    roi = np.zeros((y_mm.size, x_mm.size), dtype=bool)
    roi[np.ix_(y_idx, x_idx)] = True
    return roi


def main():
    sessions = sorted(glob.glob(SRC_PATTERN))
    if len(sessions) != 6:
        raise SystemExit(f"expected 6 v3.238 sessions, found {len(sessions)}")

    amp_stack, tof_stack, eng_stack = [], [], []
    x_mm = y_mm = None
    for sess in sessions:
        npz = next(Path(sess).glob("scan_*.npz"))
        d = np.load(npz)
        amp_stack.append(d["amplitude"].astype(np.float32))
        tof_stack.append(d["tof"].astype(np.float32))
        eng_stack.append(d["energy"].astype(np.float32))
        if x_mm is None:
            x_mm = d["x_mm"]; y_mm = d["y_mm"]
    amp = np.stack(amp_stack); tof = np.stack(tof_stack); eng = np.stack(eng_stack)

    mean_amp = np.nanmean(amp, axis=0)
    sigma_amp = np.nanstd(amp, axis=0, ddof=1)
    sigma_tof = np.nanstd(tof, axis=0, ddof=1)
    sigma_eng = np.nanstd(eng, axis=0, ddof=1)

    roi = build_roi(x_mm, y_mm)

    # ----- Figure -----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6),
                             constrained_layout=True)
    extent = [float(x_mm[0]), float(x_mm[-1]),
              float(y_mm[-1]), float(y_mm[0])]

    # Panel 0: baseline mean amplitude (grayscale, like the figure shared)
    ax = axes[0]
    vmin = float(np.percentile(mean_amp[np.isfinite(mean_amp)], 1))
    vmax = float(np.percentile(mean_amp[np.isfinite(mean_amp)], 99))
    ax.imshow(mean_amp, cmap="gray", vmin=vmin, vmax=vmax,
              extent=extent, origin="upper", aspect="equal")
    add_roi_and_dividers(ax)
    ax.set_title("6-scan mean amplitude  (baseline)", fontsize=10, loc="left")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")

    # Panels (a) (b) (c): per-pixel sigma maps
    panels = [
        (sigma_amp, "(a) σ amplitude (V)",  "σ amplitude (V)",  "mV",  1e3),
        (sigma_tof, "(b) σ ToF (µs)",        "σ ToF (µs)",       "ns",  1e3),
        (sigma_eng, "(c) σ energy",          "σ energy",          "",    1.0),
    ]
    for ax, (sigma_map, title, cbar_lbl, unit, scale) in zip(axes[1:], panels):
        fin = sigma_map[np.isfinite(sigma_map)]
        vmax = float(np.percentile(fin, 98))
        im = ax.imshow(sigma_map, cmap="magma", vmin=0,
                       vmax=max(vmax, 1e-9),
                       extent=extent, origin="upper", aspect="equal")
        plt.colorbar(im, ax=ax, label=cbar_lbl, fraction=0.046, pad=0.04)

        # ROI per-pixel stats annotation (matches user's reference style)
        roi_vals = sigma_map[roi & np.isfinite(sigma_map)]
        med = float(np.median(roi_vals)) * scale
        p95 = float(np.percentile(roi_vals, 95)) * scale
        ax.text(0.98, 0.97,
                f"ROI per-pixel σ\nmed = {med:.2f} {unit}\np95 = {p95:.2f} {unit}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="white", fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="white",
                          linewidth=0.4, pad=4, alpha=0.78))

        add_roi_and_dividers(ax)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("X (mm)")
        if ax is axes[1]:
            ax.set_ylabel("Y (mm)")

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
