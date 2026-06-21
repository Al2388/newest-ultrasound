"""Compare C-scan amplitude maps across gauge/transducer heights.

Creates a unified-color-scale contact sheet and quantitative height-selection
summary for the gauge-height scans collected on 2026-05-29.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


HEIGHT_RE = re.compile(r"gauge-height-(\d+(?:\.\d+)?)mm", re.IGNORECASE)


@dataclass
class GaugeScan:
    height_mm: float
    session_dir: Path
    npz_path: Path
    amplitude: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def find_gauge_scans(root: Path) -> list[GaugeScan]:
    scans: dict[float, GaugeScan] = {}
    for session_dir in sorted(root.glob("*")):
        if not session_dir.is_dir():
            continue
        match = HEIGHT_RE.search(session_dir.name)
        if not match:
            continue
        height_mm = float(match.group(1))
        npz_files = sorted(session_dir.glob("scan_*.npz"))
        if not npz_files:
            continue
        npz_path = npz_files[0]
        with np.load(npz_path) as z:
            scan = GaugeScan(
                height_mm=height_mm,
                session_dir=session_dir,
                npz_path=npz_path,
                amplitude=z["amplitude"].astype(float),
                x_mm=z["x_mm"].astype(float),
                y_mm=z["y_mm"].astype(float),
            )
        scans[height_mm] = scan
    return [scans[h] for h in sorted(scans)]


def load_strict_roi_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return z["mask_strict"].astype(bool)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def add_roi(ax: plt.Axes, mask: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray) -> None:
    r0, r1, c0, c1 = bbox_from_mask(mask)
    x0 = float(x_mm[c0])
    x1 = float(x_mm[c1 - 1])
    y0 = float(y_mm[r0])
    y1 = float(y_mm[r1 - 1])
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.3))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="black", linewidth=0.55, linestyle="--"))


def save_all(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def make_contact_sheet(scans: list[GaugeScan], strict_mask: np.ndarray | None, out_dir: Path) -> None:
    all_amp = np.concatenate([scan.amplitude.ravel() for scan in scans])
    vmin = float(np.nanpercentile(all_amp, 1.0))
    vmax = float(np.nanpercentile(all_amp, 99.0))

    n = len(scans)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = GridSpec(nrows, ncols + 1, figure=fig, width_ratios=[1, 1, 1, 0.045], hspace=0.24, wspace=0.12)

    last_im = None
    for i, scan in enumerate(scans):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        extent = (float(scan.x_mm.min()), float(scan.x_mm.max()), float(scan.y_mm.min()), float(scan.y_mm.max()))
        last_im = ax.imshow(scan.amplitude, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        if strict_mask is not None and strict_mask.shape == scan.amplitude.shape:
            add_roi(ax, strict_mask, scan.x_mm, scan.y_mm)
        ax.set_title(f"{scan.height_mm:g} mm", loc="left", fontweight="bold", pad=2)
        ax.set_xlabel("X (mm)")
        if col == 0:
            ax.set_ylabel("Y (mm)")
        else:
            ax.set_yticklabels([])

    for j in range(n, nrows * ncols):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        ax.axis("off")

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Amplitude (V)")
    fig.suptitle("Gauge-height amplitude comparison", x=0.02, y=0.995, ha="left", fontsize=9, fontweight="bold")
    save_all(fig, out_dir / "fig_gauge_height_amplitude_contact_sheet")
    plt.close(fig)


def quantify(scans: list[GaugeScan], strict_mask: np.ndarray | None) -> pd.DataFrame:
    rows = []
    for scan in scans:
        mask = strict_mask if strict_mask is not None and strict_mask.shape == scan.amplitude.shape else np.isfinite(scan.amplitude)
        vals = scan.amplitude[mask]
        vals = vals[np.isfinite(vals)]
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1))
        rows.append(
            {
                "height_mm": scan.height_mm,
                "session_dir": str(scan.session_dir),
                "scan_file": str(scan.npz_path),
                "roi_pixels": int(vals.size),
                "roi_mean_amplitude_v": mean,
                "roi_median_amplitude_v": float(np.nanmedian(vals)),
                "roi_std_amplitude_v": std,
                "roi_cv_percent": 100.0 * std / mean if mean else np.nan,
                "roi_p05_amplitude_v": float(np.nanpercentile(vals, 5)),
                "roi_p95_amplitude_v": float(np.nanpercentile(vals, 95)),
                "roi_p95_minus_p05_v": float(np.nanpercentile(vals, 95) - np.nanpercentile(vals, 5)),
            }
        )
    return pd.DataFrame(rows).sort_values("height_mm")


def make_quant_figure(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    x = df["height_mm"].to_numpy()

    axes[0].plot(x, df["roi_mean_amplitude_v"], marker="o", color="#225ea8", linewidth=1.4, label="mean")
    axes[0].plot(x, df["roi_median_amplitude_v"], marker="s", color="#41ab5d", linewidth=1.1, label="median")
    axes[0].set_title("ROI amplitude", loc="left", fontweight="bold", pad=3)
    axes[0].set_xlabel("Gauge height (mm)")
    axes[0].set_ylabel("Amplitude (V)")
    axes[0].legend()

    axes[1].plot(x, df["roi_p95_amplitude_v"], marker="o", color="#d95f02", linewidth=1.4)
    axes[1].plot(x, df["roi_p05_amplitude_v"], marker="o", color="#756bb1", linewidth=1.0)
    axes[1].fill_between(x, df["roi_p05_amplitude_v"], df["roi_p95_amplitude_v"], color="#fdae6b", alpha=0.25, linewidth=0)
    axes[1].set_title("ROI distribution", loc="left", fontweight="bold", pad=3)
    axes[1].set_xlabel("Gauge height (mm)")
    axes[1].set_ylabel("P05-P95 amplitude (V)")

    axes[2].plot(x, df["roi_cv_percent"], marker="o", color="#636363", linewidth=1.4)
    axes[2].set_title("Spatial uniformity", loc="left", fontweight="bold", pad=3)
    axes[2].set_xlabel("Gauge height (mm)")
    axes[2].set_ylabel("ROI CV (%)")

    for ax in axes:
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45)
        ax.set_xticks(x)
        ax.tick_params(axis="x", rotation=45)

    save_all(fig, out_dir / "fig_gauge_height_amplitude_quantification")
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    best_mean = df.loc[df["roi_mean_amplitude_v"].idxmax()]
    best_uniform = df.loc[df["roi_cv_percent"].idxmin()]
    best_balanced = df.assign(
        mean_norm=lambda x: x["roi_mean_amplitude_v"] / x["roi_mean_amplitude_v"].max(),
        cv_norm=lambda x: x["roi_cv_percent"] / x["roi_cv_percent"].max(),
    )
    best_balanced["score"] = best_balanced["mean_norm"] - 0.35 * best_balanced["cv_norm"]
    best_balanced_row = best_balanced.loc[best_balanced["score"].idxmax()]
    text = f"""# Gauge-Height Amplitude Comparison

Generated from {len(df)} C-scan gauge-height scans.

## Practical Reading

- Highest strict-ROI mean amplitude: {best_mean['height_mm']:.0f} mm ({best_mean['roi_mean_amplitude_v']:.3f} V).
- Most spatially uniform strict ROI: {best_uniform['height_mm']:.0f} mm (CV {best_uniform['roi_cv_percent']:.2f}%).
- Balanced amplitude/uniformity score: {best_balanced_row['height_mm']:.0f} mm.

For manuscript use, the contact sheet is best as a method/selection panel, while the quantification plot supports the chosen height numerically.
"""
    (out_dir / "GAUGE_HEIGHT_AMPLITUDE_SUMMARY.md").write_text(text, encoding="utf-8")


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/cscan")
    parser.add_argument("--out-dir", default="reports/experiments/gauge_height_amplitude_comparison_2026-05-29")
    parser.add_argument(
        "--strict-mask",
        default="reports/report c-scan baseline repeat noise level and roi/rectangular_roi_strictness_sweep/rectangular_roi_strictness_masks.npz",
    )
    args = parser.parse_args()

    configure_matplotlib()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scans = find_gauge_scans(data_root)
    if not scans:
        raise SystemExit(f"No gauge-height scans found under {data_root}")

    strict_mask = load_strict_roi_mask(Path(args.strict_mask))
    make_contact_sheet(scans, strict_mask, out_dir)
    df = quantify(scans, strict_mask)
    df.to_csv(out_dir / "gauge_height_amplitude_metrics.csv", index=False)
    write_markdown_table(df, out_dir / "gauge_height_amplitude_metrics.md")
    make_quant_figure(df, out_dir)
    write_summary(df, out_dir)
    print(f"Wrote gauge-height amplitude comparison to {out_dir}")


if __name__ == "__main__":
    main()
