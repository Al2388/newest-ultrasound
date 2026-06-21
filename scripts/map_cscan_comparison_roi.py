#!/usr/bin/env python
"""Build recommended comparison ROIs from a repeated C-scan noise-floor batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def _load_baseline_scan(noise_dir: Path) -> tuple[np.lib.npyio.NpzFile, dict, Path, list[str], int]:
    noise = np.load(noise_dir / "noise_floor_diffs.npz")
    scan_ids = [str(x) for x in noise["scan_ids"]]
    baseline_idx = int(noise["baseline_index"])
    baseline_id = scan_ids[baseline_idx]

    meta = None
    scan_dir = None
    for d in sorted(Path("data/raw/cscan").glob("*")):
        if not d.is_dir():
            continue
        meta_files = list(d.glob("scan_*_meta.json"))
        if not meta_files:
            continue
        meta_path = meta_files[0]
        candidate = json.load(open(meta_path, "r", encoding="utf-8"))
        if candidate.get("scan_id") == baseline_id:
            meta = candidate
            scan_dir = d
            break
    if meta is None or scan_dir is None:
        raise RuntimeError(f"Could not locate baseline scan {baseline_id}")
    return np.load(scan_dir / meta["feature_map_file"]), meta, scan_dir, scan_ids, baseline_idx


def _add_geometry(spec: dict, spans: dict, rows: int, cols: int,
                  roi_w: float, roi_h: float) -> tuple[np.ndarray, np.ndarray]:
    cs, ce = spec["col_start"], spec["col_end"]
    rs, re = spec["row_start"], spec["row_end"]
    spec["n_rows"] = re - rs
    spec["n_cols"] = ce - cs
    spec["area_fraction"] = float((re - rs) * (ce - cs) / (rows * cols))
    spec["x_mm_min"] = float(cs * roi_w / (cols - 1))
    spec["x_mm_max"] = float((ce - 1) * roi_w / (cols - 1))
    spec["y_mm_min"] = float(rs * roi_h / (rows - 1))
    spec["y_mm_max"] = float((re - 1) * roi_h / (rows - 1))

    crop = np.zeros((rows, cols), dtype=bool)
    crop[rs:re, cs:ce] = True
    th = spec["thresholds"]
    stable = (
        crop
        & (spans["amplitude"] <= th["amplitude_span_v"])
        & (spans["tof"] <= th["tof_span_us"])
        & (spans["energy"] <= th["energy_span"])
    )
    spec["stable_pixel_fraction_within_roi"] = float(stable.sum() / crop.sum())
    spec["stable_pixel_fraction_total"] = float(stable.mean())
    spec["metrics"] = {}
    for feature in ("amplitude", "tof", "energy"):
        arr = spans[feature][crop]
        spec["metrics"][feature] = {
            "median_span": float(np.nanmedian(arr)),
            "p95_span": float(np.nanpercentile(arr, 95)),
            "p99_span": float(np.nanpercentile(arr, 99)),
            "max_span": float(np.nanmax(arr)),
        }
    return crop, stable


def _save_overlay(noise_dir: Path, amp: np.ndarray, roi_w: float, roi_h: float,
                  balanced: dict, strict: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    im = ax.imshow(amp, cmap="turbo", aspect="auto", extent=[0, roi_w, roi_h, 0])
    fig.colorbar(im, ax=ax, label="Baseline amplitude (V)")
    for spec, color, label in ((balanced, "cyan", "balanced"), (strict, "lime", "strict")):
        x0, x1 = spec["x_mm_min"], spec["x_mm_max"]
        y0, y1 = spec["y_mm_min"], spec["y_mm_max"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor=color, linewidth=2.2, label=label))
    ax.set_title("Recommended comparison ROIs on baseline amplitude")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(noise_dir / "comparison_roi_overlay.png")
    plt.close(fig)


def _save_noise_score(noise_dir: Path, score: np.ndarray, roi_w: float, roi_h: float,
                      balanced: dict, strict: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    im = ax.imshow(score, cmap="viridis", aspect="auto", extent=[0, roi_w, roi_h, 0],
                   vmin=0, vmax=2)
    fig.colorbar(im, ax=ax, label="combined noise score; <=1 passes balanced threshold")
    for spec, color, label in ((balanced, "white", "balanced"), (strict, "red", "strict")):
        x0, x1 = spec["x_mm_min"], spec["x_mm_max"]
        y0, y1 = spec["y_mm_min"], spec["y_mm_max"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor=color, linewidth=2.0, label=label))
    ax.set_title("Noise-floor based comparison suitability map")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(noise_dir / "comparison_roi_noise_score.png")
    plt.close(fig)


def _save_stable_mask(noise_dir: Path, rows: int, cols: int, roi_w: float, roi_h: float,
                      crop: np.ndarray, stable: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    mask_plot = np.full((rows, cols), np.nan)
    mask_plot[crop] = 0.25
    mask_plot[stable] = 1.0
    im = ax.imshow(mask_plot, cmap="Greens", aspect="auto", extent=[0, roi_w, roi_h, 0],
                   vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="stable comparison mask")
    ax.set_title("Balanced ROI stable-pixel mask")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    fig.tight_layout()
    fig.savefig(noise_dir / "comparison_roi_stable_mask.png")
    plt.close(fig)


def build_recommendation(noise_dir: Path) -> dict:
    noise = np.load(noise_dir / "noise_floor_diffs.npz")
    spans = {
        feature: np.asarray(noise[f"{feature}_noise_floor_span"], dtype=float)
        for feature in ("amplitude", "tof", "energy")
    }
    scan, meta, _, scan_ids, baseline_idx = _load_baseline_scan(noise_dir)
    amp = np.asarray(scan["amplitude"], dtype=float)
    rows, cols = amp.shape
    roi_w = float(meta["roi_w_mm"])
    roi_h = float(meta["roi_h_mm"])
    baseline_id = scan_ids[baseline_idx]

    balanced = {
        "name": "balanced_core_roi",
        "row_start": 12,
        "row_end": rows - 12,
        "col_start": 35,
        "col_end": cols - 35,
        "thresholds": {
            "amplitude_span_v": 0.55,
            "tof_span_us": 0.18,
            "energy_span": 32.0,
        },
    }
    strict = {
        "name": "strict_core_roi",
        "row_start": 15,
        "row_end": rows - 15,
        "col_start": 50,
        "col_end": cols - 50,
        "thresholds": {
            "amplitude_span_v": 0.38,
            "tof_span_us": 0.12,
            "energy_span": 25.0,
        },
    }

    balanced_crop, balanced_stable = _add_geometry(balanced, spans, rows, cols, roi_w, roi_h)
    strict_crop, strict_stable = _add_geometry(strict, spans, rows, cols, roi_w, roi_h)

    th = balanced["thresholds"]
    score = np.nanmax(
        np.stack([
            spans["amplitude"] / th["amplitude_span_v"],
            spans["tof"] / th["tof_span_us"],
            spans["energy"] / th["energy_span"],
        ]),
        axis=0,
    )

    np.savez_compressed(
        noise_dir / "comparison_roi_masks.npz",
        balanced_core_roi=balanced_crop,
        balanced_stable_mask=balanced_stable,
        strict_core_roi=strict_crop,
        strict_stable_mask=strict_stable,
        combined_noise_score=score.astype(np.float32),
    )
    _save_overlay(noise_dir, amp, roi_w, roi_h, balanced, strict)
    _save_noise_score(noise_dir, score, roi_w, roi_h, balanced, strict)
    _save_stable_mask(noise_dir, rows, cols, roi_w, roi_h, balanced_crop, balanced_stable)

    recommendation = {
        "baseline_scan_id": baseline_id,
        "scan_shape": {"rows": rows, "cols": cols, "roi_w_mm": roi_w, "roi_h_mm": roi_h},
        "default_recommendation": (
            "balanced_core_roi plus balanced_stable_mask for pixel-level analyses"
        ),
        "balanced_core_roi": balanced,
        "strict_core_roi": strict,
        "files": {
            "masks_npz": "comparison_roi_masks.npz",
            "overlay_png": "comparison_roi_overlay.png",
            "noise_score_png": "comparison_roi_noise_score.png",
            "stable_mask_png": "comparison_roi_stable_mask.png",
        },
    }
    (noise_dir / "comparison_roi_recommendation.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )
    _write_markdown(noise_dir, recommendation)
    return recommendation


def _write_markdown(noise_dir: Path, recommendation: dict) -> None:
    balanced = recommendation["balanced_core_roi"]
    strict = recommendation["strict_core_roi"]
    lines = [
        "# Recommended C-scan Comparison ROI\n\n",
        f"Baseline used for mapping: `{recommendation['baseline_scan_id']}`.\n\n",
        "## Default: Balanced Core ROI\n\n",
        f"- Rows: `{balanced['row_start']}:{balanced['row_end']}` "
        f"({balanced['n_rows']} rows)\n",
        f"- Columns: `{balanced['col_start']}:{balanced['col_end']}` "
        f"({balanced['n_cols']} columns)\n",
        f"- Physical region: X `{balanced['x_mm_min']:.2f}-{balanced['x_mm_max']:.2f} mm`, "
        f"Y `{balanced['y_mm_min']:.2f}-{balanced['y_mm_max']:.2f} mm`\n",
        f"- Area kept: `{balanced['area_fraction'] * 100:.1f}%` of full C-scan\n",
        f"- Stable pixels within this ROI using all thresholds: "
        f"`{balanced['stable_pixel_fraction_within_roi'] * 100:.1f}%`\n",
        "- Thresholds: amplitude span <= `0.55 V`, ToF span <= `0.18 us`, "
        "energy span <= `32`.\n\n",
        "Noise span inside balanced ROI:\n\n",
        "| feature | median span | p95 span | p99 span | max span |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for feature in ("amplitude", "tof", "energy"):
        metric = balanced["metrics"][feature]
        lines.append(
            f"| {feature} | {metric['median_span']:.6g} | {metric['p95_span']:.6g} | "
            f"{metric['p99_span']:.6g} | {metric['max_span']:.6g} |\n"
        )
    lines.extend([
        "\n## Strict ROI\n\n",
        f"- Rows: `{strict['row_start']}:{strict['row_end']}`; "
        f"columns: `{strict['col_start']}:{strict['col_end']}`\n",
        f"- Physical region: X `{strict['x_mm_min']:.2f}-{strict['x_mm_max']:.2f} mm`, "
        f"Y `{strict['y_mm_min']:.2f}-{strict['y_mm_max']:.2f} mm`\n",
        f"- Area kept: `{strict['area_fraction'] * 100:.1f}%`; stable pixels within ROI: "
        f"`{strict['stable_pixel_fraction_within_roi'] * 100:.1f}%`\n",
        "- Use this stricter ROI for sensitive pixel-wise ToF or amplitude claims.\n\n",
        "## Files\n\n",
        "- `comparison_roi_overlay.png`\n",
        "- `comparison_roi_noise_score.png`\n",
        "- `comparison_roi_stable_mask.png`\n",
        "- `comparison_roi_masks.npz`\n",
        "- `comparison_roi_recommendation.json`\n",
    ])
    (noise_dir / "COMPARISON_ROI_RECOMMENDATION.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "noise_dir",
        nargs="?",
        default="reports/experiments/noise_floor_2026-05-28_13-44-01",
    )
    args = parser.parse_args()
    recommendation = build_recommendation(Path(args.noise_dir))
    print(json.dumps(recommendation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
