"""Create report-ready C-scan noise-floor and ROI figures/tables.

This script uses the 10-scan repeatability batch outputs plus the saved raw
C-scan maps to build polished assets for manuscript/report writing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


FEATURES = {
    "amplitude": {"label": "Amplitude", "unit": "V", "cmap": "magma"},
    "tof": {"label": "ToF", "unit": "us", "cmap": "viridis"},
    "energy": {"label": "Energy", "unit": "a.u.", "cmap": "plasma"},
}

BATTERY_BODY_BOUNDS_MM = {
    "x_min": 10.0,
    "x_max": 63.0,
    "y_min": 13.0,
    "y_max": 55.0,
}


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(path: Path, title: str, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |\n")


def write_latex_table(path: Path, caption: str, label: str, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    colspec = "l" * len(fieldnames)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\hline\n")
        f.write(" & ".join(fieldnames) + " \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join(str(row.get(k, "")) for k in fieldnames) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def unique_scan_records(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    seen: dict[int, dict[str, object]] = {}
    for row in summary_rows:
        idx = int(row["scan_index"])
        if idx in seen:
            continue
        seen[idx] = {
            "run": idx,
            "scan_id": row["scan_id"],
            "session_dir": Path(row["session_dir"]),
            "is_baseline": row["is_baseline"].lower() == "true",
            "temp_mean_c": float(row["temp_mean_c"]),
            "temp_min_c": float(row["temp_min_c"]),
            "temp_max_c": float(row["temp_max_c"]),
            "temp_n": int(row["temp_n"]),
        }
    return [seen[k] for k in sorted(seen)]


def load_scan_maps(records: list[dict[str, object]]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    stacks: dict[str, list[np.ndarray]] = {feature: [] for feature in FEATURES}
    x_mm = None
    y_mm = None
    for record in records:
        session_dir = Path(record["session_dir"])
        scan_id = str(record["scan_id"])
        npz_path = session_dir / f"{scan_id}.npz"
        if not npz_path.exists():
            candidates = sorted(session_dir.glob("scan_*.npz"))
            if not candidates:
                raise FileNotFoundError(f"No scan npz found in {session_dir}")
            npz_path = candidates[0]
        with np.load(npz_path) as z:
            for feature in FEATURES:
                stacks[feature].append(z[feature].astype(float))
            if x_mm is None:
                x_mm = z["x_mm"].astype(float)
                y_mm = z["y_mm"].astype(float)
    if x_mm is None or y_mm is None:
        raise RuntimeError("No scan maps were loaded.")
    return {feature: np.stack(values, axis=0) for feature, values in stacks.items()}, x_mm, y_mm


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def add_roi_rect(ax: plt.Axes, mask: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, **kwargs) -> None:
    r0, r1, c0, c1 = bbox_from_mask(mask)
    x0 = float(x_mm[c0])
    x1 = float(x_mm[c1 - 1])
    y0 = float(y_mm[r0])
    y1 = float(y_mm[r1 - 1])
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, **kwargs))


def add_bounds_rect(
    ax: plt.Axes,
    bounds: dict[str, float],
    **kwargs,
) -> None:
    ax.add_patch(
        Rectangle(
            (bounds["x_min"], bounds["y_min"]),
            bounds["x_max"] - bounds["x_min"],
            bounds["y_max"] - bounds["y_min"],
            fill=False,
            **kwargs,
        )
    )


def image_extent(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[float, float, float, float]:
    return float(x_mm.min()), float(x_mm.max()), float(y_mm.min()), float(y_mm.max())


def add_battery_focused_masks(masks: dict[str, np.ndarray], x_mm: np.ndarray, y_mm: np.ndarray) -> None:
    bounds = BATTERY_BODY_BOUNDS_MM
    x_sel = (x_mm >= bounds["x_min"]) & (x_mm <= bounds["x_max"])
    y_sel = (y_mm >= bounds["y_min"]) & (y_mm <= bounds["y_max"])
    battery_body_roi = np.outer(y_sel, x_sel).astype(bool)
    battery_body_roi &= masks["balanced_core_roi"]
    masks["battery_body_roi"] = battery_body_roi
    masks["battery_stable_mask"] = battery_body_roi & masks["balanced_stable_mask"]


def save_temperature_figure(out_dir: Path, records: list[dict[str, object]]) -> None:
    runs = np.array([int(r["run"]) for r in records])
    means = np.array([float(r["temp_mean_c"]) for r in records])
    mins = np.array([float(r["temp_min_c"]) for r in records])
    maxs = np.array([float(r["temp_max_c"]) for r in records])
    baseline_run = [int(r["run"]) for r in records if bool(r["is_baseline"])][0]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    ax.fill_between(runs, mins, maxs, color="#9ecae1", alpha=0.45, label="min-max per scan")
    ax.plot(runs, means, marker="o", color="#08519c", linewidth=2.0, label="mean per scan")
    ax.axvline(baseline_run, color="#d94801", linestyle="--", linewidth=1.4, label=f"baseline run {baseline_run}")
    ax.set_xlabel("Repeat C-scan run")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Temperature stability during 10-scan noise-floor batch")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(runs)
    ax.text(
        0.02,
        0.04,
        "Voltage condition: 3.232 V",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_temperature_stability.png")
    plt.close(fig)


def save_roi_definition_figure(
    out_dir: Path,
    records: list[dict[str, object]],
    stacks: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> None:
    baseline_idx = [i for i, r in enumerate(records) if bool(r["is_baseline"])][0]
    baseline_amp = stacks["amplitude"][baseline_idx]
    extent = image_extent(x_mm, y_mm)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), dpi=300, constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(baseline_amp, origin="lower", extent=extent, aspect="auto", cmap="turbo")
    fig.colorbar(im, ax=ax, label="Amplitude (V)", shrink=0.88)
    add_roi_rect(ax, masks["balanced_core_roi"], x_mm, y_mm, edgecolor="white", linewidth=2.0)
    add_roi_rect(ax, masks["strict_core_roi"], x_mm, y_mm, edgecolor="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("Baseline amplitude map")

    ax = axes[1]
    stable_map = masks["balanced_stable_mask"].astype(float)
    im = ax.imshow(stable_map, origin="lower", extent=extent, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Stable pixel mask", shrink=0.88, ticks=[0, 1])
    add_roi_rect(ax, masks["balanced_core_roi"], x_mm, y_mm, edgecolor="black", linewidth=2.0)
    add_roi_rect(ax, masks["strict_core_roi"], x_mm, y_mm, edgecolor="#555555", linewidth=1.5, linestyle="--")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("Stable pixels used for pixel-level analysis")

    fig.suptitle("Recommended C-scan comparison ROI", fontsize=12)
    axes[0].text(
        0.02,
        0.98,
        "white: balanced ROI\nblack dashed: strict ROI",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )
    fig.savefig(out_dir / "fig02_roi_definition_on_baseline.png")
    plt.close(fig)


def save_battery_focused_roi_figure(
    out_dir: Path,
    records: list[dict[str, object]],
    stacks: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> None:
    baseline_idx = [i for i, r in enumerate(records) if bool(r["is_baseline"])][0]
    baseline_amp = stacks["amplitude"][baseline_idx]
    extent = image_extent(x_mm, y_mm)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), dpi=300, constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(baseline_amp, origin="lower", extent=extent, aspect="auto", cmap="turbo")
    fig.colorbar(im, ax=ax, label="Amplitude (V)", shrink=0.88)
    add_roi_rect(ax, masks["balanced_core_roi"], x_mm, y_mm, edgecolor="white", linewidth=1.5)
    add_bounds_rect(ax, BATTERY_BODY_BOUNDS_MM, edgecolor="#00ffff", linewidth=2.2)
    ax.set_title("Battery-focused ROI on baseline map")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.text(
        0.02,
        0.98,
        "cyan: battery body ROI\nwhite: balanced ROI",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )

    ax = axes[1]
    battery_mask = masks["battery_stable_mask"].astype(float)
    im = ax.imshow(battery_mask, origin="lower", extent=extent, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Battery stable mask", shrink=0.88, ticks=[0, 1])
    add_bounds_rect(ax, BATTERY_BODY_BOUNDS_MM, edgecolor="black", linewidth=2.0)
    ax.set_title("Stable pixels inside battery body")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")

    fig.suptitle("Battery-focused ROI for later comparison", fontsize=12)
    fig.savefig(out_dir / "fig07_battery_focused_roi.png")
    plt.close(fig)


def save_span_maps(out_dir: Path, diffs: np.lib.npyio.NpzFile, masks: dict[str, np.ndarray], x_mm: np.ndarray, y_mm: np.ndarray) -> None:
    extent = image_extent(x_mm, y_mm)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), dpi=300, constrained_layout=True)
    for ax, feature in zip(axes, FEATURES):
        span = diffs[f"{feature}_noise_floor_span"]
        vmax = float(np.nanpercentile(span, 99.0))
        im = ax.imshow(span, origin="lower", extent=extent, aspect="auto", cmap=FEATURES[feature]["cmap"], vmin=0, vmax=vmax)
        add_roi_rect(ax, masks["balanced_core_roi"], x_mm, y_mm, edgecolor="white", linewidth=1.4)
        add_roi_rect(ax, masks["strict_core_roi"], x_mm, y_mm, edgecolor="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"{FEATURES[feature]['label']} span")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.colorbar(im, ax=ax, label=f"Span ({FEATURES[feature]['unit']})", shrink=0.88)
    fig.suptitle("Pixel-wise noise-floor span across 10 repeated scans", y=1.03, fontsize=12)
    fig.savefig(out_dir / "fig03_noise_floor_span_maps.png", bbox_inches="tight")
    plt.close(fig)


def save_sigma_maps(
    out_dir: Path,
    stacks: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> dict[str, np.ndarray]:
    extent = image_extent(x_mm, y_mm)
    sigmas = {feature: np.nanstd(stack, axis=0, ddof=1) for feature, stack in stacks.items()}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), dpi=300, constrained_layout=True)
    for ax, feature in zip(axes, FEATURES):
        sigma = sigmas[feature]
        vmax = float(np.nanpercentile(sigma, 99.0))
        im = ax.imshow(sigma, origin="lower", extent=extent, aspect="auto", cmap=FEATURES[feature]["cmap"], vmin=0, vmax=vmax)
        add_roi_rect(ax, masks["balanced_core_roi"], x_mm, y_mm, edgecolor="white", linewidth=1.4)
        add_roi_rect(ax, masks["strict_core_roi"], x_mm, y_mm, edgecolor="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"{FEATURES[feature]['label']} sigma")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.colorbar(im, ax=ax, label=f"Sigma ({FEATURES[feature]['unit']})", shrink=0.88)
    fig.suptitle("Pixel-wise repeatability sigma across 10 repeated scans", y=1.03, fontsize=12)
    fig.savefig(out_dir / "fig04_repeatability_sigma_maps.png", bbox_inches="tight")
    plt.close(fig)
    return sigmas


def save_distribution_figure(out_dir: Path, sigmas: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), dpi=300, constrained_layout=True)
    groups = [
        ("Balanced ROI", masks["balanced_core_roi"]),
        ("Stable mask", masks["balanced_stable_mask"]),
        ("Strict ROI", masks["strict_core_roi"]),
    ]
    for ax, feature in zip(axes, FEATURES):
        data = [sigmas[feature][mask].ravel() for _, mask in groups]
        ax.boxplot(data, labels=[g[0] for g in groups], showfliers=False, patch_artist=True)
        ax.set_title(f"{FEATURES[feature]['label']} local sigma")
        ax.set_ylabel(f"Sigma ({FEATURES[feature]['unit']})")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=22)
    fig.suptitle("Distribution of pixel-level repeatability noise", y=1.05, fontsize=12)
    fig.savefig(out_dir / "fig05_pixel_noise_sigma_distributions.png", bbox_inches="tight")
    plt.close(fig)


def save_roi_mean_repeatability(
    out_dir: Path,
    records: list[dict[str, object]],
    stacks: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    runs = np.array([int(r["run"]) for r in records])
    baseline_run = [int(r["run"]) for r in records if bool(r["is_baseline"])][0]
    result: dict[str, dict[str, np.ndarray]] = {}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), dpi=300, sharex=True, constrained_layout=True)
    for ax, feature in zip(axes, FEATURES):
        roi_means = np.array([float(np.nanmean(scan[masks["balanced_core_roi"]])) for scan in stacks[feature]])
        stable_means = np.array([float(np.nanmean(scan[masks["balanced_stable_mask"]])) for scan in stacks[feature]])
        roi_dev = roi_means - float(np.nanmean(roi_means))
        stable_dev = stable_means - float(np.nanmean(stable_means))
        result[feature] = {"balanced_core_roi": roi_means, "balanced_stable_mask": stable_means}
        ax.plot(runs, roi_dev, marker="o", label="balanced ROI", color="#2b8cbe")
        ax.plot(runs, stable_dev, marker="s", label="stable mask", color="#31a354")
        ax.axhline(0, color="#666666", linewidth=0.9)
        ax.axvline(baseline_run, color="#d94801", linestyle="--", linewidth=1.1)
        ax.set_ylabel(f"Delta mean ({FEATURES[feature]['unit']})")
        ax.set_title(f"{FEATURES[feature]['label']} ROI-mean repeatability")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Repeat C-scan run")
    axes[-1].set_xticks(runs)
    fig.savefig(out_dir / "fig06_roi_mean_repeatability.png", bbox_inches="tight")
    plt.close(fig)
    return result


def rounded(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def build_tables(
    out_dir: Path,
    records: list[dict[str, object]],
    masks: dict[str, np.ndarray],
    sigmas: dict[str, np.ndarray],
    diffs: np.lib.npyio.NpzFile,
    stacks: dict[str, np.ndarray],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> None:
    run_rows = []
    for record in records:
        run_rows.append(
            {
                "Run": int(record["run"]),
                "Scan ID": record["scan_id"],
                "Baseline": "yes" if bool(record["is_baseline"]) else "no",
                "Temp mean (C)": f"{float(record['temp_mean_c']):.3f}",
                "Temp range (C)": f"{float(record['temp_min_c']):.3f}-{float(record['temp_max_c']):.3f}",
                "Temp n": int(record["temp_n"]),
            }
        )
    run_fields = ["Run", "Scan ID", "Baseline", "Temp mean (C)", "Temp range (C)", "Temp n"]
    write_csv(out_dir / "table01_scan_conditions.csv", run_rows, run_fields)
    write_markdown_table(out_dir / "table01_scan_conditions.md", "Scan Conditions", run_rows, run_fields)
    write_latex_table(out_dir / "table01_scan_conditions.tex", "Repeat C-scan conditions.", "tab:cscan_noise_conditions", run_rows, run_fields)

    roi_rows = []
    for name, mask_key in [
        ("Balanced core ROI", "balanced_core_roi"),
        ("Balanced stable mask", "balanced_stable_mask"),
        ("Strict core ROI", "strict_core_roi"),
        ("Strict stable mask", "strict_stable_mask"),
        ("Battery body ROI", "battery_body_roi"),
        ("Battery stable mask", "battery_stable_mask"),
    ]:
        mask = masks[mask_key]
        r0, r1, c0, c1 = bbox_from_mask(mask)
        roi_rows.append(
            {
                "Region": name,
                "Rows": f"{r0}:{r1}",
                "Cols": f"{c0}:{c1}",
                "X (mm)": f"{x_mm[c0]:.2f}-{x_mm[c1 - 1]:.2f}",
                "Y (mm)": f"{y_mm[r0]:.2f}-{y_mm[r1 - 1]:.2f}",
                "Pixels": int(mask.sum()),
                "Area fraction": f"{mask.mean() * 100:.1f}%",
            }
        )
    roi_fields = ["Region", "Rows", "Cols", "X (mm)", "Y (mm)", "Pixels", "Area fraction"]
    write_csv(out_dir / "table02_roi_definition.csv", roi_rows, roi_fields)
    write_markdown_table(out_dir / "table02_roi_definition.md", "ROI Definition", roi_rows, roi_fields)
    write_latex_table(out_dir / "table02_roi_definition.tex", "C-scan ROI definitions.", "tab:cscan_roi_definition", roi_rows, roi_fields)

    noise_rows = []
    mask_list = [
        ("Balanced ROI", "balanced_core_roi"),
        ("Stable mask", "balanced_stable_mask"),
        ("Strict ROI", "strict_core_roi"),
        ("Strict stable", "strict_stable_mask"),
        ("Battery body", "battery_body_roi"),
        ("Battery stable", "battery_stable_mask"),
    ]
    for feature in FEATURES:
        span = diffs[f"{feature}_noise_floor_span"]
        for label, mask_key in mask_list:
            mask = masks[mask_key]
            scan_means = np.array([float(np.nanmean(scan[mask])) for scan in stacks[feature]])
            noise_rows.append(
                {
                    "Feature": FEATURES[feature]["label"],
                    "Region": label,
                    "Median sigma": rounded(float(np.nanmedian(sigmas[feature][mask]))),
                    "P95 sigma": rounded(float(np.nanpercentile(sigmas[feature][mask], 95))),
                    "Median span": rounded(float(np.nanmedian(span[mask]))),
                    "P95 span": rounded(float(np.nanpercentile(span[mask], 95))),
                    "Scan-mean sigma": rounded(float(np.nanstd(scan_means, ddof=1))),
                    "Scan-mean range": rounded(float(np.nanmax(scan_means) - np.nanmin(scan_means))),
                    "Unit": FEATURES[feature]["unit"],
                }
            )
    noise_fields = [
        "Feature",
        "Region",
        "Median sigma",
        "P95 sigma",
        "Median span",
        "P95 span",
        "Scan-mean sigma",
        "Scan-mean range",
        "Unit",
    ]
    write_csv(out_dir / "table03_noise_floor_by_roi.csv", noise_rows, noise_fields)
    write_markdown_table(out_dir / "table03_noise_floor_by_roi.md", "Noise Floor by ROI", noise_rows, noise_fields)
    write_latex_table(out_dir / "table03_noise_floor_by_roi.tex", "C-scan noise floor by ROI.", "tab:cscan_noise_floor_roi", noise_rows, noise_fields)

    model_rows = []
    for feature in FEATURES:
        mask = masks["balanced_stable_mask"]
        scan_means = np.array([float(np.nanmean(scan[masks["balanced_core_roi"]])) for scan in stacks[feature]])
        model_rows.append(
            {
                "Feature": FEATURES[feature]["label"],
                "Local noise median": rounded(float(np.nanmedian(sigmas[feature][mask]))),
                "Local noise p95": rounded(float(np.nanpercentile(sigmas[feature][mask], 95))),
                "ROI-mean sigma": rounded(float(np.nanstd(scan_means, ddof=1))),
                "ROI-mean range": rounded(float(np.nanmax(scan_means) - np.nanmin(scan_means))),
                "Unit": FEATURES[feature]["unit"],
            }
        )
    model_fields = ["Feature", "Local noise median", "Local noise p95", "ROI-mean sigma", "ROI-mean range", "Unit"]
    write_csv(out_dir / "table04_model_noise_terms.csv", model_rows, model_fields)
    write_markdown_table(out_dir / "table04_model_noise_terms.md", "Recommended Preliminary Model Noise Terms", model_rows, model_fields)
    write_latex_table(out_dir / "table04_model_noise_terms.tex", "Preliminary noise terms for model fitting.", "tab:cscan_model_noise_terms", model_rows, model_fields)

    battery_rows = []
    body = masks["battery_body_roi"]
    stable = masks["battery_stable_mask"]
    for feature in FEATURES:
        body_scan_means = np.array([float(np.nanmean(scan[body])) for scan in stacks[feature]])
        stable_scan_means = np.array([float(np.nanmean(scan[stable])) for scan in stacks[feature]])
        battery_rows.append(
            {
                "Feature": FEATURES[feature]["label"],
                "Body median sigma": rounded(float(np.nanmedian(sigmas[feature][body]))),
                "Body p95 sigma": rounded(float(np.nanpercentile(sigmas[feature][body], 95))),
                "Stable median sigma": rounded(float(np.nanmedian(sigmas[feature][stable]))),
                "Stable p95 sigma": rounded(float(np.nanpercentile(sigmas[feature][stable], 95))),
                "Body mean sigma": rounded(float(np.nanstd(body_scan_means, ddof=1))),
                "Stable mean sigma": rounded(float(np.nanstd(stable_scan_means, ddof=1))),
                "Unit": FEATURES[feature]["unit"],
            }
        )
    battery_fields = [
        "Feature",
        "Body median sigma",
        "Body p95 sigma",
        "Stable median sigma",
        "Stable p95 sigma",
        "Body mean sigma",
        "Stable mean sigma",
        "Unit",
    ]
    write_csv(out_dir / "table05_battery_focused_noise_terms.csv", battery_rows, battery_fields)
    write_markdown_table(out_dir / "table05_battery_focused_noise_terms.md", "Battery-Focused Noise Terms", battery_rows, battery_fields)
    write_latex_table(
        out_dir / "table05_battery_focused_noise_terms.tex",
        "Battery-focused C-scan noise terms.",
        "tab:cscan_battery_focused_noise",
        battery_rows,
        battery_fields,
    )


def write_asset_index(out_dir: Path) -> None:
    text = """# Report-Ready C-scan Noise/ROI Assets

Use these figures and tables for the report section that defines the C-scan
repeatability noise floor and the preliminary ROI for later model fitting.

## Recommended Figures

- `fig01_temperature_stability.png`: confirms temperature stability during the 10 repeated scans.
- `fig02_roi_definition_on_baseline.png`: shows the selected ROI and stable pixels on the baseline C-scan.
- `fig03_noise_floor_span_maps.png`: shows pixel-wise max-min noise-floor span for amplitude, ToF, and energy.
- `fig04_repeatability_sigma_maps.png`: shows pixel-wise standard deviation across the 10 scans.
- `fig05_pixel_noise_sigma_distributions.png`: compares local noise distributions in the ROI and stable mask.
- `fig06_roi_mean_repeatability.png`: shows scan-to-scan drift of ROI-averaged features.
- `fig07_battery_focused_roi.png`: restricts the analysis region to the battery body and its stable pixels.

## Recommended Tables

- `table01_scan_conditions.*`: scan IDs, baseline selection, and temperature conditions.
- `table02_roi_definition.*`: ROI coordinates and physical dimensions.
- `table03_noise_floor_by_roi.*`: detailed noise-floor statistics for each ROI.
- `table04_model_noise_terms.*`: compact values to use in preliminary model fitting.
- `table05_battery_focused_noise_terms.*`: compact values for the battery-focused ROI.

Suggested report logic:

1. Show temperature stability first.
2. Define the balanced ROI and stable mask.
3. Quantify noise using sigma and span.
4. Use the stable-mask median sigma as representative pixel-level noise.
5. Use the balanced ROI scan-mean sigma as the ROI-averaged repeatability limit.
6. For battery-only comparisons, use the battery body ROI and battery stable mask.
"""
    (out_dir / "REPORT_ASSETS_INDEX.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir",
        default="reports/report c-scan noise level and roi",
        help="Folder containing the copied noise-floor and ROI files.",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    out_dir = report_dir / "report_ready_figures_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_summary_rows(report_dir / "noise_floor_summary.csv")
    records = unique_scan_records(summary_rows)
    stacks, x_mm, y_mm = load_scan_maps(records)
    diffs = np.load(report_dir / "noise_floor_diffs.npz")
    masks_npz = np.load(report_dir / "comparison_roi_masks.npz")
    masks = {name: masks_npz[name].astype(bool) for name in masks_npz.files if name.endswith("_roi") or name.endswith("_mask")}
    add_battery_focused_masks(masks, x_mm, y_mm)

    save_temperature_figure(out_dir, records)
    save_roi_definition_figure(out_dir, records, stacks, masks, x_mm, y_mm)
    save_battery_focused_roi_figure(out_dir, records, stacks, masks, x_mm, y_mm)
    save_span_maps(out_dir, diffs, masks, x_mm, y_mm)
    sigmas = save_sigma_maps(out_dir, stacks, masks, x_mm, y_mm)
    save_distribution_figure(out_dir, sigmas, masks)
    save_roi_mean_repeatability(out_dir, records, stacks, masks)
    build_tables(out_dir, records, masks, sigmas, diffs, stacks, x_mm, y_mm)
    np.savez_compressed(
        out_dir / "battery_focused_roi_masks.npz",
        battery_body_roi=masks["battery_body_roi"],
        battery_stable_mask=masks["battery_stable_mask"],
        bounds_mm=np.array(
            [
                BATTERY_BODY_BOUNDS_MM["x_min"],
                BATTERY_BODY_BOUNDS_MM["x_max"],
                BATTERY_BODY_BOUNDS_MM["y_min"],
                BATTERY_BODY_BOUNDS_MM["y_max"],
            ],
            dtype=float,
        ),
    )
    write_asset_index(out_dir)

    manifest = {
        "source_report_dir": str(report_dir),
        "output_dir": str(out_dir),
        "n_scans": len(records),
        "baseline_scan_id": [r["scan_id"] for r in records if bool(r["is_baseline"])][0],
        "figures": sorted(p.name for p in out_dir.glob("fig*.png")),
        "tables": sorted(p.name for p in out_dir.glob("table*.*")),
    }
    (out_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
