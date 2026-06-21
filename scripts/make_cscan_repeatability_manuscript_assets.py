"""Build manuscript-ready C-scan repeatability and ROI assets.

The figure summarizes the corrected-start 6-scan repeatability experiment:
noise floor, strict rectangular ROI selection, and scan-to-scan stability.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


FEATURES = {
    "amplitude": {"label": "Amplitude", "unit": "V", "cmap": "magma"},
    "tof": {"label": "ToF", "unit": "us", "cmap": "viridis"},
    "energy": {"label": "Energy", "unit": "a.u.", "cmap": "plasma"},
}

FEATURE_ORDER = ["amplitude", "tof", "energy"]
FEATURE_COLORS = {
    "amplitude": "#3b4cc0",
    "tof": "#1b9e77",
    "energy": "#d95f02",
}

REGION_COLORS = {
    "Full map": "#bdbdbd",
    "Battery body": "#74add1",
    "Strict ROI": "#d95f02",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique_scan_records(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    for row in summary_rows:
        idx = int(row["scan_index"])
        if idx in records:
            continue
        records[idx] = {
            "run": idx,
            "scan_id": row["scan_id"],
            "session_dir": Path(row["session_dir"]),
            "is_baseline": row["is_baseline"].lower() == "true",
            "temp_mean_c": float(row["temp_mean_c"]),
            "temp_min_c": float(row["temp_min_c"]),
            "temp_max_c": float(row["temp_max_c"]),
            "temp_n": int(row["temp_n"]),
        }
    return [records[idx] for idx in sorted(records)]


def load_scan_maps(records: list[dict[str, object]]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    stacks: dict[str, list[np.ndarray]] = {feature: [] for feature in FEATURE_ORDER}
    x_mm: np.ndarray | None = None
    y_mm: np.ndarray | None = None
    for record in records:
        session_dir = Path(record["session_dir"])
        scan_id = str(record["scan_id"])
        scan_path = session_dir / f"{scan_id}.npz"
        if not scan_path.exists():
            candidates = sorted(session_dir.glob("scan_*.npz"))
            if not candidates:
                raise FileNotFoundError(f"No scan npz found in {session_dir}")
            scan_path = candidates[0]
        with np.load(scan_path) as z:
            for feature in FEATURE_ORDER:
                stacks[feature].append(z[feature].astype(float))
            if x_mm is None:
                x_mm = z["x_mm"].astype(float)
                y_mm = z["y_mm"].astype(float)
    if x_mm is None or y_mm is None:
        raise RuntimeError("No scan maps loaded.")
    return {feature: np.stack(values, axis=0) for feature, values in stacks.items()}, x_mm, y_mm


def extent_from_axes(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[float, float, float, float]:
    return float(x_mm.min()), float(x_mm.max()), float(y_mm.min()), float(y_mm.max())


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def mask_bounds_mm(mask: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray) -> dict[str, float]:
    r0, r1, c0, c1 = bbox_from_mask(mask)
    return {
        "x_min": float(x_mm[c0]),
        "x_max": float(x_mm[c1 - 1]),
        "y_min": float(y_mm[r0]),
        "y_max": float(y_mm[r1 - 1]),
    }


def draw_bounds(ax: plt.Axes, bounds: dict[str, float], **kwargs) -> None:
    ax.add_patch(
        Rectangle(
            (bounds["x_min"], bounds["y_min"]),
            bounds["x_max"] - bounds["x_min"],
            bounds["y_max"] - bounds["y_min"],
            fill=False,
            **kwargs,
        )
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def robust_limits(values: np.ndarray, mask: np.ndarray | None = None, high: float = 99.0) -> tuple[float, float]:
    data = values[mask] if mask is not None else values.ravel()
    data = data[np.isfinite(data)]
    return float(np.nanpercentile(data, 1.0)), float(np.nanpercentile(data, high))


def save_figure_all_formats(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def write_markdown_table(path: Path, title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |\n")


def write_latex_table(path: Path, caption: str, label: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    colspec = "l" * len(columns)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\hline\n")
        f.write(" & ".join(columns) + " \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join(str(row.get(col, "")) for col in columns) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def export_table_family(
    out_dir: Path,
    name: str,
    title: str,
    rows: list[dict[str, object]],
    columns: list[str],
    caption: str,
    label: str,
) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(out_dir / f"{name}.csv", index=False)
    write_markdown_table(out_dir / f"{name}.md", title, rows, columns)
    write_latex_table(out_dir / f"{name}.tex", caption, label, rows, columns)


def compute_quantification(
    stacks: dict[str, np.ndarray],
    finalized_span_maps: dict[str, np.ndarray],
    region_masks: dict[str, np.ndarray],
    baseline_idx: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    sigma_maps = {feature: np.nanstd(stacks[feature], axis=0, ddof=1) for feature in FEATURE_ORDER}
    region_rows: list[dict[str, object]] = []
    roi_mean_rows: list[dict[str, object]] = []

    for feature in FEATURE_ORDER:
        strict_sigma_p95 = float(np.nanpercentile(sigma_maps[feature][region_masks["Strict ROI"]], 95))
        for region, mask in region_masks.items():
            sigma_vals = sigma_maps[feature][mask]
            span_vals = finalized_span_maps[feature][mask]
            region_rows.append(
                {
                    "Feature": FEATURES[feature]["label"],
                    "Region": region,
                    "Pixels": int(mask.sum()),
                    "Median sigma": f"{np.nanmedian(sigma_vals):.5g}",
                    "P95 sigma": f"{np.nanpercentile(sigma_vals, 95):.5g}",
                    "P99 sigma": f"{np.nanpercentile(sigma_vals, 99):.5g}",
                    "Median span": f"{np.nanmedian(span_vals):.5g}",
                    "P95 span": f"{np.nanpercentile(span_vals, 95):.5g}",
                    "P99 span": f"{np.nanpercentile(span_vals, 99):.5g}",
                    "Unit": FEATURES[feature]["unit"],
                }
            )

        roi_values = np.array([np.nanmean(scan[region_masks["Strict ROI"]]) for scan in stacks[feature]])
        roi_shift = roi_values - roi_values[baseline_idx]
        roi_mean_rows.append(
            {
                "Feature": FEATURES[feature]["label"],
                "ROI mean sigma": f"{np.nanstd(roi_values, ddof=1):.5g}",
                "ROI mean range": f"{np.nanmax(roi_values) - np.nanmin(roi_values):.5g}",
                "Max absolute shift from baseline": f"{np.nanmax(np.abs(roi_shift)):.5g}",
                "Max shift / strict ROI p95 pixel sigma": f"{np.nanmax(np.abs(roi_shift)) / strict_sigma_p95:.3f}",
                "Unit": FEATURES[feature]["unit"],
            }
        )

    return region_rows, roi_mean_rows, sigma_maps


def make_quantification_figure(
    out_dir: Path,
    stacks: dict[str, np.ndarray],
    finalized_span_maps: dict[str, np.ndarray],
    region_masks: dict[str, np.ndarray],
    sigma_maps: dict[str, np.ndarray],
    strictness_df: pd.DataFrame,
    baseline_idx: int,
) -> None:
    regions = ["Full map", "Battery body", "Strict ROI"]
    x = np.arange(len(FEATURE_ORDER))
    width = 0.22

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    for i, region in enumerate(regions):
        vals = []
        for feature in FEATURE_ORDER:
            full_val = np.nanpercentile(sigma_maps[feature][region_masks["Full map"]], 95)
            vals.append(np.nanpercentile(sigma_maps[feature][region_masks[region]], 95) / full_val)
        ax_a.bar(x + (i - 1) * width, vals, width=width, color=REGION_COLORS[region], label=region, edgecolor="black", linewidth=0.25)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([FEATURES[f]["label"] for f in FEATURE_ORDER])
    ax_a.set_ylabel("P95 sigma / full-map P95")
    ax_a.set_ylim(0, 1.08)
    ax_a.set_title("Local noise reduction", loc="left", fontweight="bold", pad=3)
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax_a.grid(axis="y", color="#d9d9d9", linewidth=0.45)
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    for i, region in enumerate(regions):
        vals = []
        for feature in FEATURE_ORDER:
            full_val = np.nanpercentile(finalized_span_maps[feature][region_masks["Full map"]], 95)
            vals.append(np.nanpercentile(finalized_span_maps[feature][region_masks[region]], 95) / full_val)
        ax_b.bar(x + (i - 1) * width, vals, width=width, color=REGION_COLORS[region], label=region, edgecolor="black", linewidth=0.25)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([FEATURES[f]["label"] for f in FEATURE_ORDER])
    ax_b.set_ylabel("P95 span / full-map P95")
    ax_b.set_ylim(0, 1.08)
    ax_b.set_title("Worst-envelope reduction", loc="left", fontweight="bold", pad=3)
    ax_b.grid(axis="y", color="#d9d9d9", linewidth=0.45)
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    score_col = next(col for col in strictness_df.columns if col.strip() == "Score <=")
    score_limits = strictness_df[score_col].astype(float).to_numpy()
    kept = strictness_df["Battery kept %"].astype(float).to_numpy()
    p95 = strictness_df["P95 score"].astype(float).to_numpy()
    ax_c.plot(score_limits, kept, marker="o", color="#225ea8", linewidth=1.3, label="area retained")
    ax_c.set_xlabel("Allowed normalized noise score")
    ax_c.set_ylabel("Battery-body retained (%)", color="#225ea8")
    ax_c.tick_params(axis="y", labelcolor="#225ea8")
    ax_c.invert_xaxis()
    ax_c.grid(color="#d9d9d9", linewidth=0.45)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(score_limits, p95, marker="s", color="#d95f02", linewidth=1.2, label="P95 score")
    ax_c2.set_ylabel("P95 score", color="#d95f02")
    ax_c2.tick_params(axis="y", labelcolor="#d95f02")
    ax_c.set_title("Strictness sweep", loc="left", fontweight="bold", pad=3)
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    runs = np.arange(1, stacks[FEATURE_ORDER[0]].shape[0] + 1)
    for feature in FEATURE_ORDER:
        vals = np.array([np.nanmean(scan[region_masks["Strict ROI"]]) for scan in stacks[feature]])
        vals = (vals - vals[baseline_idx]) / float(np.nanpercentile(sigma_maps[feature][region_masks["Strict ROI"]], 95))
        ax_d.plot(runs, vals, marker="o", linewidth=1.2, markersize=3, color=FEATURE_COLORS[feature], label=FEATURES[feature]["label"])
    ax_d.axhline(0, color="#595959", linewidth=0.75)
    ax_d.axvline(baseline_idx + 1, color="#d95f02", linestyle="--", linewidth=1.0)
    ax_d.set_xlabel("Repeat C-scan run")
    ax_d.set_ylabel("ROI mean shift / p95 sigma")
    ax_d.set_title("Mean-feature drift", loc="left", fontweight="bold", pad=3)
    ax_d.grid(axis="y", color="#d9d9d9", linewidth=0.45)
    ax_d.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.2, columnspacing=0.8)
    add_panel_label(ax_d, "d")

    save_figure_all_formats(fig, out_dir / "fig_cscan_repeatability_quantification")
    plt.close(fig)


def make_main_figure(
    out_dir: Path,
    records: list[dict[str, object]],
    stacks: dict[str, np.ndarray],
    finalized_span_maps: dict[str, np.ndarray],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    strict_mask: np.ndarray,
    strictness_df: pd.DataFrame,
) -> dict[str, object]:
    baseline_idx = [i for i, record in enumerate(records) if bool(record["is_baseline"])][0]
    baseline_scan_id = str(records[baseline_idx]["scan_id"])
    extent = extent_from_axes(x_mm, y_mm)
    strict_bounds = mask_bounds_mm(strict_mask, x_mm, y_mm)
    runs = np.array([int(record["run"]) for record in records])
    temps = np.array([float(record["temp_mean_c"]) for record in records])

    sigma_maps = {feature: np.nanstd(stacks[feature], axis=0, ddof=1) for feature in FEATURE_ORDER}

    fig = plt.figure(figsize=(7.2, 6.9))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.04, 1.0, 0.88], hspace=0.55, wspace=1.04)

    ax_a = fig.add_subplot(gs[0, 0:3])
    im = ax_a.imshow(stacks["amplitude"][baseline_idx], origin="lower", extent=extent, aspect="auto", cmap="magma")
    draw_bounds(ax_a, strict_bounds, edgecolor="white", linewidth=1.7)
    draw_bounds(ax_a, strict_bounds, edgecolor="black", linewidth=0.7, linestyle="--")
    ax_a.set_title("Selected C-scan analysis region", loc="left", pad=3, fontweight="bold")
    ax_a.set_xlabel("X position (mm)")
    ax_a.set_ylabel("Y position (mm)")
    cb = fig.colorbar(im, ax=ax_a, fraction=0.035, pad=0.015)
    cb.ax.set_title("V", fontsize=6, pad=2)
    ax_a.text(
        strict_bounds["x_min"] + 1.3,
        strict_bounds["y_max"] - 3.0,
        "strict ROI",
        color="black",
        fontsize=5.8,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
    )
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_b.fill_between(runs, [float(r["temp_min_c"]) for r in records], [float(r["temp_max_c"]) for r in records], color="#b9d7ea", alpha=0.75, linewidth=0)
    ax_b.plot(runs, temps, marker="o", color="#225ea8", linewidth=1.3, markersize=3.0)
    ax_b.axvline(int(records[baseline_idx]["run"]), color="#d95f02", linestyle="--", linewidth=1.0)
    ax_b.set_title("Stable repeat condition", loc="left", pad=3, fontweight="bold")
    ax_b.set_xlabel("Repeat C-scan run")
    ax_b.set_ylabel("Temperature (C)", labelpad=2)
    ax_b.set_xticks(runs)
    ax_b.grid(axis="y", color="#d9d9d9", linewidth=0.45)
    ax_b.text(
        0.03,
        0.07,
        f"fixed SOC/electrochemical state\n3.232 V; T drift {temps.max() - temps.min():.3f} C",
        transform=ax_b.transAxes,
        fontsize=5.7,
        linespacing=1.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#bdbdbd", "linewidth": 0.4, "alpha": 0.92},
    )
    add_panel_label(ax_b, "b")

    noise_rows: list[dict[str, object]] = []
    for col, feature in enumerate(FEATURE_ORDER):
        ax = fig.add_subplot(gs[1, 2 * col : 2 * col + 2])
        _, vmax = robust_limits(sigma_maps[feature], strict_mask, high=99.0)
        im = ax.imshow(sigma_maps[feature], origin="lower", extent=extent, aspect="auto", cmap=FEATURES[feature]["cmap"], vmin=0, vmax=vmax)
        draw_bounds(ax, strict_bounds, edgecolor="white", linewidth=1.0)
        ax.set_title(f"{FEATURES[feature]['label']} repeatability", loc="left", pad=3, fontweight="bold")
        ax.set_xlabel("X (mm)")
        if col == 0:
            ax.set_ylabel("Y (mm)")
            add_panel_label(ax, "c")
        else:
            ax.set_yticklabels([])
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.set_title(FEATURES[feature]["unit"], fontsize=6, pad=2)

        noise_rows.append(
            {
                "Feature": FEATURES[feature]["label"],
                "Full-map median span": f"{np.nanmedian(finalized_span_maps[feature]):.4g}",
                "Full-map p95 span": f"{np.nanpercentile(finalized_span_maps[feature], 95):.4g}",
                "Strict ROI median sigma": f"{np.nanmedian(sigma_maps[feature][strict_mask]):.4g}",
                "Strict ROI p95 sigma": f"{np.nanpercentile(sigma_maps[feature][strict_mask], 95):.4g}",
                "Unit": FEATURES[feature]["unit"],
            }
        )

    ax_d = fig.add_subplot(gs[2, 0:3])
    levels_to_show = ["loose", "strict", "ultra_strict"]
    display_df = strictness_df[strictness_df["Level"].isin(levels_to_show)].copy()
    label_order = {level: i for i, level in enumerate(levels_to_show)}
    display_df["order"] = display_df["Level"].map(label_order)
    display_df = display_df.sort_values("order")
    ax_d.plot(strictness_df["Battery kept %"], strictness_df["P95 score"], color="#8c8c8c", linewidth=1.1, marker="o", markersize=2.8)
    for _, row in display_df.iterrows():
        color = "#d95f02" if row["Level"] == "strict" else "#4d4d4d"
        ax_d.scatter(row["Battery kept %"], row["P95 score"], s=26, color=color, zorder=3)
        ax_d.text(row["Battery kept %"] + 0.7, row["P95 score"] + 0.02, str(row["Level"]).replace("_", " "), fontsize=6.2, color=color)
    ax_d.set_title("ROI size-noise trade-off", loc="left", pad=3, fontweight="bold")
    ax_d.set_xlabel("Battery-body area retained (%)")
    ax_d.set_ylabel("P95 score", labelpad=2)
    ax_d.invert_xaxis()
    ax_d.grid(color="#d9d9d9", linewidth=0.45)
    add_panel_label(ax_d, "d")

    ax_e = fig.add_subplot(gs[2, 3:6])
    p95_sigma = {
        feature: float(np.nanpercentile(sigma_maps[feature][strict_mask], 95))
        for feature in FEATURE_ORDER
    }
    for feature in FEATURE_ORDER:
        values = np.array([np.nanmean(scan[strict_mask]) for scan in stacks[feature]])
        delta = (values - values[baseline_idx]) / p95_sigma[feature]
        ax_e.plot(runs, delta, marker="o", linewidth=1.25, markersize=3.0, color=FEATURE_COLORS[feature], label=FEATURES[feature]["label"])
    ax_e.axhline(0, color="#595959", linewidth=0.8)
    ax_e.axvline(int(records[baseline_idx]["run"]), color="#d95f02", linestyle="--", linewidth=1.0)
    ax_e.set_title("ROI-mean stability", loc="left", pad=3, fontweight="bold")
    ax_e.set_xlabel("Repeat C-scan run")
    ax_e.set_ylabel("Shift / p95 sigma", labelpad=2)
    ax_e.set_xticks(runs)
    ax_e.grid(axis="y", color="#d9d9d9", linewidth=0.45)
    ax_e.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.2, columnspacing=0.8)
    add_panel_label(ax_e, "e")

    stem = out_dir / "fig_cscan_repeatability_roi_definition"
    save_figure_all_formats(fig, stem)
    plt.close(fig)

    return {
        "baseline_scan_id": baseline_scan_id,
        "baseline_run": int(records[baseline_idx]["run"]),
        "strict_bounds": strict_bounds,
        "temperature_drift_c": float(temps.max() - temps.min()),
        "noise_rows": noise_rows,
    }


def export_tables(
    out_dir: Path,
    records: list[dict[str, object]],
    strictness_df: pd.DataFrame,
    noise_rows: list[dict[str, object]],
    region_quant_rows: list[dict[str, object]],
    roi_mean_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    temps = np.array([float(record["temp_mean_c"]) for record in records])
    scan_rows = [
        {"Item": "Repeated C-scans", "Value": str(len(records))},
        {"Item": "Completed C-scan rows", "Value": "144/144 for each scan"},
        {"Item": "SOC/electrochemical condition", "Value": "Fixed during repeat scans; no cycling during batch"},
        {"Item": "Voltage condition", "Value": "3.232 V at start of repeat batch"},
        {"Item": "Absolute SOC assignment", "Value": "Not independently assigned for this noise-floor batch"},
        {"Item": "Scan field", "Value": "80 mm x 72 mm"},
        {"Item": "Pitch", "Value": "0.5 mm"},
        {"Item": "Scan speed", "Value": "25 mm/s"},
        {"Item": "Acceleration", "Value": "800 mm/s2"},
        {"Item": "Baseline scan", "Value": f"run {summary['baseline_run']}, {summary['baseline_scan_id']}"},
        {"Item": "Mean temperature range", "Value": f"{temps.min():.3f}-{temps.max():.3f} C"},
        {"Item": "Mean temperature drift", "Value": f"{summary['temperature_drift_c']:.3f} C"},
    ]
    export_table_family(
        out_dir,
        "table_scan_conditions_compact",
        "Compact scan conditions",
        scan_rows,
        ["Item", "Value"],
        "Compact acquisition conditions for the corrected-start repeatability experiment, including the fixed SOC/electrochemical condition.",
        "tab:cscan-repeatability-conditions",
    )

    export_table_family(
        out_dir,
        "table_noise_floor_compact",
        "Compact C-scan noise floor",
        noise_rows,
        ["Feature", "Full-map median span", "Full-map p95 span", "Strict ROI median sigma", "Strict ROI p95 sigma", "Unit"],
        "Pixel-level repeatability noise floor for the corrected-start C-scan repeats.",
        "tab:cscan-noise-floor",
    )

    export_table_family(
        out_dir,
        "table_region_noise_quantification",
        "Region-level C-scan repeatability quantification",
        region_quant_rows,
        ["Feature", "Region", "Pixels", "Median sigma", "P95 sigma", "P99 sigma", "Median span", "P95 span", "P99 span", "Unit"],
        "Pixel-level repeatability distributions across the full map, battery body candidate region and selected strict ROI.",
        "tab:cscan-region-noise-quantification",
    )

    export_table_family(
        out_dir,
        "table_roi_mean_repeatability_quantification",
        "Strict-ROI mean repeatability quantification",
        roi_mean_rows,
        ["Feature", "ROI mean sigma", "ROI mean range", "Max absolute shift from baseline", "Max shift / strict ROI p95 pixel sigma", "Unit"],
        "Scan-to-scan repeatability of strict-ROI mean C-scan features.",
        "tab:cscan-roi-mean-repeatability",
    )

    roi_levels = ["loose", "strict", "ultra_strict"]
    roi_df = strictness_df[strictness_df["Level"].isin(roi_levels)].copy()
    roi_rows = []
    for _, row in roi_df.iterrows():
        roi_rows.append(
            {
                "ROI level": str(row["Level"]).replace("_", " "),
                "X mm": row["X mm"],
                "Y mm": row["Y mm"],
                "Battery kept %": f"{float(row['Battery kept %']):.2f}",
                "Amplitude p95 sigma": f"{float(row['Amplitude p95 sigma']):.4g}",
                "ToF p95 sigma": f"{float(row['ToF p95 sigma']):.4g}",
                "Energy p95 sigma": f"{float(row['Energy p95 sigma']):.4g}",
            }
        )
    export_table_family(
        out_dir,
        "table_roi_strictness_compact",
        "Compact ROI strictness comparison",
        roi_rows,
        ["ROI level", "X mm", "Y mm", "Battery kept %", "Amplitude p95 sigma", "ToF p95 sigma", "Energy p95 sigma"],
        "Representative rectangular ROI choices and their p95 repeatability noise.",
        "tab:cscan-roi-strictness",
    )


def write_legend_and_manifest(out_dir: Path, summary: dict[str, object]) -> None:
    bounds = summary["strict_bounds"]
    legend = f"""# Figure Legend: C-scan repeatability and ROI definition

**Conclusion.** Six corrected-start repeat C-scans quantify the measurement noise floor and justify a fixed rectangular ROI for subsequent SOC/temperature analysis.

**Figure panels.** (a) Baseline amplitude C-scan with the selected strict rectangular ROI overlaid. (b) Repeat-condition stability: the cell was kept at a fixed SOC/electrochemical state during the repeat batch, with voltage recorded as 3.232 V at the start and temperature summarized for each scan. Absolute SOC was not independently assigned for this noise-floor batch. (c) Pixel-wise repeatability sigma maps for amplitude, ToF and energy, with the same ROI overlaid. (d) Trade-off between retained battery-body area and p95 normalized noise score across rectangular ROI strictness levels. (e) ROI-mean feature shifts relative to the automatically selected baseline scan, normalized by the strict-ROI p95 pixel sigma for each feature.

**Companion quantification figure.** `fig_cscan_repeatability_quantification` provides the numerical comparison behind this figure: p95 pixel sigma, p95 max-min span, strictness sweep curves and normalized strict-ROI mean drift.

**Selected ROI.** X = {bounds['x_min']:.2f}-{bounds['x_max']:.2f} mm and Y = {bounds['y_min']:.2f}-{bounds['y_max']:.2f} mm. The strict rectangle preserves 95.45% of the battery-body candidate region while reducing edge-dominated noise.

**Export notes.** The SVG and PDF files keep text editable. The TIFF export is 600 dpi for manuscript submission.
"""
    (out_dir / "FIGURE_LEGEND_CSCAN_REPEATABILITY_ROI.md").write_text(legend, encoding="utf-8")

    manifest = {
        "figure_contract": {
            "core_conclusion": "Six corrected-start repeat C-scans quantify the C-scan noise floor and justify a fixed rectangular ROI for subsequent SOC/temperature analysis.",
            "archetype": "image plate + quant",
            "backend": "Python/matplotlib",
            "exports": ["png", "svg", "pdf", "tiff"],
        },
        "selected_roi": bounds,
        "baseline_scan_id": summary["baseline_scan_id"],
        "baseline_run": summary["baseline_run"],
        "temperature_drift_c": summary["temperature_drift_c"],
        "files": [
            "fig_cscan_repeatability_roi_definition.png",
            "fig_cscan_repeatability_roi_definition.svg",
            "fig_cscan_repeatability_roi_definition.pdf",
            "fig_cscan_repeatability_roi_definition.tiff",
            "fig_cscan_repeatability_quantification.png",
            "fig_cscan_repeatability_quantification.svg",
            "fig_cscan_repeatability_quantification.pdf",
            "fig_cscan_repeatability_quantification.tiff",
            "table_scan_conditions_compact.csv",
            "table_scan_conditions_compact.md",
            "table_scan_conditions_compact.tex",
            "table_noise_floor_compact.csv",
            "table_noise_floor_compact.md",
            "table_noise_floor_compact.tex",
            "table_region_noise_quantification.csv",
            "table_region_noise_quantification.md",
            "table_region_noise_quantification.tex",
            "table_roi_mean_repeatability_quantification.csv",
            "table_roi_mean_repeatability_quantification.md",
            "table_roi_mean_repeatability_quantification.tex",
            "table_roi_strictness_compact.csv",
            "table_roi_strictness_compact.md",
            "table_roi_strictness_compact.tex",
            "FIGURE_LEGEND_CSCAN_REPEATABILITY_ROI.md",
        ],
    }
    (out_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        default="reports/report c-scan baseline repeat noise level and roi",
        help="Final corrected-start C-scan repeatability report directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to REPORT_DIR/manuscript_assets.",
    )
    args = parser.parse_args()

    configure_matplotlib()

    report_dir = Path(args.report_dir)
    out_dir = Path(args.out_dir) if args.out_dir else report_dir / "manuscript_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = unique_scan_records(read_summary_rows(report_dir / "noise_floor_summary.csv"))
    stacks, x_mm, y_mm = load_scan_maps(records)
    baseline_idx = [i for i, record in enumerate(records) if bool(record["is_baseline"])][0]

    with np.load(report_dir / "noise_floor_diffs.npz") as z:
        finalized_span_maps = {
            feature: z[f"{feature}_noise_floor_span"].astype(float)
            for feature in FEATURE_ORDER
        }

    mask_path = report_dir / "rectangular_roi_strictness_sweep" / "rectangular_roi_strictness_masks.npz"
    with np.load(mask_path) as z:
        strict_mask = z["mask_strict"].astype(bool)
        battery_body_mask = z["mask_loose"].astype(bool)
    full_map_mask = np.ones_like(strict_mask, dtype=bool)
    region_masks = {
        "Full map": full_map_mask,
        "Battery body": battery_body_mask,
        "Strict ROI": strict_mask,
    }

    strictness_df = pd.read_csv(report_dir / "rectangular_roi_strictness_sweep" / "rectangular_roi_strictness_summary.csv")
    region_quant_rows, roi_mean_rows, sigma_maps = compute_quantification(stacks, finalized_span_maps, region_masks, baseline_idx)

    summary = make_main_figure(out_dir, records, stacks, finalized_span_maps, x_mm, y_mm, strict_mask, strictness_df)
    make_quantification_figure(out_dir, stacks, finalized_span_maps, region_masks, sigma_maps, strictness_df, baseline_idx)
    export_tables(out_dir, records, strictness_df, summary["noise_rows"], region_quant_rows, roi_mean_rows, summary)
    write_legend_and_manifest(out_dir, summary)

    print(f"Wrote manuscript assets to {out_dir}")


if __name__ == "__main__":
    main()
