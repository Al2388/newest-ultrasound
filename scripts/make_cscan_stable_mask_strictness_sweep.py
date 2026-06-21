"""Plot C-scan stable-mask strictness levels for choosing a final mask."""

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
    "amplitude": {"label": "Amplitude", "unit": "V", "base_span": 0.55},
    "tof": {"label": "ToF", "unit": "us", "base_span": 0.18},
    "energy": {"label": "Energy", "unit": "a.u.", "base_span": 32.0},
}

BATTERY_BODY_BOUNDS_MM = {
    "x_min": 10.0,
    "x_max": 63.0,
    "y_min": 13.0,
    "y_max": 55.0,
}

STRICTNESS_LEVELS = [
    ("loose", 2.00),
    ("relaxed", 1.50),
    ("current", 1.00),
    ("strict", 0.75),
    ("very_strict", 0.50),
    ("ultra_strict", 0.35),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scan_records(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    seen: dict[int, dict[str, object]] = {}
    for row in summary_rows:
        idx = int(row["scan_index"])
        if idx not in seen:
            seen[idx] = {
                "run": idx,
                "scan_id": row["scan_id"],
                "session_dir": Path(row["session_dir"]),
                "is_baseline": row["is_baseline"].lower() == "true",
            }
    return [seen[k] for k in sorted(seen)]


def load_scans(records: list[dict[str, object]]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    stacks: dict[str, list[np.ndarray]] = {feature: [] for feature in FEATURES}
    x_mm = None
    y_mm = None
    baseline_idx = 0
    for i, record in enumerate(records):
        if bool(record["is_baseline"]):
            baseline_idx = i
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
        raise RuntimeError("No scan maps loaded.")
    return {feature: np.stack(values, axis=0) for feature, values in stacks.items()}, x_mm, y_mm, baseline_idx


def extent(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[float, float, float, float]:
    return float(x_mm.min()), float(x_mm.max()), float(y_mm.min()), float(y_mm.max())


def body_mask(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    bounds = BATTERY_BODY_BOUNDS_MM
    x_sel = (x_mm >= bounds["x_min"]) & (x_mm <= bounds["x_max"])
    y_sel = (y_mm >= bounds["y_min"]) & (y_mm <= bounds["y_max"])
    return np.outer(y_sel, x_sel).astype(bool)


def add_bounds(ax: plt.Axes, bounds: dict[str, float], **kwargs) -> None:
    ax.add_patch(
        Rectangle(
            (bounds["x_min"], bounds["y_min"]),
            bounds["x_max"] - bounds["x_min"],
            bounds["y_max"] - bounds["y_min"],
            fill=False,
            **kwargs,
        )
    )


def build_noise_score(diffs: np.lib.npyio.NpzFile) -> np.ndarray:
    score_terms = []
    for feature, meta in FEATURES.items():
        span = diffs[f"{feature}_noise_floor_span"].astype(float)
        score_terms.append(span / float(meta["base_span"]))
    return np.nanmax(np.stack(score_terms, axis=0), axis=0)


def write_table(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stable Mask Strictness Summary\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[k]) for k in fieldnames) + " |\n")


def write_latex(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Stable mask strictness sweep for the battery-focused C-scan ROI.}\n")
        f.write("\\label{tab:cscan_mask_strictness_sweep}\n")
        f.write("\\begin{tabular}{llllllll}\n\\hline\n")
        f.write(" & ".join(fieldnames) + " \\\\\n\\hline\n")
        for row in rows:
            f.write(" & ".join(str(row[k]) for k in fieldnames) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def plot_mask_grid(
    out_dir: Path,
    masks: dict[str, np.ndarray],
    baseline_amp: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), dpi=300, constrained_layout=True)
    img_extent = extent(x_mm, y_mm)
    for ax, (name, _score_limit), row in zip(axes.ravel(), STRICTNESS_LEVELS, rows):
        ax.imshow(baseline_amp, origin="lower", extent=img_extent, aspect="auto", cmap="gray", vmin=0, vmax=6)
        mask = np.ma.masked_where(~masks[name], masks[name])
        ax.imshow(mask, origin="lower", extent=img_extent, aspect="auto", cmap="Greens", alpha=0.72, vmin=0, vmax=1)
        add_bounds(ax, BATTERY_BODY_BOUNDS_MM, edgecolor="#00ffff", linewidth=1.6)
        ax.set_title(f"{name}: keep {row['Battery kept %']}%")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.suptitle("Battery-focused stable mask strictness levels", fontsize=13)
    fig.savefig(out_dir / "fig_mask_strictness_levels_on_baseline.png", bbox_inches="tight")
    plt.close(fig)


def plot_score_map(
    out_dir: Path,
    score: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    vmax = float(np.nanpercentile(score, 98.0))
    im = ax.imshow(score, origin="lower", extent=extent(x_mm, y_mm), aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    add_bounds(ax, BATTERY_BODY_BOUNDS_MM, edgecolor="white", linewidth=2.0)
    ax.set_title("Combined noise score used for stable-mask strictness")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    fig.colorbar(im, ax=ax, label="max normalized span score")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mask_strictness_noise_score_map.png")
    plt.close(fig)


def plot_tradeoff(out_dir: Path, rows: list[dict[str, object]]) -> None:
    keep = np.array([float(r["Battery kept %"]) for r in rows])
    amp = np.array([float(r["Amp p95 sigma"]) for r in rows])
    tof = np.array([float(r["ToF p95 sigma"]) for r in rows])
    energy = np.array([float(r["Energy p95 sigma"]) for r in rows])
    labels = [str(r["Level"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), dpi=300, constrained_layout=True)
    series = [
        ("Amplitude", amp, "V", "#d95f0e"),
        ("ToF", tof, "us", "#2b8cbe"),
        ("Energy", energy, "a.u.", "#31a354"),
    ]
    for ax, (title, values, unit, color) in zip(axes, series):
        ax.plot(keep, values, marker="o", color=color, linewidth=2)
        for x, y, label in zip(keep, values, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Battery ROI kept (%)")
        ax.set_ylabel(f"P95 local sigma ({unit})")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.invert_xaxis()
    fig.suptitle("Strictness tradeoff: kept area vs local noise", fontsize=12)
    fig.savefig(out_dir / "fig_mask_strictness_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir",
        default="reports/report c-scan baseline repeat noise level and roi",
        help="Finalized C-scan baseline/repeat report folder.",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    out_dir = report_dir / "stable_mask_strictness_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_in = read_csv(report_dir / "noise_floor_summary.csv")
    records = scan_records(rows_in)
    stacks, x_mm, y_mm, baseline_idx = load_scans(records)
    diffs = np.load(report_dir / "noise_floor_diffs.npz")
    base_masks = np.load(report_dir / "comparison_roi_masks.npz")

    battery_body = body_mask(x_mm, y_mm) & base_masks["balanced_core_roi"].astype(bool)
    score = build_noise_score(diffs)
    sigmas = {feature: np.nanstd(stack, axis=0, ddof=1) for feature, stack in stacks.items()}
    spans = {feature: diffs[f"{feature}_noise_floor_span"].astype(float) for feature in FEATURES}

    masks: dict[str, np.ndarray] = {}
    table_rows: list[dict[str, object]] = []
    for name, score_limit in STRICTNESS_LEVELS:
        mask = battery_body & np.isfinite(score) & (score <= score_limit)
        masks[name] = mask
        scan_means = {
            feature: np.array([float(np.nanmean(scan[mask])) for scan in stacks[feature]])
            for feature in FEATURES
        }
        table_rows.append(
            {
                "Level": name,
                "Score <= ": fmt(score_limit, 3),
                "Pixels": int(mask.sum()),
                "Battery kept %": fmt(100.0 * float(mask.sum()) / float(battery_body.sum()), 4),
                "Amp med sigma": fmt(float(np.nanmedian(sigmas["amplitude"][mask]))),
                "Amp p95 sigma": fmt(float(np.nanpercentile(sigmas["amplitude"][mask], 95))),
                "ToF med sigma": fmt(float(np.nanmedian(sigmas["tof"][mask]))),
                "ToF p95 sigma": fmt(float(np.nanpercentile(sigmas["tof"][mask], 95))),
                "Energy med sigma": fmt(float(np.nanmedian(sigmas["energy"][mask]))),
                "Energy p95 sigma": fmt(float(np.nanpercentile(sigmas["energy"][mask], 95))),
                "Amp p95 span": fmt(float(np.nanpercentile(spans["amplitude"][mask], 95))),
                "ToF p95 span": fmt(float(np.nanpercentile(spans["tof"][mask], 95))),
                "Energy p95 span": fmt(float(np.nanpercentile(spans["energy"][mask], 95))),
                "Amp mean sigma": fmt(float(np.nanstd(scan_means["amplitude"], ddof=1))),
                "ToF mean sigma": fmt(float(np.nanstd(scan_means["tof"], ddof=1))),
                "Energy mean sigma": fmt(float(np.nanstd(scan_means["energy"], ddof=1))),
            }
        )

    table_fields = [
        "Level",
        "Score <= ",
        "Pixels",
        "Battery kept %",
        "Amp med sigma",
        "Amp p95 sigma",
        "ToF med sigma",
        "ToF p95 sigma",
        "Energy med sigma",
        "Energy p95 sigma",
        "Amp p95 span",
        "ToF p95 span",
        "Energy p95 span",
        "Amp mean sigma",
        "ToF mean sigma",
        "Energy mean sigma",
    ]
    write_table(out_dir / "stable_mask_strictness_summary.csv", table_rows, table_fields)
    write_markdown(out_dir / "stable_mask_strictness_summary.md", table_rows, table_fields)
    write_latex(out_dir / "stable_mask_strictness_summary.tex", table_rows, table_fields[:8])

    baseline_amp = stacks["amplitude"][baseline_idx]
    plot_mask_grid(out_dir, masks, baseline_amp, x_mm, y_mm, table_rows)
    plot_score_map(out_dir, score, x_mm, y_mm)
    plot_tradeoff(out_dir, table_rows)

    np.savez_compressed(
        out_dir / "stable_mask_strictness_masks.npz",
        battery_body_roi=battery_body,
        noise_score=score,
        bounds_mm=np.array(
            [
                BATTERY_BODY_BOUNDS_MM["x_min"],
                BATTERY_BODY_BOUNDS_MM["x_max"],
                BATTERY_BODY_BOUNDS_MM["y_min"],
                BATTERY_BODY_BOUNDS_MM["y_max"],
            ],
            dtype=float,
        ),
        **{f"mask_{name}": mask for name, mask in masks.items()},
    )

    index = """# Stable Mask Strictness Sweep

This folder compares several stable-mask strictness levels inside the
battery-focused ROI.

Noise score:

`score = max(amplitude_span/0.55 V, tof_span/0.18 us, energy_span/32)`

`current` is equivalent to the original stable-mask threshold. Lower score
limits are stricter and keep fewer pixels.

Files:

- `fig_mask_strictness_levels_on_baseline.png`
- `fig_mask_strictness_noise_score_map.png`
- `fig_mask_strictness_tradeoff.png`
- `stable_mask_strictness_summary.csv`
- `stable_mask_strictness_summary.md`
- `stable_mask_strictness_masks.npz`
"""
    (out_dir / "STRICTNESS_SWEEP_INDEX.md").write_text(index, encoding="utf-8")

    manifest = {
        "report_dir": str(report_dir),
        "out_dir": str(out_dir),
        "baseline_scan_id": records[baseline_idx]["scan_id"],
        "levels": [{"name": name, "score_limit": score_limit} for name, score_limit in STRICTNESS_LEVELS],
        "files": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "strictness_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
