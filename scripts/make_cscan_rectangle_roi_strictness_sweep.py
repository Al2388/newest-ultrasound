"""Generate rectangular ROI candidates at different strictness levels.

Unlike the stable-mask sweep, this script keeps each candidate as a complete
axis-aligned rectangle. Strictness changes the rectangle size/location, not
individual pixels inside the rectangle.
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


def unique_scan_records(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    seen: dict[int, dict[str, object]] = {}
    for row in rows:
        idx = int(row["scan_index"])
        if idx not in seen:
            seen[idx] = {
                "run": idx,
                "scan_id": row["scan_id"],
                "session_dir": Path(row["session_dir"]),
                "is_baseline": row["is_baseline"].lower() == "true",
            }
    return [seen[idx] for idx in sorted(seen)]


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
        raise RuntimeError("No scans loaded.")
    return {feature: np.stack(values, axis=0) for feature, values in stacks.items()}, x_mm, y_mm, baseline_idx


def image_extent(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[float, float, float, float]:
    return float(x_mm.min()), float(x_mm.max()), float(y_mm.min()), float(y_mm.max())


def build_noise_score(diffs: np.lib.npyio.NpzFile) -> np.ndarray:
    terms = []
    for feature, meta in FEATURES.items():
        span = diffs[f"{feature}_noise_floor_span"].astype(float)
        terms.append(span / float(meta["base_span"]))
    return np.nanmax(np.stack(terms, axis=0), axis=0)


def index_bounds_from_mm(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[int, int, int, int]:
    bounds = BATTERY_BODY_BOUNDS_MM
    cols = np.where((x_mm >= bounds["x_min"]) & (x_mm <= bounds["x_max"]))[0]
    rows = np.where((y_mm >= bounds["y_min"]) & (y_mm <= bounds["y_max"]))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def rect_bounds_mm(rect: tuple[int, int, int, int], x_mm: np.ndarray, y_mm: np.ndarray) -> dict[str, float]:
    r0, r1, c0, c1 = rect
    return {
        "x_min": float(x_mm[c0]),
        "x_max": float(x_mm[c1 - 1]),
        "y_min": float(y_mm[r0]),
        "y_max": float(y_mm[r1 - 1]),
    }


def add_rect(ax: plt.Axes, rect: tuple[int, int, int, int], x_mm: np.ndarray, y_mm: np.ndarray, **kwargs) -> None:
    b = rect_bounds_mm(rect, x_mm, y_mm)
    ax.add_patch(
        Rectangle(
            (b["x_min"], b["y_min"]),
            b["x_max"] - b["x_min"],
            b["y_max"] - b["y_min"],
            fill=False,
            **kwargs,
        )
    )


def candidate_rectangles(
    body_rect: tuple[int, int, int, int],
    min_width_fraction: float = 0.50,
    min_height_fraction: float = 0.50,
) -> list[tuple[int, int, int, int]]:
    r0, r1, c0, c1 = body_rect
    height = r1 - r0
    width = c1 - c0
    min_h = int(round(height * min_height_fraction))
    min_w = int(round(width * min_width_fraction))

    # Coarse but report-friendly search. We want interpretable boxes, not
    # single-pixel optimality, and evaluating every possible edge combination is
    # slow on the full 144 x 500 map.
    row_margins = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
    col_margins = [0, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100, 120]
    candidates = []
    for top in row_margins:
        for bottom in row_margins:
            rr0 = r0 + top
            rr1 = r1 - bottom
            if rr1 - rr0 < min_h:
                continue
            for left in col_margins:
                for right in col_margins:
                    cc0 = c0 + left
                    cc1 = c1 - right
                    if cc1 - cc0 < min_w:
                        continue
                    candidates.append((rr0, rr1, cc0, cc1))
    return candidates


def choose_rectangle(
    candidates: list[tuple[int, int, int, int]],
    score: np.ndarray,
    score_limit: float,
    body_area: int,
) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    best: tuple[int, int, int, int] | None = None
    best_stats: dict[str, float] | None = None
    fallback: tuple[int, int, int, int] | None = None
    fallback_stats: dict[str, float] | None = None

    for rect in candidates:
        r0, r1, c0, c1 = rect
        values = score[r0:r1, c0:c1]
        area = int((r1 - r0) * (c1 - c0))
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        p95 = float(np.nanpercentile(finite, 95))
        median = float(np.nanmedian(finite))
        good_fraction = float(np.mean(finite <= score_limit))
        stats = {
            "area": float(area),
            "area_fraction": float(area / body_area),
            "p95_score": p95,
            "median_score": median,
            "good_fraction": good_fraction,
            "meets_p95": float(p95 <= score_limit),
        }
        if p95 <= score_limit:
            if best is None:
                best, best_stats = rect, stats
            else:
                assert best_stats is not None
                key = (stats["area"], -stats["p95_score"], stats["good_fraction"])
                best_key = (best_stats["area"], -best_stats["p95_score"], best_stats["good_fraction"])
                if key > best_key:
                    best, best_stats = rect, stats
        if fallback is None:
            fallback, fallback_stats = rect, stats
        else:
            assert fallback_stats is not None
            # Prefer high good fraction, then larger area, then lower p95 if no rectangle meets p95.
            key = (stats["good_fraction"], stats["area"], -stats["p95_score"])
            fallback_key = (fallback_stats["good_fraction"], fallback_stats["area"], -fallback_stats["p95_score"])
            if key > fallback_key:
                fallback, fallback_stats = rect, stats

    if best is not None and best_stats is not None:
        return best, best_stats
    if fallback is None or fallback_stats is None:
        raise RuntimeError("No valid rectangle candidates found.")
    fallback_stats["meets_p95"] = 0.0
    return fallback, fallback_stats


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def summarize_rectangle(
    name: str,
    score_limit: float,
    rect: tuple[int, int, int, int],
    stats: dict[str, float],
    score: np.ndarray,
    sigmas: dict[str, np.ndarray],
    spans: dict[str, np.ndarray],
    stacks: dict[str, np.ndarray],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> dict[str, object]:
    mask = rect_mask(score.shape, rect)
    b = rect_bounds_mm(rect, x_mm, y_mm)
    row: dict[str, object] = {
        "Level": name,
        "Score <= ": fmt(score_limit, 3),
        "Rows": f"{rect[0]}:{rect[1]}",
        "Cols": f"{rect[2]}:{rect[3]}",
        "X mm": f"{b['x_min']:.2f}-{b['x_max']:.2f}",
        "Y mm": f"{b['y_min']:.2f}-{b['y_max']:.2f}",
        "Battery kept %": fmt(100.0 * stats["area_fraction"], 4),
        "P95 score": fmt(stats["p95_score"]),
        "Good pixel %": fmt(100.0 * stats["good_fraction"], 4),
        "Meets p95": "yes" if stats["meets_p95"] else "no",
    }
    for feature, meta in FEATURES.items():
        scan_means = np.array([float(np.nanmean(scan[mask])) for scan in stacks[feature]])
        row[f"{meta['label']} p95 sigma"] = fmt(float(np.nanpercentile(sigmas[feature][mask], 95)))
        row[f"{meta['label']} p95 span"] = fmt(float(np.nanpercentile(spans[feature][mask], 95)))
        row[f"{meta['label']} mean sigma"] = fmt(float(np.nanstd(scan_means, ddof=1)))
    return row


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Rectangular ROI Strictness Summary\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[k]) for k in fieldnames) + " |\n")


def write_latex(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Rectangular C-scan ROI candidates across noise strictness levels.}\n")
        f.write("\\label{tab:cscan_rectangular_roi_strictness}\n")
        f.write("\\begin{tabular}{llllllll}\n\\hline\n")
        f.write(" & ".join(fieldnames) + " \\\\\n\\hline\n")
        for row in rows:
            f.write(" & ".join(str(row[k]) for k in fieldnames) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def plot_all_rectangles(
    out_dir: Path,
    baseline_amp: np.ndarray,
    rects: dict[str, tuple[int, int, int, int]],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=300)
    im = ax.imshow(baseline_amp, origin="lower", extent=image_extent(x_mm, y_mm), aspect="auto", cmap="turbo", vmin=0, vmax=6)
    fig.colorbar(im, ax=ax, label="Amplitude (V)")
    colors = {
        "loose": "#ffff33",
        "relaxed": "#fdae61",
        "current": "#00ffff",
        "strict": "#1a9850",
        "very_strict": "#2c7bb6",
        "ultra_strict": "#762a83",
    }
    for name, rect in rects.items():
        add_rect(ax, rect, x_mm, y_mm, edgecolor=colors.get(name, "white"), linewidth=2.0, label=name)
    ax.set_title("Rectangular ROI candidates by strictness")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_rectangular_roi_all_levels.png")
    plt.close(fig)


def plot_rectangle_panels(
    out_dir: Path,
    baseline_amp: np.ndarray,
    rects: dict[str, tuple[int, int, int, int]],
    rows: list[dict[str, object]],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), dpi=300, constrained_layout=True)
    img_extent = image_extent(x_mm, y_mm)
    row_by_name = {str(row["Level"]): row for row in rows}
    for ax, (name, rect) in zip(axes.ravel(), rects.items()):
        ax.imshow(baseline_amp, origin="lower", extent=img_extent, aspect="auto", cmap="gray", vmin=0, vmax=6)
        add_rect(ax, rect, x_mm, y_mm, edgecolor="#00ffff", linewidth=2.2)
        row = row_by_name[name]
        ax.set_title(f"{name}: keep {row['Battery kept %']}%, p95 score {row['P95 score']}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.suptitle("Complete rectangular ROI candidates", fontsize=13)
    fig.savefig(out_dir / "fig_rectangular_roi_level_panels.png", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(out_dir: Path, rows: list[dict[str, object]]) -> None:
    keep = np.array([float(row["Battery kept %"]) for row in rows])
    amp = np.array([float(row["Amplitude p95 sigma"]) for row in rows])
    tof = np.array([float(row["ToF p95 sigma"]) for row in rows])
    energy = np.array([float(row["Energy p95 sigma"]) for row in rows])
    labels = [str(row["Level"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), dpi=300, constrained_layout=True)
    for ax, title, values, unit, color in [
        (axes[0], "Amplitude", amp, "V", "#d95f0e"),
        (axes[1], "ToF", tof, "us", "#2b8cbe"),
        (axes[2], "Energy", energy, "a.u.", "#31a354"),
    ]:
        ax.plot(keep, values, marker="o", linewidth=2, color=color)
        for x, y, label in zip(keep, values, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Battery-body area kept (%)")
        ax.set_ylabel(f"P95 local sigma ({unit})")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.invert_xaxis()
    fig.suptitle("Rectangular ROI strictness tradeoff", fontsize=12)
    fig.savefig(out_dir / "fig_rectangular_roi_tradeoff.png", bbox_inches="tight")
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
    out_dir = report_dir / "rectangular_roi_strictness_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_in = read_csv(report_dir / "noise_floor_summary.csv")
    records = unique_scan_records(rows_in)
    stacks, x_mm, y_mm, baseline_idx = load_scans(records)
    diffs = np.load(report_dir / "noise_floor_diffs.npz")
    score = build_noise_score(diffs)
    sigmas = {feature: np.nanstd(stack, axis=0, ddof=1) for feature, stack in stacks.items()}
    spans = {feature: diffs[f"{feature}_noise_floor_span"].astype(float) for feature in FEATURES}

    body_rect = index_bounds_from_mm(x_mm, y_mm)
    body_area = (body_rect[1] - body_rect[0]) * (body_rect[3] - body_rect[2])
    candidates = candidate_rectangles(body_rect)

    rects: dict[str, tuple[int, int, int, int]] = {}
    summary_rows: list[dict[str, object]] = []
    for name, score_limit in STRICTNESS_LEVELS:
        rect, stats = choose_rectangle(candidates, score, score_limit, body_area)
        rects[name] = rect
        summary_rows.append(summarize_rectangle(name, score_limit, rect, stats, score, sigmas, spans, stacks, x_mm, y_mm))

    fieldnames = [
        "Level",
        "Score <= ",
        "Rows",
        "Cols",
        "X mm",
        "Y mm",
        "Battery kept %",
        "P95 score",
        "Good pixel %",
        "Meets p95",
        "Amplitude p95 sigma",
        "ToF p95 sigma",
        "Energy p95 sigma",
        "Amplitude p95 span",
        "ToF p95 span",
        "Energy p95 span",
        "Amplitude mean sigma",
        "ToF mean sigma",
        "Energy mean sigma",
    ]
    write_csv(out_dir / "rectangular_roi_strictness_summary.csv", summary_rows, fieldnames)
    write_markdown(out_dir / "rectangular_roi_strictness_summary.md", summary_rows, fieldnames)
    write_latex(out_dir / "rectangular_roi_strictness_summary.tex", summary_rows, fieldnames[:8])

    baseline_amp = stacks["amplitude"][baseline_idx]
    plot_all_rectangles(out_dir, baseline_amp, rects, x_mm, y_mm)
    plot_rectangle_panels(out_dir, baseline_amp, rects, summary_rows, x_mm, y_mm)
    plot_tradeoff(out_dir, summary_rows)

    masks = {f"mask_{name}": rect_mask(score.shape, rect) for name, rect in rects.items()}
    bounds = {
        name: rect_bounds_mm(rect, x_mm, y_mm)
        for name, rect in rects.items()
    }
    np.savez_compressed(
        out_dir / "rectangular_roi_strictness_masks.npz",
        noise_score=score,
        **masks,
        **{f"bounds_{name}": np.array([b["x_min"], b["x_max"], b["y_min"], b["y_max"]], dtype=float) for name, b in bounds.items()},
    )

    index = """# Rectangular ROI Strictness Sweep

This folder compares complete rectangular ROI candidates. Unlike the stable
pixel-mask sweep, each level keeps every pixel inside its rectangle.

Selection rule:

For each strictness score limit, the script searches inside the battery-body
region for the largest rectangle whose 95th percentile combined noise score is
below that limit.

Combined score:

`score = max(amplitude_span/0.55 V, tof_span/0.18 us, energy_span/32)`

Files:

- `fig_rectangular_roi_all_levels.png`
- `fig_rectangular_roi_level_panels.png`
- `fig_rectangular_roi_tradeoff.png`
- `rectangular_roi_strictness_summary.csv`
- `rectangular_roi_strictness_summary.md`
- `rectangular_roi_strictness_masks.npz`
"""
    (out_dir / "RECTANGULAR_ROI_SWEEP_INDEX.md").write_text(index, encoding="utf-8")

    manifest = {
        "report_dir": str(report_dir),
        "out_dir": str(out_dir),
        "baseline_scan_id": records[baseline_idx]["scan_id"],
        "levels": [{"name": name, "score_limit": limit, "bounds_mm": bounds[name]} for name, limit in STRICTNESS_LEVELS],
        "files": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "rectangular_roi_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
