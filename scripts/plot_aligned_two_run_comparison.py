"""
Make supervisor-friendly aligned comparison plots for two cycling A-scan runs.

Elapsed time is not ideal for comparing repeatability because one run may reach
cutoffs slightly earlier. This script builds a shared cycle-coordinate axis from
Maccor step/branch information:

  initial discharge -> low rest -> full charge -> high rest -> full discharge
  -> low rest -> partial charge

Both experiments are then plotted on the same x-axis so voltage, SOC,
temperature, and selected ultrasound features line up by cycling state.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    ("18-5", Path("reports/experiments/18-5_feature_exploration/ascan_feature_table.csv"), "#2563eb"),
    ("21-5", Path("reports/experiments/21-5_feature_exploration/ascan_feature_table.csv"), "#dc2626"),
]


SEGMENTS = [
    (2, "initial discharge"),
    (3, "low rest"),
    (4, "charge"),
    (5, "CV"),
    (6, "high rest"),
    (7, "discharge"),
    (8, "low rest"),
    (9, "partial charge"),
]


FEATURES = [
    ("h5_tof_us", "TOF (us)"),
    ("energy_centroid_time_us", "Energy-centroid time (us)"),
    ("late_energy", "Late energy"),
    ("early_energy", "Early energy"),
    ("spectral_entropy", "Spectral entropy"),
    ("bandpower_3", "Bandpower 3"),
]


def robust_norm(y: pd.Series) -> np.ndarray:
    arr = y.to_numpy(dtype=float)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        return np.full_like(arr, np.nan)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def add_cycle_x(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cycle_x"] = np.nan
    width = 1.0
    gap = 0.04
    start = 0.0
    centers = []
    labels = []
    boundaries = []
    for step, label in SEGMENTS:
        m = out["Step"].eq(step)
        if not m.any():
            start += width + gap
            continue
        t = out.loc[m, "time_h"].to_numpy(dtype=float)
        denom = float(np.nanmax(t) - np.nanmin(t))
        if denom <= 1e-12:
            phase = np.zeros_like(t)
        else:
            phase = (t - np.nanmin(t)) / denom
        out.loc[m, "cycle_x"] = start + phase * width
        centers.append(start + 0.5 * width)
        labels.append(label)
        boundaries.append(start)
        start += width + gap
    boundaries.append(start - gap)
    return out, np.asarray(centers), labels, np.asarray(boundaries)


def smooth_for_plot(x: np.ndarray, y: np.ndarray, n: int = 75) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 3:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = min(n, max(5, len(y) // 6))
    if n <= 3:
        return x, y
    k = np.ones(n) / n
    ys = np.convolve(y, k, mode="same")
    h = n // 2
    ys[:h] = np.nan
    ys[-h:] = np.nan
    return x, ys


def plot_aligned(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    centers = labels = boundaries = None
    for label, path, color in RUNS:
        df = pd.read_csv(path)
        df, c, l, b = add_cycle_x(df)
        loaded.append((label, df, color))
        centers, labels, boundaries = c, l, b

    fig, axs = plt.subplots(6, 1, figsize=(15, 13), dpi=160, sharex=True)
    for label, df, color in loaded:
        x = df["cycle_x"].to_numpy(dtype=float)
        for ax, col, ylabel, norm in [
            (axs[0], "Voltage", "Voltage (V)", False),
            (axs[1], "soc_pct_clipped", "SOC (%)", False),
            (axs[2], "temperature_c", "Temperature (C)", False),
            (axs[3], "h5_tof_us", "TOF shift\nnorm.", True),
            (axs[4], "late_energy", "Late energy\nnorm.", True),
            (axs[5], "spectral_entropy", "Spectral entropy\nnorm.", True),
        ]:
            if col not in df:
                continue
            y = robust_norm(df[col]) if norm else df[col].to_numpy(dtype=float)
            xs, ys = smooth_for_plot(x, y, 75)
            ax.plot(xs, ys, color=color, lw=1.5, label=label)
            ax.set_ylabel(ylabel)
    for ax in axs:
        for boundary in boundaries:
            ax.axvline(boundary, color="#9ca3af", lw=0.7, alpha=0.55)
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8)
    axs[0].set_title("18-5 vs 21-5 aligned by cycling segment")
    axs[-1].set_xticks(centers)
    axs[-1].set_xticklabels(labels, rotation=25, ha="right")
    axs[-1].set_xlabel("Aligned cycling segment")
    fig.tight_layout()
    fig.savefig(out_dir / "aligned_cycle_context_and_features.png")
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(16, 8), dpi=160, sharex=True)
    for ax, (feature, title) in zip(axs.ravel(), FEATURES):
        for label, df, color in loaded:
            if feature not in df:
                continue
            x = df["cycle_x"].to_numpy(dtype=float)
            y = robust_norm(df[feature])
            xs, ys = smooth_for_plot(x, y, 75)
            ax.plot(xs, ys, color=color, lw=1.5, label=label)
        for boundary in boundaries:
            ax.axvline(boundary, color="#9ca3af", lw=0.7, alpha=0.45)
        ax.set_title(title)
        ax.set_ylabel("normalized")
        ax.grid(alpha=0.22)
    for ax in axs[-1]:
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    handles, legend_labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2)
    fig.suptitle("Feature repeatability on aligned cycling axis", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "aligned_feature_repeatability_panels.png")
    plt.close(fig)

    # Also provide a SOC-as-x view, separated by branch, for direct calibration thinking.
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), dpi=160)
    for ax, (feature, title) in zip(axs.ravel(), FEATURES):
        for branch, ls in [("charge", "-"), ("discharge", "--")]:
            for label, df, color in loaded:
                if feature not in df:
                    continue
                m = df["branch"].eq(branch)
                x = df.loc[m, "soc_pct_clipped"].to_numpy(dtype=float)
                y = robust_norm(df.loc[m, feature])
                xs, ys = smooth_for_plot(x, y, 35)
                ax.plot(xs, ys, color=color, lw=1.4, ls=ls, label=f"{label} {branch}")
        ax.set_title(title)
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel("normalized")
        ax.grid(alpha=0.22)
    handles, legend_labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4)
    fig.suptitle("Feature-vs-SOC alignment by branch", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "aligned_soc_branch_feature_panels.png")
    plt.close(fig)


def main() -> None:
    out = Path("reports/experiments/two_run_aligned_comparison")
    plot_aligned(out)
    print(out.resolve())


if __name__ == "__main__":
    main()
