"""
Compare two experiments on a current-start aligned x-axis.

The x-axis is real cycler time, zeroed at the first row where the cycler current
rises from rest to about 0.086 A. This preserves the true step durations while
still aligning the two experiments at the same cycling event.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    (
        "18-5",
        Path("data/experiments/18-5/cycler/19-5 cycle.txt"),
        Path("reports/experiments/18-5_feature_exploration/ascan_feature_table.csv"),
        "#2563eb",
    ),
    (
        "21-5",
        Path("data/experiments/21-5/cycler/21-5 cycle.txt"),
        Path("reports/experiments/21-5_feature_exploration/ascan_feature_table.csv"),
        "#dc2626",
    ),
]


FEATURES = [
    ("h5_tof_us", "TOF norm."),
    ("energy_centroid_time_us", "Energy-centroid\nnorm."),
    ("late_energy", "Late energy\nnorm."),
    ("spectral_entropy", "Spectral entropy\nnorm."),
]


def read_cycler(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skiprows=6, engine="python").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["Rec", "Step", "Capacity", "Energy", "Current", "Voltage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["DPT Time"], dayfirst=True, errors="coerce")
    mode = df["MD"].astype(str).str.strip()
    df["signed_current_a"] = 0.0
    df.loc[mode.eq("C"), "signed_current_a"] = df.loc[mode.eq("C"), "Current"]
    df.loc[mode.eq("D"), "signed_current_a"] = -df.loc[mode.eq("D"), "Current"]
    active = df["Current"].abs().ge(0.08)
    if not active.any():
        raise ValueError(f"No active current row found in {path}")
    t0 = df.loc[active.idxmax(), "time"]
    df["active_time_h"] = (df["time"] - t0).dt.total_seconds() / 3600.0
    return df


def add_active_time(feature_df: pd.DataFrame, cycler: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.copy()
    # Feature table has cycler_elapsed_s from cycler first row. Convert by using
    # the cycler active start offset in that same frame.
    active_start_h = float(cycler.loc[cycler["Current"].abs().ge(0.08), "active_time_h"].iloc[0])
    # This is zero by construction, but keep the code explicit and robust.
    first_time = cycler["time"].iloc[0]
    active_time = cycler.loc[cycler["Current"].abs().ge(0.08), "time"].iloc[0]
    active_offset_s = (active_time - first_time).total_seconds()
    out["active_time_h"] = (out["cycler_elapsed_s"].to_numpy(dtype=float) - active_offset_s) / 3600.0
    return out


def robust_norm(y: pd.Series) -> np.ndarray:
    arr = y.to_numpy(dtype=float)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        return np.full_like(arr, np.nan)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def smooth_xy(x: np.ndarray, y: np.ndarray, win: int = 75) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < win:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    k = np.ones(win) / win
    ys = np.convolve(y, k, mode="same")
    h = win // 2
    ys[:h] = np.nan
    ys[-h:] = np.nan
    return x, ys


def step_events(cycler: pd.DataFrame) -> list[tuple[float, str]]:
    labels = {
        2: "D1 start",
        3: "D1 end",
        4: "charge start",
        5: "CV",
        6: "high rest",
        7: "D2 start",
        8: "D2 end",
        9: "partial charge",
        10: "end",
    }
    events = []
    for step, label in labels.items():
        rows = cycler[cycler["Step"].eq(step)]
        if rows.empty:
            continue
        events.append((float(rows["active_time_h"].iloc[0]), label))
    # Add final row as end marker when the file has no explicit final step.
    end_x = float(cycler["active_time_h"].iloc[-1])
    if not events or abs(events[-1][0] - end_x) > 1e-3:
        events.append((end_x, "end"))

    deduped = []
    for x, label in events:
        if deduped and abs(deduped[-1][0] - x) <= 1e-3 and deduped[-1][1] == label:
            continue
        deduped.append((x, label))
    return deduped


def write_event_table(loaded: list[tuple[str, pd.DataFrame, pd.DataFrame, str]], out_dir: Path) -> None:
    rows = []
    for run_label, cycler, _, _ in loaded:
        for x, event in step_events(cycler):
            rows.append({"run": run_label, "event": event, "hours_since_0p086a_start": x})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "current_start_event_timing.csv", index=False)


def main() -> None:
    out_dir = Path("reports/experiments/current_start_aligned_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for label, cycler_path, feature_path, color in RUNS:
        cycler = read_cycler(cycler_path)
        features = add_active_time(pd.read_csv(feature_path), cycler)
        loaded.append((label, cycler, features, color))
    write_event_table(loaded, out_dir)

    fig, axs = plt.subplots(8, 1, figsize=(15, 16), dpi=160, sharex=True)
    for label, cycler, features, color in loaded:
        axs[0].plot(cycler["active_time_h"], cycler["signed_current_a"], color=color, lw=1.0, label=label)
        axs[1].plot(cycler["active_time_h"], cycler["Voltage"], color=color, lw=1.0, label=label)
        axs[2].plot(features["active_time_h"], features["soc_pct_clipped"], color=color, lw=1.2, label=label)
        axs[3].plot(features["active_time_h"], features["temperature_c"], color=color, lw=1.0, label=label)
        for ax, (feature, ylabel) in zip(axs[4:], FEATURES):
            x = features["active_time_h"].to_numpy(dtype=float)
            y = robust_norm(features[feature])
            xs, ys = smooth_xy(x, y, 75)
            ax.plot(xs, ys, color=color, lw=1.4, label=label)

    axs[0].axhline(0, color="#111827", lw=0.6)
    axs[0].set_ylabel("Current (A)")
    axs[1].set_ylabel("Voltage (V)")
    axs[2].set_ylabel("SOC (%)")
    axs[3].set_ylabel("Temp (C)")
    for ax, (_, ylabel) in zip(axs[4:], FEATURES):
        ax.set_ylabel(ylabel)

    # Event labels: use 21-5 labels because it includes the full 0.3 Ah partial charge.
    event_source = loaded[1][1]
    event_y_offsets = [-0.18, -0.27, -0.36]
    for i, (x, label) in enumerate(step_events(event_source)):
        for ax in axs:
            ax.axvline(x, color="#9ca3af", lw=0.7, alpha=0.5)
        axs[-1].text(
            x,
            event_y_offsets[i % len(event_y_offsets)],
            label,
            rotation=35,
            ha="right",
            va="top",
            fontsize=8,
            transform=axs[-1].get_xaxis_transform(),
        )

    for ax in axs:
        ax.grid(alpha=0.23)
        ax.legend(loc="best", fontsize=8)

    axs[0].set_title("18-5 vs 21-5 aligned by first active 0.086 A current")
    axs[-1].set_xlabel("Hours since first active current row (Current = 0.086 A)")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_dir / "current_start_aligned_context_features.png")
    plt.close(fig)

    # A more compact SOC-overlaid feature plot, useful for calibration discussion.
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), dpi=160)
    feature_set = [
        ("h5_tof_us", "TOF"),
        ("energy_centroid_time_us", "Energy-centroid time"),
        ("late_energy", "Late energy"),
        ("early_energy", "Early energy"),
        ("spectral_entropy", "Spectral entropy"),
        ("bandpower_3", "Bandpower 3"),
    ]
    for ax, (feature, title) in zip(axs.ravel(), feature_set):
        for branch, ls in [("charge", "-"), ("discharge", "--")]:
            for label, _, features, color in loaded:
                m = features["branch"].eq(branch)
                x = features.loc[m, "soc_pct_clipped"].to_numpy(dtype=float)
                y = robust_norm(features.loc[m, feature])
                xs, ys = smooth_xy(x, y, 35)
                ax.plot(xs, ys, color=color, ls=ls, lw=1.4, label=f"{label} {branch}")
        ax.set_title(title)
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel("normalized")
        ax.grid(alpha=0.23)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Feature-vs-SOC comparison by branch", y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_dir / "soc_branch_feature_comparison.png")
    plt.close(fig)

    print(out_dir.resolve())


if __name__ == "__main__":
    main()
