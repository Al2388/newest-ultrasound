"""
Compare feature sensitivity and repeatability across two A-scan cycling runs.

Inputs are the per-run outputs from scripts/explore_ascan_features.py:
  - ascan_feature_table.csv
  - feature_sensitivity_scores.csv

The comparison asks:
  1. Which features are more SOC/branch-sensitive than temperature-sensitive?
  2. Which feature-vs-SOC curves repeat between experiments?
  3. Which features are good candidates for later scans?
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTEXT_COLS = {
    "ascan_index",
    "ascan_unix_s",
    "ascan_elapsed_h",
    "cycler_elapsed_s",
    "Step",
    "Capacity",
    "Energy",
    "Current",
    "Voltage",
    "MD",
    "signed_current_a",
    "relative_q_ah",
    "soc_pct",
    "soc_pct_clipped",
    "branch",
    "temperature_c",
    "time_h",
}


PREFERRED_FEATURES = [
    "h5_tof_us",
    "h5_tof_us_envelope",
    "ncc_shift_us",
    "energy_centroid_time_us",
    "early_energy",
    "late_energy",
    "early_late_energy_ratio",
    "h5_energy",
    "rms_v",
    "spectral_bandwidth_mhz",
    "spectral_entropy",
    "spectral_centroid_mhz",
    "bandpower_0",
    "bandpower_1",
    "bandpower_2",
    "bandpower_3",
    "bandpower_4",
    "pca_score_1",
    "pca_score_3",
    "cosine_similarity_ref",
]


def load_run(path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(path / "ascan_feature_table.csv")
    score = pd.read_csv(path / "feature_sensitivity_scores.csv")
    table["experiment"] = label
    score["experiment"] = label
    return table, score


def robust_norm(y: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(y, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        return np.full_like(y, np.nan, dtype=float)
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)


def binned_curve(
    df: pd.DataFrame,
    feature: str,
    branch: str,
    bins: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    m = df["branch"].eq(branch) & np.isfinite(df[feature]) & np.isfinite(df["soc_pct_clipped"])
    x = df.loc[m, "soc_pct_clipped"].to_numpy(dtype=float)
    y = df.loc[m, feature].to_numpy(dtype=float)
    if normalize:
        y = robust_norm(y)
    out = np.full(len(bins) - 1, np.nan, dtype=float)
    which = np.digitize(x, bins) - 1
    for i in range(len(out)):
        vals = y[which == i]
        if vals.size >= 3:
            out[i] = float(np.nanmedian(vals))
    return out


def curve_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 5:
        return np.nan, np.nan, int(m.sum())
    aa, bb = a[m], b[m]
    corr = float(np.corrcoef(aa, bb)[0, 1]) if np.nanstd(aa) > 0 and np.nanstd(bb) > 0 else np.nan
    rmse = float(np.sqrt(np.nanmean((aa - bb) ** 2)))
    return corr, rmse, int(m.sum())


def build_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    score_a: pd.DataFrame,
    score_b: pd.DataFrame,
) -> pd.DataFrame:
    sa = score_a.set_index("feature")
    sb = score_b.set_index("feature")
    common = sorted(set(sa.index) & set(sb.index))
    bins = np.arange(0, 102, 2)
    rows = []
    for feature in common:
        if feature in CONTEXT_COLS or feature not in df_a.columns or feature not in df_b.columns:
            continue
        row = {"feature": feature}
        for prefix, s in [("a", sa.loc[feature]), ("b", sb.loc[feature])]:
            row[f"{prefix}_soc_after_temp"] = float(s["partial_r2_soc_after_temp"])
            row[f"{prefix}_temp_after_soc"] = float(s["partial_r2_temp_after_soc"])
            row[f"{prefix}_span_to_noise"] = float(s["span_to_noise"])
            row[f"{prefix}_charge_corr_soc"] = float(s["charge_corr_soc"])
            row[f"{prefix}_discharge_corr_soc"] = float(s["discharge_corr_soc"])
            row[f"{prefix}_rest_corr_temp"] = float(s["rest_corr_temp"])
        for branch in ["charge", "discharge"]:
            ca = binned_curve(df_a, feature, branch, bins)
            cb = binned_curve(df_b, feature, branch, bins)
            corr, rmse, n = curve_similarity(ca, cb)
            row[f"{branch}_curve_corr"] = corr
            row[f"{branch}_curve_rmse"] = rmse
            row[f"{branch}_overlap_bins"] = n
        row["mean_soc_after_temp"] = np.nanmean([row["a_soc_after_temp"], row["b_soc_after_temp"]])
        row["mean_temp_after_soc"] = np.nanmean([row["a_temp_after_soc"], row["b_temp_after_soc"]])
        row["soc_minus_temp"] = row["mean_soc_after_temp"] - row["mean_temp_after_soc"]
        row["mean_log_span_to_noise"] = np.nanmean(
            np.log10(np.clip([row["a_span_to_noise"], row["b_span_to_noise"]], 1, None))
        )
        row["mean_curve_corr"] = np.nanmean([row["charge_curve_corr"], row["discharge_curve_corr"]])
        row["mean_curve_rmse"] = np.nanmean([row["charge_curve_rmse"], row["discharge_curve_rmse"]])
        row["recommendation_score"] = (
            row["soc_minus_temp"]
            + 0.35 * np.nan_to_num(row["mean_curve_corr"], nan=0.0)
            + 0.10 * row["mean_log_span_to_noise"]
            - 0.20 * np.nan_to_num(row["mean_curve_rmse"], nan=0.5)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("recommendation_score", ascending=False)


def plot_cycle_context(df_a: pd.DataFrame, df_b: pd.DataFrame, out: Path) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(13, 8), dpi=160, sharex=False)
    for df, label, color in [(df_a, "18-5", "#2563eb"), (df_b, "21-5", "#dc2626")]:
        axs[0].plot(df["time_h"], df["Voltage"], color=color, lw=0.9, label=label)
        axs[1].plot(df["time_h"], df["soc_pct_clipped"], color=color, lw=1.1, label=label)
        axs[2].plot(df["time_h"], df["temperature_c"], color=color, lw=0.9, label=label)
    axs[0].set_ylabel("Voltage (V)")
    axs[1].set_ylabel("SOC (%)")
    axs[2].set_ylabel("Temperature (C)")
    axs[2].set_xlabel("Elapsed time (h)")
    axs[0].set_title("Cycle and temperature context for repeatability comparison")
    for ax in axs:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "cycle_temperature_context.png")
    plt.close(fig)


def plot_rankings(comp: pd.DataFrame, out: Path) -> None:
    top = comp.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9), dpi=160)
    ax.barh(top["feature"], top["recommendation_score"], color="#047857")
    ax.set_xlabel("Combined score: SOC distinct + repeatable + high SNR")
    ax.set_title("Best feature candidates across both experiments")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "feature_candidate_ranking.png")
    plt.close(fig)

    cols = [
        "mean_soc_after_temp",
        "mean_temp_after_soc",
        "soc_minus_temp",
        "charge_curve_corr",
        "discharge_curve_corr",
        "mean_log_span_to_noise",
    ]
    heat = comp.head(25).set_index("feature")[cols]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(heat)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["SOC|T", "T|SOC", "SOC-T", "charge repeat", "disch repeat", "log SNR"], rotation=35, ha="right")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature sensitivity and repeatability matrix")
    fig.tight_layout()
    fig.savefig(out / "feature_repeatability_matrix.png")
    plt.close(fig)


def plot_feature_curves(df_a: pd.DataFrame, df_b: pd.DataFrame, features: list[str], out: Path) -> None:
    bins = np.arange(0, 102, 2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ncols = 3
    nrows = math.ceil(len(features) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), dpi=160, squeeze=False)
    for ax, feature in zip(axs.ravel(), features):
        for df, label, color, ls in [
            (df_a, "18-5 charge", "#2563eb", "-"),
            (df_b, "21-5 charge", "#dc2626", "-"),
            (df_a, "18-5 discharge", "#2563eb", "--"),
            (df_b, "21-5 discharge", "#dc2626", "--"),
        ]:
            branch = "charge" if "charge" in label and "discharge" not in label else "discharge"
            curve = binned_curve(df, feature, branch, bins, normalize=True)
            ax.plot(centers, curve, color=color, ls=ls, lw=1.7, label=label)
        ax.set_title(feature)
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel("normalized feature")
        ax.grid(alpha=0.25)
    for ax in axs.ravel()[len(features) :]:
        ax.axis("off")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "top_feature_soc_curves_repeatability.png")
    plt.close(fig)


def write_report(comp: pd.DataFrame, df_a: pd.DataFrame, df_b: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Two-Experiment Ultrasound Feature Repeatability",
        "",
        "## Data Used",
        f"- 18-5 rows: {len(df_a):,}; SOC {df_a['soc_pct_clipped'].min():.1f}-{df_a['soc_pct_clipped'].max():.1f}%; temperature {df_a['temperature_c'].min():.2f}-{df_a['temperature_c'].max():.2f} C",
        f"- 21-5 rows: {len(df_b):,}; SOC {df_b['soc_pct_clipped'].min():.1f}-{df_b['soc_pct_clipped'].max():.1f}%; temperature {df_b['temperature_c'].min():.2f}-{df_b['temperature_c'].max():.2f} C",
        "",
        "## Best Candidate Features",
        "",
        "| Feature | Combined score | mean SOC after temp | mean temp after SOC | SOC-temp gap | charge repeat r | discharge repeat r |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in comp.head(20).iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['recommendation_score']:.3f} | {r['mean_soc_after_temp']:.3f} | "
            f"{r['mean_temp_after_soc']:.3f} | {r['soc_minus_temp']:.3f} | "
            f"{r['charge_curve_corr']:.3f} | {r['discharge_curve_corr']:.3f} |"
        )
    lines += [
        "",
        "## How To Read This",
        "- `mean SOC after temp`: how much variation remains explainable by SOC/branch once temperature is already included.",
        "- `mean temp after SOC`: how much temperature still explains after SOC/branch is included.",
        "- `SOC-temp gap`: positive values are better for SOC features.",
        "- `charge/discharge repeat r`: correlation between the normalized feature-vs-SOC curves in the two experiments.",
        "",
        "## Suggested Shortlist For Supervisor Slides",
        "- Use one TOF feature: `h5_tof_us` or `energy_centroid_time_us`.",
        "- Use one energy redistribution feature: `early_late_energy_ratio` or `late_energy`.",
        "- Use one spectral feature: `spectral_bandwidth_mhz` or `spectral_entropy`.",
        "- Keep amplitude-only features secondary because they are more vulnerable to coupling and clipping.",
        "",
        "## Caveat",
        "Temperature is not independently controlled here, so this proves within-run separation and repeatability, not a universal chamber-calibrated temperature coefficient.",
    ]
    (out / "TWO_EXPERIMENT_FEATURE_REPEATABILITY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", default="reports/experiments/18-5_feature_exploration")
    parser.add_argument("--run-b", default="reports/experiments/21-5_feature_exploration")
    parser.add_argument("--label-a", default="18-5")
    parser.add_argument("--label-b", default="21-5")
    parser.add_argument("--out-dir", default="reports/experiments/two_run_feature_repeatability")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_a, score_a = load_run(Path(args.run_a), args.label_a)
    df_b, score_b = load_run(Path(args.run_b), args.label_b)
    comp = build_comparison(df_a, df_b, score_a, score_b)
    comp.to_csv(out / "feature_repeatability_scores.csv", index=False)

    plot_cycle_context(df_a, df_b, out)
    plot_rankings(comp, out)
    preferred = [f for f in PREFERRED_FEATURES if f in set(comp["feature"])]
    top = []
    for feature in preferred:
        if feature in set(comp.head(40)["feature"]):
            top.append(feature)
    for feature in comp["feature"].head(12):
        if feature not in top:
            top.append(feature)
    plot_feature_curves(df_a, df_b, top[:12], out)
    write_report(comp, df_a, df_b, out)
    print(out.resolve())


if __name__ == "__main__":
    main()
