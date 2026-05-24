"""
Cross-session SOC-vs-temperature feature analysis for two repeat cycling runs.

Goal (for picking acoustic features to keep in future scans): rank A-scan
features by how strongly they track SOC while staying insensitive to
temperature, and quantify whether the two repeat sessions are reproducible.

Two repeats of the *same* protocol let us break the SOC/temperature confound
that a single session cannot: at a matched SOC, any feature difference between
the sessions is NOT due to SOC, so it is a clean readout of temperature (plus
drift/noise) sensitivity.

Three analyses
--------------
1. Within-session sensitivity (pooled charge+discharge, both sessions):
     partial R2 of a flexible SOC block (cubic + branch) vs partial R2 of
     temperature. SOC-temp collinearity is reported to justify the partials.
2. Cross-session repeatability: each feature is interpolated onto a common SOC
     grid per session/branch; we report the between-session RMS disagreement
     both absolutely and after removing a constant offset (a fixed offset is
     recalibratable), relative to the feature's SOC-driven span.
3. Temperature attribution: at matched SOC, regress dFeature(B-A) on
     dTemperature(B-A). The slope is an SOC-free temperature sensitivity and
     the R2 says how much of the non-repeatability temperature explains.

Outputs (to reports/experiments/repeatability/):
  feature_ranking.csv, REPEATABILITY_REPORT.md, and presentation PNGs.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES = {
    "18-5": "reports/experiments/19-5_feature_exploration/ascan_feature_table.csv",
    "21-5": "reports/experiments/21-5_feature_exploration/ascan_feature_table.csv",
}
OUT = Path("reports/experiments/repeatability")

CONTEXT_COLS = {
    "ascan_index", "ascan_unix_s", "ascan_elapsed_h", "cycler_elapsed_s",
    "Step", "Capacity", "Energy", "Current", "Voltage", "MD",
    "signed_current_a", "relative_q_ah", "soc_pct", "soc_pct_clipped",
    "branch", "temperature_c", "time_h",
}
# Features defined relative to a per-session reference waveform / PCA basis /
# tracking anchor: their absolute value is not comparable across sessions.
SESSION_RELATIVE = {
    "ncc_shift_us", "ncc_corr_ref", "cosine_similarity_ref", "mse_to_ref",
    "rmse_to_ref", "dtw_proxy_l1_to_ref", "h5_tof_us", "h5_tof_us_envelope",
    "h5_tracking_lag_samples", "h5_tracking_corr",
}
SESSION_RELATIVE_PREFIX = ("pca_score_", "pca_explained_")
# Quality/clip diagnostics, not physical observables.
DROP = {"h5_clip_fraction", "h5_raw_clip_fraction", "h5_n_averaged",
        "h5_n_rejected", "ringing_count"}

SOC_GRID = np.arange(5.0, 95.0 + 1e-9, 1.0)  # overlap region both sessions cover


# ----------------------------------------------------------------------------
def load() -> dict[str, pd.DataFrame]:
    out = {}
    for name, path in TABLES.items():
        df = pd.read_csv(path)
        df = df[df["branch"].isin(["charge", "discharge"])].copy()
        out[name] = df
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in CONTEXT_COLS or c in DROP:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def is_cross_session(col: str) -> bool:
    if col in SESSION_RELATIVE:
        return False
    return not col.startswith(SESSION_RELATIVE_PREFIX)


# ---- partial R2 helpers ----------------------------------------------------
def _sse(y: np.ndarray, X: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sum((y - X @ coef) ** 2))


def partial_r2(df: pd.DataFrame, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return partial R2(SOC|temp), partial R2(temp|SOC), std SOC slope, std temp slope."""
    soc = df["soc_pct_clipped"].to_numpy(float) / 100.0
    temp = df["temperature_c"].to_numpy(float)
    charge = (df["branch"].to_numpy() == "charge").astype(float)
    m = np.isfinite(y) & np.isfinite(soc) & np.isfinite(temp)
    if m.sum() < 30:
        return np.nan, np.nan, np.nan, np.nan
    y, soc, temp, charge = y[m], soc[m], temp[m], charge[m]
    one = np.ones(m.sum())
    soc_block = np.column_stack([soc, soc**2, soc**3, charge, soc * charge])
    X_full = np.column_stack([one, soc_block, temp])
    X_soc  = np.column_stack([one, soc_block])
    X_temp = np.column_stack([one, temp])
    sse_full, sse_soc, sse_temp = _sse(y, X_full), _sse(y, X_soc), _sse(y, X_temp)
    pr2_soc  = (sse_temp - sse_full) / sse_temp if sse_temp > 0 else np.nan
    pr2_temp = (sse_soc  - sse_full) / sse_soc  if sse_soc  > 0 else np.nan
    # standardized linear slopes (descriptive)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    A = np.column_stack([one, (soc - soc.mean()) / (soc.std() + 1e-12),
                         (temp - temp.mean()) / (temp.std() + 1e-12)])
    c, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return pr2_soc, pr2_temp, float(c[1]), float(c[2])


# ---- cross-session repeatability on a common SOC grid ----------------------
def grid_interp(df: pd.DataFrame, col: str, branch: str) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate feature and temperature onto SOC_GRID for one branch."""
    sub = df[df["branch"] == branch]
    s = sub["soc_pct_clipped"].to_numpy(float)
    y = sub[col].to_numpy(float)
    t = sub["temperature_c"].to_numpy(float)
    order = np.argsort(s)
    s, y, t = s[order], y[order], t[order]
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() < 10:
        nan = np.full_like(SOC_GRID, np.nan)
        return nan, nan
    f = np.interp(SOC_GRID, s[ok], y[ok], left=np.nan, right=np.nan)
    tt = np.interp(SOC_GRID, s[np.isfinite(s) & np.isfinite(t)],
                   t[np.isfinite(s) & np.isfinite(t)], left=np.nan, right=np.nan)
    return f, tt


def cross_session(dfa: pd.DataFrame, dfb: pd.DataFrame, col: str) -> dict:
    fa_c, ta_c = grid_interp(dfa, col, "charge")
    fb_c, tb_c = grid_interp(dfb, col, "charge")
    fa_d, ta_d = grid_interp(dfa, col, "discharge")
    fb_d, tb_d = grid_interp(dfb, col, "discharge")

    fA = np.concatenate([fa_c, fa_d]); fB = np.concatenate([fb_c, fb_d])
    tA = np.concatenate([ta_c, ta_d]); tB = np.concatenate([tb_c, tb_d])
    dF = fB - fA
    dT = tB - tA
    m = np.isfinite(dF)
    if m.sum() < 20:
        return {k: np.nan for k in ("abs_rms", "shape_rms", "soc_span",
                                    "discriminability", "dFdT", "temp_expl_r2")}
    dF = dF[m]; dT = dT[m]
    mean_curve = np.nanmean(np.column_stack([fA, fB]), axis=1)
    soc_span = float(np.nanmax(mean_curve) - np.nanmin(mean_curve))
    abs_rms = float(np.sqrt(np.mean(dF**2)))
    shape_rms = float(np.sqrt(np.mean((dF - dF.mean()) ** 2)))  # offset removed
    # temperature attribution: dF ~ a + b dT
    okT = np.isfinite(dT)
    if okT.sum() > 10 and np.std(dT[okT]) > 1e-9:
        A = np.column_stack([np.ones(okT.sum()), dT[okT]])
        c, *_ = np.linalg.lstsq(A, dF[okT], rcond=None)
        pred = A @ c
        ss = np.sum((dF[okT] - pred) ** 2)
        st = np.sum((dF[okT] - dF[okT].mean()) ** 2)
        dFdT = float(c[1]); temp_r2 = float(1 - ss / st) if st > 0 else np.nan
    else:
        dFdT, temp_r2 = np.nan, np.nan
    discrim = soc_span / shape_rms if shape_rms > 0 else np.nan
    return {"abs_rms": abs_rms, "shape_rms": shape_rms, "soc_span": soc_span,
            "discriminability": discrim, "dFdT": dFdT, "temp_expl_r2": temp_r2}


# ----------------------------------------------------------------------------
def collinearity(pooled: pd.DataFrame) -> dict:
    out = {}
    for key, sub in [("pooled", pooled)] + [
        (f"{s}/{b}", pooled[(pooled.session == s) & (pooled.branch == b)])
        for s in TABLES for b in ("charge", "discharge")
    ]:
        s = sub["soc_pct_clipped"].to_numpy(float)
        t = sub["temperature_c"].to_numpy(float)
        m = np.isfinite(s) & np.isfinite(t)
        out[key] = float(np.corrcoef(s[m], t[m])[0, 1]) if m.sum() > 5 else np.nan
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    for name, df in data.items():
        df["session"] = name
    pooled = pd.concat(data.values(), ignore_index=True)

    coll = collinearity(pooled)
    feats = [c for c in feature_columns(pooled) if c in data["18-5"].columns
             and c in data["21-5"].columns]

    rows = []
    for col in feats:
        y = pooled[col].to_numpy(float)
        pr2_soc, pr2_temp, b_soc, b_temp = partial_r2(pooled, y)
        rec = {"feature": col, "cross_session_comparable": is_cross_session(col),
               "partial_r2_soc": pr2_soc, "partial_r2_temp": pr2_temp,
               "std_slope_soc": b_soc, "std_slope_temp": b_temp,
               "soc_selectivity": (pr2_soc / (pr2_soc + pr2_temp)
                                   if np.isfinite(pr2_soc) and np.isfinite(pr2_temp)
                                   and (pr2_soc + pr2_temp) > 0 else np.nan)}
        if is_cross_session(col):
            rec.update(cross_session(data["18-5"], data["21-5"], col))
        else:
            rec.update({k: np.nan for k in ("abs_rms", "shape_rms", "soc_span",
                                            "discriminability", "dFdT", "temp_expl_r2")})
        rows.append(rec)

    rank = pd.DataFrame(rows)
    # Composite: reward SOC partials & cross-session discriminability,
    # penalize temperature partial R2. Only meaningful for comparable features.
    d = rank["discriminability"].replace([np.inf, -np.inf], np.nan)
    rank["composite_score"] = (
        rank["partial_r2_soc"].fillna(0)
        * np.log10(d.clip(lower=1).fillna(1))
        * (1.0 - rank["partial_r2_temp"].fillna(1).clip(0, 1))
    )
    rank.loc[~rank["cross_session_comparable"], "composite_score"] = np.nan
    rank = rank.sort_values("composite_score", ascending=False, na_position="last")
    rank.to_csv(OUT / "feature_ranking.csv", index=False)

    make_plots(data, pooled, rank, coll)
    write_report(rank, coll, data)
    print("Wrote", OUT / "feature_ranking.csv")
    print("Wrote", OUT / "REPEATABILITY_REPORT.md")
    print("\nTop 12 SOC-selective, repeatable features:")
    show = rank[rank.cross_session_comparable].head(12)
    print(show[["feature", "partial_r2_soc", "partial_r2_temp", "soc_selectivity",
                "discriminability", "temp_expl_r2", "composite_score"]].to_string(index=False))


# ----------------------------------------------------------------------------
def make_plots(data, pooled, rank, coll) -> None:
    cmp = rank[rank.cross_session_comparable].copy()

    # --- 1. SOC/temp collinearity scatter -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    colors = {("18-5", "charge"): "#1d4ed8", ("18-5", "discharge"): "#60a5fa",
              ("21-5", "charge"): "#c2410c", ("21-5", "discharge"): "#fb923c"}
    for (s, b), c in colors.items():
        sub = pooled[(pooled.session == s) & (pooled.branch == b)]
        ax.scatter(sub["soc_pct_clipped"], sub["temperature_c"], s=4, alpha=0.3,
                   color=c, label=f"{s} {b}")
    ax.set_xlabel("SOC (%)"); ax.set_ylabel("Temperature (C)")
    ax.set_title(f"SOC-temperature coverage  (pooled r = {coll['pooled']:+.2f})")
    ax.legend(fontsize=8, markerscale=2); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "1_soc_temp_collinearity.png"); plt.close(fig)

    # --- 2. selectivity map -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.scatter(cmp["partial_r2_temp"], cmp["partial_r2_soc"], s=28,
               c=cmp["composite_score"], cmap="viridis")
    lim = 1.0
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.8)
    for _, r in cmp.head(10).iterrows():
        ax.annotate(r["feature"], (r["partial_r2_temp"], r["partial_r2_soc"]),
                    fontsize=7, alpha=0.9)
    ax.set_xlabel("partial R2 — temperature (lower is better)")
    ax.set_ylabel("partial R2 — SOC (higher is better)")
    ax.set_title("Feature selectivity: SOC vs temperature\n(top-left = SOC-specific)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "2_selectivity_map.png"); plt.close(fig)

    # --- 3. repeatability overlay of top features ---------------------------
    top = cmp.head(6)["feature"].tolist()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), dpi=150, squeeze=False)
    for ax, col in zip(axs.ravel(), top):
        for s, style in [("18-5", "-"), ("21-5", "--")]:
            for b, c in [("charge", "#2563eb"), ("discharge", "#dc2626")]:
                f, _ = grid_interp(data[s], col, b)
                ax.plot(SOC_GRID, f, style, color=c, lw=1.2,
                        label=f"{s} {b}")
        ax.set_title(col, fontsize=9); ax.set_xlabel("SOC (%)"); ax.grid(alpha=0.25)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Cross-session repeatability of top SOC features (solid=18-5, dashed=21-5)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "3_repeatability_top_features.png"); plt.close(fig)

    # --- 4. composite ranking bar -------------------------------------------
    bar = cmp.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    ax.barh(bar["feature"], bar["composite_score"], color="#047857")
    ax.set_xlabel("composite score  (SOC partial R2 x log10 discriminability x (1 - temp partial R2))")
    ax.set_title("Recommended features for SOC (high = SOC-specific & repeatable)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "4_recommended_features.png"); plt.close(fig)


def write_report(rank, coll, data) -> None:
    cmp = rank[rank.cross_session_comparable]
    lines = [
        "# SOC vs Temperature Feature Sensitivity & Cross-Session Repeatability",
        "",
        "Two repeat runs of the identical cycling protocol (18-5 and 21-5), "
        "restricted to the full charge and full discharge. Features are ranked "
        "by how specifically they track SOC (not temperature) and how well they "
        "reproduce between the two sessions.",
        "",
        "## SOC-temperature collinearity (why the partials are trustworthy)",
        "",
        "| subset | r(SOC, temp) |", "|---|---:|",
    ]
    for k, v in coll.items():
        lines.append(f"| {k} | {v:+.2f} |")
    lines += [
        "",
        f"Pooling both sessions and both branches drops the SOC-temp correlation "
        f"to r={coll['pooled']:+.2f}, so the partial-R2 split between SOC and "
        "temperature is meaningful (they are not strongly collinear in the pooled set).",
        "",
        "## Top features — SOC-specific and repeatable",
        "",
        "| feature | partial R2 SOC | partial R2 temp | SOC selectivity | discriminability | temp explains x-session (R2) | composite |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in cmp.head(15).iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['partial_r2_soc']:.3f} | {r['partial_r2_temp']:.3f} | "
            f"{r['soc_selectivity']:.2f} | {r['discriminability']:.1f} | "
            f"{r['temp_expl_r2']:.2f} | {r['composite_score']:.2f} |")
    lines += [
        "",
        "## How to read the columns",
        "- **partial R2 SOC**: unique feature variance explained by SOC (cubic + branch) after temperature is already in the model. Higher = stronger SOC information.",
        "- **partial R2 temp**: the reverse — unique variance explained by temperature after SOC. Lower = less temperature-contaminated.",
        "- **SOC selectivity** = partial R2 SOC / (SOC + temp). >0.5 means SOC dominates.",
        "- **discriminability** = SOC-driven span / offset-removed cross-session RMS. ~how many distinguishable SOC levels survive run-to-run reproducibility. Higher = more repeatable relative to its useful range.",
        "- **temp explains x-session (R2)**: at matched SOC, how much of the 18-5 vs 21-5 difference is explained by the temperature difference. High here means the run-to-run drift IS temperature (so temperature-compensation would help); low means the residual is other noise/drift.",
        "",
        "## Caveats",
        "- Reference-relative features (ncc/cosine/mse to ref, PCA scores, tracked relative ToF) are excluded from cross-session metrics because their absolute value is session-relative; use `h5_tof_us_absolute` for ToF.",
        "- Cross-session alignment assumes the loggers started together (confirmed within ~1 min).",
        "- A constant per-feature offset is treated as recalibratable (discriminability uses offset-removed RMS); absolute RMS is in the CSV (`abs_rms`) if you need calibration-free reproducibility.",
        "- LFP is hysteretic: charge and discharge are compared within their own branch, so hysteresis is not counted as a repeatability error.",
    ]
    (OUT / "REPEATABILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
