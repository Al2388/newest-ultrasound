"""Full 35C analysis suite -- mirrors the 25C charge-focus suite.

Produces:
  03_beta_soc_map/      per-pixel beta_S = dX/dSOC maps (3 modalities)
  04_pca_svd/           SVD spatial modes + temporal scores (3 modalities)
  07_tab_heterogeneity/ X-band series, beta_S per band, beta_S(x) profile
  08_rest_gradient_decay/ (tab_prox - tab_distal) vs rest time, per plateau

All under reports/longrun_cycling_35c_charge_focus/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_orig_imshow = matplotlib.axes.Axes.imshow


def _patched_imshow(self, X, *args, **kwargs):
    extent = kwargs.get("extent")
    if extent is not None and len(extent) == 4 and extent[2] > extent[3]:
        kwargs["extent"] = [extent[0], extent[1], extent[3], extent[2]]
        kwargs.setdefault("origin", "lower")
    return _orig_imshow(self, X, *args, **kwargs)


matplotlib.axes.Axes.imshow = _patched_imshow

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"

UNITS = {
    "amplitude": ("mV",   1e3, NOISE_FLOOR["amp_sigma_roi_mv"], NOISE_FLOOR["amp_sigma_pixp95_mv"]),
    "tof":       ("ns",   1e3, NOISE_FLOOR["tof_sigma_roi_ns"], NOISE_FLOOR["tof_sigma_pixp95_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"], NOISE_FLOOR["energy_sigma_pixp95"]),
}

REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]
SOC_COLORS = {20: "tab:blue", 40: "tab:green", 60: "tab:orange", 80: "tab:red"}

# Tab geometry from project memory (tabs on X-large edge)
SUBREGIONS = {
    "tab-distal":   (14.0, 30.0),
    "interior":     (30.0, 50.0),
    "tab-proximal": (50.0, 64.5),
}


def _roi_mean(arr, mask):
    return np.nanmean(arr.reshape(arr.shape[0], -1)[:, mask.reshape(-1)], axis=1)


def _xband_in_roi(roi, x_mm, x_lo, x_hi):
    cl = int(np.searchsorted(x_mm, x_lo)); ch = int(np.searchsorted(x_mm, x_hi))
    m = np.zeros_like(roi, dtype=bool)
    m[:, cl:ch] = True
    return m & roi


def _roi_bounds(roi, x_mm, y_mm):
    rows = np.where(roi.any(axis=1))[0]
    cols = np.where(roi.any(axis=0))[0]
    return (float(x_mm[cols.min()]), float(x_mm[cols.max()]),
            float(y_mm[rows.min()]), float(y_mm[rows.max()]))


def load_data():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    return {
        "amp": d["amplitude"], "tof": d["tof"], "eng": d["energy"],
        "x_mm": d["x_mm"], "y_mm": d["y_mm"], "roi": d["roi_mask"],
        "meta": meta,
    }


# ============================================================
# 03. Per-pixel beta_S map  (dX/dSOC at constant T)
# ============================================================
def analysis_03_beta_S(D):
    out = OUT_ROOT / "03_beta_soc_map"
    out.mkdir(exist_ok=True)
    print(f"\n[03] beta_S per-pixel maps -> {out.name}/")

    meta = D["meta"]; roi = D["roi"]
    rest_mask = meta["step"].isin(REST_STEPS).values
    rest_idx = np.where(rest_mask)[0]
    soc = meta.iloc[rest_idx]["soc_plateau_label"].values.astype(float)
    Tc = meta.iloc[rest_idx]["line_T_mean_c"].values.astype(float)

    arrs = {"amplitude": D["amp"], "tof": D["tof"], "energy": D["eng"]}

    for mod, arr3d in arrs.items():
        unit, scale, sigma_roi, sigma_pix_p95 = UNITS[mod]
        sigma_pix_p95_native = sigma_pix_p95 / scale     # convert from "mV/ns" back to V/us
        flat = arr3d[rest_idx].reshape(len(rest_idx), -1).astype(np.float64)
        mask_flat = roi.reshape(-1)
        Y = flat[:, mask_flat]
        finite = np.isfinite(Y).all(axis=0)
        Y_fit = Y[:, finite]

        # Design: f = beta_S * SOC + beta_T * T + c
        X = np.column_stack([soc, Tc, np.ones_like(soc)])
        coeffs, *_ = np.linalg.lstsq(X, Y_fit, rcond=None)
        beta_S = coeffs[0]  # shape (n_finite,)

        # Scatter back to full map
        full = np.full(mask_flat.sum(), np.nan, dtype=np.float64)
        full[finite] = beta_S
        beta_map = np.full(roi.shape, np.nan, dtype=np.float64)
        beta_map[roi] = full

        # Display in native plot units
        beta_map_disp = beta_map * scale

        # Significance: |beta_S * SOC_range| > 2 * sigma_pix_p95 (in plot units)
        soc_range = float(soc.max() - soc.min())   # 60
        signif = np.abs(beta_map_disp * soc_range) > 2 * sigma_pix_p95

        x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi, D["x_mm"], D["y_mm"])
        extent = [D["x_mm"].min(), D["x_mm"].max(), D["y_mm"].max(), D["y_mm"].min()]
        vmax = float(np.nanpercentile(np.abs(beta_map_disp), 98))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        ax = axes[0]
        im = ax.imshow(beta_map_disp, extent=extent, aspect="equal",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo],
                "k--", lw=0.6, alpha=0.7)
        ax.set_title(f"β_S  ({unit}/%SoC)\nROI mean = {np.nanmean(beta_map_disp[roi]):+.4f} {unit}/%SoC",
                     fontsize=10)
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
        fig.colorbar(im, ax=ax, label=f"β_S [{unit}/%SoC]")

        ax = axes[1]
        ax.imshow(signif.astype(float), extent=extent, aspect="equal",
                  cmap="Greys", vmin=0, vmax=1)
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo],
                "r--", lw=0.6, alpha=0.7)
        frac = float(np.nanmean(signif[roi]))
        ax.set_title(f"|β_S × ΔSOC| > 2·σ_p95   ({frac*100:.1f}% of ROI pixels)", fontsize=10)
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

        fig.suptitle(f"35°C  per-pixel β_S map  ·  {mod}  (fit on {len(rest_idx)} rest scans, "
                     "with simultaneous β_T regression)", fontsize=11)
        fig.savefig(out / f"beta_S_{mod}.png", dpi=130)
        plt.close(fig)
        print(f"  {mod:<10s} beta_S map: ROI mean = {np.nanmean(beta_map_disp[roi]):+.4g} {unit}/%SoC,  "
              f"significant fraction = {frac*100:.1f}%")


# ============================================================
# 04. PCA / SVD on full 93-scan stack
# ============================================================
def analysis_04_pca(D):
    out = OUT_ROOT / "04_pca_svd"
    out.mkdir(exist_ok=True)
    print(f"\n[04] PCA / SVD -> {out.name}/")

    meta = D["meta"]; roi = D["roi"]
    arrs = {"amplitude": D["amp"], "tof": D["tof"], "energy": D["eng"]}
    n_t = D["amp"].shape[0]
    t = (meta["timestamp"] - meta["timestamp"].iloc[0]).dt.total_seconds().values / 3600.0  # hours

    x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi, D["x_mm"], D["y_mm"])
    extent = [D["x_mm"].min(), D["x_mm"].max(), D["y_mm"].max(), D["y_mm"].min()]

    for mod, arr3d in arrs.items():
        unit, scale, *_ = UNITS[mod]
        mask_flat = roi.reshape(-1)
        flat_roi = arr3d.reshape(n_t, -1)[:, mask_flat]
        finite = np.isfinite(flat_roi).all(axis=0)
        X_in = flat_roi[:, finite].astype(np.float64)
        mean = X_in.mean(axis=0, keepdims=True)
        Xc = X_in - mean
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        expl = s**2 / (s**2).sum()
        k = 6
        scores = U[:, :k] * s[:k]                    # (n_t, k)
        loadings = Vt[:k]                            # (k, n_finite)

        # 6-mode panel: top row scree+top mode score, then 2x3 spatial modes
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.6, 1.6])

        # Scree
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(range(1, 11), expl[:10] * 100, color="tab:gray")
        ax.set_xlabel("mode"); ax.set_ylabel("explained variance [%]")
        ax.set_title(f"scree  (top 1: {expl[0]*100:.1f}%, top 3: {expl[:3].sum()*100:.1f}%)",
                     fontsize=9)
        ax.grid(alpha=0.3)

        # Temporal scores (top 3)
        for i, ax_col in enumerate([1, 2, 3]):
            ax = fig.add_subplot(gs[0, ax_col])
            ax.plot(t, scores[:, i], "o-", color="tab:blue", lw=1, ms=3)
            # Mark plateau steps
            for step_label, soc_label, color in zip(REST_STEPS, SOC_LABELS,
                                                    [SOC_COLORS[s] for s in SOC_LABELS]):
                sel = meta[meta["step"] == step_label]
                if not sel.empty:
                    t0 = t[sel.index[0]]; t1 = t[sel.index[-1]]
                    ax.axvspan(t0, t1, color=color, alpha=0.12,
                               label=f"SOC {soc_label}%" if i == 0 else None)
            ax.set_title(f"PC{i+1} score  ({expl[i]*100:.1f}%)", fontsize=9)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=6, loc="best")
            ax.set_xlabel("time [h]")

        # Spatial modes (top 6 in 2x3)
        for i in range(6):
            ax = fig.add_subplot(gs[1 + i // 3, i % 3])
            full = np.full(mask_flat.sum(), np.nan, dtype=np.float64)
            full[finite] = loadings[i]
            mode_map = np.full(roi.shape, np.nan, dtype=np.float64)
            mode_map[roi] = full
            vmax = float(np.nanpercentile(np.abs(mode_map), 99))
            im = ax.imshow(mode_map, extent=extent, aspect="equal",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo],
                    [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=0.6, alpha=0.7)
            ax.set_title(f"mode {i+1}  ({expl[i]*100:.1f}%)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"35°C  PCA / SVD on {n_t} scans  ·  {mod}  "
                     f"(mean-centred per pixel, then SVD)", fontsize=11)
        fig.savefig(out / f"pca_{mod}.png", dpi=130)
        plt.close(fig)
        print(f"  {mod:<10s} top-3 expl var = "
              f"{expl[0]*100:.1f}% / {expl[1]*100:.1f}% / {expl[2]*100:.1f}%")


# ============================================================
# 07. Tab heterogeneity: per-band β_S + β_S(x) profile
# ============================================================
def analysis_07_tab_het(D):
    out = OUT_ROOT / "07_tab_heterogeneity"
    out.mkdir(exist_ok=True)
    print(f"\n[07] tab heterogeneity -> {out.name}/")

    meta = D["meta"]; roi = D["roi"]
    subregions = {n: _xband_in_roi(roi, D["x_mm"], lo, hi)
                  for n, (lo, hi) in SUBREGIONS.items()}
    arrs = {"amplitude": D["amp"], "tof": D["tof"], "energy": D["eng"]}

    # Use rest equilibrium = last 3 scans of each plateau, per band, per modality
    eq_per_plateau = {soc: {} for soc in SOC_LABELS}
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        sel = meta[meta["step"] == step].sort_values("timestamp").index.values
        last3 = sel[-3:]
        for mod, arr3d in arrs.items():
            for name, m in subregions.items():
                vals = _roi_mean(arr3d[last3], m)
                eq_per_plateau[soc][(mod, name)] = float(np.mean(vals))

    # Plot ROI-mean per band vs SOC (3 modalities)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    band_colors = {"tab-distal": "tab:blue", "interior": "tab:green", "tab-proximal": "tab:red"}
    beta_table = []
    for ax, mod in zip(axes, ["amplitude", "tof", "energy"]):
        unit, scale, sigma_roi, _ = UNITS[mod]
        for name in SUBREGIONS:
            ys = np.array([eq_per_plateau[soc][(mod, name)] for soc in SOC_LABELS]) * scale
            slope, intercept = np.polyfit(SOC_LABELS, ys, 1)
            ax.plot(SOC_LABELS, ys, "o-", color=band_colors[name], ms=8,
                    label=f"{name}: β_S = {slope:+.4f} {unit}/%SoC")
            beta_table.append({"modality": mod, "band": name, "beta_S": slope, "unit": unit})
        ax.set_xlabel("SOC [%]")
        ax.set_ylabel(f"end-of-rest mean {mod} [{unit}]")
        ax.set_title(f"{mod}  (σ_ROI = {sigma_roi:.2g})")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    fig.suptitle("35°C  tab-distal / interior / tab-proximal  ·  equilibrium vs SOC (4 rest plateaus)",
                 fontsize=11)
    fig.savefig(out / "band_series.png", dpi=130)
    plt.close(fig)

    # β_S(x) profile across X: for each x-column slice, compute β_S from 4 equilibria
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, mod in zip(axes, ["amplitude", "tof", "energy"]):
        unit, scale, sigma_roi, sigma_pix_p95 = UNITS[mod]
        arr3d = arrs[mod]
        # equilibrium maps per plateau (mean of last 3 scans)
        eq_maps = []
        for step in REST_STEPS:
            sel = meta[meta["step"] == step].sort_values("timestamp").index.values
            eq_maps.append(np.nanmean(arr3d[sel[-3:]], axis=0))
        eq_stack = np.stack(eq_maps)              # (4, ny, nx)
        # per-column fit (mask by roi columns)
        nx = eq_stack.shape[-1]
        beta_x = np.full(nx, np.nan)
        for col in range(nx):
            col_mask_y = roi[:, col]
            if col_mask_y.sum() < 5: continue
            ys = np.nanmean(eq_stack[:, col_mask_y, col], axis=1)
            if not np.all(np.isfinite(ys)): continue
            slope, _ = np.polyfit(SOC_LABELS, ys, 1)
            beta_x[col] = slope * scale         # in plot units per %SoC
        ax.plot(D["x_mm"], beta_x, "-", color="tab:blue", lw=1.3)
        ax.fill_between(D["x_mm"], -2*sigma_pix_p95/60, 2*sigma_pix_p95/60,
                        color="gray", alpha=0.2,
                        label=f"±2σ_p95 / 60")
        for name, (lo, hi) in SUBREGIONS.items():
            ax.axvspan(lo, hi, color=band_colors[name], alpha=0.08)
        ax.set_xlabel("X [mm]   (tab side -> larger X)")
        ax.set_ylabel(f"β_S [{unit}/%SoC]  (Y-averaged within ROI)")
        ax.set_title(f"{mod}", fontsize=10)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.suptitle("35°C  β_S(x) profile across cell (Y-averaged inside ROI per column)", fontsize=11)
    fig.savefig(out / "betaS_x_profile.png", dpi=130)
    plt.close(fig)

    # text summary
    df = pd.DataFrame(beta_table)
    df.to_csv(out / "beta_S_per_band.csv", index=False)
    print(f"  band-level beta_S table -> beta_S_per_band.csv")
    for mod in ["amplitude", "tof", "energy"]:
        sub = df[df.modality == mod]
        print(f"  {mod:<10s}: " + ", ".join(
            f"{r['band']} beta_S={r['beta_S']:+.4f} {r['unit']}/%SoC"
            for _, r in sub.iterrows()))


# ============================================================
# 08. Rest gradient decay: (tab_prox - tab_distal) vs rest time
# ============================================================
def analysis_08_grad_decay(D):
    out = OUT_ROOT / "08_rest_gradient_decay"
    out.mkdir(exist_ok=True)
    print(f"\n[08] rest gradient decay -> {out.name}/")

    meta = D["meta"]; roi = D["roi"]
    subregions = {n: _xband_in_roi(roi, D["x_mm"], lo, hi)
                  for n, (lo, hi) in SUBREGIONS.items()}
    arrs = {"amplitude": D["amp"], "tof": D["tof"], "energy": D["eng"]}

    rows = []
    for mod, arr3d in arrs.items():
        unit, scale, sigma_roi, _ = UNITS[mod]
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.0), constrained_layout=True, sharex=True)
        for col, (step, soc) in enumerate(zip(REST_STEPS, SOC_LABELS)):
            sel = meta[meta["step"] == step].sort_values("timestamp")
            idx = sel.index.values
            t = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
            prox = _roi_mean(arr3d[idx], subregions["tab-proximal"]) * scale
            dist = _roi_mean(arr3d[idx], subregions["tab-distal"])   * scale
            grad = prox - dist
            grad_d = grad - grad[0]
            # linear fit on (t, grad_d)
            slope, intercept = np.polyfit(t, grad_d, 1)
            fit = slope * t + intercept
            ax = axes[col]
            ax.plot(t, grad_d, "o-", color=SOC_COLORS[soc], lw=1.3, ms=5,
                    label="(prox − distal) Δ")
            ax.plot(t, fit, "--", color="k", lw=1, alpha=0.7,
                    label=f"slope {slope:+.4f} {unit}/min")
            ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
            ax.fill_between(t, -2*sigma_roi, 2*sigma_roi, color="gray", alpha=0.15)
            ax.set_title(f"SOC {soc}% rest", fontsize=10, color=SOC_COLORS[soc], fontweight="bold")
            ax.set_xlabel("rest time [min]")
            if col == 0:
                ax.set_ylabel(f"Δ(prox − distal) [{unit}]\n(σ_ROI = {sigma_roi:.2g})")
            ax.legend(fontsize=7, loc="best")
            rows.append({"modality": mod, "SOC%": soc,
                         "slope_per_min": float(slope),
                         "end_value": float(grad_d[-1]),
                         "end_value_sigma": float(grad_d[-1] / sigma_roi),
                         "unit": unit})
        fig.suptitle(f"35°C  rest gradient decay  ·  {mod}  "
                     "(tab-prox − tab-distal Δ since rest start; linear trend fit)",
                     fontsize=11)
        fig.savefig(out / f"decay_{mod}.png", dpi=130)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out / "gradient_decay_summary.csv", index=False)

    # summary plot: slope vs SOC for each modality
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    for ax, mod in zip(axes, ["amplitude", "tof", "energy"]):
        unit = UNITS[mod][0]
        sub = df[df.modality == mod]
        ax.bar(sub["SOC%"].astype(str), sub["slope_per_min"], color=[SOC_COLORS[s] for s in sub["SOC%"]])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("SOC plateau")
        ax.set_ylabel(f"linear slope [{unit}/min]")
        ax.set_title(f"{mod}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("35°C  (prox − distal) gradient slope per plateau", fontsize=11)
    fig.savefig(out / "summary_vs_plateau.png", dpi=130)
    plt.close(fig)
    print(f"  per-modality decay panels + summary_vs_plateau.png + CSV written")


def main():
    print("=" * 60)
    print("35°C full analysis suite (mirroring 25°C charge-focus)")
    print("=" * 60)
    D = load_data()
    print(f"\nLoaded: stack n_t={D['amp'].shape[0]}, ROI px={int(D['roi'].sum())}")

    analysis_03_beta_S(D)
    analysis_04_pca(D)
    analysis_07_tab_het(D)
    analysis_08_grad_decay(D)


if __name__ == "__main__":
    main()
