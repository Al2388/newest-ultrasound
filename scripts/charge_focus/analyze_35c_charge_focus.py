"""Run paper-grade analyses on 35°C charge-focus subset.

Outputs:
- within_rest_evolution_4plateaus.png    — ΔV/Δamp/ΔToF/Δenergy vs rest time for 4 plateaus
- within_rest_reference_subtracted.png   — cell-specific signal at each plateau
- equilibrium_vs_soc.png                 — end-of-rest ROI mean vs SOC plateau
- equilibrium_maps_<mod>.png             — 4-panel equilibrium maps per modality
- consecutive_diff_<mod>.png             — equilibrium plateau differences
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
    "amplitude": ("mV", 1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns", 1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}


def _xy_band(roi, x_mm, y_mm, x_lo, x_hi, y_lo, y_hi, exclude=True):
    cl = int(np.searchsorted(x_mm, x_lo)); ch = int(np.searchsorted(x_mm, x_hi))
    rl = int(np.searchsorted(y_mm, y_lo)); rh = int(np.searchsorted(y_mm, y_hi))
    m = np.zeros_like(roi, dtype=bool)
    m[rl:rh, cl:ch] = True
    if exclude:
        m &= ~roi
    return m


def _roi_bounds(roi, x_mm, y_mm):
    rows = np.where(roi.any(axis=1))[0]; cols = np.where(roi.any(axis=0))[0]
    return float(x_mm[cols.min()]), float(x_mm[cols.max()]), float(y_mm[rows.min()]), float(y_mm[rows.max()])


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    amp = d["amplitude"]; tof = d["tof"]; energy = d["energy"]
    x_mm = d["x_mm"]; y_mm = d["y_mm"]; roi = d["roi_mask"]
    arrs = {"amplitude": amp, "tof": tof, "energy": energy}

    refs = {
        "far-left":  _xy_band(roi, x_mm, y_mm, 0, 10, 0, 72),
        "far-right": _xy_band(roi, x_mm, y_mm, 70, 80, 0, 72),
        "far-top":   _xy_band(roi, x_mm, y_mm, 0, 80, 0, 12),
        "far-bot":   _xy_band(roi, x_mm, y_mm, 0, 80, 58, 72),
    }

    def band_series(arr, mask):
        return np.nanmean(arr.reshape(arr.shape[0], -1)[:, mask.reshape(-1)], axis=1)

    # ============== Per-plateau within-rest evolution ==============
    rest_steps = [5, 7, 9, 11]
    soc_labels = [20, 40, 60, 80]

    fig, axes = plt.subplots(4, 4, figsize=(16, 11), constrained_layout=True, sharex=True)
    summary = ["# 35°C charge-focus analyses (SOC 20→80 + rests)\n\n",
               "## Within-rest evolution per plateau\n\n",
               "All rests are ~117 min, 13 scans. End-of-rest drift vs σ_ROI noise floor.\n\n"]

    plateau_tables = {mod: [] for mod in ["amplitude", "tof", "energy"]}

    for col, (step, soc) in enumerate(zip(rest_steps, soc_labels)):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        idx = sel.index.values
        t = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0

        # Row 0: voltage
        V = sel["voltage_at_scan"].values * 1000
        dV = V - V[0]
        T_C = sel["line_T_mean_c"].values
        dT_mC = (T_C - T_C[0]) * 1000

        ax = axes[0, col]
        ax2 = ax.twinx()
        l1, = ax.plot(t, dV, "o-", color="tab:red", lw=1.4, ms=5, label="ΔV [mV]")
        l2, = ax2.plot(t, dT_mC, "s--", color="tab:purple", lw=0.8, ms=4, alpha=0.7, label="ΔT [mC]")
        ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
        ax.set_title(f"SOC ≈ {soc}%  rest (step {step})", fontsize=10)
        if col == 0:
            ax.set_ylabel("ΔV (red) [mV]\nΔT (purple) [mC]")

        # Rows 1-3: acoustic modalities with cell-specific (ROI − mean reference)
        for r, mod in enumerate(["amplitude", "tof", "energy"], start=1):
            unit, scale, sigma = UNITS[mod]
            roi_series = band_series(arrs[mod][idx], roi) * scale
            ref_means = []
            for label, m in refs.items():
                ref_means.append(band_series(arrs[mod][idx], m) * scale)
            ref_mean_series = np.mean(ref_means, axis=0)
            d_roi = roi_series - roi_series[0]
            d_ref = ref_mean_series - ref_mean_series[0]
            d_specific = d_roi - d_ref

            ax = axes[r, col]
            ax.plot(t, d_roi, "o-", color="tab:gray", lw=1, ms=4, label="raw cell ROI")
            ax.plot(t, d_ref, "o--", color="tab:olive", lw=0.8, ms=3, label="mean of 4 refs", alpha=0.7)
            ax.plot(t, d_specific, "o-", color="tab:red", lw=1.8, ms=5, label="cell − refs")
            ax.fill_between(t, -2*sigma, 2*sigma, color="gray", alpha=0.15)
            ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"Δ{mod} [{unit}]\n(σ_ROI = {sigma:.2g})")
            if r == 1 and col == 3:
                ax.legend(fontsize=6.5, loc="best")

            plateau_tables[mod].append({
                "SOC%": soc, "ΔV_end_mV": float(dV[-1]), "ΔT_end_mC": float(dT_mC[-1]),
                "cell_raw_end": float(d_roi[-1]), "mean_ref_end": float(d_ref[-1]),
                "cell_specific_end": float(d_specific[-1]),
                "cell_specific_sigma": float(d_specific[-1]/sigma),
                "rate_late_early": float(np.mean(np.abs(np.diff(roi_series))[-3:]) / max(np.mean(np.abs(np.diff(roi_series))[:3]), 1e-9)),
            })
        axes[-1, col].set_xlabel("rest time [min]")

    fig.suptitle("35°C — within-rest evolution at 4 SOC plateaus  (cell-specific = ROI − mean of 4 reference regions)", fontsize=12)
    fig.savefig(OUT_ROOT / "within_rest_evolution_4plateaus.png", dpi=130)
    plt.close(fig)

    # Tables
    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, sigma = UNITS[mod]
        summary.append(f"### {mod}  (σ_ROI = {sigma:.3g} {unit})\n\n")
        summary.append("| SOC | ΔV (mV) | ΔT (mC) | cell raw end | ref mean end | cell − ref end | end-z (σ_ROI) | rate late/early |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in plateau_tables[mod]:
            summary.append(f"| {row['SOC%']}% | {row['ΔV_end_mV']:+.1f} | {row['ΔT_end_mC']:+.1f} | {row['cell_raw_end']:+.3g} | {row['mean_ref_end']:+.3g} | {row['cell_specific_end']:+.3g} | {row['cell_specific_sigma']:+.1f} | {row['rate_late_early']:.2f} |\n")
        summary.append("\n")

    # ============== Equilibrium maps ==============
    end_idx = []
    end_socs = []
    for step, soc in zip(rest_steps, soc_labels):
        sel = meta[meta["step"] == step].sort_values("timestamp")
        end_idx.append(sel.index.values[-1])
        end_socs.append(soc)
    end_idx = np.array(end_idx)
    end_socs = np.array(end_socs)

    # Equilibrium maps per modality (4-panel)
    x_lo, x_hi, y_lo, y_hi = _roi_bounds(roi, x_mm, y_mm)
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]

    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, _ = UNITS[mod]
        arr = arrs[mod][end_idx]
        vmin = float(np.nanpercentile(arr[:, roi], 2))
        vmax = float(np.nanpercentile(arr[:, roi], 98))
        fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), constrained_layout=True)
        for i, ax in enumerate(axes):
            im = ax.imshow(arr[i], extent=extent, aspect="equal", cmap="turbo", vmin=vmin, vmax=vmax)
            ax.plot([x_lo,x_hi,x_hi,x_lo,x_lo], [y_lo,y_lo,y_hi,y_hi,y_lo], "k--", lw=0.6, alpha=0.7)
            ax.set_title(f"SOC = {end_socs[i]}%", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.85, location="right", label=mod)
        fig.suptitle(f"35°C equilibrium maps — {mod}  (end-of-rest at each plateau)", fontsize=11)
        fig.savefig(OUT_ROOT / f"equilibrium_maps_{mod}.png", dpi=130)
        plt.close(fig)

    # Consecutive equilibrium diff (4 SOCs → 3 pair diffs)
    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, sigma = UNITS[mod]
        arr = arrs[mod][end_idx]
        diffs = [(arr[i+1] - arr[i]) * scale for i in range(3)]
        vmax = max(float(np.nanpercentile(np.abs(c[roi]), 99)) for c in diffs)
        vmax = max(vmax, 3 * sigma)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
        for i, ax in enumerate(axes):
            im = ax.imshow(diffs[i], extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.plot([x_lo,x_hi,x_hi,x_lo,x_lo], [y_lo,y_lo,y_hi,y_hi,y_lo], "k--", lw=0.6, alpha=0.7)
            roi_mean = float(np.nanmean(diffs[i][roi]))
            z = roi_mean/sigma
            ax.set_title(f"SOC {end_socs[i]} → {end_socs[i+1]}%\nΔROI = {roi_mean:+.3g} {unit} ({z:+.1f}σ)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.85, location="right", label=f"Δ{mod} [{unit}]")
        fig.suptitle(f"35°C consecutive equilibrium differences — {mod}", fontsize=11)
        fig.savefig(OUT_ROOT / f"consecutive_diff_{mod}.png", dpi=130)
        plt.close(fig)

    # ============== Equilibrium vs SOC plot (3 modalities) ==============
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, mod in zip(axes, ["amplitude", "tof", "energy"]):
        unit, scale, sigma = UNITS[mod]
        arr = arrs[mod][end_idx]
        roi_mean = np.array([float(np.nanmean(a[roi])) for a in arr]) * scale
        slope, intercept = np.polyfit(end_socs, roi_mean, 1)
        line = np.linspace(end_socs.min(), end_socs.max(), 50)
        ax.plot(end_socs, roi_mean, "o-", color="#222", ms=8, lw=1.5)
        ax.plot(line, slope*line+intercept, "--", color="tab:blue", alpha=0.7,
                label=f"linear: {slope:+.4f} {unit}/%SoC")
        for i, soc in enumerate(end_socs):
            ax.annotate(f"SOC {soc}%", (soc, roi_mean[i]), fontsize=8, xytext=(5,5), textcoords="offset points")
        ax.set_xlabel("SOC [%]")
        ax.set_ylabel(f"end-of-rest ROI mean {mod} [{unit}]")
        ax.set_title(f"{mod}  (σ_ROI={sigma:.2g})")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
        # residuals
        resid = roi_mean - (slope*end_socs+intercept)
        zmax = np.max(np.abs(resid))/sigma
        summary.append(f"\n## Equilibrium {mod}: slope = {slope:+.4f} {unit}/%SoC, max residual = {zmax:.1f} σ_ROI\n")
    fig.suptitle("35°C — end-of-rest equilibrium ROI mean vs SOC", fontsize=12)
    fig.savefig(OUT_ROOT / "equilibrium_vs_soc.png", dpi=130)
    plt.close(fig)

    with (OUT_ROOT / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n\n" + "".join(summary))
    print(f"wrote outputs to {OUT_ROOT}/")


if __name__ == "__main__":
    main()
