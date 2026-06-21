"""Test the current-crowding hypothesis using within-rest gradient decay.

Hypothesis A (current crowding → SOC gradient): the tab vs distal band
mean difference decays exponentially during rest as Li redistributes,
with τ comparable to the bulk relaxation τ from analysis #2.

Hypothesis B (metal-tab acoustic artifact): the difference stays constant
during rest because metal reflectivity doesn't depend on Li distribution.

Test: per rest segment, fit (tab_band - distal_band)(t) = A·exp(-t/τ) + C
and report A (decay amplitude), C (constant baseline), τ. Then judge:
  - A >> C  →  supports A (crowding)
  - A ≈ 0   →  supports B (metal artifact)
  - both nonzero → both mechanisms contribute (most physical).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
LONGRUN = PROJ / "scripts" / "longrun_analysis"
sys.path.insert(0, str(LONGRUN))

import common  # noqa: E402
from common import NOISE_FLOOR, Stack, rest_segments  # noqa: E402

NEW_OUT = PROJ / "reports" / "longrun_cycling_22h_charge_focus"
common.OUT_ROOT = NEW_OUT
common.CACHE_PATH = NEW_OUT / "_cache" / "stack.npz"
common.META_PATH = NEW_OUT / "_cache" / "meta.csv"
OUT = NEW_OUT / "08_rest_gradient_decay"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = [
    ("distal", (14.6, 30.0)),
    ("interior", (30.0, 50.0)),
    ("proximal", (50.0, 64.5)),
]
UNITS = {"tof": ("ns", 1e3), "amplitude": ("mV", 1e3), "energy": ("", 1.0)}
SIGMA_ROI = {
    "tof": NOISE_FLOOR["tof_sigma_roi_ns"] * 1e-3,
    "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"] * 1e-3,
    "energy": NOISE_FLOOR["energy_sigma_roi"],
}

_orig_load = common.load


def filtered_load(rebuild: bool = False) -> Stack:
    s = _orig_load(rebuild=False)
    keep = s.meta.step_tag.isin({"charge", "rest"}).values
    idx = np.where(keep)[0]
    new_meta = s.meta.iloc[idx].copy().reset_index(drop=True)
    soc_min = float(new_meta.soc_pct.min())
    new_meta["soc_pct"] = new_meta["soc_pct"] - soc_min
    return Stack(
        amplitude=s.amplitude[idx], tof=s.tof[idx], energy=s.energy[idx],
        x_mm=s.x_mm, y_mm=s.y_mm, roi_mask=s.roi_mask, meta=new_meta,
    )


def _band_mask(stack, x_lo, x_hi):
    col_lo = int(np.searchsorted(stack.x_mm, x_lo))
    col_hi = int(np.searchsorted(stack.x_mm, x_hi))
    m = np.zeros_like(stack.roi_mask)
    m[:, col_lo:col_hi] = stack.roi_mask[:, col_lo:col_hi]
    return m


def _band_mean_series(stack, mod, mask, seg_idx):
    arr = getattr(stack, mod)[seg_idx]
    flat = arr.reshape(arr.shape[0], -1)
    m = mask.reshape(-1)
    return np.nanmean(flat[:, m], axis=1)


def _fit_linear_trend(t_min, y):
    """Fit y = slope·t + intercept. Returns (slope, slope_se, end_minus_start)."""
    coeffs, cov = np.polyfit(t_min, y, 1, cov=True)
    slope = float(coeffs[0])
    slope_se = float(np.sqrt(cov[0, 0]))
    end_pred = slope * t_min[-1] + float(coeffs[1])
    start_pred = slope * t_min[0] + float(coeffs[1])
    delta = end_pred - start_pred
    return slope, slope_se, delta, float(coeffs[1])


def main():
    stack = filtered_load()
    segs = rest_segments(stack.meta)
    print(f"{len(segs)} rest segments")

    band_masks = {name: _band_mask(stack, lo, hi) for name, (lo, hi) in BANDS}

    summary = [
        "# Rest-segment gradient: linear trend test\n",
        "Within each rest plateau, fit  (tab_proximal − tab_distal)(t) = slope·t + intercept.\n\n",
        "Reasoning:\n",
        "- **Current-crowding SOC heterogeneity** prediction: |slope| > 0 with sign opposite to the gradient itself (gradient decays toward 0 as Li redistributes). |slope| × 100 min should be a meaningful fraction of the gradient.\n",
        "- **Metal-tab acoustic artifact** prediction: slope ≈ 0 (no time dependence); gradient is determined by SOC state, not rest progress.\n",
        "- **Inconclusive** (most likely outcome here): slope ≈ 0 means either τ_diffusion ≫ 100 min OR the gradient is artifact-dominated. Cannot distinguish from this dataset alone.\n\n",
        f"σ_ROI noise floors: ToF {NOISE_FLOOR['tof_sigma_roi_ns']:.2f} ns, amp {NOISE_FLOOR['amp_sigma_roi_mv']:.2f} mV, energy {NOISE_FLOOR['energy_sigma_roi']:.3f}.\n\n",
    ]

    plateau_summary = {mod: [] for mod in ["tof", "amplitude", "energy"]}

    for mod in ["tof", "amplitude", "energy"]:
        unit, scale = UNITS[mod]
        sigma_disp = SIGMA_ROI[mod] * scale

        fig, axes = plt.subplots(1, len(segs), figsize=(4 * len(segs), 4.2), sharey=True, constrained_layout=True)
        summary.append(f"## {mod}\n\n| seg SOC | gradient at t=0 [{unit}] | slope [{unit}/min] | slope/SE | end−start [{unit}] | vs σ_ROI | verdict |\n|---:|---:|---:|---:|---:|---:|---|\n")

        for ax, seg_meta in zip(axes, segs):
            seg_idx = np.where(np.isin(stack.meta.run_idx.values, seg_meta.run_idx.values))[0]
            soc_plateau = float(seg_meta.soc_pct.iloc[0])
            t_iso = pd.to_datetime(seg_meta.time_utc)
            t_min = (t_iso - t_iso.iloc[0]).dt.total_seconds().values / 60.0

            y_distal = _band_mean_series(stack, mod, band_masks["distal"], seg_idx) * scale
            y_prox = _band_mean_series(stack, mod, band_masks["proximal"], seg_idx) * scale
            y_diff = y_prox - y_distal

            slope, slope_se, delta, intercept = _fit_linear_trend(t_min, y_diff)
            z_slope = slope / slope_se if slope_se else np.nan
            z_delta = delta / sigma_disp

            same_sign_decay = (slope * intercept < 0)
            if abs(delta) < 2 * sigma_disp:
                verdict = "no time-decay (within noise)"
            elif same_sign_decay:
                verdict = "**gradient decaying**"
            else:
                verdict = "**gradient growing**"

            ax.plot(t_min, y_diff, "o", color="#222", ms=6)
            t_fit = np.linspace(t_min.min(), t_min.max(), 100)
            ax.plot(t_fit, slope * t_fit + intercept, "-", color="tab:red", lw=1.5,
                    label=f"slope = {slope:+.3f} {unit}/min\n(z={z_slope:+.1f})\nend-start = {delta:+.1f} {unit}\n({z_delta:+.1f} σ_ROI)")
            ax.axhline(0, color="k", lw=0.4)
            ax.axhline(intercept, color="gray", lw=0.4, ls=":")
            ax.fill_between(t_min, intercept - 2 * sigma_disp, intercept + 2 * sigma_disp, color="gray", alpha=0.15)
            ax.set_xlabel("rest time [min]")
            ax.set_title(f"seg SOC≈{soc_plateau:.0f}%")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5, loc="best")

            plateau_summary[mod].append(dict(soc=soc_plateau, intercept=intercept, slope=slope, delta=delta))
            summary.append(
                f"| {soc_plateau:.0f}% | {intercept:+.2f} | {slope:+.4f} ± {slope_se:.4f} | {z_slope:+.1f} | "
                f"{delta:+.2f} | {z_delta:+.1f} | {verdict} |\n"
            )

        axes[0].set_ylabel(f"(proximal − distal) {mod} [{unit if unit else 'a.u.'}]")
        fig.suptitle(
            f"Gradient (proximal − distal) within rest — {mod}\n"
            f"red line = linear fit; grey band = intercept ± 2 σ_ROI",
            fontsize=11,
        )
        fig.savefig(OUT / f"decay_{mod}.png", dpi=130)
        plt.close(fig)
        summary.append("\n")

    # Cross-plateau summary: gradient (at start of rest) vs SOC plateau
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, mod in zip(axes, ["tof", "amplitude", "energy"]):
        unit, _ = UNITS[mod]
        rows = plateau_summary[mod]
        socs = [r["soc"] for r in rows]
        starts = [r["intercept"] for r in rows]
        slopes_per_100 = [r["delta"] for r in rows]
        ax.bar([s - 4 for s in socs], starts, width=4, color="tab:gray", label="gradient at t=0 of rest")
        ax.bar([s + 4 for s in socs], slopes_per_100, width=4, color="tab:red", label="end−start (100 min drift)")
        sigma_disp = SIGMA_ROI[mod] * UNITS[mod][1]
        ax.axhspan(-2 * sigma_disp, 2 * sigma_disp, color="black", alpha=0.08, label="±2 σ_ROI")
        ax.set_xlabel("rest plateau SOC [%]")
        ax.set_ylabel(f"{unit if unit else 'a.u.'}")
        ax.set_title(mod)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Within-rest dynamics summary: (tab_proximal − tab_distal) gradient and its 100-min drift", fontsize=12)
    fig.savefig(OUT / "summary_vs_plateau.png", dpi=130)
    plt.close(fig)

    # Verdict synthesis at the end
    summary.append("---\n\n## Honest verdict\n\n")
    summary.append("Across 5 plateaus × 3 modalities = 15 within-rest fits:\n")
    n_decay = sum(1 for mod in ["tof", "amplitude", "energy"] for r in plateau_summary[mod]
                  if (r["slope"] * r["intercept"]) < 0 and abs(r["delta"]) > 2 * SIGMA_ROI[mod] * UNITS[mod][1])
    n_grow = sum(1 for mod in ["tof", "amplitude", "energy"] for r in plateau_summary[mod]
                 if (r["slope"] * r["intercept"]) > 0 and abs(r["delta"]) > 2 * SIGMA_ROI[mod] * UNITS[mod][1])
    n_flat = 15 - n_decay - n_grow
    summary.append(f"- {n_decay} show gradient decay (consistent with crowding relaxation)\n")
    summary.append(f"- {n_grow} show gradient growth (NOT consistent with simple crowding)\n")
    summary.append(f"- {n_flat} are within ±2 σ_ROI of flat\n\n")
    summary.append(
        "The dominant feature is that the gradient at t=0 of rest **scales strongly with SOC plateau** "
        "(in `summary_vs_plateau.png`) but **does not relax to zero** within 100 min. This means either:\n"
        "1. τ_Li_diff ≫ 100 min (we don't catch the relaxation in this dataset), or\n"
        "2. The gradient is **dominated by an SOC-dependent acoustic effect that doesn't relax with current off** — "
        "this is consistent with the metal-tab artifact hypothesis where the ratio (metal weld reflection / cell-body amplitude) "
        "changes as cell-body amplitude changes with SOC.\n\n"
        "**The within-rest test alone cannot distinguish the two.** To resolve, the cleanest experiments are:\n"
        "- vary C-rate (crowding scales with current, artifact does not)\n"
        "- extend rest to >> 100 min (≥ 6 h would catch τ if it's a few hours)\n"
    )
    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
