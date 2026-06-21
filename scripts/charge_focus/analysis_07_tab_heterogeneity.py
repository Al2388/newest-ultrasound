"""Tab-proximal vs tab-distal SOC heterogeneity probe.

Tabs are confirmed on the right edge of the C-scan (high X). The user
observed in the amplitude GIF that the tab region's amplitude grows
(or shrinks less) with SOC compared to the cell interior.

This script:
  (1) Splits the ROI into 3 vertical bands — tab-proximal (right),
      interior (middle), tab-distal (left).
  (2) Computes ROI-mean amplitude / ToF / energy vs SOC in each band
      for rest scans (equilibrium) and charge scans (dynamic) separately.
  (3) Fits linear β_S in each band, tests if β_S_tab differs from
      β_S_distal at the σ_ROI noise floor.
  (4) Also produces a per-band-and-modality x-profile of β_S from the
      existing per-pixel β_S maps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
LONGRUN = PROJ / "scripts" / "longrun_analysis"
sys.path.insert(0, str(LONGRUN))

import common  # noqa: E402
from common import NOISE_FLOOR, Stack  # noqa: E402

NEW_OUT = PROJ / "reports" / "longrun_cycling_22h_charge_focus"
common.OUT_ROOT = NEW_OUT
common.CACHE_PATH = NEW_OUT / "_cache" / "stack.npz"
common.META_PATH = NEW_OUT / "_cache" / "meta.csv"

OUT = NEW_OUT / "07_tab_heterogeneity"
OUT.mkdir(parents=True, exist_ok=True)

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


BANDS = [
    ("tab-distal (left)",   (14.6, 30.0)),
    ("interior (middle)",   (30.0, 50.0)),
    ("tab-proximal (right)", (50.0, 64.5)),
]
BAND_COLORS = ["#1f77b4", "#7f7f7f", "#d62728"]

UNITS = {"tof": ("ns", 1e3), "amplitude": ("mV", 1e3), "energy": ("", 1.0)}
SIGMA_ROI = {
    "tof": NOISE_FLOOR["tof_sigma_roi_ns"] * 1e-3,
    "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"] * 1e-3,
    "energy": NOISE_FLOOR["energy_sigma_roi"],
}


def _band_mask(stack: Stack, x_lo: float, x_hi: float) -> np.ndarray:
    col_lo = int(np.searchsorted(stack.x_mm, x_lo))
    col_hi = int(np.searchsorted(stack.x_mm, x_hi))
    m = np.zeros_like(stack.roi_mask)
    m[:, col_lo:col_hi] = stack.roi_mask[:, col_lo:col_hi]
    return m


def _series_band(stack: Stack, mod: str, band_mask: np.ndarray) -> np.ndarray:
    arr = getattr(stack, mod)
    flat = arr.reshape(arr.shape[0], -1)
    m = band_mask.reshape(-1)
    return np.nanmean(flat[:, m], axis=1)


def _fit_slope(soc: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coeffs, cov = np.polyfit(soc, y, 1, cov=True)
    return float(coeffs[0]), float(np.sqrt(cov[0, 0]))


def _plot_band_series(stack: Stack):
    meta = stack.meta
    soc = meta.soc_pct.values
    rest_mask = meta.step_tag.values == "rest"
    charge_mask = meta.step_tag.values == "charge"

    band_masks = [_band_mask(stack, lo, hi) for _, (lo, hi) in BANDS]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    summary = ["# Tab-region heterogeneity analysis\n"]
    summary.append("Bands (x-range, ROI restricted):\n")
    for (name, (lo, hi)) in BANDS:
        summary.append(f"- {name}: X ∈ [{lo:.1f}, {hi:.1f}] mm\n")
    summary.append(
        f"\nTab is on the right edge (high X). σ_ROI noise floors: "
        f"ToF {NOISE_FLOOR['tof_sigma_roi_ns']:.2f} ns, amp {NOISE_FLOOR['amp_sigma_roi_mv']:.2f} mV, "
        f"energy {NOISE_FLOOR['energy_sigma_roi']:.3f}.\n\n"
    )

    for col, mod in enumerate(["tof", "amplitude", "energy"]):
        unit, scale = UNITS[mod]
        ax_rest = axes[0, col]
        ax_charge = axes[1, col]
        sigma_disp = SIGMA_ROI[mod] * scale

        summary.append(f"## {mod}\n\n| band | rest β_S [{unit}/%SoC] | charge β_S | (chg − rest) at SOC=50 |\n|---|---:|---:|---:|\n")
        rest_slopes = []
        for (name, _), bm, color in zip(BANDS, band_masks, BAND_COLORS):
            series = _series_band(stack, mod, bm)
            rest_y = series[rest_mask] * scale
            chg_y = series[charge_mask] * scale
            rest_soc = soc[rest_mask]
            chg_soc = soc[charge_mask]

            rb, rse = _fit_slope(rest_soc, rest_y)
            cb, cse = _fit_slope(chg_soc, chg_y)
            rest_slopes.append((name, rb, rse, cb, cse))

            ax_rest.plot(rest_soc, rest_y, "o", color=color, alpha=0.7, label=f"{name}")
            line = np.linspace(rest_soc.min(), rest_soc.max(), 50)
            ax_rest.plot(line, rb * line + np.polyfit(rest_soc, rest_y, 1)[1], "-", color=color, alpha=0.6)
            ax_charge.plot(chg_soc, chg_y, "o", color=color, alpha=0.7, label=f"{name}")
            ax_charge.plot(line, cb * line + np.polyfit(chg_soc, chg_y, 1)[1], "-", color=color, alpha=0.6)

            # value at SOC=50 for delta (chg - rest)
            soc_eval = 50.0
            v_rest = rb * soc_eval + np.polyfit(rest_soc, rest_y, 1)[1]
            v_chg = cb * soc_eval + np.polyfit(chg_soc, chg_y, 1)[1]
            delta = v_chg - v_rest
            summary.append(f"| {name} | {rb:+.4f} ± {rse:.4f} | {cb:+.4f} ± {cse:.4f} | {delta:+.3f} ({delta/sigma_disp:+.1f}σ_ROI) |\n")

        # significance: is tab β_S different from distal β_S?
        if len(rest_slopes) == 3:
            name_d, sl_d, se_d, _, _ = rest_slopes[0]
            name_t, sl_t, se_t, _, _ = rest_slopes[2]
            diff = sl_t - sl_d
            diff_se = np.sqrt(se_d ** 2 + se_t ** 2)
            z = diff / diff_se if diff_se else np.nan
            summary.append(
                f"\n**β_S gradient tab vs distal**: {diff:+.4f} {unit}/%SoC  ({z:+.1f}σ vs fit error). "
                f"|Δβ_S × 80%SoC| = {abs(diff)*80:.2f} {unit}, vs σ_ROI = {sigma_disp:.3g} {unit}. "
                f"{'**Spatially significant**' if abs(diff)*80 > 2 * sigma_disp else 'within noise'}.\n\n"
            )

        ax_rest.set_xlabel("SOC [%]")
        ax_rest.set_ylabel(f"band-mean {mod} [{unit if unit else 'a.u.'}]")
        ax_rest.set_title(f"REST — {mod}")
        ax_rest.grid(alpha=0.3)
        ax_rest.legend(fontsize=7)
        ax_charge.set_xlabel("SOC [%]")
        ax_charge.set_ylabel(f"band-mean {mod} [{unit if unit else 'a.u.'}]")
        ax_charge.set_title(f"CHARGE — {mod}")
        ax_charge.grid(alpha=0.3)
        ax_charge.legend(fontsize=7)
    fig.suptitle("Sub-ROI band evolution: tab-distal (blue) → interior (gray) → tab-proximal (red)", fontsize=12)
    fig.savefig(OUT / "band_series.png", dpi=130)
    plt.close(fig)

    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    return summary


def _plot_x_profile_betaS(stack: Stack):
    """Profile β_S along X by averaging the per-pixel β_S map over Y inside ROI."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, mod in zip(axes, ["tof", "amplitude", "energy"]):
        unit, scale = UNITS[mod]
        npz_path = NEW_OUT / "03_beta_soc_map" / f"beta_S_{mod}.npz"
        if not npz_path.exists():
            ax.text(0.5, 0.5, f"{npz_path.name} missing", ha="center")
            continue
        d = np.load(npz_path)
        beta_S = d["beta_S"]
        mask = stack.roi_mask
        col_sum = np.where(mask, beta_S, np.nan)
        x_profile = np.nanmean(col_sum, axis=0) * scale
        x_std = np.nanstd(col_sum, axis=0) * scale
        x = stack.x_mm
        finite = np.isfinite(x_profile)
        ax.plot(x[finite], x_profile[finite], "-", color="#222")
        ax.fill_between(x[finite], (x_profile - x_std)[finite], (x_profile + x_std)[finite], alpha=0.2, color="#222")
        ax.axhline(0, color="k", lw=0.6)
        for (name, (lo, hi)), color in zip(BANDS, BAND_COLORS):
            ax.axvspan(lo, hi, color=color, alpha=0.10, label=name if mod == "tof" else None)
        ax.set_xlabel("X [mm]   (tabs at X ≈ 64+ →)")
        ax.set_ylabel(f"β_S along X-row mean [{unit}/%SoC]")
        ax.set_title(mod)
        ax.grid(alpha=0.3)
        if mod == "tof":
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("β_S profile along X (rest-only pixel fit, Y-averaged inside ROI)", fontsize=12)
    fig.savefig(OUT / "betaS_x_profile.png", dpi=130)
    plt.close(fig)


def main() -> None:
    stack = filtered_load()
    _plot_band_series(stack)
    _plot_x_profile_betaS(stack)
    print(f"wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
