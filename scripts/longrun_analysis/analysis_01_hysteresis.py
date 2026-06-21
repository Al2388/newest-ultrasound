"""Hysteresis / current-direction asymmetry maps.

Two flavors given the data layout (discharge only covers SOC=[-16,-1]):
(A) charge vs discharge maps within the overlap window SOC=[-16,-1]
(B) charge (under current) vs rest plateau (equilibrium) at matched SOC

Outputs to reports/longrun_cycling_22h_analysis/01_hysteresis/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NOISE_FLOOR, OUT_ROOT, load, roi_bounds_mm  # noqa: E402

OUT = OUT_ROOT / "01_hysteresis"
OUT.mkdir(parents=True, exist_ok=True)

MODALITIES = ["tof", "amplitude", "energy"]
UNITS = {"tof": "µs", "amplitude": "V", "energy": "(a.u.)"}
SIGMA_ROI = {"tof": NOISE_FLOOR["tof_sigma_roi_ns"] * 1e-3, "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"] * 1e-3, "energy": NOISE_FLOOR["energy_sigma_roi"]}


def _mean_map(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.full(arr.shape[1:], np.nan, dtype=np.float32)
    return np.nanmean(arr[indices], axis=0)


def _diff_panel(stack, mod: str, group_a_idx, group_b_idx, label_a: str, label_b: str, soc_label: str):
    arr = getattr(stack, mod)
    map_a = _mean_map(arr, group_a_idx)
    map_b = _mean_map(arr, group_b_idx)
    diff = map_a - map_b

    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]
    sigma_roi = SIGMA_ROI[mod]
    roi_mean_diff = float(np.nanmean(diff[stack.roi_mask]))
    roi_sd_diff = float(np.nanstd(diff[stack.roi_mask]))
    z_pix = roi_mean_diff / sigma_roi if sigma_roi > 0 else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    common_kwargs = dict(extent=extent, aspect="equal", cmap="turbo")
    im0 = axes[0].imshow(map_a, **common_kwargs)
    axes[0].set_title(f"{label_a}\n{soc_label} (n={len(group_a_idx)})")
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label=UNITS[mod])

    im1 = axes[1].imshow(map_b, **common_kwargs)
    axes[1].set_title(f"{label_b}\n{soc_label} (n={len(group_b_idx)})")
    plt.colorbar(im1, ax=axes[1], shrink=0.85, label=UNITS[mod])

    vmax = float(np.nanpercentile(np.abs(diff), 99))
    im2 = axes[2].imshow(diff, extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title(
        f"({label_a}) − ({label_b})\n"
        f"ROI Δ={roi_mean_diff*(1e3 if mod!='energy' else 1):.3f} "
        f"{'mV' if mod=='amplitude' else ('ns' if mod=='tof' else '')}  z={z_pix:.1f}σ"
    )
    plt.colorbar(im2, ax=axes[2], shrink=0.85, label=UNITS[mod])

    for ax in axes:
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=1, alpha=0.7)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
    fig.suptitle(f"Hysteresis  {mod}  —  {soc_label}", fontsize=12)
    return fig, dict(roi_mean_diff=roi_mean_diff, roi_sd_diff=roi_sd_diff, z=z_pix, n_a=len(group_a_idx), n_b=len(group_b_idx))


def main() -> None:
    stack = load()
    meta = stack.meta
    soc = meta.soc_pct.values

    summary_lines = ["# Hysteresis analysis\n"]

    # (A) Direct charge vs discharge in overlap window
    overlap = (soc >= -16.5) & (soc <= -0.5)
    charge_idx = np.where(overlap & (meta.step_tag.values == "charge"))[0]
    discharge_idx = np.where(overlap & (meta.step_tag.values == "discharge"))[0]
    summary_lines.append(f"## (A) charge vs discharge, SOC ∈ [-16, -1]\n")
    summary_lines.append(f"charge n={len(charge_idx)}, discharge n={len(discharge_idx)}\n")
    soc_label_a = f"SOC ∈ [-16, -1] %"
    for mod in MODALITIES:
        fig, stats = _diff_panel(stack, mod, charge_idx, discharge_idx, "charge", "discharge", soc_label_a)
        fig.savefig(OUT / f"A_chg_minus_dischg_{mod}.png", dpi=130)
        plt.close(fig)
        summary_lines.append(f"- **{mod}** Δ_ROI = {stats['roi_mean_diff']:.4g}  ({stats['z']:.1f}σ_ROI)\n")

    # (B) Charge (under current) vs rest equilibrium, matched SOC bands
    bins = [(-20, -10), (-10, 0), (0, 15), (15, 30), (30, 45), (45, 60)]
    summary_lines.append("\n## (B) charge under-current vs rest equilibrium, matched SOC bins\n")
    summary_lines.append("Δ = charge_map − rest_map at matched SOC. Isolates kinetic / current-driven component.\n")
    for lo, hi in bins:
        in_bin = (soc >= lo) & (soc < hi)
        ci = np.where(in_bin & (meta.step_tag.values == "charge"))[0]
        ri = np.where(in_bin & (meta.step_tag.values == "rest"))[0]
        if len(ci) == 0 or len(ri) == 0:
            continue
        soc_label = f"SOC ∈ [{lo}, {hi}) %"
        for mod in MODALITIES:
            fig, stats = _diff_panel(stack, mod, ci, ri, "charge (current on)", "rest (equilibrium)", soc_label)
            fig.savefig(OUT / f"B_chg_minus_rest_SOC{lo:+04d}_{hi:+04d}_{mod}.png", dpi=120)
            plt.close(fig)
            summary_lines.append(f"- **{mod}** SOC[{lo},{hi})  Δ_ROI = {stats['roi_mean_diff']:.4g}  ({stats['z']:.1f}σ_ROI)  n_chg={stats['n_a']} n_rest={stats['n_b']}\n")

    (OUT / "README.md").write_text("".join(summary_lines), encoding="utf-8")
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
