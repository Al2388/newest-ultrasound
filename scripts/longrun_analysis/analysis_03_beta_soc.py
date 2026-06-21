"""Per-pixel SOC sensitivity (beta_S) map using REST scans only.

At every pixel inside the ROI, fit  f = beta_S * SOC + beta_T * T + c
using the 60 rest scans (current=0, equilibrium). Plots beta_S map + a
significance mask  |beta_S * SOC_range| > 2 * sigma_p95.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NOISE_FLOOR, OUT_ROOT, load, roi_bounds_mm  # noqa: E402

OUT = OUT_ROOT / "03_beta_soc_map"
OUT.mkdir(parents=True, exist_ok=True)

SIGMA_PIX_P95 = {
    "tof": NOISE_FLOOR["tof_sigma_pixp95_ns"] * 1e-3,
    "amplitude": NOISE_FLOOR["amp_sigma_pixp95_mv"] * 1e-3,
    "energy": NOISE_FLOOR["energy_sigma_pixp95"],
}
UNITS_PER_SOC = {"tof": "µs/%SoC", "amplitude": "V/%SoC", "energy": "/%SoC"}


def _fit_pixelwise(arr3d_rest: np.ndarray, soc: np.ndarray, temp: np.ndarray, roi_mask: np.ndarray):
    n_t, ny, nx = arr3d_rest.shape
    X = np.column_stack([soc, temp, np.ones_like(soc)]).astype(np.float64)

    flat = arr3d_rest.reshape(n_t, -1).astype(np.float64)
    mask_flat = roi_mask.reshape(-1)
    Y_roi = flat[:, mask_flat]
    finite = np.isfinite(Y_roi).all(axis=0)
    Y_fit = Y_roi[:, finite]

    coeffs, _, _, _ = np.linalg.lstsq(X, Y_fit, rcond=None)
    pred = X @ coeffs
    residuals = Y_fit - pred
    sse = (residuals ** 2).sum(axis=0)
    sst = ((Y_fit - Y_fit.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(sst > 0, 1.0 - sse / sst, np.nan)

    se = np.sqrt(sse / max(n_t - 3, 1))
    XtX_inv = np.linalg.inv(X.T @ X)

    def _scatter(values, full_shape):
        full = np.full(full_shape[0] * full_shape[1], np.nan, dtype=np.float64)
        roi_vals = np.full(int(mask_flat.sum()), np.nan, dtype=np.float64)
        roi_vals[finite] = values
        full[mask_flat] = roi_vals
        return full.reshape(full_shape).astype(np.float32)

    beta_S = _scatter(coeffs[0], (ny, nx))
    beta_T = _scatter(coeffs[1], (ny, nx))
    intercept = _scatter(coeffs[2], (ny, nx))
    r2_map = _scatter(r2, (ny, nx))
    se_beta_S = _scatter(se * np.sqrt(XtX_inv[0, 0]), (ny, nx))

    return dict(beta_S=beta_S, beta_T=beta_T, intercept=intercept, r2=r2_map, se_beta_S=se_beta_S)


def _plot(stack, mod: str, fit: dict, soc_range: float):
    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]
    sigma_p = SIGMA_PIX_P95[mod]

    beta_S = fit["beta_S"]
    dyn = beta_S * soc_range
    significant = np.abs(dyn) > 2 * sigma_p

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    p99 = np.nanpercentile(np.abs(beta_S[stack.roi_mask]), 99)
    im = axes[0, 0].imshow(beta_S, extent=extent, aspect="equal", cmap="RdBu_r", vmin=-p99, vmax=p99)
    axes[0, 0].set_title(f"β_S  [{UNITS_PER_SOC[mod]}]\n(SOC sensitivity, rest-only fit, n={fit['n']})")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.85)

    p99d = np.nanpercentile(np.abs(dyn[stack.roi_mask]), 99)
    im = axes[0, 1].imshow(dyn, extent=extent, aspect="equal", cmap="RdBu_r", vmin=-p99d, vmax=p99d)
    axes[0, 1].set_title(f"β_S · ΔSOC ({soc_range:.0f} %SoC span)")
    plt.colorbar(im, ax=axes[0, 1], shrink=0.85)

    sig_show = np.where(significant & stack.roi_mask, np.sign(dyn), 0)
    im = axes[1, 0].imshow(sig_show, extent=extent, aspect="equal", cmap="RdBu_r", vmin=-1, vmax=1)
    pct = float(significant[stack.roi_mask].mean()) * 100
    axes[1, 0].set_title(f"significant pixels  |β_S·ΔSOC| > 2σ_p95\n({pct:.1f}% of ROI)")
    plt.colorbar(im, ax=axes[1, 0], shrink=0.85, ticks=[-1, 0, 1])

    im = axes[1, 1].imshow(fit["r2"], extent=extent, aspect="equal", cmap="viridis", vmin=0, vmax=1)
    r2_med = float(np.nanmedian(fit["r2"][stack.roi_mask]))
    axes[1, 1].set_title(f"R² (per pixel)   median = {r2_med:.2f}")
    plt.colorbar(im, ax=axes[1, 1], shrink=0.85)

    for ax in axes.ravel():
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=1, alpha=0.7)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
    fig.suptitle(f"Per-pixel SOC sensitivity (rest only) — {mod}", fontsize=12)
    return fig


def main() -> None:
    stack = load()
    meta = stack.meta
    rest_mask = meta.step_tag.values == "rest"
    soc = meta.soc_pct.values[rest_mask]
    temp = meta.temp_at_scan.values[rest_mask]
    rest_idx = np.where(rest_mask)[0]
    print(f"rest scans: n={len(rest_idx)}, SOC=[{soc.min():.1f},{soc.max():.1f}], T=[{temp.min():.2f},{temp.max():.2f}]")
    soc_range = float(soc.max() - soc.min())

    summary = ["# Per-pixel β_S map (rest only)\n", f"n_rest={len(rest_idx)}, SOC span={soc_range:.1f}%, T span={temp.max()-temp.min():.2f}°C\n\n"]
    summary.append("| modality | β_S median (ROI) | β_S p5/p95 | significant pixels | R² median |\n")
    summary.append("|---|---:|---:|---:|---:|\n")

    for mod in ["tof", "amplitude", "energy"]:
        arr = getattr(stack, mod)[rest_idx]
        fit = _fit_pixelwise(arr, soc, temp, stack.roi_mask)
        fit["n"] = len(rest_idx)
        fig = _plot(stack, mod, fit, soc_range)
        fig.savefig(OUT / f"beta_S_{mod}.png", dpi=140)
        plt.close(fig)

        beta_roi = fit["beta_S"][stack.roi_mask]
        med = float(np.nanmedian(beta_roi))
        p5, p95 = np.nanpercentile(beta_roi, [5, 95])
        dyn = fit["beta_S"] * soc_range
        sig_pct = float((np.abs(dyn[stack.roi_mask]) > 2 * SIGMA_PIX_P95[mod]).mean()) * 100
        r2_med = float(np.nanmedian(fit["r2"][stack.roi_mask]))
        summary.append(f"| {mod} | {med:+.4g} | {p5:+.3g} / {p95:+.3g} | {sig_pct:.1f}% | {r2_med:.2f} |\n")

        np.savez_compressed(OUT / f"beta_S_{mod}.npz", **{k: v for k, v in fit.items() if k != "n"})

    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
