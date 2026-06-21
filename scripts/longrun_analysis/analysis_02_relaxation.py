"""Rest relaxation: per-pixel exponential time-constant maps.

For each of the 5 rest segments (12 scans each), fit
    f(t) = f_inf + dF * exp(-t / tau)
at every pixel using ROI-mean-detrended residuals as a sanity check first,
then per-pixel curve_fit on ToF and amplitude. Outputs per-segment tau maps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_ROOT, load, rest_segments, roi_bounds_mm  # noqa: E402

OUT = OUT_ROOT / "02_relaxation"
OUT.mkdir(parents=True, exist_ok=True)


def _exp(t, f_inf, dF, tau):
    return f_inf + dF * np.exp(-t / tau)


def _fit_pixel(t, y):
    if not np.all(np.isfinite(y)):
        return np.nan, np.nan, np.nan, np.nan
    y0, y1 = y[0], y[-1]
    p0 = (y1, y0 - y1, max((t[-1] - t[0]) / 3, 60.0))
    try:
        popt, _ = curve_fit(_exp, t, y, p0=p0, maxfev=400)
        f_inf, dF, tau = popt
        if tau <= 0 or tau > 10 * (t[-1] - t[0]):
            return np.nan, np.nan, np.nan, np.nan
        resid = y - _exp(t, *popt)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        return float(f_inf), float(dF), float(tau), rms
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, np.nan


def _segment_times_s(meta_seg: pd.DataFrame) -> np.ndarray:
    t = pd.to_datetime(meta_seg.time_utc)
    return (t - t.iloc[0]).dt.total_seconds().values.astype(np.float64)


def _fit_segment(arr3d: np.ndarray, t: np.ndarray, roi_mask: np.ndarray) -> dict:
    n_t, ny, nx = arr3d.shape
    out_finf = np.full((ny, nx), np.nan, dtype=np.float32)
    out_dF = np.full((ny, nx), np.nan, dtype=np.float32)
    out_tau = np.full((ny, nx), np.nan, dtype=np.float32)
    out_rms = np.full((ny, nx), np.nan, dtype=np.float32)

    rows, cols = np.where(roi_mask)
    for r, c in zip(rows, cols):
        y = arr3d[:, r, c].astype(np.float64)
        f_inf, dF, tau, rms = _fit_pixel(t, y)
        out_finf[r, c] = f_inf
        out_dF[r, c] = dF
        out_tau[r, c] = tau
        out_rms[r, c] = rms
    return dict(f_inf=out_finf, dF=out_dF, tau=out_tau, rms=out_rms)


def _plot_segment(seg_idx: int, soc_at_start: float, mod: str, t: np.ndarray, roi_mean_series: np.ndarray, fit: dict, stack, save_path: Path):
    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(t / 60, roi_mean_series, "o-", color="#222")
    ax.set_xlabel("time since segment start [min]")
    ax.set_ylabel(f"ROI-mean {mod}")
    ax.set_title(f"ROI-mean relaxation, seg {seg_idx+1} (SOC≈{soc_at_start:.1f}%)")
    ax.grid(alpha=0.3)

    tau_min = fit["tau"] / 60.0
    p_lo, p_hi = np.nanpercentile(tau_min[stack.roi_mask], [5, 95])
    im = axes[0, 1].imshow(tau_min, extent=extent, aspect="equal", cmap="viridis", vmin=p_lo, vmax=p_hi)
    axes[0, 1].set_title(f"τ map [min]   p5–p95: {p_lo:.1f}–{p_hi:.1f}")
    plt.colorbar(im, ax=axes[0, 1], shrink=0.85, label="min")

    dF_abs = np.abs(fit["dF"])
    p99 = np.nanpercentile(dF_abs[stack.roi_mask], 99)
    im = axes[1, 0].imshow(fit["dF"], extent=extent, aspect="equal", cmap="RdBu_r", vmin=-p99, vmax=p99)
    axes[1, 0].set_title("ΔF (initial offset from equilibrium)")
    plt.colorbar(im, ax=axes[1, 0], shrink=0.85)

    im = axes[1, 1].imshow(fit["rms"], extent=extent, aspect="equal", cmap="magma")
    axes[1, 1].set_title("fit RMS residual")
    plt.colorbar(im, ax=axes[1, 1], shrink=0.85)

    for ax in axes[:, 1:].ravel():
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=1, alpha=0.7)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")

    fig.suptitle(f"Relaxation segment {seg_idx+1}  ·  {mod}  ·  SOC≈{soc_at_start:.1f}%", fontsize=12)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main() -> None:
    stack = load()
    segs = rest_segments(stack.meta)
    print(f"found {len(segs)} rest segments, sizes: {[len(s) for s in segs]}")

    summary = ["# Rest relaxation analysis\n", f"{len(segs)} rest segments. Per-pixel exponential fit f(t)=f∞+ΔF·exp(-t/τ).\n\n"]
    summary.append("| seg | SOC start [%] | duration [min] | n | ROI τ_med [min] (ToF) | ROI ΔF_med (ToF, ns) |\n")
    summary.append("|---|---:|---:|---:|---:|---:|\n")

    for i, seg in enumerate(segs):
        run_idxs = seg.run_idx.values
        sel = np.isin(stack.meta.run_idx.values, run_idxs)
        sel_idx = np.where(sel)[0]
        t = _segment_times_s(seg)
        soc0 = float(seg.soc_pct.iloc[0])
        dur_min = (t[-1] - t[0]) / 60.0

        for mod in ["tof", "amplitude"]:
            arr3d = getattr(stack, mod)[sel_idx]
            fit = _fit_segment(arr3d, t, stack.roi_mask)
            roi_mean_series = np.array([np.nanmean(a[stack.roi_mask]) for a in arr3d])
            _plot_segment(i, soc0, mod, t, roi_mean_series, fit, stack, OUT / f"seg{i+1:02d}_SOC{soc0:+05.1f}_{mod}.png")
            if mod == "tof":
                tau_roi = fit["tau"][stack.roi_mask]
                dF_roi = fit["dF"][stack.roi_mask]
                tau_med_min = float(np.nanmedian(tau_roi)) / 60.0
                dF_med_ns = float(np.nanmedian(dF_roi)) * 1e3
                summary.append(f"| {i+1} | {soc0:+.1f} | {dur_min:.1f} | {len(seg)} | {tau_med_min:.1f} | {dF_med_ns:+.2f} |\n")

            np.savez_compressed(
                OUT / f"seg{i+1:02d}_{mod}_fit.npz",
                f_inf=fit["f_inf"], dF=fit["dF"], tau=fit["tau"], rms=fit["rms"], t=t,
            )

    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
