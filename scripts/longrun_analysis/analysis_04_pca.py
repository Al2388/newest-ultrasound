"""SVD/PCA decomposition of the 147-frame stack.

Per modality: stack ROI pixels across all 147 frames into a (147 x N_pix)
matrix, mean-center per pixel, then SVD. Plots:
- explained variance scree (top 10)
- top 6 spatial modes (loadings) as maps
- top 6 temporal scores against time, with SOC overlay
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_ROOT, load, roi_bounds_mm  # noqa: E402

OUT = OUT_ROOT / "04_pca_svd"
OUT.mkdir(parents=True, exist_ok=True)


def _pca(arr3d: np.ndarray, roi_mask: np.ndarray, k: int = 6):
    n_t, ny, nx = arr3d.shape
    mask_flat = roi_mask.reshape(-1)
    flat_roi = arr3d.reshape(n_t, -1)[:, mask_flat]
    finite = np.isfinite(flat_roi).all(axis=0)
    X_in = flat_roi[:, finite]
    mean = X_in.mean(axis=0, keepdims=True)
    X = X_in - mean
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    expl = s ** 2 / (s ** 2).sum()

    scores = U[:, :k] * s[:k]
    loadings_in = Vt[:k]

    n_roi = int(mask_flat.sum())
    loadings_roi = np.full((k, n_roi), np.nan, dtype=np.float32)
    loadings_roi[:, finite] = loadings_in.astype(np.float32)

    full_loadings = np.full((k, ny * nx), np.nan, dtype=np.float32)
    full_loadings[:, mask_flat] = loadings_roi
    full_loadings = full_loadings.reshape(k, ny, nx)
    return scores, full_loadings, expl


def _plot(stack, mod: str, scores, loadings, expl):
    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]
    meta = stack.meta
    t_min = (pd.to_datetime(meta.time_utc) - pd.to_datetime(meta.time_utc.iloc[0])).dt.total_seconds().values / 60.0

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 6)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.bar(range(1, 11), expl[:10] * 100, color="#444")
    ax.set_xlabel("component")
    ax.set_ylabel("variance explained [%]")
    ax.set_title(f"scree (top10), cum top6 = {expl[:6].sum()*100:.1f}%")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2:6])
    ax2 = ax.twinx()
    ax.plot(t_min, meta.soc_pct.values, color="tab:blue", label="SOC [%]")
    ax.set_ylabel("SOC [%]", color="tab:blue")
    ax.set_xlabel("time [min]")
    ax2.plot(t_min, meta.voltage_at_scan.values, color="tab:red", alpha=0.6, label="V")
    ax2.set_ylabel("voltage [V]", color="tab:red")
    ax.set_title("SOC and voltage trajectory")
    ax.grid(alpha=0.3)

    for k in range(6):
        ax = fig.add_subplot(gs[1 + k // 3, (k % 3) * 2: (k % 3) * 2 + 1])
        load_k = loadings[k]
        p99 = np.nanpercentile(np.abs(load_k[stack.roi_mask]), 99)
        im = ax.imshow(load_k, extent=extent, aspect="equal", cmap="RdBu_r", vmin=-p99, vmax=p99)
        ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=0.8, alpha=0.7)
        ax.set_title(f"PC{k+1} loading ({expl[k]*100:.1f}%)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.85)

        ax = fig.add_subplot(gs[1 + k // 3, (k % 3) * 2 + 1])
        ax.plot(t_min, scores[:, k], "-", color="#222")
        ax.set_title(f"PC{k+1} score(t)")
        ax.grid(alpha=0.3)
        if k // 3 == 1:
            ax.set_xlabel("time [min]")

    fig.suptitle(f"PCA of 147 C-scan frames — {mod}", fontsize=13)
    return fig


def main() -> None:
    stack = load()
    summary = ["# PCA/SVD decomposition\n", "Mean-centered per pixel inside ROI, then SVD on (147 × N_roi).\n\n"]
    summary.append("| modality | PC1 var% | PC1-3 cum% | PC1-6 cum% |\n|---|---:|---:|---:|\n")

    for mod in ["tof", "amplitude", "energy"]:
        arr = getattr(stack, mod)
        scores, loadings, expl = _pca(arr, stack.roi_mask, k=6)
        fig = _plot(stack, mod, scores, loadings, expl)
        fig.savefig(OUT / f"pca_{mod}.png", dpi=140)
        plt.close(fig)
        np.savez_compressed(OUT / f"pca_{mod}.npz", scores=scores, loadings=loadings, expl=expl)
        summary.append(f"| {mod} | {expl[0]*100:.1f} | {expl[:3].sum()*100:.1f} | {expl[:6].sum()*100:.1f} |\n")

    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"wrote PCA outputs to {OUT}")


if __name__ == "__main__":
    main()
