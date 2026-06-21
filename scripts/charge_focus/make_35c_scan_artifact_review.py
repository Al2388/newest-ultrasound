"""Create a scan-by-scan 35C artifact review PDF.

Each PDF page shows one scan with the three modalities side by side. A compact
CSV is written alongside it with simple row-artifact checks to help prioritize
manual inspection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJ = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"
OUT_DIR = OUT_ROOT / "06_animation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_OUT = OUT_DIR / "all_scans_artifact_review.pdf"
CSV_OUT = OUT_DIR / "all_scans_artifact_review_flags.csv"

MODS = ("tof", "amplitude", "energy")
UNITS = {"tof": "us", "amplitude": "V", "energy": "(a.u.)"}
CMAP = "turbo"


def _row_flags(arr: np.ndarray, roi: np.ndarray, vmin: float) -> tuple[int, int, int]:
    """Return nan/zero/low-clipped row counts inside ROI for one scan."""
    nan_rows = 0
    zero_rows = 0
    low_clip_rows = 0
    for r in range(arr.shape[0]):
        if not roi[r].any():
            continue
        vals = arr[r, roi[r]]
        finite = np.isfinite(vals)
        if not finite.any():
            nan_rows += 1
            continue
        if np.mean(finite & (np.abs(vals) < 1e-12)) >= 0.5:
            zero_rows += 1
        if np.all(vals[finite] <= vmin):
            low_clip_rows += 1
    return nan_rows, zero_rows, low_clip_rows


def main() -> None:
    print("loading 35C cache ...")
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])

    roi = d["roi_mask"].astype(bool)
    x_mm = d["x_mm"]
    y_mm = d["y_mm"]
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]

    roi_rows = np.where(roi.any(axis=1))[0]
    roi_cols = np.where(roi.any(axis=0))[0]
    x_lo, x_hi = float(x_mm[roi_cols.min()]), float(x_mm[roi_cols.max()])
    y_lo, y_hi = float(y_mm[roi_rows.min()]), float(y_mm[roi_rows.max()])

    arrs = {m: d[m] for m in MODS}
    vrange = {
        m: (
            float(np.nanpercentile(arrs[m][:, roi], 5)),
            float(np.nanpercentile(arrs[m][:, roi], 95)),
        )
        for m in MODS
    }

    print(f"writing flags -> {CSV_OUT}")
    flag_rows = []
    n_scans = arrs["amplitude"].shape[0]
    for s in range(n_scans):
        row = {
            "scan_idx_zero_based": s,
            "run_idx_original": int(meta.loc[s, "run_idx"]),
            "timestamp": meta.loc[s, "timestamp"],
            "step_tag": meta.loc[s, "step_tag"],
        }
        for mod in MODS:
            nan_n, zero_n, low_n = _row_flags(arrs[mod][s], roi, vrange[mod][0])
            row[f"{mod}_nan_rows"] = nan_n
            row[f"{mod}_zero_rows"] = zero_n
            row[f"{mod}_all_low_clip_rows"] = low_n
        flag_rows.append(row)
    flags = pd.DataFrame(flag_rows)
    flags.to_csv(CSV_OUT, index=False)

    print(f"writing PDF -> {PDF_OUT}")
    with PdfPages(PDF_OUT) as pdf:
        for s in range(n_scans):
            fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)
            for ax, mod in zip(axes, MODS):
                vmin, vmax = vrange[mod]
                im = ax.imshow(
                    arrs[mod][s],
                    extent=extent,
                    aspect="equal",
                    cmap=CMAP,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    origin="lower",
                )
                ax.set_title(mod)
                ax.set_xlabel("X [mm]")
                if ax is axes[0]:
                    ax.set_ylabel("Y [mm]")
                fig.colorbar(im, ax=ax, shrink=0.82, label=UNITS[mod])

            flag = flags.loc[s]
            total_flags = int(
                sum(
                    flag[f"{m}_{kind}"]
                    for m in MODS
                    for kind in ("nan_rows", "zero_rows", "all_low_clip_rows")
                )
            )
            title_color = "tab:red" if total_flags else "#222"
            fig.suptitle(
                "35C scan artifact review | "
                f"scan {s:03d} | original r{int(meta.loc[s, 'run_idx']):03d} | "
                f"{meta.loc[s, 'timestamp']} | {meta.loc[s, 'step_tag']} | "
                f"auto-flag rows={total_flags}",
                fontsize=12,
                color=title_color,
            )
            pdf.savefig(fig, dpi=170)
            plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
