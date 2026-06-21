"""Reference-region drift check during SOC=80% rest.

Critical sanity check before claiming the within-rest acoustic changes
are 'battery internal state changing'.

Defines several OFF-CELL bands (oil-only or fixture-only), computes their
ROI-mean trajectory during the 100-min rest, and compares to:
  - canonical 50x40 ROI (cell interior)
  - tab-distal / interior / tab-proximal sub-bands
  - cell temperature from cycler probe (temp_at_scan)

If off-cell bands also drift in the same direction / same timing as the
cell ROI -> the 'acceleration' is measurement-system drift, not battery.
If off-cell bands are flat while ROI moves -> the change is the battery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
OUT = NEW_OUT / "11_reference_check"
OUT.mkdir(parents=True, exist_ok=True)

UNITS = {
    "amplitude": ("mV", 1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns", 1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

_orig_load = common.load


def filtered_load() -> Stack:
    s = _orig_load(rebuild=False)
    keep = s.meta.step_tag.isin({"charge", "rest"}).values
    idx = np.where(keep)[0]
    m = s.meta.iloc[idx].copy().reset_index(drop=True)
    m["soc_pct"] = m["soc_pct"] - m["soc_pct"].min()
    return Stack(amplitude=s.amplitude[idx], tof=s.tof[idx], energy=s.energy[idx],
                 x_mm=s.x_mm, y_mm=s.y_mm, roi_mask=s.roi_mask, meta=m)


def _rect_mask(stack, x_lo, x_hi, y_lo, y_hi):
    cl = int(np.searchsorted(stack.x_mm, x_lo)); ch = int(np.searchsorted(stack.x_mm, x_hi))
    rl = int(np.searchsorted(stack.y_mm, y_lo)); rh = int(np.searchsorted(stack.y_mm, y_hi))
    m = np.zeros_like(stack.roi_mask, dtype=bool)
    m[rl:rh, cl:ch] = True
    return m


def _band_mask_x(stack, x_lo, x_hi):
    cl = int(np.searchsorted(stack.x_mm, x_lo)); ch = int(np.searchsorted(stack.x_mm, x_hi))
    m = np.zeros_like(stack.roi_mask)
    m[:, cl:ch] = stack.roi_mask[:, cl:ch]
    return m


def main():
    stack = filtered_load()
    segs = rest_segments(stack.meta)
    i_seg = int(np.argmin([abs(s.soc_pct.iloc[0] - 80.0) for s in segs]))
    seg_meta = segs[i_seg]
    soc = float(seg_meta.soc_pct.iloc[0])
    seg_idx = np.where(np.isin(stack.meta.run_idx.values, seg_meta.run_idx.values))[0]
    t_iso = pd.to_datetime(seg_meta.time_utc)
    t_min = (t_iso - t_iso.iloc[0]).dt.total_seconds().values / 60.0
    print(f"chose seg #{i_seg+1}, SOC ~ {soc:.1f}%, n_scans={len(seg_meta)}, dur={t_min[-1]:.0f} min")

    # Define test regions
    regions = {
        # In-cell bands (the signal regions)
        "ROI whole (cell body)":     ("cell", stack.roi_mask),
        "ROI tab-distal X14-30":     ("cell", _band_mask_x(stack, 14.6, 30)),
        "ROI interior  X30-50":      ("cell", _band_mask_x(stack, 30, 50)),
        "ROI tab-prox  X50-64.5":    ("cell", _band_mask_x(stack, 50, 64.5)),
        # Off-cell references
        "OFF: far-left X0-10":       ("ref",  _rect_mask(stack, 0, 10, 5, 67)),
        "OFF: far-right X70-80":     ("ref",  _rect_mask(stack, 70, 80, 5, 67)),
        "OFF: top edge Y0-10":       ("ref",  _rect_mask(stack, 5, 75, 0, 10)),
        "OFF: bottom edge Y62-72":   ("ref",  _rect_mask(stack, 5, 75, 62, 72)),
    }

    # Compute mean trajectory
    series = {}
    for name, (kind, mask) in regions.items():
        d = {}
        for mod in ["amplitude", "tof", "energy"]:
            arr = getattr(stack, mod)[seg_idx]
            flat = arr.reshape(arr.shape[0], -1)
            m = mask.reshape(-1) & np.isfinite(arr[0]).reshape(-1)
            mean = np.array([float(np.nanmean(f[m])) for f in flat])
            d[mod] = mean - mean[0]
        d["kind"] = kind
        d["n_pixels"] = int(mask.sum())
        series[name] = d

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    for r, mod in enumerate(["amplitude", "tof", "energy"]):
        unit, scale, sigma = UNITS[mod]
        ax_cell = axes[r, 0]
        ax_ref = axes[r, 1]
        for name, d in series.items():
            kind = d["kind"]
            y = d[mod] * scale
            n = d["n_pixels"]
            ax = ax_cell if kind == "cell" else ax_ref
            ax.plot(t_min, y, "o-", lw=1.2, ms=4, label=f"{name}  (n={n})")
        for ax in (ax_cell, ax_ref):
            ax.axhline(0, color="k", lw=0.4)
            ax.fill_between(t_min, -2 * sigma * scale, 2 * sigma * scale, color="gray", alpha=0.15,
                            label=f"+/-2 sigma_ROI = {2*sigma*scale:.2g}")
            ax.set_xlabel("rest time [min]")
            ax.set_ylabel(f"Delta {mod} [{unit}]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        ax_cell.set_title(f"CELL bands  -  {mod}  (Delta from scan[0])")
        ax_ref.set_title(f"OFF-CELL reference bands  -  {mod}")

    fig.suptitle(f"Reference-region drift check  -  SOC~{soc:.0f}% rest, {t_min[-1]:.0f} min\n"
                 f"If OFF-CELL bands drift like CELL bands -> measurement-system drift, not battery", fontsize=12)
    fig.savefig(OUT / "reference_check_trajectories.png", dpi=130)
    plt.close(fig)

    # Temperature evolution panel
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    T = seg_meta.temp_at_scan.values
    dT = (T - T[0]) * 1000  # mC
    ax.plot(t_min, dT, "o-", color="tab:purple", ms=5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("rest time [min]")
    ax.set_ylabel("Delta cycler-probe T [mC]")
    ax.set_title(f"Cell-side temperature during SOC~{soc:.0f}% rest  -  total {dT[-1]:+.1f} mC")
    ax.grid(alpha=0.3)
    # mark 54-min transition point user is worried about
    ax.axvline(54, color="tab:red", ls=":", lw=1, alpha=0.7, label="t=54 min (acoustic inflection)")
    ax.legend()
    fig.savefig(OUT / "cell_temperature_during_rest.png", dpi=130)
    plt.close(fig)

    # Write summary
    lines = [f"# Reference-region check  --  SOC~{soc:.0f}% rest\n\n"]
    lines.append(f"100-min rest, 12 scans, voltage drift {(seg_meta.voltage_at_scan.iloc[-1]-seg_meta.voltage_at_scan.iloc[0])*1000:+.1f} mV, T drift {dT[-1]:+.1f} mC\n\n")
    lines.append("## Final cumulative drift (t=99 min, scan[t]-scan[0]):\n\n")
    lines.append("| region | n_pix | Amp [mV] | ToF [ns] | Energy |\n|---|---:|---:|---:|---:|\n")
    for name, d in series.items():
        a = d["amplitude"][-1] * 1e3
        tf = d["tof"][-1] * 1e3
        e = d["energy"][-1]
        lines.append(f"| {name} | {d['n_pixels']} | {a:+.2f} | {tf:+.2f} | {e:+.3f} |\n")
    lines.append("\n## Verdict logic\n")
    lines.append("- OFF bands flat (|Delta| < 2 sigma_ROI) + CELL bands large -> cell change is REAL\n")
    lines.append("- OFF bands drift similarly to CELL bands -> measurement-system drift (oil temp 2nd-order, coupling, etc.)\n")
    lines.append("- Mixed -> partial system drift; cell change residual is the difference\n")
    (OUT / "README.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
