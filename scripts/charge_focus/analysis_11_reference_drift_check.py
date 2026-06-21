"""Reference region drift check for SOC=80% rest.

Critical diagnostic for the "rate accelerates" finding: if non-cell
reference regions ALSO drift in the same 54->99 min window, the apparent
"acceleration" is a system artifact (bath thermal lag, coupling change,
fixture drift), not cell physics. If reference is flat while ROI drifts,
the cell physics is real.

Reference bands defined as pixels OUTSIDE the canonical ROI:
- far-left fixture / tank wall reflection (X < 10 mm)
- far-right fixture (X > 70 mm)
- far-top / far-bottom outside cell body
- oil-only band: pixels between cell edge and fixture (if separable)

Also pulls in-scan line temperature (from NPZ line_temperature_mean_c) to
test if bath temperature has a slow second-order drift across the 100 min.
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
OUT = NEW_OUT / "11_reference_drift_check"
OUT.mkdir(parents=True, exist_ok=True)

ROOT = PROJ / "data" / "raw" / "cscan"

_orig_load = common.load


def filtered_load() -> Stack:
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


def _xy_band(stack, x_lo, x_hi, y_lo, y_hi) -> np.ndarray:
    cl = int(np.searchsorted(stack.x_mm, x_lo)); ch = int(np.searchsorted(stack.x_mm, x_hi))
    rl = int(np.searchsorted(stack.y_mm, y_lo)); rh = int(np.searchsorted(stack.y_mm, y_hi))
    m = np.zeros_like(stack.roi_mask)
    m[rl:rh, cl:ch] = True
    # exclude any pixel inside the canonical cell-body ROI
    m &= ~stack.roi_mask
    return m


def _band_series(stack, mod, seg_idx, mask):
    arr = getattr(stack, mod)[seg_idx]
    flat = arr.reshape(arr.shape[0], -1)
    m = mask.reshape(-1)
    return np.nanmean(flat[:, m], axis=1)


def _pull_line_temperature(seg_meta, target_y_mm=36.0):
    """Get per-scan in-scan temperature: median across lines (bath) + line variability."""
    tmean_per_scan = []
    tspan_per_scan = []
    for _, row in seg_meta.iterrows():
        sess_dir = ROOT / Path(row["session_dir"].replace("\\", "/")).name
        if not sess_dir.exists():
            sess_dir = PROJ / Path(row["session_dir"].replace("\\", "/"))
        npz = next(sess_dir.glob("scan_*.npz"))
        d = np.load(npz)
        tm = d["line_temperature_mean_c"]
        tmin = d["line_temperature_min_c"]
        tmax = d["line_temperature_max_c"]
        tmean_per_scan.append(float(np.nanmean(tm)))
        tspan_per_scan.append(float(np.nanmax(tmax) - np.nanmin(tmin)))
    return np.array(tmean_per_scan), np.array(tspan_per_scan)


def main():
    stack = filtered_load()
    segs = rest_segments(stack.meta)
    i = int(np.argmin([abs(s.soc_pct.iloc[0] - 80.0) for s in segs]))
    seg = segs[i]
    seg_idx = np.where(np.isin(stack.meta.run_idx.values, seg.run_idx.values))[0]
    t_iso = pd.to_datetime(seg.time_utc)
    t_min = (t_iso - t_iso.iloc[0]).dt.total_seconds().values / 60.0

    # Build reference masks: regions outside ROI but within scan
    # Far-left fixture: X 0-10 mm
    # Far-right fixture: X 70-80 mm
    # Far-top: Y 0-12 mm (cross full X range outside ROI)
    # Far-bottom: Y 58-72 mm
    masks = {
        "far-left fixture (X<10)":    _xy_band(stack, 0, 10, 0, 72),
        "far-right fixture (X>70)":   _xy_band(stack, 70, 80, 0, 72),
        "far-top (Y<12)":             _xy_band(stack, 0, 80, 0, 12),
        "far-bottom (Y>58)":          _xy_band(stack, 0, 80, 58, 72),
        "oil ring (just outside ROI)": _xy_band(stack, 6, 70, 6, 64) & ~_xy_band(stack, 14.6, 64.5, 16.11, 55.38),
        "CELL ROI (the test signal)": stack.roi_mask,
    }
    # The oil ring needs to be off-ROI specifically
    oil_ring_box = np.zeros_like(stack.roi_mask)
    cl_in = int(np.searchsorted(stack.x_mm, 14.59))
    ch_in = int(np.searchsorted(stack.x_mm, 64.45))
    rl_in = int(np.searchsorted(stack.y_mm, 16.11))
    rh_in = int(np.searchsorted(stack.y_mm, 55.38))
    box_outer = np.zeros_like(stack.roi_mask)
    cl_o = int(np.searchsorted(stack.x_mm, 11))
    ch_o = int(np.searchsorted(stack.x_mm, 69))
    rl_o = int(np.searchsorted(stack.y_mm, 12))
    rh_o = int(np.searchsorted(stack.y_mm, 60))
    box_outer[rl_o:rh_o, cl_o:ch_o] = True
    box_inner = np.zeros_like(stack.roi_mask)
    box_inner[rl_in:rh_in, cl_in:ch_in] = True
    masks["oil ring (outside cell)"] = box_outer & ~box_inner & ~stack.roi_mask
    masks.pop("oil ring (just outside ROI)", None)

    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("region pixel counts:", counts)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True, sharex=True)
    units = {"amplitude": ("mV", 1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
             "tof":       ("ns", 1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
             "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"])}

    summary = ["# Reference-region drift check (SOC=80% rest)\n\n",
               "If reference regions drift in the same 54-99 min window as the ROI,\n",
               "the apparent acceleration is system artifact (bath/coupling), not cell physics.\n\n"]

    for r, mod in enumerate(["amplitude", "tof", "energy"]):
        unit, scale, sigma = units[mod]
        ax = axes[r]
        summary.append(f"## {mod}  (sigma_ROI = {sigma:.3g} {unit})\n\n")
        summary.append(f"| region | Δ@t=18 min | Δ@t=54 min | Δ@t=99 min | rate ratio (late/early) |\n|---|---:|---:|---:|---:|\n")
        for label, m in masks.items():
            if m.sum() == 0:
                continue
            y = _band_series(stack, mod, seg_idx, m) * scale
            d = y - y[0]
            cd = np.abs(np.diff(y))
            early = float(np.mean(cd[:3]))
            late = float(np.mean(cd[-3:]))
            ratio = late / max(early, 1e-9)
            ls = "-" if label == "CELL ROI (the test signal)" else "--"
            lw = 2.0 if label == "CELL ROI (the test signal)" else 1.0
            ax.plot(t_min, d, ls, lw=lw, marker="o", ms=4, label=f"{label} (n={int(m.sum())})")
            v18 = float(d[2]); v54 = float(d[6]); v99 = float(d[-1])
            summary.append(f"| {label} | {v18:+.3g} | {v54:+.3g} | {v99:+.3g} | {ratio:.2f} |\n")
        sigma_disp = sigma
        ax.fill_between(t_min, -2 * sigma_disp, 2 * sigma_disp, color="gray", alpha=0.15, label="ROI 2σ noise floor")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel(f"Δ{mod} from t=0 [{unit}]")
        ax.set_title(f"{mod} drift during SOC≈80% rest — cell ROI vs reference regions")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        summary.append("\n")
    axes[-1].set_xlabel("rest time [min]")
    fig.savefig(OUT / "reference_drift_amp_tof_energy.png", dpi=130)
    plt.close(fig)

    # Cell-side thermocouple from in-scan line_temperature_mean_c
    cp_idx = np.where(np.isin(stack.meta.run_idx.values, seg.run_idx.values))[0]
    seg_meta_full = stack.meta.iloc[cp_idx].copy().reset_index(drop=True)
    try:
        tmean_per_scan, tspan_per_scan = _pull_line_temperature(seg_meta_full)
        cycler_T_C = seg.temp_at_scan.values

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True, sharex=True)
        axes[0].plot(t_min, (tmean_per_scan - tmean_per_scan[0]) * 1000, "o-", color="tab:red", label="line_temperature_mean (TC08 on cell)")
        axes[0].plot(t_min, (cycler_T_C - cycler_T_C[0]) * 1000, "s--", color="tab:blue", label="cycler probe T")
        axes[0].axhline(0, color="k", lw=0.4)
        axes[0].set_ylabel("ΔT from t=0 [mC]")
        axes[0].set_title("Cell-side temperature during SOC≈80% rest")
        axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].plot(t_min, tspan_per_scan * 1000, "o-", color="#222")
        axes[1].set_ylabel("intra-scan T span [mC]")
        axes[1].set_xlabel("rest time [min]")
        axes[1].set_title("intra-scan temperature range (proxy for ongoing thermal transients during a scan)")
        axes[1].grid(alpha=0.3)
        fig.savefig(OUT / "cell_temperature_during_rest.png", dpi=130)
        plt.close(fig)

        summary.append(f"\n## Cell-side temperature (during this rest)\n\n")
        summary.append(f"- TC08 line-mean ΔT(0→99 min): **{(tmean_per_scan[-1]-tmean_per_scan[0])*1000:+.1f} mC**\n")
        summary.append(f"- Cycler probe ΔT(0→99 min): **{(cycler_T_C[-1]-cycler_T_C[0])*1000:+.1f} mC**\n")
        summary.append(f"- TC08 first-50min drift: {(tmean_per_scan[6]-tmean_per_scan[0])*1000:+.1f} mC; last-50min: {(tmean_per_scan[-1]-tmean_per_scan[6])*1000:+.1f} mC\n")
        summary.append(f"- Mean intra-scan T span: {tspan_per_scan.mean()*1000:.1f} mC (worst {tspan_per_scan.max()*1000:.1f} mC)\n")
    except Exception as e:
        print(f"could not load line temperatures: {e}")
        summary.append(f"\n## Cell-side temperature  — could not load: {e}\n")

    (OUT / "README.md").write_text("".join(summary), encoding="utf-8")
    print(f"wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
