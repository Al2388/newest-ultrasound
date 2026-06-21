"""35C: within-rest evolution of acoustic features, per SOC plateau and sub-band.

3 x 3 grid (rows = sub-band, cols = modality).  In each cell:
  x  = step_time_relative_min  (time since the rest step started, minutes)
  y  = feature(t) - feature(t0)   (anchored so each plateau starts at 0)
  one curve per SOC plateau (20 / 40 / 60 / 80 %)
  exponential fit overlaid: y(t) = f_inf * (1 - exp(-t/tau))
  tau per (band, modality, SOC) printed and saved as CSV.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus"
OUT_PNG = OUT_DIR / "within_rest_per_band.png"
OUT_CSV = OUT_DIR / "within_rest_per_band_tau.csv"

SUBREGIONS = {
    "tab-distal (left)":    (14.0, 30.0),
    "interior (middle)":    (30.0, 50.0),
    "tab-proximal (right)": (50.0, 64.5),
}
SOC_COLORS = {20.0: "#3b75c6", 40.0: "#3aa86c", 60.0: "#e08e1c", 80.0: "#c34141"}
MODS   = ["tof", "amplitude", "energy"]
SCALES = {"tof": 1e3, "amplitude": 1e3, "energy": 1.0}
UNITS  = {"tof": "ns", "amplitude": "mV", "energy": "a.u."}


def _xband(roi, x_mm, lo, hi):
    cl = int(np.searchsorted(x_mm, lo)); ch = int(np.searchsorted(x_mm, hi))
    m = np.zeros_like(roi, dtype=bool); m[:, cl:ch] = True
    return m & roi


def _fit_relax(t, y):
    """y(t) = f_inf * (1 - exp(-t/tau)), with t,y anchored so y(0)=0."""
    if t.size < 4 or not np.all(np.isfinite(y)):
        return np.nan, np.nan, np.nan
    p0 = (y[-1] if abs(y[-1]) > 1e-9 else np.sign(y.mean()) * max(abs(y).max(), 1e-6),
          max((t[-1] - t[0]) / 3, 5.0))
    try:
        popt, _ = curve_fit(lambda x, f_inf, tau: f_inf * (1 - np.exp(-x / tau)),
                            t, y, p0=p0, maxfev=2000)
        f_inf, tau = popt
        if tau <= 0 or tau > 20 * (t[-1] - t[0]) or not np.isfinite(tau):
            return np.nan, np.nan, np.nan
        resid = y - f_inf * (1 - np.exp(-t / tau))
        rms = float(np.sqrt(np.mean(resid ** 2)))
        return float(f_inf), float(tau), rms
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    roi, x_mm = d["roi_mask"], d["x_mm"]
    arrs = {"tof": d["tof"], "amplitude": d["amplitude"], "energy": d["energy"]}

    # Per-scan band means
    sub_masks = {name: _xband(roi, x_mm, lo, hi)
                 for name, (lo, hi) in SUBREGIONS.items()}
    band_means = {}
    for mod, arr in arrs.items():
        flat = arr.reshape(arr.shape[0], -1)
        for name, mask in sub_masks.items():
            band_means[(name, mod)] = (np.nanmean(flat[:, mask.reshape(-1)], axis=1)
                                        * SCALES[mod])

    rest = meta[meta["step_tag"] == "rest"].copy()
    rest["soc_plateau_label"] = rest["soc_plateau_label"].astype(float)
    plateaus = sorted(SOC_COLORS.keys())

    fig, axes = plt.subplots(len(SUBREGIONS), len(MODS),
                              figsize=(15, 11), sharex=True,
                              constrained_layout=True)

    summary_rows = []

    for r, (band_name, _) in enumerate(SUBREGIONS.items()):
        for c, mod in enumerate(MODS):
            ax = axes[r, c]
            yfull = band_means[(band_name, mod)]
            for soc in plateaus:
                sel = (rest["soc_plateau_label"].values == soc)
                if sel.sum() < 3:
                    continue
                order = np.argsort(rest["step_time_relative_min"].values[sel])
                t = rest["step_time_relative_min"].values[sel][order]
                y = yfull[rest.index.values[sel][order]]
                t0, y0 = t[0], y[0]
                t_rel = t - t0
                y_rel = y - y0

                color = SOC_COLORS[soc]
                ax.plot(t_rel, y_rel, "o", color=color, ms=4, alpha=0.85, zorder=3)

                f_inf, tau, rms = _fit_relax(t_rel, y_rel)
                if np.isfinite(tau):
                    xs = np.linspace(0, t_rel.max(), 200)
                    ax.plot(xs, f_inf * (1 - np.exp(-xs / tau)),
                            "-", color=color, lw=1.6, alpha=0.9,
                            label=f"SOC {int(soc)}%  τ={tau:.1f} min  Δ∞={f_inf:+.2f}")
                else:
                    ax.plot([], [], "-", color=color, lw=1.6, alpha=0.9,
                            label=f"SOC {int(soc)}%  (fit failed)")
                summary_rows.append({
                    "band": band_name, "modality": mod, "soc_pct": soc,
                    "tau_min": tau, "delta_inf": f_inf, "fit_rms": rms,
                    "n_scans": int(sel.sum()), "t_max_min": float(t_rel.max()),
                    "unit": UNITS[mod],
                })

            ax.axhline(0, color="k", lw=0.5, alpha=0.6)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7.5, loc="best", framealpha=0.85)
            if r == 0:
                ax.set_title(mod.upper(), fontsize=12, weight="bold")
            if r == len(SUBREGIONS) - 1:
                ax.set_xlabel("time into rest [min]")
            if c == 0:
                ax.set_ylabel(f"{band_name}\nΔ {mod} [{UNITS[mod]}]")
            else:
                ax.set_ylabel(f"Δ {mod} [{UNITS[mod]}]")

    fig.suptitle("35°C  within-rest evolution per SOC plateau and sub-band\n"
                 "Δ feature = feature(t) − feature(t₀);  fit  y(t) = Δ∞ · (1 − e^{−t/τ})",
                 fontsize=12)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"wrote {OUT_CSV}")

    print("\n=== tau summary (minutes) -- within-rest relaxation, 35C ===")
    piv = df.pivot_table(index=["band", "modality"], columns="soc_pct",
                          values="tau_min")
    with pd.option_context("display.float_format", "{:.1f}".format):
        print(piv)
    print("\n=== dF_inf summary (asymptotic drift from rest-start) ===")
    pivd = df.pivot_table(index=["band", "modality"], columns="soc_pct",
                           values="delta_inf")
    with pd.option_context("display.float_format", "{:+.2f}".format):
        print(pivd)


if __name__ == "__main__":
    main()
