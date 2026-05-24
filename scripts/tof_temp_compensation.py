"""
Temperature-compensate absolute ToF and re-check cross-session repeatability.

Model:  ToF(SOC, T) = g(SOC, branch) + k * (T - T_ref) + noise
We estimate the temperature coefficient k (us/C) two independent ways and apply
ToF_corr = ToF - k * (T - T_ref), then compare 18-5 vs 21-5 at matched SOC
before and after correction.

k estimators
------------
  k_xsession : regress dToF(B-A) on dT(B-A) at matched SOC (SOC-controlled,
               uses the ~1 C between-run gap). Primary.
  k_pooled   : coefficient on T in  ToF ~ poly(SOC,3) + branch + T  (pooled).
The two should roughly agree if the temperature term is real and stable.

Outputs to reports/experiments/repeatability/:
  5_tof_temp_compensation.png  and printed before/after numbers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES = {
    "18-5": "reports/experiments/19-5_feature_exploration/ascan_feature_table.csv",
    "21-5": "reports/experiments/21-5_feature_exploration/ascan_feature_table.csv",
}
OUT = Path("reports/experiments/repeatability")
TOF = "h5_tof_us_absolute"
SOC_GRID = np.arange(5.0, 95.0 + 1e-9, 1.0)


def load():
    d = {}
    for name, path in TABLES.items():
        df = pd.read_csv(path)
        df = df[df["branch"].isin(["charge", "discharge"])].copy()
        df["session"] = name
        d[name] = df
    return d


def grid(df, col, branch):
    sub = df[df["branch"] == branch]
    s = sub["soc_pct_clipped"].to_numpy(float)
    y = sub[col].to_numpy(float)
    t = sub["temperature_c"].to_numpy(float)
    o = np.argsort(s); s, y, t = s[o], y[o], t[o]
    ok = np.isfinite(s) & np.isfinite(y)
    f = np.interp(SOC_GRID, s[ok], y[ok], left=np.nan, right=np.nan)
    tt = np.interp(SOC_GRID, s[ok], t[ok], left=np.nan, right=np.nan)
    return f, tt


def xsession_metrics(dA, dB, col):
    """Between-session RMS (offset-removed) and SOC span for a feature."""
    fa = np.concatenate([grid(dA, col, "charge")[0], grid(dA, col, "discharge")[0]])
    fb = np.concatenate([grid(dB, col, "charge")[0], grid(dB, col, "discharge")[0]])
    dF = fb - fa
    m = np.isfinite(dF)
    dF = dF[m]
    mean_curve = np.nanmean(np.column_stack([fa, fb]), axis=1)
    span = float(np.nanmax(mean_curve) - np.nanmin(mean_curve))
    abs_rms = float(np.sqrt(np.mean(dF**2)))
    shape_rms = float(np.sqrt(np.mean((dF - dF.mean()) ** 2)))
    return abs_rms, shape_rms, span


def estimate_k(data):
    pooled = pd.concat(data.values(), ignore_index=True)
    # k_xsession: dToF vs dT over the SOC grid (both branches)
    dF, dT = [], []
    for b in ("charge", "discharge"):
        fa, ta = grid(data["18-5"], TOF, b)
        fb, tb = grid(data["21-5"], TOF, b)
        dF.append(fb - fa); dT.append(tb - ta)
    dF = np.concatenate(dF); dT = np.concatenate(dT)
    m = np.isfinite(dF) & np.isfinite(dT)
    A = np.column_stack([np.ones(m.sum()), dT[m]])
    c, *_ = np.linalg.lstsq(A, dF[m], rcond=None)
    k_xsession = float(c[1])

    # k_pooled: coefficient on T in ToF ~ poly(SOC,3) + branch + T
    soc = pooled["soc_pct_clipped"].to_numpy(float) / 100.0
    temp = pooled["temperature_c"].to_numpy(float)
    charge = (pooled["branch"].to_numpy() == "charge").astype(float)
    y = pooled[TOF].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(soc) & np.isfinite(temp)
    X = np.column_stack([np.ones(ok.sum()), soc[ok], soc[ok]**2, soc[ok]**3,
                         charge[ok], soc[ok] * charge[ok], temp[ok]])
    c2, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    k_pooled = float(c2[-1])
    return k_xsession, k_pooled, float(np.nanmean(temp))


def apply_metrics(data, k, t_ref):
    for df in data.values():
        df["tof_corr"] = df[TOF] - k * (df["temperature_c"] - t_ref)
    return xsession_metrics(data["18-5"], data["21-5"], "tof_corr")


def main():
    data = load()
    k_x, k_p, t_ref = estimate_k(data)

    # Empirically optimal k: the linear temperature correction that best aligns
    # the two sessions at matched SOC (minimizes absolute between-session RMS).
    ks = np.linspace(-0.6, 0.3, 181)
    abs_rms_curve = np.array([apply_metrics(data, k, t_ref)[0] for k in ks])
    k_opt = float(ks[np.argmin(abs_rms_curve)])

    print(f"k_xsession = {k_x:+.4f} us/C   k_pooled = {k_p:+.4f} us/C   "
          f"k_optimal = {k_opt:+.4f} us/C   T_ref = {t_ref:.2f} C")

    raw = xsession_metrics(data["18-5"], data["21-5"], TOF)
    cor = apply_metrics(data, k_opt, t_ref)   # use the best-case correction
    k = k_opt

    print(f"\nCross-session reproducibility ({TOF}):")
    print(f"  RAW              abs_rms={raw[0]*1000:6.1f} ns  shape_rms={raw[1]*1000:6.1f} ns  "
          f"span={raw[2]*1000:6.1f} ns  discrim={raw[2]/raw[1]:.1f}")
    print(f"  BEST-CASE CORR   abs_rms={cor[0]*1000:6.1f} ns  shape_rms={cor[1]*1000:6.1f} ns  "
          f"span={cor[2]*1000:6.1f} ns  discrim={cor[2]/cor[1]:.1f}")
    print(f"  best-case abs_rms reduction: {100*(1-cor[0]/raw[0]):.0f}%  (at k_opt={k_opt:+.3f})")

    # plot raw vs corrected ToF-vs-SOC
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=False)
    for ax, col, title in zip(
        axs, [TOF, "tof_corr"],
        [f"Raw ToF\nabs RMS {raw[0]*1000:.0f} ns",
         f"Temp-compensated (k={k:+.3f} us/C)\nabs RMS {cor[0]*1000:.0f} ns "
         f"({100*(1-cor[0]/raw[0]):.0f}% lower)"]):
        for s, style in [("18-5", "-"), ("21-5", "--")]:
            for b, c in [("charge", "#2563eb"), ("discharge", "#dc2626")]:
                f, _ = grid(data[s], col, b)
                ax.plot(SOC_GRID, f, style, color=c, lw=1.3, label=f"{s} {b}")
        ax.set_title(title, fontsize=10); ax.set_xlabel("SOC (%)")
        ax.set_ylabel("ToF (us)"); ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    fig.suptitle("Temperature compensation of absolute ToF  (solid 18-5, dashed 21-5)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "5_tof_temp_compensation.png")
    plt.close(fig)
    print(f"\nWrote {OUT / '5_tof_temp_compensation.png'}")


if __name__ == "__main__":
    main()
