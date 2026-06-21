"""35C: within-rest evolution, 4x4 panels (Δtemp + modality x SOC), Δ-anchored.

Same layout as the reference figure (columns = SOC 20 / 40 / 60 / 80 %),
with a top row showing the per-scan cell-line temperature envelope and
three modality rows (ToF / amplitude / energy) below.

Every curve is anchored so the first scan of each plateau is at Δ = 0.
Acoustic rows carry a translucent gray ±2σ_ROI noise-floor band so that
sub-band excursions are interpretable as significant vs noise.

Each acoustic cell carries three colored sub-band traces:
  tab-distal (green), mid (purple), tab-prox (orange).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus"
OUT_PNG = OUT_DIR / "within_rest_socs_panels_delta.png"

SUBREGIONS = {
    "tab-distal": (14.0, 30.0, "#2e9c54"),
    "mid":        (30.0, 50.0, "#7d6cc3"),
    "tab-prox":   (50.0, 64.5, "#e08e1c"),
}
SOC_PLATEAUS = [20.0, 40.0, 60.0, 80.0]
MODS   = ["tof", "amplitude", "energy"]
SCALES = {"tof": 1e3, "amplitude": 1e3, "energy": 1.0}
UNITS  = {"tof": "ns", "amplitude": "mV", "energy": "a.u."}
SIGMA  = {                    # σ_ROI per modality, scaled to display units
    "tof":       NOISE_FLOOR["tof_sigma_roi_ns"],      # ns
    "amplitude": NOISE_FLOOR["amp_sigma_roi_mv"],      # mV
    "energy":    NOISE_FLOOR["energy_sigma_roi"],      # a.u.
}
T_COLOR = "#b03060"

# Safety buffer applied to BOTH the noise floor and the thermal envelope.
BUFFER = 2.0
# ToF thermal sensitivity: median cell-region ToF shifted ~800 ns over 10.87 C
# between the 25.0 C and 35.87 C calibration scans.
TOF_THERMAL_NS_PER_C = 73.6
THERMAL_BAND_COLOR   = "#b03060"


def _xband(roi, x_mm, lo, hi):
    cl = int(np.searchsorted(x_mm, lo)); ch = int(np.searchsorted(x_mm, hi))
    m = np.zeros_like(roi, dtype=bool); m[:, cl:ch] = True
    return m & roi


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    roi, x_mm = d["roi_mask"], d["x_mm"]
    arrs = {"tof": d["tof"], "amplitude": d["amplitude"], "energy": d["energy"]}

    sub_masks = {name: _xband(roi, x_mm, lo, hi)
                 for name, (lo, hi, _) in SUBREGIONS.items()}
    band_means = {}
    for mod, arr in arrs.items():
        flat = arr.reshape(arr.shape[0], -1)
        for name, mask in sub_masks.items():
            band_means[(name, mod)] = (np.nanmean(flat[:, mask.reshape(-1)], axis=1)
                                        * SCALES[mod])

    rest = meta[meta["step_tag"] == "rest"].copy()
    rest["soc_plateau_label"] = rest["soc_plateau_label"].astype(float)

    plt.rcParams.update({
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.9,
        "axes.edgecolor":    "#444444",
        "axes.labelcolor":   "#222222",
        "xtick.color":       "#444444",
        "ytick.color":       "#444444",
        "axes.titlecolor":   "#111111",
        "figure.dpi":        140,
    })

    n_rows = len(MODS)
    fig, axes = plt.subplots(n_rows, len(SOC_PLATEAUS),
                              figsize=(17, 14), sharex=True, sharey="row",
                              gridspec_kw=dict(hspace=0.22, wspace=0.08))

    # Per-row symmetric y-limits so all 4 SOC columns of a modality share scale
    per_row_max = {mod: 0.0 for mod in MODS}
    cache_curves = {}     # (mod_idx, c, band) -> (t_rel, y_rel)
    cache_temp   = {}     # c -> (t_rel, dT_mC)
    dT_max_mc    = 0.0

    for c, soc in enumerate(SOC_PLATEAUS):
        sel = rest["soc_plateau_label"].values == soc
        if sel.sum() < 3:
            continue
        order = np.argsort(rest["step_time_relative_min"].values[sel])
        sel_idx = rest.index.values[sel][order]
        t = rest["step_time_relative_min"].values[sel][order]
        t_rel = t - t[0]

        # Temperature envelope (mC), anchored at rest start
        T_C = rest["line_T_mean_c"].values[sel][order]
        dT_mC = (T_C - T_C[0]) * 1000.0
        cache_temp[c] = (t_rel, dT_mC)
        dT_max_mc = max(dT_max_mc, float(np.nanmax(np.abs(dT_mC))))

        for r, mod in enumerate(MODS):
            for band_name in SUBREGIONS:
                y = band_means[(band_name, mod)][sel_idx]
                y_rel = y - y[0]
                cache_curves[(r, c, band_name)] = (t_rel, y_rel)
                m = float(np.nanmax(np.abs(y_rel)))
                if m > per_row_max[mod]:
                    per_row_max[mod] = m

    # ---- Acoustic rows --------------------------------------------------------
    seen_legend = set()
    for r, mod in enumerate(MODS):
        ax_row = r
        unit = UNITS[mod]
        sigma = SIGMA[mod]
        y_lim = max(per_row_max[mod] * 1.15, 2.5 * sigma) if per_row_max[mod] > 0 else 1.0

        for c, soc in enumerate(SOC_PLATEAUS):
            ax = axes[ax_row, c]
            # Buffered noise-floor band  (BUFFER x sigma_ROI)
            band = BUFFER * sigma
            ax.axhspan(-band, +band, color="#7f7f7f", alpha=0.15,
                        lw=0, zorder=1,
                        label=f"±{BUFFER:.0f}·σ_ROI = ±{band:.2g} {unit}"
                              if (c == 0) else None)

            # Thermal envelope -- only meaningful for ToF (sensitivity 73.6 ns/C
            # from the cell-region calibration figure).  Drawn as a pink
            # time-varying band: +/- BUFFER * sens * |dT(t)|.
            if mod == "tof" and c in cache_temp:
                t_rel_T, dT_mC = cache_temp[c]
                thermal_halfband = BUFFER * TOF_THERMAL_NS_PER_C * np.abs(dT_mC) / 1000.0
                ax.fill_between(t_rel_T, -thermal_halfband, +thermal_halfband,
                                 color=THERMAL_BAND_COLOR, alpha=0.18, lw=0,
                                 zorder=2,
                                 label=(f"±{BUFFER:.0f}·{TOF_THERMAL_NS_PER_C:.1f} ns/°C · |ΔT|"
                                        if c == 0 else None))

            ax.axhline(0, color="#666", lw=0.7, alpha=0.7, zorder=3)
            ax.grid(axis="y", alpha=0.22, color="#cccccc", lw=0.6, zorder=0)

            for band_name, (_, _, color) in SUBREGIONS.items():
                if (r, c, band_name) not in cache_curves:
                    continue
                t_rel, y_rel = cache_curves[(r, c, band_name)]
                lbl = band_name if (band_name not in seen_legend) else None
                seen_legend.add(band_name)
                ax.plot(t_rel, y_rel, "-", color=color, lw=2.0, alpha=0.95,
                         zorder=4, label=lbl)
                ax.scatter(t_rel, y_rel, color=color, s=26, alpha=0.95,
                            edgecolors="white", linewidths=0.6, zorder=5)

            ax.set_ylim(-y_lim, y_lim)
            ax.set_xlim(-3, 122)
            ax.tick_params(direction="out", length=3.5)
            if ax_row == 0:
                ax.set_title(f"{int(soc)} % SOC", fontsize=12, weight="bold", pad=8)
            if ax_row == n_rows - 1:
                ax.set_xlabel("rest time (min)")
            if c == 0:
                ax.set_ylabel(f"Δ {mod.upper()}  ({unit})",
                              fontsize=12, weight="bold", labelpad=10)
                ax.text(-0.34, 0.5, "anchored at t₀",
                         transform=ax.transAxes, rotation=90,
                         ha="center", va="center",
                         fontsize=9, color="#888888")

    # Combined legend: temperature row + acoustic bands + noise band
    handles, labels = [], []
    seen = set()
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l in seen:
                continue
            seen.add(l)
            handles.append(h); labels.append(l)
    leg = fig.legend(handles, labels, loc="upper center",
                      ncol=len(handles), bbox_to_anchor=(0.5, 0.97),
                      fontsize=11, frameon=False, handlelength=2.2,
                      columnspacing=2.0)
    for line in leg.get_lines():
        try:
            line.set_linewidth(3.0)
        except AttributeError:
            pass

    fig.suptitle("35 °C  ·  within-rest evolution per SOC plateau and sub-band",
                  fontsize=14, weight="bold", y=1.005)
    fig.text(0.5, 0.965,
              "each cell anchored so Δ feature = 0 at the first scan of the plateau;  "
              f"y-axes shared per row;  grey band = ±{BUFFER:.0f}·σ_ROI noise floor;  "
              f"pink band on ToF row = ±{BUFFER:.0f}·{TOF_THERMAL_NS_PER_C:.1f} ns/°C · |ΔT(t)| "
              "thermal envelope",
              ha="center", va="bottom", fontsize=10, color="#666666")

    fig.subplots_adjust(top=0.90, left=0.07, right=0.985, bottom=0.07)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
