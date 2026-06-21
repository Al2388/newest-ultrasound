"""35C: within-rest evolution as a stacked waterfall -- one figure, three panels.

Each modality (ToF / amplitude / energy) gets one panel.  Inside each panel,
the four SOC plateaus (80/60/40/20%) are stacked top-to-bottom as offset
"strips" sharing the same x-axis.  Each strip carries three colored curves
for the sub-bands.  Curves are anchored so each plateau starts at Δ = 0,
making the *shape* of the relaxation (not the absolute level) the visible
quantity.

Conventions:
  - SOC ordering top→bottom: 80 / 60 / 40 / 20 (high → low SOC reads top→bottom).
  - Per-modality offset chosen from the data's own range × 1.6, so curves
    cannot touch across strips.
  - Light alternating background per strip + SOC label on the left margin.
  - Zero-line per strip is dashed grey.
  - Bands colored: tab-distal green, interior purple, tab-prox orange,
    matching the reference figure the user shared.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus"
OUT_PNG = OUT_DIR / "within_rest_waterfall.png"

SUBREGIONS = {
    "tab-distal": (14.0, 30.0, "#2e9c54"),
    "interior":   (30.0, 50.0, "#7d6cc3"),
    "tab-prox":   (50.0, 64.5, "#e08e1c"),
}
SOC_PLATEAUS = [80.0, 60.0, 40.0, 20.0]   # top-to-bottom
MODS   = ["tof", "amplitude", "energy"]
SCALES = {"tof": 1e3, "amplitude": 1e3, "energy": 1.0}
UNITS  = {"tof": "ns", "amplitude": "mV", "energy": "a.u."}


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

    # Collect per-curve relative data first so we can scale offsets sanely
    per_curve = {mod: {} for mod in MODS}  # (band, soc) -> (t_rel, y_rel)
    x_max_global = 0.0
    for soc in SOC_PLATEAUS:
        sel = rest["soc_plateau_label"].values == soc
        if sel.sum() < 3:
            continue
        order = np.argsort(rest["step_time_relative_min"].values[sel])
        t = rest["step_time_relative_min"].values[sel][order]
        t_rel = t - t[0]
        x_max_global = max(x_max_global, float(t_rel.max()))
        for band_name in SUBREGIONS:
            y = band_means[(band_name, MODS[0])][rest.index.values[sel][order]]
            for mod in MODS:
                y_full = band_means[(band_name, mod)][rest.index.values[sel][order]]
                per_curve[mod][(band_name, soc)] = (t_rel, y_full - y_full[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), constrained_layout=True)

    for ax, mod in zip(axes, MODS):
        unit = UNITS[mod]
        # Per-modality dynamic range -> strip offset
        all_y = np.concatenate([per_curve[mod][k][1] for k in per_curve[mod]])
        full_range = float(np.nanmax(all_y) - np.nanmin(all_y))
        offset_step = max(full_range * 1.6, 1e-9)

        # Stripe backgrounds + zero lines + SOC labels
        for i, soc in enumerate(SOC_PLATEAUS):
            y_center = (len(SOC_PLATEAUS) - 1 - i) * offset_step
            ax.axhspan(y_center - offset_step / 2, y_center + offset_step / 2,
                       facecolor=("#f7f7fa" if i % 2 == 0 else "#ffffff"),
                       edgecolor="none", zorder=0)
            ax.axhline(y_center, color="#999999", lw=0.7, ls="--",
                        alpha=0.7, zorder=1)
            ax.text(-0.022 * x_max_global, y_center,
                     f"SOC\n{int(soc)}%",
                     ha="right", va="center", fontsize=10,
                     color="#333", weight="bold")

        # Curves
        seen = set()
        for i, soc in enumerate(SOC_PLATEAUS):
            y_center = (len(SOC_PLATEAUS) - 1 - i) * offset_step
            for band_name, (_, _, color) in SUBREGIONS.items():
                if (band_name, soc) not in per_curve[mod]:
                    continue
                t_rel, y_rel = per_curve[mod][(band_name, soc)]
                lbl = band_name if band_name not in seen else None
                seen.add(band_name)
                ax.plot(t_rel, y_rel + y_center,
                         color=color, lw=2.0, alpha=0.95,
                         label=lbl, zorder=3)
                ax.scatter(t_rel, y_rel + y_center,
                            color=color, s=14, alpha=0.95,
                            edgecolors="white", linewidths=0.4, zorder=4)

        ax.set_xlim(-0.04 * x_max_global, x_max_global * 1.02)
        ymin = -offset_step / 2
        ymax = (len(SOC_PLATEAUS) - 1) * offset_step + offset_step / 2
        ax.set_ylim(ymin, ymax)

        # Hide y-tick labels (offsets aren't meaningful) but add a small
        # scale bar showing what one strip's height corresponds to.
        ax.set_yticks([])
        # Scale bar in lower-right of the panel
        sb_y_lo = ymin + offset_step * 0.10
        sb_y_hi = sb_y_lo + offset_step * 0.5
        sb_x = x_max_global * 0.985
        ax.plot([sb_x, sb_x], [sb_y_lo, sb_y_hi],
                 color="black", lw=2.2, zorder=5)
        ax.text(sb_x - x_max_global * 0.012,
                 (sb_y_lo + sb_y_hi) / 2,
                 f"{(sb_y_hi - sb_y_lo):.1f} {unit}",
                 ha="right", va="center", fontsize=9, color="black")

        ax.set_xlabel("time into rest (min)")
        ax.set_title(f"{mod.upper()}  (Δ {mod} vs t)", fontsize=12, weight="bold")
        ax.grid(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=3,
                   bbox_to_anchor=(0.99, 1.08))

    fig.suptitle("35°C  within-rest evolution per SOC plateau and sub-band  —  waterfall "
                 "(each strip Δ-anchored so the relaxation shape is the visible quantity)",
                 fontsize=12, y=1.02)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
