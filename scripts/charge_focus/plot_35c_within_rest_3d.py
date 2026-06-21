"""35C: within-rest evolution as 3D ribbons -- one figure per modality.

Each modality (ToF / amplitude / energy) gets a single 3D axes with:
  x = rest_time (min)
  y = SOC (%)             -- 4 discrete planes at 20 / 40 / 60 / 80
  z = delta_feature(t) = feature(t) - feature(t0)   anchored at the rest start

Three colored curves per SOC plane: tab-distal (green), interior (purple),
tab-prox (orange).  A translucent vertical SOC slab is drawn behind each
plane to give visual depth; the projection on the (x, SOC) floor at z = 0
shows the trajectory shape independent of magnitude.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CACHE_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus"
OUT_PNG = OUT_DIR / "within_rest_3d.png"

SUBREGIONS = {
    "tab-distal": (14.0, 30.0, "#2e9c54"),
    "interior":   (30.0, 50.0, "#7d6cc3"),
    "tab-prox":   (50.0, 64.5, "#e08e1c"),
}
SOC_PLATEAUS = [20.0, 40.0, 60.0, 80.0]
MODS   = ["tof", "amplitude", "energy"]
SCALES = {"tof": 1e3, "amplitude": 1e3, "energy": 1.0}
UNITS  = {"tof": "ns", "amplitude": "mV", "energy": "a.u."}


def _xband(roi, x_mm, lo, hi):
    cl = int(np.searchsorted(x_mm, lo)); ch = int(np.searchsorted(x_mm, hi))
    m = np.zeros_like(roi, dtype=bool); m[:, cl:ch] = True
    return m & roi


def _draw_slab(ax, soc, x_max, z_lo, z_hi, color="#bbbbbb", alpha=0.05):
    """Translucent vertical SOC slab so the eye sees a 'plane' per plateau."""
    X = np.array([[0, x_max], [0, x_max]])
    Y = np.array([[soc, soc], [soc, soc]])
    Z = np.array([[z_lo, z_lo], [z_hi, z_hi]])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0,
                     antialiased=False, shade=False)


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

    fig = plt.figure(figsize=(22, 7.5))

    for col_i, mod in enumerate(MODS):
        ax = fig.add_subplot(1, 3, col_i + 1, projection="3d")
        unit = UNITS[mod]

        # Precompute global z range for slab heights
        z_all = []
        per_curve = {}
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
                y = band_means[(band_name, mod)][rest.index.values[sel][order]]
                y_rel = y - y[0]
                per_curve[(soc, band_name)] = (t_rel, y_rel)
                z_all.append(y_rel)
        z_lo = float(np.nanmin([z.min() for z in z_all])) * 1.1
        z_hi = float(np.nanmax([z.max() for z in z_all])) * 1.1
        if z_lo >= z_hi:
            z_lo, z_hi = -1.0, 1.0

        # SOC slabs (decorative depth)
        for soc in SOC_PLATEAUS:
            _draw_slab(ax, soc, x_max_global, z_lo, z_hi)
            # Zero-line on the slab so the eye has an anchor
            ax.plot([0, x_max_global], [soc, soc], [0, 0],
                    color="#999999", lw=0.6, alpha=0.6)

        # Curves
        seen_labels = set()
        for (soc, band_name), (t_rel, y_rel) in per_curve.items():
            color = SUBREGIONS[band_name][2]
            lbl = band_name if band_name not in seen_labels else None
            seen_labels.add(band_name)
            ax.plot(t_rel, [soc] * len(t_rel), y_rel,
                    color=color, lw=2.2, alpha=0.95, label=lbl, zorder=4)
            ax.scatter(t_rel, [soc] * len(t_rel), y_rel,
                        color=color, s=22, alpha=0.95, edgecolors="white",
                        linewidths=0.4, depthshade=False, zorder=5)
            # Floor projection: shape of trajectory only
            ax.plot(t_rel, [soc] * len(t_rel), [z_lo] * len(t_rel),
                    color=color, lw=0.9, alpha=0.35, zorder=2)

        ax.set_xlabel("rest time (min)", labelpad=8)
        ax.set_ylabel("SOC (%)", labelpad=8)
        ax.set_zlabel(f"Δ {mod} ({unit})", labelpad=6)
        ax.set_title(mod.upper(), fontsize=13, weight="bold", pad=4)
        ax.set_xlim(0, x_max_global)
        ax.set_ylim(15, 85)
        ax.set_zlim(z_lo, z_hi)
        ax.set_yticks(SOC_PLATEAUS)
        ax.view_init(elev=22, azim=-62)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.7))
        ax.grid(True, alpha=0.25)

    fig.suptitle("35°C  within-rest evolution per SOC plateau and sub-band  —  "
                 "Δ feature = feature(t) − feature(t₀)",
                 fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
