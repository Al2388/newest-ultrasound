"""Regenerate scan_amp.png / scan_tof.png / scan_eng.png in each charge-focus
session directory from the PATCHED cache stack.

Mirrors the webapp's `_save_plot` style (turbo cmap, 5-95 percentile, NaN white,
nearest interpolation) so the new PNGs look like the originals -- just without
the missing-line artefacts.

Originals are backed up as scan_<mod>_original.png before being overwritten.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
CACHE = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"

CMAP = "turbo"
MOD_TO_PNG = {"amplitude": "scan_amp.png",
              "tof":       "scan_tof.png",
              "energy":    "scan_eng.png"}
MOD_TO_LABEL = {"amplitude": "Amplitude",
                "tof":       "ToF",
                "energy":    "Energy"}


def render_one(img, out_path, label, line_total):
    """Match webapp _save_plot output style."""
    if not np.any(np.isfinite(img)):
        return
    cm = matplotlib.colormaps.get_cmap(CMAP).copy()
    cm.set_bad("white")
    finite = img[np.isfinite(img)]
    vmin, vmax = (np.percentile(finite, [5, 95]) if finite.size >= 16 else (0.0, 1.0))

    # roi geometry matches webapp config; for now use the C-scan extent
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    im = ax.imshow(img, cmap=cm, origin="upper", aspect="equal",
                   extent=[0, 80, 72, 0],
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"{label}  (Line {line_total}/{line_total})  [patched]")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    d = np.load(CACHE / "stack.npz")
    meta = pd.read_csv(CACHE / "meta.csv")
    arrs = {"amplitude": d["amplitude"],
            "tof":       d["tof"],
            "energy":    d["energy"]}

    n_rendered = 0
    n_backed_up = 0
    for i, row in meta.iterrows():
        sess = PROJ / row["session_dir"].replace("\\", "/")
        if not sess.exists():
            print(f"  WARN: session dir missing: {sess}")
            continue
        for mod, png_name in MOD_TO_PNG.items():
            target = sess / png_name
            backup = sess / png_name.replace(".png", "_original.png")
            if target.exists() and not backup.exists():
                shutil.copy2(target, backup)
                n_backed_up += 1
            render_one(arrs[mod][i], target,
                       MOD_TO_LABEL[mod], line_total=144)
            n_rendered += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>2}/93] sessions done")

    print(f"\ndone. rendered {n_rendered} PNGs, backed up {n_backed_up} originals.")
    print(f"original PNGs preserved as scan_*_original.png in each session dir.")


if __name__ == "__main__":
    main()
