"""Re-run all longrun analyses keeping only charge + rest-between-charge.

Drops discharge (11 scans) and transition (15 scans), re-zeros SOC so that
the first remaining scan is at SOC=0%. Then runs analyses 2-6 (analysis 01
hysteresis no longer makes sense — needs discharge).

Outputs to reports/longrun_cycling_22h_charge_focus/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
LONGRUN = PROJ / "scripts" / "longrun_analysis"
sys.path.insert(0, str(LONGRUN))

import common  # noqa: E402
from common import Stack  # noqa: E402

NEW_OUT = PROJ / "reports" / "longrun_cycling_22h_charge_focus"
NEW_OUT.mkdir(parents=True, exist_ok=True)
common.OUT_ROOT = NEW_OUT
common.CACHE_PATH = NEW_OUT / "_cache" / "stack.npz"
common.META_PATH = NEW_OUT / "_cache" / "meta.csv"

_KEEP_TAGS = {"charge", "rest"}
_orig_load = common.load


def filtered_load(rebuild: bool = False) -> Stack:
    s = _orig_load(rebuild=False)
    keep = s.meta.step_tag.isin(_KEEP_TAGS).values
    idx = np.where(keep)[0]
    new_meta = s.meta.iloc[idx].copy().reset_index(drop=True)
    soc_min = float(new_meta.soc_pct.min())
    new_meta["soc_pct"] = new_meta["soc_pct"] - soc_min
    print(f"charge+rest filter: kept {len(new_meta)}/{len(s.meta)} scans; SOC shifted by {-soc_min:+.2f}%")
    print(f"  new SOC range: [{new_meta.soc_pct.min():.1f}, {new_meta.soc_pct.max():.1f}]%")
    print(f"  charge n={int((new_meta.step_tag=='charge').sum())}, rest n={int((new_meta.step_tag=='rest').sum())}")
    return Stack(
        amplitude=s.amplitude[idx],
        tof=s.tof[idx],
        energy=s.energy[idx],
        x_mm=s.x_mm,
        y_mm=s.y_mm,
        roi_mask=s.roi_mask,
        meta=new_meta,
    )


common.load = filtered_load

import analysis_02_relaxation as A2  # noqa: E402
import analysis_03_beta_soc as A3  # noqa: E402
import analysis_04_pca as A4  # noqa: E402
import analysis_05_reversibility as A5  # noqa: E402
import analysis_06_animation as A6  # noqa: E402

ANALYSES = [
    ("02_relaxation", A2),
    ("03_beta_soc_map", A3),
    ("04_pca_svd", A4),
    ("05_equilibrium_evolution", A5),
    ("06_animation", A6),
]


def main() -> None:
    only = set(sys.argv[1:]) or None
    for name, mod in ANALYSES:
        if only and not any(name.startswith(o) or o in name for o in only):
            continue
        print(f"\n=== {name} ===")
        mod.main()
    print("\nall done.")


if __name__ == "__main__":
    main()
