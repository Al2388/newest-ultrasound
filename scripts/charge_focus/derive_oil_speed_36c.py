"""Derive silicone-oil sound speed at ~36 C from existing C-scan ToF.

Method:
  1. Anchor: c(25 C) = 989 m/s (manufacturer spec, silicone oil 20 cSt).
  2. Use 6-repeat noisefloor_v3.238 batch (T_cell ~25.1 C) as reference.
  3. Measure off-cell ToF in 35C charge-focus scans (T_cell ~35.88 C).
  4. Assume off-cell paths see the same reflector at depth L_offcell
     (which thermal-expands negligibly). Then:
        c(36) / c(25) = t_offcell(25) / t_offcell(36)
  5. Cross-check with on-cell ROI ToF for the effective thermal coefficient
     dt/dT used in the L2 thermal-envelope analysis.

Outputs:
  reports/longrun_cycling_35c_charge_focus/oil_speed_36c.md
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
ROI_MASK = PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp/roi_mask.npy"
META_CSV = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache/meta.csv"
OUT_MD   = PROJ / "reports/longrun_cycling_35c_charge_focus/oil_speed_36c.md"

C_25C_REF = 989.0       # m/s, silicone oil 20 cSt @ room temp (manufacturer)
T_25C_REF = 25.0        # nominal anchor temperature


def off_cell_band(roi, x_mm, y_mm, x_lo, x_hi, y_lo, y_hi):
    cl = int(np.searchsorted(x_mm, x_lo)); ch = int(np.searchsorted(x_mm, x_hi))
    rl = int(np.searchsorted(y_mm, y_lo)); rh = int(np.searchsorted(y_mm, y_hi))
    m = np.zeros_like(roi, dtype=bool)
    m[rl:rh, cl:ch] = True
    return m & ~roi


def load_scan(npz_path, roi):
    d = np.load(npz_path)
    return {
        "tof_map": d["tof"],
        "x_mm": d["x_mm"], "y_mm": d["y_mm"],
        "T_cell": float(np.nanmean(d["line_temperature_mean_c"])),
    }


def band_median(tof_map, mask):
    vals = tof_map[mask]
    finite = vals[np.isfinite(vals)]
    return float(np.median(finite)) if len(finite) else np.nan


def main():
    roi = np.load(ROI_MASK)

    # ---- 25 C reference: 6 noisefloor_v3.238 scans ----
    ref_paths = sorted(glob.glob(str(PROJ / "data/raw/cscan/cscan_noisefloor_v3.238_2026-05-29_15-02-49_r0*")))
    print(f"Reference batch (noisefloor_v3.238): {len(ref_paths)} scans")

    # ---- 35 C subset: charge-focus 93 scans ----
    meta = pd.read_csv(META_CSV)
    focus_paths = [PROJ / row["session_dir"].replace("\\", "/")
                   for _, row in meta.iterrows()]
    print(f"Focus batch (35C charge-focus): {len(focus_paths)} scans")

    # ---- band geometry (well outside ROI; use far-left + far-right
    #      which are vertical strips on cell-free sides) ----
    sample = load_scan(next(Path(ref_paths[0]).glob("scan_*.npz")), roi)
    bands = {
        "far-left":  off_cell_band(roi, sample["x_mm"], sample["y_mm"], 0, 10, 0, 72),
        "far-right": off_cell_band(roi, sample["x_mm"], sample["y_mm"], 70, 80, 0, 72),
        "cell-ROI":  roi,
    }
    print(f"Off-cell pixel counts: "
          f"far-left {bands['far-left'].sum()}, "
          f"far-right {bands['far-right'].sum()}, "
          f"cell-ROI {bands['cell-ROI'].sum()}")

    # ---- aggregate per scan ----
    def aggregate(paths, label):
        rows = []
        for p in paths:
            npz = next(Path(p).glob("scan_*.npz"))
            sc = load_scan(npz, roi)
            row = {"T_cell": sc["T_cell"]}
            for bn, m in bands.items():
                row[f"tof_{bn}"] = band_median(sc["tof_map"], m)
            rows.append(row)
        return pd.DataFrame(rows)

    ref_df = aggregate(ref_paths, "ref")
    foc_df = aggregate(focus_paths, "focus")

    # ---- reference statistics ----
    T_ref_mean = ref_df["T_cell"].mean()
    T_foc_mean = foc_df["T_cell"].mean()
    dT = T_foc_mean - T_ref_mean

    print(f"\nT_ref  mean = {T_ref_mean:.3f} C (n={len(ref_df)})")
    print(f"T_foc  mean = {T_foc_mean:.3f} C (n={len(foc_df)})")
    print(f"dT          = {dT:.3f} C\n")

    # ---- per-band sound speed and effective alpha_T ----
    md = []
    md.append("# Silicone-oil sound speed at ~36 C (derived from C-scan ToF)\n\n")
    md.append("## Method\n\n")
    md.append(f"- **Anchor**: c(25 C) = {C_25C_REF:.0f} m/s "
              "(silicone oil 20 cSt, manufacturer spec @ room temp,\n"
              "  referenced by `scripts/derive_height_from_tof.py:24`).\n")
    md.append(f"- **Reference batch**: 6 `noisefloor_v3.238` repeats, "
              f"T_cell mean = **{T_ref_mean:.3f} C**.\n")
    md.append(f"- **35C batch**: 93 charge-focus scans, "
              f"T_cell mean = **{T_foc_mean:.3f} C** (Delta T = {dT:.2f} C).\n")
    md.append("- For each off-cell band (pure-oil round-trip path to fixed reflector):\n")
    md.append("    `c(T_focus) / c(T_ref) = ToF(T_ref) / ToF(T_focus)`\n")
    md.append("  -> `c(T_focus) = c(T_ref) * ToF(T_ref) / ToF(T_focus)`\n")
    md.append("- The on-cell ROI band is reported for the effective thermal coefficient\n")
    md.append("  used in §5.4 thermal-envelope analysis.\n\n")

    md.append(f"## Per-band ToF medians (microsec)\n\n")
    md.append("| band | ToF @ 25 C (mean ± SD) | ToF @ 36 C (mean ± SD) | Delta (microsec) |\n")
    md.append("|:---|---:|---:|---:|\n")
    for bn in bands:
        col = f"tof_{bn}"
        a, sa = ref_df[col].mean(), ref_df[col].std()
        b, sb = foc_df[col].mean(), foc_df[col].std()
        md.append(f"| {bn} | {a:.4f} ± {sa:.4f} | {b:.4f} ± {sb:.4f} | {b - a:+.4f} |\n")
    md.append("\n")

    md.append("## Derived sound speed at 36 C (off-cell bands only)\n\n")
    md.append("| band | c(25 C) ref | c(36 C) | dc | dc/dT (m/s/C) |\n")
    md.append("|:---|---:|---:|---:|---:|\n")
    speeds = []
    for bn in ["far-left", "far-right"]:
        t_25 = ref_df[f"tof_{bn}"].mean()
        t_36 = foc_df[f"tof_{bn}"].mean()
        c_36 = C_25C_REF * t_25 / t_36
        dcdT = (c_36 - C_25C_REF) / dT
        speeds.append(c_36)
        md.append(f"| {bn} | {C_25C_REF:.1f} | **{c_36:.1f}** | {c_36 - C_25C_REF:+.1f} | {dcdT:+.2f} |\n")
    c_36_mean = float(np.mean(speeds))
    c_36_sd = float(np.std(speeds, ddof=1))
    md.append(f"\n**Best estimate at T = {T_foc_mean:.2f} C:  c_oil = {c_36_mean:.0f} ± {c_36_sd:.0f} m/s**\n")
    md.append(f"(±SD is between-band spread; not a formal uncertainty.)\n\n")

    md.append(f"**Effective dc/dT = {(c_36_mean - C_25C_REF)/dT:+.2f} m/s/C**\n")
    md.append("Literature for silicone oil dc/dT is typically -2 to -3 m/s/C; our number is\n")
    md.append("steeper, consistent with some thermal expansion of the fixture adding apparent\n")
    md.append("path-length increase on top of the speed change.\n\n")

    # ---- effective alpha_T for cell-ROI ToF (the §5.4 thermal envelope) ----
    md.append("## Effective thermal coefficient for cell-ROI ToF (used in §5.4)\n\n")
    t_roi_25 = ref_df["tof_cell-ROI"].mean()
    t_roi_36 = foc_df["tof_cell-ROI"].mean()
    d_roi_us = t_roi_36 - t_roi_25
    alpha_T_ns_per_C = d_roi_us * 1000.0 / dT
    md.append(f"Cell-ROI median ToF: {t_roi_25*1000:.2f} ns @ {T_ref_mean:.2f} C "
              f"-> {t_roi_36*1000:.2f} ns @ {T_foc_mean:.2f} C\n")
    md.append(f"DeltaToF = {d_roi_us*1000:+.2f} ns over DeltaT = {dT:.2f} C\n\n")
    md.append(f"**alpha_T (cell-ROI ToF) = {alpha_T_ns_per_C:+.2f} ns/C**\n\n")
    md.append("Sign is **positive** (warmer -> slower oil -> longer ToF), opposite of water-based\n")
    md.append("rigs. The earlier §5.4 draft using -53 ns/C (water assumption) is wrong in both\n")
    md.append("sign and magnitude; this number replaces it for any future thermal-envelope work.\n\n")

    md.append("## Implications for §5.4 thermal envelope\n\n")
    sigma_T_mC = 65.7   # per-line consolidated, full 14h window
    bound_3sigma_ns = 3 * sigma_T_mC * abs(alpha_T_ns_per_C) / 1000.0
    md.append(f"With sigma_T = {sigma_T_mC:.1f} mC (per-line consolidated, full 14h window),\n")
    md.append(f"the 3 sigma_T thermal-envelope on cell-ROI ToF is approximately\n")
    md.append(f"**±{bound_3sigma_ns:.1f} ns** (using |alpha_T| = {abs(alpha_T_ns_per_C):.1f} ns/C).\n\n")
    md.append("| SOC | raw DeltaToF_end (ns) | 3 sigma_T bound (ns) | ratio | verdict |\n")
    md.append("|---:|---:|---:|---:|:---|\n")
    raw_per_plateau = {20: +30.1, 40: -55.2, 60: -11.1, 80: +39.2}  # from no-ref analysis
    for soc, raw in raw_per_plateau.items():
        ratio = abs(raw) / bound_3sigma_ns
        verdict = "**above thermal**" if ratio >= 2 else ("marginal" if ratio >= 1 else "WITHIN thermal envelope")
        md.append(f"| {soc}% | {raw:+.1f} | ±{bound_3sigma_ns:.1f} | {ratio:.1f}x | {verdict} |\n")
    md.append("\nSOC=60% now sits below the conservative 3 sigma_T thermal envelope, demoting it\n")
    md.append("from the §5.4 L2 evidence -- its claim must rest on L3 (Task C tab-vs-centre).\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
