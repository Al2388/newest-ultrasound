"""Dump the underlying data for fig_evolution_amp_energy as CSVs.

Generates:
  data_amp_energy_per_scan.csv  -- per (SOC, scan) row: t_rest, amp ROI median,
                                    energy ROI median + companion sigma_ROI
  data_amp_energy_fits.csv      -- per (SOC, feature) row: Theil-Sen slope,
                                    intercept, slope*120 min, n_sigma_ROI
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import siegelslopes

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

CACHE = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
OUT_DIR = PROJ / "reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROI_X_MM = (14.6, 64.5)
ROI_Y_MM = (16.1, 55.4)
REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]

SIG_AMP = NOISE_FLOOR["amp_sigma_roi_mv"]     # 2.219 mV
SIG_ENG = NOISE_FLOOR["energy_sigma_roi"]      # 0.0814 a.u.


def build_roi(x_mm, y_mm):
    x_idx = np.where((x_mm >= ROI_X_MM[0]) & (x_mm <= ROI_X_MM[1]))[0]
    y_idx = np.where((y_mm >= ROI_Y_MM[0]) & (y_mm <= ROI_Y_MM[1]))[0]
    roi = np.zeros((y_mm.size, x_mm.size), dtype=bool)
    roi[np.ix_(y_idx, x_idx)] = True
    return roi


def main():
    d = np.load(CACHE / "stack.npz")
    meta = pd.read_csv(CACHE / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    roi = build_roi(d["x_mm"], d["y_mm"])
    amp_mV = d["amplitude"] * 1e3
    eng_au = d["energy"]

    per_scan_rows = []
    fit_rows = []

    for step, soc in zip(REST_STEPS, SOC_LABELS):
        rest = meta[meta["step"] == step].sort_values("timestamp").reset_index()
        t = (rest["timestamp"] - rest["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        idx = rest["index"].values
        amp_med = np.array([float(np.nanmedian(amp_mV[i][roi])) for i in idx])
        eng_med = np.array([float(np.nanmedian(eng_au[i][roi])) for i in idx])
        v_at_scan = rest["voltage_at_scan"].values * 1000
        T_at_scan = rest["line_T_mean_c"].values
        run_idx = rest["run_idx"].values

        for k in range(len(rest)):
            per_scan_rows.append({
                "SOC%": soc,
                "step": step,
                "run_idx": int(run_idx[k]),
                "scan_id": rest["scan_id"].iloc[k],
                "t_rest_min": float(t[k]),
                "amp_med_ROI_mV": float(amp_med[k]),
                "energy_med_ROI_au": float(eng_med[k]),
                "voltage_mV": float(v_at_scan[k]),
                "T_cell_C": float(T_at_scan[k]),
            })

        # Theil-Sen fits
        sa, ba = siegelslopes(amp_med, t)
        se, be = siegelslopes(eng_med, t)
        dAmp_120 = float(sa * 120)
        dEng_120 = float(se * 120)
        fit_rows.append({
            "SOC%": soc,
            "feature": "amplitude",
            "unit": "mV",
            "n_scans": int(len(t)),
            "TheilSen_slope_per_min": float(sa),
            "TheilSen_intercept": float(ba),
            "delta_120min": dAmp_120,
            "sigma_ROI": float(SIG_AMP),
            "n_sigma_ROI": dAmp_120 / SIG_AMP,
        })
        fit_rows.append({
            "SOC%": soc,
            "feature": "energy",
            "unit": "a.u.",
            "n_scans": int(len(t)),
            "TheilSen_slope_per_min": float(se),
            "TheilSen_intercept": float(be),
            "delta_120min": dEng_120,
            "sigma_ROI": float(SIG_ENG),
            "n_sigma_ROI": dEng_120 / SIG_ENG,
        })

    per_scan = pd.DataFrame(per_scan_rows)
    fits = pd.DataFrame(fit_rows)

    out_pscan = OUT_DIR / "data_amp_energy_per_scan.csv"
    out_fits  = OUT_DIR / "data_amp_energy_fits.csv"
    per_scan.to_csv(out_pscan, index=False, float_format="%.6g")
    fits.to_csv(out_fits, index=False, float_format="%.6g")
    print(f"wrote {out_pscan}  ({len(per_scan)} rows)")
    print(f"wrote {out_fits}  ({len(fits)} rows)")

    print("\nfit summary:")
    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
