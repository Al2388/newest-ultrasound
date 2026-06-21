"""Dump the underlying data for fig_evolution_tof as CSVs.

Generates:
  fig_evolution_tof_per_scan_data.csv  -- per (SOC, scan) row:
       t_rest_min, ROI-median ToF (ns), drift Δ since rest start,
       drift normalised to σ_ROI, voltage, cell T
  fig_evolution_tof_fits.csv           -- per (SOC) row:
       Theil-Sen slope, intercept, Δ_120min, σ_T, 3σ_T envelope,
       thermal_multiple, n_σ_ROI
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

SIG_TOF_NS = NOISE_FLOOR["tof_sigma_roi_ns"]    # 4.700 ns
ALPHA_T_NS_PER_C = +73.5                         # in-situ thermal coefficient


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
    tof_ns = d["tof"] * 1e3

    per_scan_rows = []
    fit_rows = []

    for step, soc in zip(REST_STEPS, SOC_LABELS):
        rest = meta[meta["step"] == step].sort_values("timestamp").reset_index()
        t = (rest["timestamp"] - rest["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        idx = rest["index"].values
        tof_med = np.array([float(np.nanmedian(tof_ns[i][roi])) for i in idx])
        drift = tof_med - tof_med[0]
        z = drift / SIG_TOF_NS

        for k in range(len(rest)):
            per_scan_rows.append({
                "SOC%": soc,
                "step": step,
                "run_idx": int(rest["run_idx"].iloc[k]),
                "scan_id": rest["scan_id"].iloc[k],
                "t_rest_min": float(t[k]),
                "ToF_med_ROI_ns": float(tof_med[k]),
                "dToF_since_start_ns": float(drift[k]),
                "z_sigmaROI": float(z[k]),
                "voltage_mV": float(rest["voltage_at_scan"].iloc[k]) * 1000,
                "T_cell_C": float(rest["line_T_mean_c"].iloc[k]),
            })

        # Theil-Sen on absolute ToF (ns) vs rest time (min)
        slope, intercept = siegelslopes(tof_med, t)
        d120 = slope * 120

        # Per-plateau sigma_T and thermal envelope
        sigma_T_mC = float(rest["line_T_mean_c"].std(ddof=1) * 1000)
        env_3sigT_ns = 3 * (sigma_T_mC / 1000) * ALPHA_T_NS_PER_C
        thermal_mult = d120 / env_3sigT_ns if abs(env_3sigT_ns) > 0 else float("inf")
        n_sigma = d120 / SIG_TOF_NS

        fit_rows.append({
            "SOC%": soc,
            "n_scans": int(len(t)),
            "t_first_min": float(t[0]),
            "t_last_min":  float(t[-1]),
            "TheilSen_slope_ns_per_min": float(slope),
            "TheilSen_intercept_ns": float(intercept),
            "delta_120min_ns": float(d120),
            "sigma_ROI_ToF_ns": SIG_TOF_NS,
            "n_sigma_ROI": float(n_sigma),
            "sigma_T_mC": sigma_T_mC,
            "alpha_T_ns_per_C": ALPHA_T_NS_PER_C,
            "thermal_envelope_3sigT_ns": float(env_3sigT_ns),
            "thermal_multiple": float(thermal_mult),
        })

    per_scan = pd.DataFrame(per_scan_rows)
    fits     = pd.DataFrame(fit_rows)

    out_pscan = OUT_DIR / "fig_evolution_tof_per_scan_data.csv"
    out_fits  = OUT_DIR / "fig_evolution_tof_fits.csv"
    per_scan.to_csv(out_pscan, index=False, float_format="%.6g")
    fits.to_csv(out_fits, index=False, float_format="%.6g")
    print(f"wrote {out_pscan}  ({len(per_scan)} rows)")
    print(f"wrote {out_fits}  ({len(fits)} rows)\n")

    # Pretty print summary
    print("ToF fit summary per SOC:\n")
    cols = ["SOC%", "n_scans", "TheilSen_slope_ns_per_min",
            "delta_120min_ns", "sigma_T_mC", "thermal_envelope_3sigT_ns",
            "thermal_multiple", "n_sigma_ROI"]
    print(fits[cols].to_string(index=False))


if __name__ == "__main__":
    main()
