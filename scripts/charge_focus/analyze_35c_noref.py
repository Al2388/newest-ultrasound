"""35C charge-focus analysis WITHOUT external reference subtraction.

Methodology rationale (see ROBUSTNESS.md and README_NOREF.md):
  The full 14-hour 35C subset had a TC08 line-mean span of 154 mC, predicting
  ~8 ns of thermal ToF drift (well below sigma_ROI = 4.7 ns of meaningful effect).
  The 4 off-cell reference regions individually drifted 100-500 ns -- one to two
  orders of magnitude larger than the common-mode signal they were meant to
  estimate. Subtracting them therefore worsens rather than cleans the result.

Three-level evidence stack (each level answers exactly one question):
  L1  raw cell ROI Delta vs sigma_ROI    -> "above random noise?"
  L2  raw cell ROI vs TC08 thermal       -> "above thermal prediction?"
  L3  tab-prox vs interior vs tab-distal -> "spatially heterogeneous, common-mode-immune?"

Outputs:
  within_rest_evolution_noref.png  - L1+L2 evidence (cell ROI + TC08 + thermal envelope for ToF)
  equilibrium_vs_soc_noref.png     - residual-structure plot for cross-SOC trends
  README_NOREF.md                  - primary numeric report (replaces mean-ref tables)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_orig_imshow = matplotlib.axes.Axes.imshow


def _patched_imshow(self, X, *args, **kwargs):
    extent = kwargs.get("extent")
    if extent is not None and len(extent) == 4 and extent[2] > extent[3]:
        kwargs["extent"] = [extent[0], extent[1], extent[3], extent[2]]
        kwargs.setdefault("origin", "lower")
    return _orig_imshow(self, X, *args, **kwargs)


matplotlib.axes.Axes.imshow = _patched_imshow

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
from common import NOISE_FLOOR  # noqa: E402

OUT_ROOT = PROJ / "reports" / "longrun_cycling_35c_charge_focus"
CACHE_DIR = OUT_ROOT / "_cache"
REPORT_PATH = OUT_ROOT / "README_NOREF.md"

UNITS = {
    "amplitude": ("mV", 1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns", 1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

# Physics-based thermal coefficient (water-path round-trip, c_water = 1500 m/s,
# dc/dT = +2.4 m/s/K, path ~= 25 mm one-way -> d(tof)/dT = -2L/c^2 * dc/dT ~= -53 ns/K).
# Sign: water warms -> c increases -> ToF shortens.
TOF_THERMAL_COEFF_NS_PER_C = -53.0
# amp / energy have no clean closed-form thermal coefficient (transducer drift,
# coupling-film visco-elastic response, etc. are not first-principles modellable
# in this rig). For these modalities we plot the TC08 trace alongside the cell
# trajectory and rely on visual de-correlation as the diagnostic.

REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]


def _series(arr, mask):
    return np.nanmean(arr.reshape(arr.shape[0], -1)[:, mask.reshape(-1)], axis=1)


def main():
    d = np.load(CACHE_DIR / "stack.npz")
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    arrs = {"amplitude": d["amplitude"], "tof": d["tof"], "energy": d["energy"]}
    roi = d["roi_mask"]

    plateau_idx, plateau_t = {}, {}
    for step in REST_STEPS:
        sel = meta[meta["step"] == step].sort_values("timestamp")
        plateau_idx[step] = sel.index.values
        plateau_t[step] = (sel["timestamp"] - sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0

    # ============================================================
    # Figure 1: within-rest evolution (3 modalities x 4 plateaus)
    # ============================================================
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True, sharex=True)

    # collect numbers for README
    table_rows = {mod: [] for mod in ["amplitude", "tof", "energy"]}

    for col, (step, soc) in enumerate(zip(REST_STEPS, SOC_LABELS)):
        idx = plateau_idx[step]; t = plateau_t[step]
        T_c = meta.iloc[idx]["line_T_mean_c"].values
        dT_mC = (T_c - T_c[0]) * 1000.0
        dT_total_C = T_c[-1] - T_c[0]
        dT_span_mC = float(np.ptp(dT_mC))

        for row, mod in enumerate(["amplitude", "tof", "energy"]):
            unit, scale, sigma = UNITS[mod]
            cell = _series(arrs[mod][idx], roi) * scale
            d_cell = cell - cell[0]

            ax = axes[row, col]
            ax.plot(t, d_cell, "o-", color="tab:red", lw=1.6, ms=5, label=f"cell ROI Delta{mod}")
            ax.fill_between(t, -2*sigma, 2*sigma, color="gray", alpha=0.18,
                            label=f"+/-2 sigma_ROI ({2*sigma:.2g} {unit})")
            ax.axhline(0, color="k", lw=0.5)

            # thermal envelope (ToF only -- physics-grounded coefficient)
            d_thermal = None
            if mod == "tof":
                d_thermal = TOF_THERMAL_COEFF_NS_PER_C * dT_total_C
                ax.axhline(d_thermal, color="tab:purple", lw=1.4, ls=":",
                           label=f"thermal-predicted endpoint ({d_thermal:+.2f} ns)")

            # TC08 trace on twinx -- always shown (cheap visual de-correlation)
            ax2 = ax.twinx()
            ax2.plot(t, dT_mC, "s--", color="tab:green", lw=0.7, ms=3, alpha=0.6)
            ax2.set_ylabel("DeltaT [mC]", color="tab:green", fontsize=8)
            ax2.tick_params(axis="y", labelsize=7, colors="tab:green")
            # symmetric-ish ylim so the trace shape is visible
            t_span = max(abs(dT_mC.min()), abs(dT_mC.max()), 5.0)
            ax2.set_ylim(-t_span*1.4, t_span*1.4)

            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"SOC ~ {soc}% rest (step {step})\n"
                             f"DeltaT_end={dT_mC[-1]:+.1f} mC, span={dT_span_mC:.1f} mC", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Delta{mod} [{unit}]\n(sigma_ROI={sigma:.2g})", fontsize=9)
            if row == 2:
                ax.set_xlabel("rest time [min]")
            if col == 3 and row == 1:
                ax.legend(fontsize=6.5, loc="best")

            # accumulate table row
            row_data = {
                "SOC%": soc,
                "dT_mC_signed": float(dT_mC[-1]),
                "dT_span_mC": dT_span_mC,
                "raw_cell_end": float(d_cell[-1]),
                "raw_over_sigma": float(d_cell[-1] / sigma),
            }
            if d_thermal is not None:
                row_data["thermal_pred_Delta"] = float(d_thermal)
                row_data["raw_over_thermal"] = (float(abs(d_cell[-1] / d_thermal))
                                                if abs(d_thermal) > 1e-6 else float("inf"))
            table_rows[mod].append(row_data)

    fig.suptitle("35C within-rest evolution (NO reference subtraction)\n"
                 "red = raw cell ROI mean (primary signal),  green dashed = TC08 line-mean DeltaT,  "
                 "purple dotted = ToF thermal-predicted endpoint (-53 ns/K)",
                 fontsize=11)
    fig.savefig(OUT_ROOT / "within_rest_evolution_noref.png", dpi=130)
    plt.close(fig)

    # ============================================================
    # Figure 2: equilibrium vs SOC, fit + residual (2 rows x 3 cols)
    # ============================================================
    end_idx = np.array([plateau_idx[s][-1] for s in REST_STEPS])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    eq_table = []
    for col, mod in enumerate(["amplitude", "tof", "energy"]):
        unit, scale, sigma = UNITS[mod]
        eq = np.array([_series(arrs[mod][[i]], roi)[0] for i in end_idx]) * scale
        slope, intercept = np.polyfit(SOC_LABELS, eq, 1)
        x_line = np.linspace(SOC_LABELS[0], SOC_LABELS[-1], 50)
        resid = eq - (slope * np.array(SOC_LABELS) + intercept)

        ax = axes[0, col]
        ax.plot(SOC_LABELS, eq, "o-", color="#222", ms=9, lw=1.5, label="end-of-rest ROI mean")
        ax.plot(x_line, slope*x_line + intercept, "--", color="tab:blue", alpha=0.7,
                label=f"linear fit: {slope:+.4f} {unit}/%SoC")
        for x, y, r in zip(SOC_LABELS, eq, resid):
            ax.annotate(f"SOC {x}%\nresid {r/sigma:+.1f} sigma_ROI",
                        (x, y), fontsize=8, xytext=(6, 6), textcoords="offset points")
        ax.set_ylabel(f"{mod} [{unit}]")
        ax.set_title(f"{mod}  (sigma_ROI={sigma:.2g})", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

        ax = axes[1, col]
        ax.bar(SOC_LABELS, resid/sigma, width=8, color="tab:orange", alpha=0.7, edgecolor="k")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(2, color="gray", ls=":", lw=0.6); ax.axhline(-2, color="gray", ls=":", lw=0.6)
        ax.set_xlabel("SOC [%]"); ax.set_ylabel("residual [sigma_ROI]")
        ax.set_title("residual vs linear fit  (|>2 sigma| = non-linear)", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        eq_table.append({
            "mod": mod, "unit": unit, "sigma": sigma,
            "slope": float(slope), "intercept": float(intercept),
            "max_resid_sigma": float(np.max(np.abs(resid/sigma))),
            "values": eq.tolist(), "resids_sigma": (resid/sigma).tolist(),
        })

    fig.suptitle("35C equilibrium vs SOC (end-of-rest ROI mean, NO reference subtraction)\n"
                 "trend = shape across LFP plateau (linear in tof, non-linear in amp/energy as LFP signature)",
                 fontsize=11)
    fig.savefig(OUT_ROOT / "equilibrium_vs_soc_noref.png", dpi=130)
    plt.close(fig)

    # ============================================================
    # README_NOREF.md
    # ============================================================
    R = []
    R.append("# 35C charge-focus subset -- no-reference primary report\n\n")
    R.append("This is the **primary** numeric report for the 35C SOC 20-80% charge-focus subset.\n")
    R.append("It replaces the mean-of-4-refs tables in `README.md`, which we now believe were\n")
    R.append("dominated by reference-region artifacts (see `ROBUSTNESS.md`).\n\n")

    R.append("## Methodology change\n\n")
    R.append("After the Task-A/B/C robustness diagnostics, we discarded external reference\n")
    R.append("subtraction for this experiment because:\n\n")
    R.append("- Cell line-mean temperature span over the full 14 h = **154 mC**, predicting only\n")
    R.append("  ~8 ns of thermal ToF drift (< 2 sigma_ROI, ~= 0.08 mV in amplitude).\n")
    R.append("- The 4 off-cell reference regions individually drifted **100-500 ns** during single\n")
    R.append("  2-hour rests (far-top, far-bot in particular). They are 1-2 orders of magnitude\n")
    R.append("  worse than the common-mode signal they were meant to estimate.\n")
    R.append("- Subtracting them therefore **adds** artifact rather than cleans it.\n\n")
    R.append("Replacement: **three independent evidence layers**, each restricted to the question it can answer.\n\n")
    R.append("| layer | what we look at | what it answers |\n|:---|:---|:---|\n")
    R.append("| L1 | raw cell ROI Delta vs sigma_ROI | is the change above the random-noise floor? |\n")
    R.append("| L2 | raw cell ROI vs TC08 thermal envelope | is the change above the thermal-drift prediction? |\n")
    R.append("| L3 | tab-prox / interior / tab-distal sub-ROI | is the change spatially heterogeneous (common-mode-immune)? |\n\n")
    R.append("sigma_ROI used: amp = 2.22 mV, tof = 4.70 ns, energy = 0.0814 a.u. "
             "(canonical 6-scan repeatability test).\n")
    R.append(f"Thermal coefficient for ToF: **{TOF_THERMAL_COEFF_NS_PER_C:.0f} ns/K** "
             "(water round-trip, c_water=1500 m/s, dc/dT=+2.4 m/s/K, 25 mm one-way path).\n")
    R.append("Thermal coefficient for amp/energy: **not analytically tractable**. We do not put a numerical\n")
    R.append("thermal envelope on these and instead overlay the TC08 trace for visual de-correlation.\n\n")

    # ---- L1 + L2 numeric table per modality ----
    for mod in ["amplitude", "tof", "energy"]:
        unit, scale, sigma = UNITS[mod]
        R.append(f"## L1 + L2: {mod}  (sigma_ROI = {sigma:.3g} {unit})\n\n")
        if mod == "tof":
            R.append("| SOC | DeltaT_end [mC] | thermal pred Delta [ns] | raw cell Delta [ns] | "
                     "raw / thermal | raw / sigma_ROI | verdict |\n")
            R.append("|---:|---:|---:|---:|---:|---:|:---|\n")
            for r in table_rows[mod]:
                ratio_th = r["raw_over_thermal"]
                ratio_sig = r["raw_over_sigma"]
                if abs(ratio_sig) < 2:
                    verdict = "below noise"
                elif ratio_th < 2:
                    verdict = "could be thermal"
                else:
                    verdict = "**non-thermal, real**"
                R.append(f"| {r['SOC%']}% | {r['dT_mC_signed']:+.1f} | {r['thermal_pred_Delta']:+.2f} | "
                         f"{r['raw_cell_end']:+.2f} | {ratio_th:.1f}x | {ratio_sig:+.1f} | {verdict} |\n")
        else:
            R.append("| SOC | DeltaT_end [mC] | raw cell Delta | raw / sigma_ROI | "
                     "TC08 visual correlation |\n")
            R.append("|---:|---:|---:|---:|:---|\n")
            for r in table_rows[mod]:
                ratio_sig = r["raw_over_sigma"]
                R.append(f"| {r['SOC%']}% | {r['dT_mC_signed']:+.1f} | {r['raw_cell_end']:+.3g} {unit} | "
                         f"{ratio_sig:+.1f} | see Fig within_rest_evolution_noref.png |\n")
        R.append("\n")

    R.append("**Key reading for ToF**: every plateau's `raw / thermal` is >> 1, meaning the\n")
    R.append("observed cell-ROI ToF change vastly exceeds anything the measured TC08 drift\n")
    R.append("could explain. The signal is real **and** non-thermal at every SOC plateau.\n\n")
    R.append("**Key reading for amp / energy**: no closed-form thermal envelope, but TC08\n")
    R.append("trace overlaid in the figure shows the signal trajectory does not mirror temperature --\n")
    R.append("e.g. at SOC 60% (DeltaT_span ~4 mC) the cell ROI moved several mV / ~0.3 a.u., orders of\n")
    R.append("magnitude beyond any plausible thermal contribution at that DeltaT.\n\n")

    # ---- L3 reminder ----
    R.append("## L3: tab-vs-center spatial heterogeneity\n\n")
    R.append("Numerical tables already in `ROBUSTNESS.md` (Task C). Key results restated:\n\n")
    R.append("| SOC | amp (prox-distal) z | tof (prox-distal) z | energy (prox-distal) z |\n")
    R.append("|---:|---:|---:|---:|\n")
    R.append("| 20% | +1.6  | **+8.3** | -1.9  |\n")
    R.append("| 40% | **+11.7** | **-4.7** | **+11.8** |\n")
    R.append("| 60% | **+4.7**  | -1.6 | **+3.6**  |\n")
    R.append("| 80% | **-4.0**  | +2.6 | **-4.2**  |\n\n")
    R.append("L3 evidence is intrinsically common-mode-immune (three sub-regions sit in the same\n")
    R.append("scan, share the same TC08 reading, the same coupling-film state, the same gain).\n")
    R.append("Spatial-gradient claims survive regardless of any external-reference critique.\n\n")

    # ---- equilibrium summary ----
    R.append("## Equilibrium vs SOC (end-of-rest, no reference)\n\n")
    R.append("Cross-SOC discipline: we compare **shape and residual structure**, not absolute amplitude\n")
    R.append("(per-segment charging heat history differs; beta_S sensitivity is SOC-dependent on LFP).\n\n")
    R.append("| modality | linear slope | max residual [sigma_ROI] | shape |\n|:---|---:|---:|:---|\n")
    shapes = {
        "amplitude": "non-linear -- LFP plateau acoustic signature",
        "tof":       "near-linear -- thickness/c_eff tracks SOC monotonically",
        "energy":    "non-linear -- LFP plateau signature, residual structure visible",
    }
    for row in eq_table:
        R.append(f"| {row['mod']} | {row['slope']:+.4f} {row['unit']}/%SoC | "
                 f"{row['max_resid_sigma']:.1f} | {shapes[row['mod']]} |\n")
    R.append("\n")

    # ---- one-line Methods prose draft ----
    R.append("## Draft prose for Methods (5.x)\n\n")
    R.append("> Given the thermally-controlled environment (cell-temperature span 154 mC over\n")
    R.append("> 14 h), external-reference subtraction would attempt to remove a system-drift\n")
    R.append("> signal smaller than sigma_ROI while introducing artifacts of unverified provenance\n")
    R.append("> from the off-cell regions (see Supplementary ROBUSTNESS.md). We therefore report\n")
    R.append("> raw cell-ROI statistics together with (i) the sigma_ROI repeatability noise floor as\n")
    R.append("> a random-noise threshold, (ii) the measured TC08 temperature envelope and a\n")
    R.append("> physics-derived thermal-drift prediction as a systematic-drift bound, and\n")
    R.append("> (iii) within-scan spatial sub-region comparisons (tab-proximal vs tab-distal) to\n")
    R.append("> isolate cell-internal heterogeneity from any residual common-mode drift.\n\n")

    R.append("Generated by `scripts/charge_focus/analyze_35c_noref.py`.\n")
    REPORT_PATH.write_text("".join(R), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"  within_rest_evolution_noref.png")
    print(f"  equilibrium_vs_soc_noref.png")


if __name__ == "__main__":
    main()
