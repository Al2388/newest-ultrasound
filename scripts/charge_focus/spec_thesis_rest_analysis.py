"""Thesis rest-analysis spec runner -- full analysis per the specification.

Outputs everything required by Section 10 of the spec:
  - sigma_ROI from N=6 noisefloor_v3.238 repeats
  - Temperature stats (overall + per plateau)
  - Layer 1 table: rest-evolution per SOC per feature, with robust linear fit
  - Layer 2a: end-of-rest spatial contrast (tab-prox - tab-distal)
  - Layer 2b: contrast evolution during rest (robust linear fit)
  - Figures A-F as PDFs (300 dpi vector)
  - Underlying CSVs/JSONs
  - Sanity checks

Aggregation: MEDIAN over ROI (per spec; robust to outliers).
Robust fit: scipy.stats.siegelslopes (per spec; no endpoint differencing).
ToF reported in ns, amp in mV, energy in a.u. throughout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import siegelslopes

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))

CACHE = PROJ / "reports/longrun_cycling_35c_charge_focus/_cache"
ROI_JSON = PROJ / "reports/experiments/roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp/roi_summary.json"
OUT = PROJ / "reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---- constants per spec ----
ALPHA_T_NS_PER_C = +73.5     # ns/degC, in-situ measured cell-ROI ToF coefficient
ROI_X_MM = (14.6, 64.5)
ROI_Y_MM = (16.1, 55.4)
SUBREGIONS_X = {
    "tab-distal":   (14.6, 30.0),
    "interior":     (30.0, 50.0),
    "tab-proximal": (50.0, 64.5),
}
REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]
SOC_COLORS = {20: "tab:blue", 40: "tab:green", 60: "tab:orange", 80: "tab:red"}

# Feature unit conversions to spec units (ns/mV/a.u.)
TO_NS = 1e3   # us -> ns
TO_MV = 1e3   # V  -> mV
ENG_SCALE = 1.0


def _x_in(x_mm, lo, hi):
    return np.where((x_mm >= lo) & (x_mm <= hi))[0]


def _y_in(y_mm, lo, hi):
    return np.where((y_mm >= lo) & (y_mm <= hi))[0]


def load_data():
    d = np.load(CACHE / "stack.npz")
    meta = pd.read_csv(CACHE / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    return d, meta


def build_roi_subregions(d):
    x_mm = d["x_mm"]; y_mm = d["y_mm"]
    # Geometric ROI from spec bounds (NOT the canonical roi_mask; spec says
    # "use these fixed bounds in millimetres")
    x_idx = _x_in(x_mm, *ROI_X_MM)
    y_idx = _y_in(y_mm, *ROI_Y_MM)
    roi = np.zeros((y_mm.size, x_mm.size), dtype=bool)
    roi[np.ix_(y_idx, x_idx)] = True
    subregions = {}
    for name, (lo, hi) in SUBREGIONS_X.items():
        m = np.zeros_like(roi)
        m[np.ix_(y_idx, _x_in(x_mm, lo, hi))] = True
        subregions[name] = m
    return roi, subregions, x_mm, y_mm


def get_sigma_ROI_from_N6():
    """sigma_ROI from N=6 noisefloor_v3.238 repeats, per spec."""
    j = json.loads(ROI_JSON.read_text())
    agg = j["roi_aggregated_sigma"]
    return {
        "tof_ns":  agg["tof_us"] * TO_NS,            # us -> ns
        "amp_mV":  agg["amplitude_V"] * TO_MV,       # V  -> mV
        "eng_au":  agg["energy"],
        "n_repeats": j["n_scans"],
        "source": "roi_aggregated_sigma in roi_summary.json (noisefloor_v3.238)",
    }


def median_in_mask(arr3d, mask):
    """Per-scan median over pixels in mask. arr3d shape (n, ny, nx)."""
    n = arr3d.shape[0]
    out = np.empty(n)
    flat_mask = mask.reshape(-1)
    for i in range(n):
        a = arr3d[i].reshape(-1)[flat_mask]
        out[i] = float(np.nanmedian(a))
    return out


def robust_fit(t_min, y):
    """Theil-Sen via scipy siegelslopes. Returns (slope_per_min, intercept)."""
    finite = np.isfinite(t_min) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan"), float("nan")
    slope, intercept = siegelslopes(y[finite], t_min[finite])
    return float(slope), float(intercept)


def per_plateau_temperature_stats(meta, t_per_line_dict):
    """Per-plateau cell T stats using per-scan line_T_mean_c (no cycler csv here).

    Spec §7 expects per-line resolution if available; we use per-scan T (already
    averaged over each line's temperatures), which is the highest resolution in the
    cache. The watchdog log is a separate, higher-resolution stream noted in §9.
    """
    rows = []
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        sel = meta[meta["step"] == step].copy()
        T_vals = sel["line_T_mean_c"].values
        rows.append({
            "SOC%": soc,
            "T_mean_C": float(np.mean(T_vals)),
            "sigma_T_mC": float(np.std(T_vals, ddof=1) * 1000),
            "T_span_mC": float((T_vals.max() - T_vals.min()) * 1000),
            "n_scans": int(len(T_vals)),
        })
    return rows


def main():
    d, meta = load_data()
    print(f"loaded stack: {d['amplitude'].shape}, meta rows: {len(meta)}")

    # Build geometric ROI from spec bounds
    roi, subregions, x_mm, y_mm = build_roi_subregions(d)
    print(f"ROI pixels: {int(roi.sum())}")
    for name, m in subregions.items():
        print(f"  {name:<14s}: {int(m.sum())} px")

    # Sigma_ROI from N=6
    sigma_ROI = get_sigma_ROI_from_N6()
    print(f"\nsigma_ROI (N={sigma_ROI['n_repeats']} repeats):")
    print(f"  ToF       = {sigma_ROI['tof_ns']:.3f} ns")
    print(f"  amplitude = {sigma_ROI['amp_mV']:.3f} mV")
    print(f"  energy    = {sigma_ROI['eng_au']:.5f} a.u.")

    # Feature arrays in spec units
    amp = d["amplitude"] * TO_MV   # mV
    tof = d["tof"] * TO_NS         # ns
    eng = d["energy"]              # a.u.

    # Per-scan ROI median for each feature (full ROI + sub-regions)
    scans_meta = meta.copy()
    scans_meta["amp_med_ROI"] = median_in_mask(amp, roi)
    scans_meta["tof_med_ROI"] = median_in_mask(tof, roi)
    scans_meta["eng_med_ROI"] = median_in_mask(eng, roi)
    for sname, smask in subregions.items():
        scans_meta[f"amp_med_{sname}"] = median_in_mask(amp, smask)
        scans_meta[f"tof_med_{sname}"] = median_in_mask(tof, smask)
        scans_meta[f"eng_med_{sname}"] = median_in_mask(eng, smask)

    # Per-plateau temperature stats
    T_stats = per_plateau_temperature_stats(scans_meta, None)
    T_overall = {
        "T_mean_C": float(scans_meta.loc[scans_meta["step_tag"]=="rest", "line_T_mean_c"].mean()),
        "sigma_T_mC": float(scans_meta.loc[scans_meta["step_tag"]=="rest", "line_T_mean_c"].std(ddof=1) * 1000),
        "T_min_C": float(scans_meta.loc[scans_meta["step_tag"]=="rest", "line_T_mean_c"].min()),
        "T_max_C": float(scans_meta.loc[scans_meta["step_tag"]=="rest", "line_T_mean_c"].max()),
    }

    print(f"\nTemperature overall (rest scans only):")
    print(f"  mean = {T_overall['T_mean_C']:.4f} C")
    print(f"  sigma = {T_overall['sigma_T_mC']:.1f} mC")
    print(f"  range = {T_overall['T_min_C']:.4f} -> {T_overall['T_max_C']:.4f} C")
    print(f"\nPer-plateau T stats:")
    for r in T_stats:
        print(f"  SOC {r['SOC%']}%  mean={r['T_mean_C']:.4f} C  sigma={r['sigma_T_mC']:.1f} mC  span={r['T_span_mC']:.0f} mC")

    # ================================================================
    # LAYER 1: rest evolution within each SOC
    # ================================================================
    print(f"\n{'='*70}")
    print("LAYER 1: rest evolution per SOC")
    print(f"{'='*70}")

    layer1 = []
    for step, soc, Trow in zip(REST_STEPS, SOC_LABELS, T_stats):
        rest_sel = scans_meta[scans_meta["step"] == step].sort_values("timestamp")
        t_rest = (rest_sel["timestamp"] - rest_sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        # ToF
        y_tof = rest_sel["tof_med_ROI"].values
        slope_tof, _ = robust_fit(t_rest, y_tof)
        dToF_120 = slope_tof * 120.0
        sigma_T_C = Trow["sigma_T_mC"] / 1000
        envelope_3sigT_ns = 3 * sigma_T_C * ALPHA_T_NS_PER_C
        n_sigma_tof = dToF_120 / sigma_ROI["tof_ns"]
        thermal_mult_tof = dToF_120 / envelope_3sigT_ns if abs(envelope_3sigT_ns) > 0 else float("inf")
        # Amp
        y_amp = rest_sel["amp_med_ROI"].values
        slope_amp, _ = robust_fit(t_rest, y_amp)
        dAmp_120 = slope_amp * 120.0
        n_sigma_amp = dAmp_120 / sigma_ROI["amp_mV"]
        # Energy
        y_eng = rest_sel["eng_med_ROI"].values
        slope_eng, _ = robust_fit(t_rest, y_eng)
        dEng_120 = slope_eng * 120.0
        n_sigma_eng = dEng_120 / sigma_ROI["eng_au"]

        layer1.append({
            "SOC%": soc,
            "n_scans": int(len(rest_sel)),
            "t_first_min": float(t_rest[0]),
            "t_last_min": float(t_rest[-1]),
            "sigma_T_mC": Trow["sigma_T_mC"],
            "thermal_envelope_3sigT_ToF_ns": float(envelope_3sigT_ns),
            "dToF_fit_120min_ns": float(dToF_120),
            "ToF_thermal_multiple": float(thermal_mult_tof),
            "ToF_n_sigma": float(n_sigma_tof),
            "ToF_sign": "+" if dToF_120 > 0 else "-",
            "dAmp_fit_120min_mV": float(dAmp_120),
            "Amp_n_sigma": float(n_sigma_amp),
            "dEng_fit_120min_au": float(dEng_120),
            "Eng_n_sigma": float(n_sigma_eng),
        })
        print(f"\nSOC {soc}%  (n={len(rest_sel)} scans, t = {t_rest[0]:.1f} -> {t_rest[-1]:.1f} min)")
        print(f"  sigma_T = {Trow['sigma_T_mC']:.1f} mC,  3sigma_T thermal envelope (ToF) = {envelope_3sigT_ns:+.2f} ns")
        print(f"  dToF_120min = {dToF_120:+.3f} ns  (thermal mult = {thermal_mult_tof:+.2f}x, "
              f"noise n_sigma = {n_sigma_tof:+.2f})")
        print(f"  dAmp_120min = {dAmp_120:+.3f} mV  (n_sigma = {n_sigma_amp:+.2f})")
        print(f"  dEng_120min = {dEng_120:+.4f} a.u. (n_sigma = {n_sigma_eng:+.2f})")

    pd.DataFrame(layer1).to_csv(OUT / "layer1_rest_evolution.csv", index=False)

    # ================================================================
    # LAYER 2a: end-of-rest sub-region medians + contrast
    # ================================================================
    print(f"\n{'='*70}")
    print("LAYER 2a: end-of-rest sub-region medians + contrast (tab-prox - tab-distal)")
    print(f"{'='*70}")

    layer2a = []
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        rest_sel = scans_meta[scans_meta["step"] == step].sort_values("timestamp")
        end_row = rest_sel.iloc[-1]
        for feat, ns_per_unit, unit in [
            ("tof", sigma_ROI["tof_ns"], "ns"),
            ("amp", sigma_ROI["amp_mV"], "mV"),
            ("eng", sigma_ROI["eng_au"], "a.u."),
        ]:
            tab_distal = float(end_row[f"{feat}_med_tab-distal"])
            interior  = float(end_row[f"{feat}_med_interior"])
            tab_prox  = float(end_row[f"{feat}_med_tab-proximal"])
            contrast = tab_prox - tab_distal
            n_sig = contrast / ns_per_unit
            layer2a.append({
                "SOC%": soc, "feature": feat,
                "tab-distal":   tab_distal,
                "interior":     interior,
                "tab-proximal": tab_prox,
                "contrast (prox-distal)": contrast,
                "n_sigma_ROI": n_sig,
            })
        print(f"\nSOC {soc}% end-of-rest:")
        for r in layer2a[-3:]:
            print(f"  {r['feature']:<3s}:  distal={r['tab-distal']:>10.3f}  "
                  f"interior={r['interior']:>10.3f}  prox={r['tab-proximal']:>10.3f}  "
                  f"contrast={r['contrast (prox-distal)']:+.3f}  ({r['n_sigma_ROI']:+.1f} sigma_ROI)")
    pd.DataFrame(layer2a).to_csv(OUT / "layer2a_end_of_rest_contrast.csv", index=False)

    # ================================================================
    # LAYER 2b: contrast evolution during rest (robust fit)
    # ================================================================
    print(f"\n{'='*70}")
    print("LAYER 2b: contrast (prox-distal) evolution during rest, robust fit slope x 120 min")
    print(f"{'='*70}")

    layer2b = []
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        rest_sel = scans_meta[scans_meta["step"] == step].sort_values("timestamp")
        t_rest = (rest_sel["timestamp"] - rest_sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        for feat, sigma_val, unit in [
            ("tof", sigma_ROI["tof_ns"], "ns"),
            ("amp", sigma_ROI["amp_mV"], "mV"),
            ("eng", sigma_ROI["eng_au"], "a.u."),
        ]:
            contrast = (rest_sel[f"{feat}_med_tab-proximal"].values
                        - rest_sel[f"{feat}_med_tab-distal"].values)
            slope, _ = robust_fit(t_rest, contrast)
            d120 = slope * 120.0
            n_sig = d120 / sigma_val
            layer2b.append({
                "SOC%": soc, "feature": feat,
                "slope_per_min": slope,
                "delta_contrast_120min": d120,
                "n_sigma_ROI": n_sig,
            })
        print(f"\nSOC {soc}%:")
        for r in layer2b[-3:]:
            print(f"  {r['feature']:<3s}  d(prox-distal)_120min = "
                  f"{r['delta_contrast_120min']:+.3f}  ({r['n_sigma_ROI']:+.1f} sigma_ROI)")
    pd.DataFrame(layer2b).to_csv(OUT / "layer2b_contrast_evolution.csv", index=False)

    # ================================================================
    # SANITY CHECKS
    # ================================================================
    sanity = {}
    rest_mask_meta = scans_meta["step_tag"] == "rest"
    rest_idx = np.where(rest_mask_meta)[0]

    # amp saturation inside ROI
    amp_roi_pixels = amp[rest_idx][:, roi]
    sanity["amp_max_in_ROI_mV"] = float(np.nanmax(amp_roi_pixels))
    sanity["amp_p99_in_ROI_mV"] = float(np.nanpercentile(amp_roi_pixels, 99))
    sanity["amp_pixels_at_5000mV"] = int(np.sum(amp_roi_pixels >= 5000.0))
    sanity["amp_saturated_in_ROI"] = bool(sanity["amp_pixels_at_5000mV"] > 0)

    # ToF range inside ROI
    tof_roi_pixels = tof[rest_idx][:, roi]
    sanity["tof_min_in_ROI_us"] = float(np.nanmin(tof_roi_pixels)) / 1000   # back to us
    sanity["tof_max_in_ROI_us"] = float(np.nanmax(tof_roi_pixels)) / 1000
    sanity["tof_median_in_ROI_us"] = float(np.nanmedian(tof_roi_pixels)) / 1000

    # 20% plateau T anomaly
    soc20_T = T_stats[0]["T_mean_C"]
    others_T = np.mean([T_stats[i]["T_mean_C"] for i in [1, 2, 3]])
    sanity["T20%_minus_TOthers_mC"] = float((soc20_T - others_T) * 1000)
    sanity["T20%_slightly_higher"] = bool(soc20_T > others_T)

    print(f"\n{'='*70}")
    print("SANITY CHECKS")
    print(f"{'='*70}")
    for k, v in sanity.items():
        print(f"  {k}: {v}")

    (OUT / "sanity_checks.json").write_text(json.dumps(sanity, indent=2))

    # ================================================================
    # FIGURES A - F  (PDF, 300 dpi)
    # ================================================================
    print(f"\n{'='*70}")
    print("FIGURES A-F")
    print(f"{'='*70}")

    plt.rcParams.update({
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
    })

    # ---- Figure A: ToF evolution 2x2 (PRIMARY) ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True, sharex=True)
    for ax, (step, soc, Trow, l1) in zip(axes.ravel(),
                                          zip(REST_STEPS, SOC_LABELS, T_stats, layer1),
                                          ):
        pass
    # rebuild iteration since zip got nested awkwardly
    iter_data = list(zip(REST_STEPS, SOC_LABELS, T_stats, layer1))
    for ax, (step, soc, Trow, l1) in zip(axes.ravel(), iter_data):
        rest_sel = scans_meta[scans_meta["step"] == step].sort_values("timestamp")
        t_rest = (rest_sel["timestamp"] - rest_sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        y_tof = rest_sel["tof_med_ROI"].values
        slope, intercept = robust_fit(t_rest, y_tof)
        t_line = np.array([t_rest.min(), t_rest.max()])
        fit_line = slope * t_line + intercept
        # Center bands at y0 = intercept (~start)
        y0 = intercept
        ax.fill_between(t_rest, y0 - 3*l1["thermal_envelope_3sigT_ToF_ns"]/3,
                        y0 + 3*l1["thermal_envelope_3sigT_ToF_ns"]/3,
                        color="tab:orange", alpha=0.12,
                        label=f"3sigT thermal envelope (±{l1['thermal_envelope_3sigT_ToF_ns']:.2f} ns)")
        ax.fill_between(t_rest, y0 - sigma_ROI["tof_ns"], y0 + sigma_ROI["tof_ns"],
                        color="gray", alpha=0.25,
                        label=f"sigma_ROI band (±{sigma_ROI['tof_ns']:.2f} ns)")
        ax.scatter(t_rest, y_tof, c="black", s=22, zorder=3, label="ROI median")
        ax.plot(t_line, fit_line, "-", color=SOC_COLORS[soc], lw=1.6, zorder=2,
                label=f"Theil-Sen fit (slope×120 = {l1['dToF_fit_120min_ns']:+.2f} ns)")
        ax.set_title(f"SOC {soc}%", color=SOC_COLORS[soc], fontweight="bold")
        ax.set_xlabel("rest time [min]")
        ax.set_ylabel("ROI-median ToF [ns]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best", framealpha=0.85)
        ax.text(0.02, 0.04,
                f"thermal mult = {l1['ToF_thermal_multiple']:+.2f}\n"
                f"n_sigma_ROI = {l1['ToF_n_sigma']:+.2f}",
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
    fig.suptitle("Figure A — ToF rest evolution per SOC (35°C charge-focus)", fontsize=11)
    fig.savefig(FIG_DIR / "fig_evolution_tof.pdf")
    plt.close(fig)
    print(f"  wrote fig_evolution_tof.pdf")

    # ---- Figure B: amplitude + energy 2x2 (twin y) ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True, sharex=True)
    for ax, (step, soc, l1) in zip(axes.ravel(), zip(REST_STEPS, SOC_LABELS, layer1)):
        rest_sel = scans_meta[scans_meta["step"] == step].sort_values("timestamp")
        t_rest = (rest_sel["timestamp"] - rest_sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        y_amp = rest_sel["amp_med_ROI"].values
        y_eng = rest_sel["eng_med_ROI"].values
        slope_a, int_a = robust_fit(t_rest, y_amp)
        slope_e, int_e = robust_fit(t_rest, y_eng)
        t_line = np.array([t_rest.min(), t_rest.max()])

        # amp on left axis
        ax.scatter(t_rest, y_amp, c="tab:red", s=20, label="amp ROI median")
        ax.plot(t_line, slope_a*t_line+int_a, "-", color="tab:red", lw=1.6)
        ax.fill_between(t_rest, int_a - sigma_ROI["amp_mV"], int_a + sigma_ROI["amp_mV"],
                        color="tab:red", alpha=0.15)
        ax.set_ylabel("amp ROI median [mV]", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        # energy on right axis
        ax2 = ax.twinx()
        ax2.scatter(t_rest, y_eng, c="tab:blue", marker="^", s=18, label="energy ROI median")
        ax2.plot(t_line, slope_e*t_line+int_e, "-", color="tab:blue", lw=1.6)
        ax2.fill_between(t_rest, int_e - sigma_ROI["eng_au"], int_e + sigma_ROI["eng_au"],
                         color="tab:blue", alpha=0.15)
        ax2.set_ylabel("energy ROI median [a.u.]", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        ax.set_title(f"SOC {soc}%", color=SOC_COLORS[soc], fontweight="bold")
        ax.set_xlabel("rest time [min]")
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.04,
                f"dAmp_120 = {l1['dAmp_fit_120min_mV']:+.1f} mV ({l1['Amp_n_sigma']:+.1f}σ)\n"
                f"dEng_120 = {l1['dEng_fit_120min_au']:+.3f} ({l1['Eng_n_sigma']:+.1f}σ)",
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
    fig.suptitle("Figure B — amplitude + energy rest evolution per SOC", fontsize=11)
    fig.savefig(FIG_DIR / "fig_evolution_amp_energy.pdf")
    plt.close(fig)
    print(f"  wrote fig_evolution_amp_energy.pdf")

    # ---- Figure C: end-of-rest ToF maps (PRIMARY spatial) ----
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), constrained_layout=True)
    end_tof_maps = []
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        rest_sel = scans_meta[scans_meta["step"] == step].sort_values("timestamp")
        end_idx = rest_sel.index[-1]
        m = tof[end_idx].astype(np.float32).copy()
        m[~roi] = np.nan
        end_tof_maps.append((soc, m))
    # shared color scale
    pooled = np.concatenate([m[roi] for _, m in end_tof_maps])
    vmin = float(np.nanpercentile(pooled, 5))
    vmax = float(np.nanpercentile(pooled, 95))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad((1, 1, 1, 0))
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]
    for ax, (soc, m) in zip(axes, end_tof_maps):
        im = ax.imshow(m, extent=extent, aspect="equal", cmap=cmap,
                       vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
        # subregion dividers
        for xb in (30.0, 50.0):
            ax.axvline(xb, color="white", ls="--", lw=1.2)
        ax.set_xlim(ROI_X_MM[0]-1, ROI_X_MM[1]+1)
        ax.set_ylim(ROI_Y_MM[1]+1, ROI_Y_MM[0]-1)
        ax.set_title(f"SOC {soc}%", color=SOC_COLORS[soc], fontweight="bold")
        ax.set_xlabel("X [mm]   (tabs at X ≈ 64+ →)")
        if ax is axes[0]:
            ax.set_ylabel("Y [mm]")
    fig.colorbar(im, ax=axes, shrink=0.85, location="right", label="ToF [ns]")
    fig.suptitle("Figure C — End-of-rest ToF maps (ROI only)", fontsize=11)
    fig.savefig(FIG_DIR / "fig_endrest_maps_tof.pdf")
    plt.close(fig)
    print(f"  wrote fig_endrest_maps_tof.pdf")

    # ---- Figure D: spatial contrast vs SOC ----
    fig, (ax_tof, ax_amp) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    # ToF contrast on left
    tof_contrast = [r["contrast (prox-distal)"] for r in layer2a if r["feature"] == "tof"]
    ax_tof.plot(SOC_LABELS, tof_contrast, "o-", color="black", ms=10, lw=1.6)
    ax_tof.axhline(0, color="k", lw=0.5)
    ax_tof.fill_between([15, 85], -sigma_ROI["tof_ns"], sigma_ROI["tof_ns"],
                        color="gray", alpha=0.2, label=f"±σ_ROI = {sigma_ROI['tof_ns']:.2f} ns")
    ax_tof.set_xlabel("SOC [%]"); ax_tof.set_ylabel("(tab-prox − tab-distal)  ToF [ns]")
    ax_tof.set_xlim(15, 85)
    ax_tof.grid(alpha=0.3); ax_tof.legend(fontsize=8)
    ax_tof.set_title("ToF spatial contrast")
    # amp / energy on right (two y-axes)
    amp_contrast = [r["contrast (prox-distal)"] for r in layer2a if r["feature"] == "amp"]
    eng_contrast = [r["contrast (prox-distal)"] for r in layer2a if r["feature"] == "eng"]
    ax_amp.plot(SOC_LABELS, amp_contrast, "s-", color="tab:red", ms=9, lw=1.5,
                label="amp [mV]")
    ax_amp.axhline(0, color="k", lw=0.5)
    ax_amp.fill_between([15, 85], -sigma_ROI["amp_mV"], sigma_ROI["amp_mV"],
                        color="tab:red", alpha=0.10)
    ax_amp.set_xlabel("SOC [%]")
    ax_amp.set_ylabel("(tab-prox − tab-distal) amp [mV]", color="tab:red")
    ax_amp.tick_params(axis="y", labelcolor="tab:red")
    ax_amp.set_xlim(15, 85); ax_amp.grid(alpha=0.3)
    ax_eng = ax_amp.twinx()
    ax_eng.plot(SOC_LABELS, eng_contrast, "^--", color="tab:blue", ms=9, lw=1.5,
                label="energy [a.u.]")
    ax_eng.set_ylabel("(tab-prox − tab-distal) energy [a.u.]", color="tab:blue")
    ax_eng.tick_params(axis="y", labelcolor="tab:blue")
    ax_amp.set_title("amplitude + energy spatial contrast")
    fig.suptitle("Figure D — End-of-rest spatial contrast (tab-prox − tab-distal) vs SOC",
                 fontsize=11)
    fig.savefig(FIG_DIR / "fig_spatial_contrast_vs_soc.pdf")
    plt.close(fig)
    print(f"  wrote fig_spatial_contrast_vs_soc.pdf")

    # ---- Figure E: contrast evolution 2x2 (ToF) ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True, sharex=True)
    for ax, (step, soc, l2b_subset) in zip(
            axes.ravel(),
            zip(REST_STEPS, SOC_LABELS,
                [[r for r in layer2b if r["SOC%"]==s and r["feature"]=="tof"][0]
                 for s in SOC_LABELS])):
        rest_sel = scans_meta[scans_meta["step"] == step].sort_values("timestamp")
        t_rest = (rest_sel["timestamp"] - rest_sel["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        contrast = (rest_sel["tof_med_tab-proximal"].values
                    - rest_sel["tof_med_tab-distal"].values)
        slope, intercept = robust_fit(t_rest, contrast)
        ax.scatter(t_rest, contrast, c="black", s=22, zorder=3)
        t_line = np.array([t_rest.min(), t_rest.max()])
        ax.plot(t_line, slope*t_line + intercept, "-",
                color=SOC_COLORS[soc], lw=1.6,
                label=f"slope×120 = {l2b_subset['delta_contrast_120min']:+.2f} ns "
                      f"({l2b_subset['n_sigma_ROI']:+.1f}σ_ROI)")
        ax.fill_between(t_rest, intercept-sigma_ROI["tof_ns"], intercept+sigma_ROI["tof_ns"],
                        color="gray", alpha=0.2)
        ax.axhline(intercept, color="k", lw=0.4)
        ax.set_title(f"SOC {soc}%", color=SOC_COLORS[soc], fontweight="bold")
        ax.set_xlabel("rest time [min]"); ax.set_ylabel("(tab-prox − tab-distal) ToF [ns]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.suptitle("Figure E — Spatial contrast evolution during rest (ToF)", fontsize=11)
    fig.savefig(FIG_DIR / "fig_contrast_evolution.pdf")
    plt.close(fig)
    print(f"  wrote fig_contrast_evolution.pdf")

    # ---- Figure F: per-line acquisition time for one rest (optional) ----
    rep_step = 5
    rep_rest = scans_meta[scans_meta["step"]==rep_step].sort_values("timestamp")
    rep_sess = PROJ / rep_rest.iloc[0]["session_dir"].replace("\\", "/")
    rep_npz_path = next(rep_sess.glob("scan_*.npz"))
    rep_d = np.load(rep_npz_path)
    line_times = rep_d["line_unix_center_s"]
    line_dur_min = (line_times - line_times[0]) / 60.0
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(np.arange(len(line_times)), line_dur_min, "o-", color="tab:blue", ms=3, lw=0.8)
    ax.set_xlabel("line index (y, 0 = first)")
    ax.set_ylabel("line acquisition time [min from first line]")
    ax.set_title(f"Figure F — Per-line acquisition time within one scan\n"
                 f"({rep_npz_path.name})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.annotate(f"total scan duration ≈ {line_dur_min[-1]:.1f} min",
                xy=(len(line_times)-1, line_dur_min[-1]),
                xytext=(0.5, 0.05), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9)
    fig.savefig(FIG_DIR / "fig_acqtime_check.pdf")
    plt.close(fig)
    print(f"  wrote fig_acqtime_check.pdf")

    # ================================================================
    # SAVE summary JSON
    # ================================================================
    summary = {
        "sigma_ROI": sigma_ROI,
        "temperature_overall": T_overall,
        "temperature_per_plateau": T_stats,
        "layer1_rest_evolution": layer1,
        "layer2a_endrest_contrast": layer2a,
        "layer2b_contrast_evolution": layer2b,
        "sanity_checks": sanity,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote summary -> {OUT/'summary.json'}")
    print(f"\nall outputs in {OUT}")


if __name__ == "__main__":
    main()
