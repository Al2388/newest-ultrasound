"""Re-render all spec figures (A-F) with much cleaner, publication-ready style.

Improvements over v1:
  - No twin-y axes (split into separate panels instead)
  - Consistent colour palette and typography across all figures
  - Larger labels, cleaner band visualisations
  - Each panel has its own title with quantitative summary
  - Clear sub-region dividers on maps
  - Thermal envelope drawn as a soft horizontal band, not stacked

Reads the underlying numbers from the spec analysis output, so this script
is purely cosmetic -- the numbers are unchanged from the spec run.
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
SPEC_OUT = PROJ / "reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis"
FIG_DIR = SPEC_OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- constants ----
ALPHA_T_NS_PER_C = +73.5
ROI_X_MM = (14.6, 64.5)
ROI_Y_MM = (16.1, 55.4)
SUBREGIONS_X = {
    "tab-distal":   (14.6, 30.0),
    "interior":     (30.0, 50.0),
    "tab-proximal": (50.0, 64.5),
}
REST_STEPS = [5, 7, 9, 11]
SOC_LABELS = [20, 40, 60, 80]

# ---- consistent style ----
SOC_COLOR = {
    20: "#3b75c6",   # blue
    40: "#3aa86c",   # green
    60: "#e08e1c",   # orange
    80: "#c34141",   # red
}
FEATURE_COLOR = {
    "tof": "#3b3b3b",      # near-black
    "amp": "#c34141",      # red
    "eng": "#3b75c6",      # blue
}
BAND_COLOR_NOISE = "#9e9e9e"   # gray for sigma_ROI band
BAND_COLOR_THERMAL = "#f1c060"  # warm yellow for thermal envelope

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":      300,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "figure.facecolor": "white",
})


def load_inputs():
    d = np.load(CACHE / "stack.npz")
    meta = pd.read_csv(CACHE / "meta.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    summary = json.loads((SPEC_OUT / "summary.json").read_text())
    return d, meta, summary


def _x_in(x_mm, lo, hi): return np.where((x_mm >= lo) & (x_mm <= hi))[0]
def _y_in(y_mm, lo, hi): return np.where((y_mm >= lo) & (y_mm <= hi))[0]


def build_masks(d):
    x_mm = d["x_mm"]; y_mm = d["y_mm"]
    x_idx = _x_in(x_mm, *ROI_X_MM); y_idx = _y_in(y_mm, *ROI_Y_MM)
    roi = np.zeros((y_mm.size, x_mm.size), dtype=bool)
    roi[np.ix_(y_idx, x_idx)] = True
    sub = {}
    for n, (lo, hi) in SUBREGIONS_X.items():
        m = np.zeros_like(roi); m[np.ix_(y_idx, _x_in(x_mm, lo, hi))] = True
        sub[n] = m
    return roi, sub, x_mm, y_mm


def med_in(arr3d, mask):
    out = np.empty(arr3d.shape[0])
    flat = mask.reshape(-1)
    for i in range(arr3d.shape[0]):
        out[i] = float(np.nanmedian(arr3d[i].reshape(-1)[flat]))
    return out


def rfit(t, y):
    f = np.isfinite(t) & np.isfinite(y)
    if f.sum() < 3:
        return float("nan"), float("nan")
    s, b = siegelslopes(y[f], t[f])
    return float(s), float(b)


def _decorate_panel(ax, soc):
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(direction="out", color="#444")
    ax.grid(True, alpha=0.18, linewidth=0.7)


def add_soc_badge(ax, soc):
    """Coloured SOC badge in the corner."""
    ax.text(0.97, 0.96, f"SOC {soc}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=SOC_COLOR[soc],
                      edgecolor="none"))


# ============================================================
# Figure A — ToF rest evolution per SOC (PRIMARY)
# ============================================================
def figure_A(d, meta, summary, roi):
    tof_ns = d["tof"] * 1e3
    sR = summary["sigma_ROI"]["tof_ns"]
    layer1 = {r["SOC%"]: r for r in summary["layer1_rest_evolution"]}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True,
                             sharex=True)
    for ax, (step, soc) in zip(axes.ravel(), zip(REST_STEPS, SOC_LABELS)):
        rest = meta[meta["step"] == step].sort_values("timestamp")
        idx = rest.index.values
        t = (rest["timestamp"] - rest["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        y = med_in(tof_ns[idx], roi)
        slope, intercept = rfit(t, y)

        L = layer1[soc]
        env = abs(L["thermal_envelope_3sigT_ToF_ns"])
        y0 = float(intercept)

        # noise band (sigma_ROI) -- subtle gray
        ax.fill_between(t, y0 - sR, y0 + sR, color=BAND_COLOR_NOISE, alpha=0.18, zorder=1,
                        label=f"±σ_ROI = ±{sR:.2f} ns")
        # thermal envelope -- warm yellow, lighter
        ax.fill_between(t, y0 - env, y0 + env, color=BAND_COLOR_THERMAL, alpha=0.22, zorder=0,
                        label=f"3σ_T thermal env. = ±{env:.2f} ns")

        # data
        ax.scatter(t, y, color=SOC_COLOR[soc], s=42, zorder=5,
                   edgecolors="white", linewidths=0.8, label="ROI median")
        # fit
        tline = np.array([t.min(), t.max()])
        ax.plot(tline, slope*tline + intercept, "-", color=SOC_COLOR[soc],
                lw=2.0, zorder=4,
                label=f"Theil–Sen fit")

        # annotation
        sign = "↑" if L["dToF_fit_120min_ns"] > 0 else "↓"
        ax.text(0.03, 0.04,
                f"Δ_120min = {L['dToF_fit_120min_ns']:+.1f} ns {sign}\n"
                f"thermal × = {L['ToF_thermal_multiple']:+.1f}\n"
                f"noise n_σ = {L['ToF_n_sigma']:+.1f}",
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", lw=0.6))

        ax.set_title(f"SOC {soc} % rest", color=SOC_COLOR[soc], fontweight="bold")
        ax.set_xlabel("rest time [min]")
        ax.set_ylabel("ROI-median ToF [ns]")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)
        _decorate_panel(ax, soc)

    fig.suptitle("Figure A   Rest evolution of cell-ROI ToF, per SOC plateau\n"
                 "(noise floor: σ_ROI = 4.70 ns;  ToF temperature coefficient: α_T = +73.5 ns/°C)",
                 fontsize=11, y=1.01)
    out = FIG_DIR / "fig_evolution_tof.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ============================================================
# Figure B — amplitude + energy rest evolution (split, NO twin-y)
# ============================================================
def figure_B(d, meta, summary, roi):
    amp_mV = d["amplitude"] * 1e3
    eng_au = d["energy"]
    sR_amp = summary["sigma_ROI"]["amp_mV"]
    sR_eng = summary["sigma_ROI"]["eng_au"]
    layer1 = {r["SOC%"]: r for r in summary["layer1_rest_evolution"]}

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True,
                             sharex=True)
    # Row 0 = amplitude; Row 1 = energy
    for col, (step, soc) in enumerate(zip(REST_STEPS, SOC_LABELS)):
        rest = meta[meta["step"] == step].sort_values("timestamp")
        idx = rest.index.values
        t = (rest["timestamp"] - rest["timestamp"].iloc[0]).dt.total_seconds().values / 60.0

        # amplitude
        ax = axes[0, col]
        y = med_in(amp_mV[idx], roi)
        s, b = rfit(t, y)
        ax.fill_between(t, b - sR_amp, b + sR_amp, color=BAND_COLOR_NOISE, alpha=0.18,
                        label=f"±σ_ROI = ±{sR_amp:.2f} mV" if col == 0 else None)
        ax.scatter(t, y, color=SOC_COLOR[soc], s=38, edgecolors="white", linewidths=0.7, zorder=4)
        tl = np.array([t.min(), t.max()])
        ax.plot(tl, s*tl+b, "-", color=SOC_COLOR[soc], lw=2.0, zorder=3)
        L = layer1[soc]
        ax.set_title(f"SOC {soc} % — amp", color=SOC_COLOR[soc], fontweight="bold")
        ax.set_ylabel("ROI-median amplitude [mV]" if col == 0 else "")
        ax.text(0.03, 0.04,
                f"Δ_120 = {L['dAmp_fit_120min_mV']:+.1f} mV\n"
                f"n_σ   = {L['Amp_n_sigma']:+.1f}",
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888", lw=0.5))
        _decorate_panel(ax, soc)

        # energy
        ax = axes[1, col]
        y = med_in(eng_au[idx], roi)
        s, b = rfit(t, y)
        ax.fill_between(t, b - sR_eng, b + sR_eng, color=BAND_COLOR_NOISE, alpha=0.18,
                        label=f"±σ_ROI = ±{sR_eng:.3g} a.u." if col == 0 else None)
        ax.scatter(t, y, color=SOC_COLOR[soc], s=38, edgecolors="white", linewidths=0.7, zorder=4)
        ax.plot(tl, s*tl+b, "-", color=SOC_COLOR[soc], lw=2.0, zorder=3)
        ax.set_title(f"SOC {soc} % — energy", color=SOC_COLOR[soc], fontweight="bold")
        ax.set_xlabel("rest time [min]")
        ax.set_ylabel("ROI-median energy [a.u.]" if col == 0 else "")
        ax.text(0.03, 0.04,
                f"Δ_120 = {L['dEng_fit_120min_au']:+.3f}\n"
                f"n_σ   = {L['Eng_n_sigma']:+.1f}",
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888", lw=0.5))
        _decorate_panel(ax, soc)

    axes[0, 0].legend(loc="upper right", fontsize=7.5)
    axes[1, 0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Figure B   Rest evolution of cell-ROI amplitude (top) and energy (bottom), per SOC plateau",
                 fontsize=11, y=1.005)
    out = FIG_DIR / "fig_evolution_amp_energy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ============================================================
# Figure C — end-of-rest ToF maps (PRIMARY spatial)
# ============================================================
def figure_C(d, meta, roi, x_mm, y_mm):
    tof_ns = d["tof"] * 1e3

    end_maps = []
    for step, soc in zip(REST_STEPS, SOC_LABELS):
        rest = meta[meta["step"] == step].sort_values("timestamp")
        end_idx = rest.index[-1]
        m = tof_ns[end_idx].astype(np.float32).copy()
        m[~roi] = np.nan
        end_maps.append((soc, m))

    pooled = np.concatenate([m[roi] for _, m in end_maps])
    vmin = float(np.nanpercentile(pooled, 5))
    vmax = float(np.nanpercentile(pooled, 95))

    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad((1, 1, 1, 0))
    extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.0), constrained_layout=True)
    for ax, (soc, m) in zip(axes, end_maps):
        im = ax.imshow(m, extent=extent, aspect="equal", cmap=cmap,
                       vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
        # sub-region divider dashes
        for xb in (30.0, 50.0):
            ax.axvline(xb, color="white", ls="--", lw=1.2, alpha=0.85)
        # sub-region labels at top
        y_lbl = ROI_Y_MM[0] - 0.5
        for cx, lbl in [((ROI_X_MM[0]+30)/2, "tab-distal"),
                        ((30+50)/2, "interior"),
                        ((50+ROI_X_MM[1])/2, "tab-prox")]:
            ax.text(cx, y_lbl, lbl, ha="center", va="bottom",
                    fontsize=8, color="black", fontweight="bold")
        # tab indicator arrow
        ax.annotate("tabs →", xy=(ROI_X_MM[1]+0.5, (ROI_Y_MM[0]+ROI_Y_MM[1])/2),
                    fontsize=8, va="center", ha="left", color="#333")
        ax.set_xlim(ROI_X_MM[0]-1.0, ROI_X_MM[1]+5.0)
        ax.set_ylim(ROI_Y_MM[1]+1.0, ROI_Y_MM[0]-3.5)
        ax.set_title(f"SOC {soc} %", color=SOC_COLOR[soc], fontweight="bold", fontsize=12)
        ax.set_xlabel("X [mm]")
        if ax is axes[0]:
            ax.set_ylabel("Y [mm]")
        ax.tick_params(direction="out")
    fig.colorbar(im, ax=axes, shrink=0.82, location="right", label="ToF [ns]",
                 pad=0.02)
    fig.suptitle("Figure C   End-of-rest ToF maps (ROI only) at each SOC plateau\n"
                 f"shared scale {vmin:.0f}–{vmax:.0f} ns;  dashed lines at X = 30, 50 mm divide sub-regions",
                 fontsize=11, y=1.04)
    out = FIG_DIR / "fig_endrest_maps_tof.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ============================================================
# Figure D — spatial contrast vs SOC (3 separate panels, no twin)
# ============================================================
def figure_D(summary):
    sR = summary["sigma_ROI"]
    l2a = summary["layer2a_endrest_contrast"]
    by_feat = {f: [r for r in l2a if r["feature"] == f] for f in ("tof", "amp", "eng")}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    panel_meta = [
        ("tof", "ToF", "ns",      sR["tof_ns"], FEATURE_COLOR["tof"]),
        ("amp", "amplitude", "mV", sR["amp_mV"], FEATURE_COLOR["amp"]),
        ("eng", "energy", "a.u.",  sR["eng_au"], FEATURE_COLOR["eng"]),
    ]
    for ax, (feat, fname, unit, sR_v, color) in zip(axes, panel_meta):
        ys = [r["contrast (prox-distal)"] for r in by_feat[feat]]
        xs = [r["SOC%"] for r in by_feat[feat]]
        # ±sigma_ROI noise band
        ax.fill_between([15, 85], -sR_v, sR_v, color=BAND_COLOR_NOISE,
                        alpha=0.20, label=f"±σ_ROI = ±{sR_v:.3g} {unit}")
        ax.axhline(0, color="#444", lw=0.7)
        # connect-the-dots line
        ax.plot(xs, ys, "-", color=color, lw=1.6, alpha=0.4, zorder=2)
        # large dots colored by SOC
        for xi, yi in zip(xs, ys):
            ax.scatter([xi], [yi], color=SOC_COLOR[xi], s=110, zorder=4,
                       edgecolors="white", linewidths=1.4)
        # value labels
        for r in by_feat[feat]:
            ax.annotate(f"{r['contrast (prox-distal)']:+.1f}\n({r['n_sigma_ROI']:+.0f}σ)",
                        (r["SOC%"], r["contrast (prox-distal)"]),
                        textcoords="offset points", xytext=(0, -22),
                        ha="center", fontsize=7.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(15, 85)
        ax.set_xticks(SOC_LABELS)
        ax.set_xlabel("SOC [%]")
        ax.set_ylabel(f"contrast (tab-prox − tab-distal)  [{unit}]")
        ax.set_title(fname, fontweight="bold", color=color)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.18, linewidth=0.7)

    fig.suptitle("Figure D   End-of-rest spatial contrast (tab-proximal − tab-distal) vs SOC",
                 fontsize=11, y=1.02)
    out = FIG_DIR / "fig_spatial_contrast_vs_soc.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ============================================================
# Figure E — contrast evolution during rest (PRIMARY spatial dynamic)
# Show ToF + amp + energy as 3 separate sub-panel rows per SOC = 4 rows x 3 cols
# ============================================================
def figure_E(d, meta, summary, sub):
    tof_ns = d["tof"] * 1e3
    amp_mV = d["amplitude"] * 1e3
    eng_au = d["energy"]
    sR = summary["sigma_ROI"]
    l2b = {(r["SOC%"], r["feature"]): r for r in summary["layer2b_contrast_evolution"]}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True, sharex=True)
    for ax, (step, soc) in zip(axes.ravel(), zip(REST_STEPS, SOC_LABELS)):
        rest = meta[meta["step"] == step].sort_values("timestamp")
        idx = rest.index.values
        t = (rest["timestamp"] - rest["timestamp"].iloc[0]).dt.total_seconds().values / 60.0

        prox = med_in(tof_ns[idx], sub["tab-proximal"])
        dist = med_in(tof_ns[idx], sub["tab-distal"])
        contrast = prox - dist
        s, b = rfit(t, contrast)

        L = l2b[(soc, "tof")]
        sR_v = sR["tof_ns"]

        # noise band centered at intercept
        ax.fill_between(t, b - sR_v, b + sR_v, color=BAND_COLOR_NOISE, alpha=0.20)
        ax.axhline(b, color="#666", lw=0.4, ls="--", alpha=0.6)
        ax.scatter(t, contrast, color=SOC_COLOR[soc], s=44, edgecolors="white",
                   linewidths=0.8, zorder=4)
        tl = np.array([t.min(), t.max()])
        ax.plot(tl, s*tl + b, "-", color=SOC_COLOR[soc], lw=2.0, zorder=3)

        sign = "↑" if L["delta_contrast_120min"] > 0 else "↓"
        ax.text(0.03, 0.04,
                f"Δ_120 = {L['delta_contrast_120min']:+.1f} ns {sign}\n"
                f"n_σ   = {L['n_sigma_ROI']:+.1f}",
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888", lw=0.5))

        ax.set_title(f"SOC {soc} % — ToF (prox − distal) contrast",
                     color=SOC_COLOR[soc], fontweight="bold")
        ax.set_xlabel("rest time [min]")
        ax.set_ylabel("ToF contrast [ns]")
        _decorate_panel(ax, soc)

    fig.suptitle("Figure E   Within-rest evolution of ToF spatial contrast (tab-prox − tab-distal)\n"
                 "robust Theil–Sen fit;  gray band = ±σ_ROI ToF",
                 fontsize=11, y=1.01)
    out = FIG_DIR / "fig_contrast_evolution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ============================================================
# Figure F — per-line acquisition-time ramp
# ============================================================
def figure_F(meta):
    rest = meta[meta["step"] == 5].sort_values("timestamp")
    sess = PROJ / rest.iloc[0]["session_dir"].replace("\\", "/")
    npz = next(sess.glob("scan_*.npz"))
    arr = np.load(npz)
    lt = arr["line_unix_center_s"]
    t_min = (lt - lt[0]) / 60.0
    line_idx = np.arange(len(lt))

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.plot(line_idx, t_min, "-", color="#3b75c6", lw=1.6, zorder=3)
    ax.scatter(line_idx[::8], t_min[::8], color="#3b75c6", s=14, zorder=4)
    # per-line spacing annotation
    dt_per_line_s = float(np.median(np.diff(lt)))
    total_min = float(t_min[-1])
    ax.annotate(f"total scan duration ≈ {total_min:.1f} min\n"
                f"per-line spacing ≈ {dt_per_line_s:.2f} s",
                xy=(0.97, 0.04), xycoords="axes fraction",
                ha="right", va="bottom",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", lw=0.6))
    ax.set_xlim(0, len(lt))
    ax.set_xlabel("line index  (y axis,  0 = first line)")
    ax.set_ylabel("elapsed time since first line [min]")
    ax.set_title("Figure F   Per-line acquisition time within one scan\n"
                 "(supports: acquisition-time gradient is in y, not x)",
                 fontsize=11)
    ax.grid(alpha=0.20, linewidth=0.7)
    out = FIG_DIR / "fig_acqtime_check.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


def main():
    d, meta, summary = load_inputs()
    roi, sub, x_mm, y_mm = build_masks(d)

    print("re-rendering figures with cleaner style:")
    figure_A(d, meta, summary, roi)
    figure_B(d, meta, summary, roi)
    figure_C(d, meta, roi, x_mm, y_mm)
    figure_D(summary)
    figure_E(d, meta, summary, sub)
    figure_F(meta)
    print(f"\nall outputs in {FIG_DIR}")


if __name__ == "__main__":
    main()
