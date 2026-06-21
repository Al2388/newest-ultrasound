"""All-rest by-band analysis for the 35C continuous run.

Question: during each 2-h rest segment (after each charge step), does the
ultrasound signal drift, and is that drift uniform across the cell or
concentrated in the tab-proximal / interior / tab-distal X-band?

Reuses:
  - 3 X-bands from analysis_07_tab_heterogeneity.py (tabs at high X)
  - canonical 50x40 mm ROI mask from project-canonical-roi memory
  - sigma_ROI noise floor from longrun_analysis/common.NOISE_FLOOR
  - rest-window identification from the Maccor cycler log

Output:
  reports/longrun_cycling_35c_2026-05-30_19-47-06_analysis/all_rest_by_band/
    drift_per_soc_per_band.png   (4 SOC rows x 3 modality cols)
    drift_summary.csv            (one row per SOC x band x modality)
    README.md                    (verdict text)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))
sys.path.insert(0, str(PROJ / "scripts"))
from common import NOISE_FLOOR  # noqa: E402
from extract_charge_discharge import read_cycler  # noqa: E402


BATCH_DIR = PROJ / "reports" / "experiments" / "longrun_cycling_35c_2026-05-30_19-47-06"
CHECKPOINT = BATCH_DIR / "checkpoint.json"
CYCLER = PROJ / "data" / "raw" / "cycler" / "LFP860_35degrees.002.txt"
ROI_MASK_PATH = PROJ / "reports" / "experiments" / "roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp" / "roi_mask.npy"
OUT = PROJ / "reports" / "longrun_cycling_35c_2026-05-30_19-47-06_analysis" / "all_rest_by_band"
OUT.mkdir(parents=True, exist_ok=True)


# Bands match analysis_07_tab_heterogeneity. Tabs at high X (right).
BANDS = [
    ("tab-proximal", 50.0, 64.5, "#d62728"),   # red
    ("interior",     30.0, 50.0, "#7f7f7f"),   # gray
    ("tab-distal",   14.6, 30.0, "#1f77b4"),   # blue
]

UNITS = {
    "amplitude": ("mV",   1e3, NOISE_FLOOR["amp_sigma_roi_mv"]),
    "tof":       ("ns",   1e3, NOISE_FLOOR["tof_sigma_roi_ns"]),
    "energy":    ("a.u.", 1.0, NOISE_FLOOR["energy_sigma_roi"]),
}

# 4 rest segments to analyze. Pulled from cycler enumeration (step,md).
# step indices follow Maccor numbering in LFP860_35degrees.002.txt.
REST_TARGETS = [
    ("SOC 20%", 5,  "after charge 1"),
    ("SOC 40%", 7,  "after charge 2"),
    ("SOC 60%", 9,  "after charge 3"),
    ("SOC 80%", 11, "after charge 4"),
]


def _scan_ts_from_dir(name: str) -> datetime | None:
    m = re.search(r"_r\d{3}_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$", name)
    if not m:
        return None
    d, hh, mm, ss = m.groups()
    return datetime.strptime(f"{d} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")


def _step_window(cyc: dict, step_no: int) -> tuple[datetime, datetime] | None:
    """Return wall-clock start/end of the contiguous block where step == step_no."""
    s = cyc["step"]; t0 = cyc["t0"]; el = cyc["elapsed_s"]
    idx = np.where(s == step_no)[0]
    if idx.size == 0:
        return None
    return t0 + timedelta(seconds=float(el[idx[0]])), t0 + timedelta(seconds=float(el[idx[-1]]))


def _band_mask(roi_mask: np.ndarray, x_mm: np.ndarray, x_lo: float, x_hi: float) -> np.ndarray:
    col_lo = int(np.searchsorted(x_mm, x_lo))
    col_hi = int(np.searchsorted(x_mm, x_hi))
    m = np.zeros_like(roi_mask, dtype=bool)
    m[:, col_lo:col_hi] = roi_mask[:, col_lo:col_hi]
    return m


def _band_series(arrs: dict, band_mask: np.ndarray) -> dict:
    out = {}
    m = band_mask.reshape(-1)
    for k, a in arrs.items():
        flat = a.reshape(a.shape[0], -1)
        out[k] = np.nanmean(flat[:, m], axis=1)
    return out


def _select_scans_in_window(cp: dict, t_lo: datetime, t_hi: datetime) -> list[tuple]:
    """Return list of (run_idx, sess_dir, ts) for completed scans whose acq ts is in [t_lo, t_hi]."""
    out = []
    for c in cp["completed"]:
        sess_dir = PROJ / Path(c["session_dir"].replace("\\", "/"))
        ts = _scan_ts_from_dir(sess_dir.name)
        if ts is None:
            continue
        if t_lo <= ts <= t_hi:
            out.append((c["run_idx"], sess_dir, ts))
    out.sort(key=lambda r: r[2])
    return out


def _load_stack(scans: list[tuple]) -> tuple[dict, list[datetime], np.ndarray, np.ndarray]:
    arrs = {"amplitude": [], "tof": [], "energy": []}
    line_T = []
    x_mm = y_mm = None
    times = []
    for _, sess_dir, ts in scans:
        npz_path = next(sess_dir.glob("scan_*.npz"), None)
        if npz_path is None:
            continue
        d = np.load(npz_path)
        arrs["amplitude"].append(d["amplitude"].astype(np.float32))
        arrs["tof"].append(d["tof"].astype(np.float32))
        arrs["energy"].append(d["energy"].astype(np.float32))
        if "line_temperature_mean_c" in d.files:
            line_T.append(float(np.nanmean(d["line_temperature_mean_c"])))
        else:
            line_T.append(np.nan)
        if x_mm is None:
            x_mm = d["x_mm"].astype(np.float32)
            y_mm = d["y_mm"].astype(np.float32)
        times.append(ts)
    for k in arrs:
        arrs[k] = np.stack(arrs[k], axis=0)
    return arrs, times, x_mm, np.array(line_T)


def main() -> None:
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    cyc = read_cycler(str(CYCLER))
    roi_mask = np.load(ROI_MASK_PATH)

    # Build figure: 4 rows (SOC) x 3 cols (modality), all share x = rest-time-in-min
    fig, axes = plt.subplots(len(REST_TARGETS), 3, figsize=(15, 12),
                             constrained_layout=True, sharex=True)

    csv_rows = ["soc_label,band,modality,unit,delta_end_minus_start,delta_in_sigma_roi,n_scans,rest_duration_min,T_drift_mC\n"]
    md_lines = [
        "# 35C all-rest by-band drift analysis\n\n",
        "Rest windows pulled from Maccor cycler `LFP860_35degrees.002.txt`; ",
        "C-scans matched by acquisition timestamp; band masks ANDed with canonical 50x40 mm ROI.\n\n",
        f"σ_ROI noise floor: amp {NOISE_FLOOR['amp_sigma_roi_mv']:.2f} mV, "
        f"ToF {NOISE_FLOOR['tof_sigma_roi_ns']:.2f} ns, "
        f"energy {NOISE_FLOOR['energy_sigma_roi']:.3g}.\n\n",
    ]

    for row, (soc_label, step_no, descr) in enumerate(REST_TARGETS):
        win = _step_window(cyc, step_no)
        if win is None:
            print(f"[skip] {soc_label}: step {step_no} not found in cycler")
            continue
        t_lo, t_hi = win
        scans = _select_scans_in_window(cp, t_lo, t_hi + timedelta(minutes=10))
        print(f"{soc_label} (step {step_no}, {descr}): {len(scans)} scans in window {t_lo} -> {t_hi}")
        if len(scans) < 3:
            print(f"  too few scans, skipping")
            continue

        arrs, times, x_mm, line_T = _load_stack(scans)
        rest_min = np.array([(t - times[0]).total_seconds() / 60.0 for t in times])
        T_drift_mC = (line_T - line_T[0]) * 1000.0 if np.isfinite(line_T[0]) else np.zeros_like(line_T)

        md_lines.append(
            f"## {soc_label}  ({descr}, step {step_no})\n\n"
            f"- window: {t_lo:%Y-%m-%d %H:%M:%S} -> {t_hi:%Y-%m-%d %H:%M:%S}  ({(t_hi-t_lo).total_seconds()/60:.0f} min)\n"
            f"- n scans: {len(scans)}, span captured: {rest_min[-1]:.1f} min\n"
            f"- cell-side T drift over window: {T_drift_mC[-1]:+.1f} mC "
            f"(start {line_T[0]:.3f} → end {line_T[-1]:.3f} C)\n\n"
            f"| band | modality | Δ@end [unit] | Δ in σ_ROI |\n|---|---|---:|---:|\n"
        )

        band_masks = [(name, _band_mask(roi_mask, x_mm, lo, hi), color)
                      for (name, lo, hi, color) in BANDS]

        for col, mod in enumerate(["amplitude", "tof", "energy"]):
            ax = axes[row, col]
            unit, scale, sigma = UNITS[mod]
            for name, bmask, color in band_masks:
                if bmask.sum() == 0:
                    continue
                series = _band_series({mod: arrs[mod]}, bmask)[mod] * scale
                d_series = series - series[0]
                ax.plot(rest_min, d_series, "o-", color=color, lw=1.5, ms=4, label=name)
                delta_end = float(d_series[-1])
                md_lines.append(
                    f"| {name} | {mod} | {delta_end:+.3g} {unit} | {delta_end/sigma:+.2f} |\n"
                )
                csv_rows.append(
                    f"{soc_label},{name},{mod},{unit},{delta_end:.4g},"
                    f"{delta_end/sigma:.3f},{len(scans)},{rest_min[-1]:.1f},{T_drift_mC[-1]:.1f}\n"
                )
            ax.fill_between(rest_min, -2 * sigma, 2 * sigma, color="gray",
                            alpha=0.15, label="±2σ_ROI" if (row == 0 and col == 0) else None)
            ax.axhline(0, color="k", lw=0.4)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"{mod}  (σ_ROI = {sigma:.3g} {unit})")
            if col == 0:
                ax.set_ylabel(f"{soc_label}\nΔ{mod} [{unit}]")
            if row == len(REST_TARGETS) - 1:
                ax.set_xlabel("rest time [min]")
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="best")
        md_lines.append("\n")

    fig.suptitle("35°C continuous run · within-rest US drift by SOC × cell band\n"
                 "(red = tab-proximal, gray = interior, blue = tab-distal; ±2σ_ROI grey)", fontsize=12)
    out_png = OUT / "drift_per_soc_per_band.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    (OUT / "drift_summary.csv").write_text("".join(csv_rows), encoding="utf-8")
    (OUT / "README.md").write_text("".join(md_lines), encoding="utf-8")
    print(f"\nwrote {out_png}")
    print(f"wrote {OUT / 'drift_summary.csv'}")
    print(f"wrote {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
