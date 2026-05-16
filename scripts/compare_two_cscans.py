"""
Pixel-by-pixel comparison of two C-scan sessions.

Compares the previous scan (V_prev) against the latest scan (V_curr) for the
same battery at two states of charge. Produces:
  - Per-feature global statistics (mean, std, RMSE, MAE, correlation)
  - Per-pixel difference maps saved as PNG (latest - previous)
  - Histogram of pixel-wise differences
  - Quick verdict on whether the scanner is producing repeatable, physically
    sensible results between SOC steps.

Usage:
    python analysis/compare_two_cscans.py
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREV = ("cscan_scan_2026-05-14_14-46-55", "scan_1778772194", 3.209)
CURR = ("cscan_scan_2026-05-14_22-22-19", "scan_1778799518", 3.366)
OUT_DIR = os.path.join(ROOT, "analysis", "compare_out")
os.makedirs(OUT_DIR, exist_ok=True)


def load(session_dir: str, scan_id: str) -> dict:
    npz = np.load(os.path.join(ROOT, "data", "cscan", session_dir, f"{scan_id}.npz"))
    with open(os.path.join(ROOT, "data", "cscan", session_dir, f"{scan_id}_meta.json")) as f:
        meta = json.load(f)
    return {
        "amp": npz["amplitude"],
        "tof": npz["tof"],
        "eng": npz["energy"],
        "x": npz["x_mm"],
        "y": npz["y_mm"],
        "meta": meta,
    }


def feature_stats(name: str, a_prev: np.ndarray, a_curr: np.ndarray, unit: str) -> dict:
    d = a_curr - a_prev
    m = np.isfinite(d) & np.isfinite(a_prev) & np.isfinite(a_curr)
    d_m = d[m]
    p = a_prev[m]
    c = a_curr[m]
    corr = np.corrcoef(p, c)[0, 1] if p.size else float("nan")
    stats = {
        "name": name,
        "unit": unit,
        "prev_mean": float(p.mean()),
        "prev_std": float(p.std()),
        "curr_mean": float(c.mean()),
        "curr_std": float(c.std()),
        "mean_delta": float(d_m.mean()),
        "median_delta": float(np.median(d_m)),
        "std_delta": float(d_m.std()),
        "rmse": float(np.sqrt(np.mean(d_m**2))),
        "mae": float(np.mean(np.abs(d_m))),
        "corr": float(corr),
        "p5_delta": float(np.percentile(d_m, 5)),
        "p95_delta": float(np.percentile(d_m, 95)),
        "frac_pixels_increased": float(np.mean(d_m > 0)),
        "n_pixels": int(m.sum()),
    }
    return stats


def print_stats(s: dict) -> None:
    u = s["unit"]
    print(f"\n{s['name']} ({u})")
    print(f"  prev    mean={s['prev_mean']:+.4g}  std={s['prev_std']:.4g}")
    print(f"  curr    mean={s['curr_mean']:+.4g}  std={s['curr_std']:.4g}")
    print(f"  delta   mean={s['mean_delta']:+.4g}  median={s['median_delta']:+.4g}  std={s['std_delta']:.4g}")
    print(f"  RMSE   {s['rmse']:.4g}{u}    MAE   {s['mae']:.4g}{u}")
    print(f"  corr(prev, curr) = {s['corr']:.4f}")
    print(f"  delta 5..95 percentile  =  [{s['p5_delta']:+.4g}, {s['p95_delta']:+.4g}]")
    print(f"  fraction pixels increased = {s['frac_pixels_increased']*100:.1f}%")
    print(f"  pixels compared = {s['n_pixels']}")


def save_diff_map(name: str, a_prev: np.ndarray, a_curr: np.ndarray,
                  x: np.ndarray, y: np.ndarray, unit: str, fname: str) -> None:
    d = a_curr - a_prev
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return
    lim = float(np.percentile(np.abs(finite), 99))
    if lim == 0:
        lim = 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = [float(x.min()), float(x.max()), float(y.max()), float(y.min())]

    vmin = float(min(a_prev.min(), a_curr.min()))
    vmax = float(max(a_prev.max(), a_curr.max()))

    im0 = axes[0].imshow(a_prev, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title(f"{name} — prev (3.209 V)")
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    plt.colorbar(im0, ax=axes[0], label=unit)

    im1 = axes[1].imshow(a_curr, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title(f"{name} — curr (3.366 V)")
    axes[1].set_xlabel("x (mm)")
    plt.colorbar(im1, ax=axes[1], label=unit)

    im2 = axes[2].imshow(d, extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
    axes[2].set_title(f"Δ {name} (curr − prev)")
    axes[2].set_xlabel("x (mm)")
    plt.colorbar(im2, ax=axes[2], label=f"Δ {unit}")

    fig.suptitle(f"C-scan {name}: 3.209 V → 3.366 V", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=140)
    plt.close(fig)


def save_diff_hist(stats_list: list[tuple[str, np.ndarray, np.ndarray, str]], fname: str) -> None:
    fig, axes = plt.subplots(1, len(stats_list), figsize=(5 * len(stats_list), 4))
    if len(stats_list) == 1:
        axes = [axes]
    for ax, (name, a_prev, a_curr, unit) in zip(axes, stats_list):
        d = (a_curr - a_prev).ravel()
        d = d[np.isfinite(d)]
        lim = float(np.percentile(np.abs(d), 99))
        ax.hist(d, bins=120, range=(-lim, lim), color="steelblue", edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(float(d.mean()), color="red", linewidth=1.2, label=f"mean = {d.mean():+.3g}")
        ax.axvline(float(np.median(d)), color="orange", linewidth=1.2, linestyle="--",
                   label=f"median = {np.median(d):+.3g}")
        ax.set_xlabel(f"Δ {name} ({unit})")
        ax.set_ylabel("pixels")
        ax.set_title(f"Δ {name} histogram")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=140)
    plt.close(fig)


def main() -> None:
    print("Loading scans...")
    prev = load(PREV[0], PREV[1])
    curr = load(CURR[0], CURR[1])

    assert prev["amp"].shape == curr["amp"].shape, "shape mismatch between scans"
    assert np.allclose(prev["x"], curr["x"]) and np.allclose(prev["y"], curr["y"]), \
        "axis mismatch between scans"

    print(f"\nPrev: {PREV[0]}  ({PREV[2]} V)")
    print(f"Curr: {CURR[0]}  ({CURR[2]} V)")
    print(f"Grid: {prev['amp'].shape[0]} × {prev['amp'].shape[1]}  "
          f"({prev['meta']['roi_w_mm']} × {prev['meta']['roi_h_mm']} mm, "
          f"{prev['meta']['pitch_mm']} mm pitch)")

    stats = []
    for name, key, unit in [("amplitude", "amp", "V"),
                            ("tof", "tof", "µs"),
                            ("energy", "eng", "")]:
        s = feature_stats(name, prev[key], curr[key], unit)
        stats.append(s)
        print_stats(s)
        save_diff_map(name, prev[key], curr[key], prev["x"], prev["y"], unit,
                      f"diff_{name}.png")

    save_diff_hist([("amplitude", prev["amp"], curr["amp"], "V"),
                    ("tof",       prev["tof"], curr["tof"], "µs"),
                    ("energy",    prev["eng"], curr["eng"], "")],
                   "diff_histograms.png")

    with open(os.path.join(OUT_DIR, "compare_summary.json"), "w") as f:
        json.dump({
            "prev": {"session": PREV[0], "voltage_v": PREV[2]},
            "curr": {"session": CURR[0], "voltage_v": CURR[2]},
            "delta_v": CURR[2] - PREV[2],
            "features": stats,
        }, f, indent=2)

    print(f"\nSaved diff maps and summary to: {OUT_DIR}")

    print("\n--- Verdict ---")
    amp = stats[0]; tof = stats[1]; eng = stats[2]
    verdicts = []
    if amp["corr"] > 0.9:
        verdicts.append(f"  [OK] Amplitude map correlated (r={amp['corr']:.3f}) - same battery, repeatable raster.")
    else:
        verdicts.append(f"  [!!] Amplitude correlation r={amp['corr']:.3f} is low - alignment or hardware drift?")
    if tof["corr"] > 0.9:
        verdicts.append(f"  [OK] ToF map correlated (r={tof['corr']:.3f}).")
    else:
        verdicts.append(f"  [!!] ToF correlation r={tof['corr']:.3f} is low.")
    if tof["mean_delta"] < 0:
        verdicts.append(f"  [OK] ToF decreased by {abs(tof['mean_delta'])*1000:.1f} ns on average - "
                        f"consistent with charging (sound speed rises).")
    else:
        verdicts.append(f"  [??] ToF increased by {tof['mean_delta']*1000:.1f} ns - unexpected sign for charging.")
    if amp["mean_delta"] > 0:
        verdicts.append(f"  [OK] Amplitude rose by {amp['mean_delta']:.4f} V on average.")
    else:
        verdicts.append(f"  [??] Amplitude dropped by {abs(amp['mean_delta']):.4f} V on average.")
    for v in verdicts:
        print(v)


if __name__ == "__main__":
    main()
