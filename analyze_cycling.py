"""
Cycling-data analysis and plotting for A-scan HDF5 archives.

Reads a v2.0 A-scan HDF5 file, computes summary statistics for ToF, amplitude,
and energy, and saves a multi-panel time-series plot showing how each feature
evolves over the cycling experiment. Event annotations from the JSON sidecar
(if present) are overlaid as vertical markers.

Usage
-----
  python analyze_cycling.py path/to/session.h5
  python analyze_cycling.py path/to/session.h5 --smooth 120 --out custom.png

Programmatic
------------
  from analyze_cycling import plot_cycling_features
  plot_cycling_features("data/ascan/<session>/<session>.h5")
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_session(h5_path: str) -> dict:
    """Read all per-snapshot scalar arrays plus session attributes from an HDF5."""
    with h5py.File(h5_path, "r") as f:
        data = {
            "timestamps": f["timestamps"][:],
            "tof_us":     f["tof_us"][:],
            "amplitude":  f["amplitude"][:],
            "energy":     f["energy"][:],
            "n_averaged": f["n_averaged"][:] if "n_averaged" in f else None,
            "attrs":      dict(f.attrs),
        }
    return data


def _load_annotations(h5_path: str, annotations_path: str | None) -> list[dict]:
    """Resolve the companion annotations JSON. Returns [] if none found."""
    if annotations_path is None:
        # Default sibling filename: <stem>_annotations.json
        annotations_path = str(Path(h5_path).with_suffix("")) + "_annotations.json"
    if not os.path.exists(annotations_path):
        return []
    try:
        with open(annotations_path) as fp:
            return json.load(fp)
    except Exception:
        return []


def _rolling_mean(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Centred rolling mean with edge-handling. Returns NaN where the window
    would extend past the data — keeps the smoothed line visually honest."""
    if window_samples <= 1:
        return x.astype(np.float32, copy=True)
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    smoothed = np.convolve(x.astype(np.float64), kernel, mode="same")
    half = window_samples // 2
    smoothed[:half] = np.nan
    smoothed[-half:] = np.nan
    return smoothed.astype(np.float32)


def _format_summary(data: dict) -> str:
    """Build a one-line-per-feature summary string."""
    ts  = data["timestamps"]
    tof = data["tof_us"]
    amp = data["amplitude"]
    eng = data["energy"]
    dur_h = (ts[-1] - ts[0]) / 3600 if len(ts) > 1 else 0.0

    lines = [
        f"  rows:        {len(ts):,}",
        f"  duration:    {dur_h:.2f} hours",
        "",
        f"  ToF        mean={tof.mean():.4f} us   peak-to-peak={tof.max()-tof.min():.4f} us   drift={tof[-100:].mean()-tof[:100].mean():+.4f} us",
        f"  Amplitude  mean={amp.mean():.4f} V    peak-to-peak={amp.max()-amp.min():.4f} V    drift={amp[-100:].mean()-amp[:100].mean():+.4f} V",
        f"  Energy     mean={eng.mean():.2f}      peak-to-peak={eng.max()-eng.min():.2f}      drift={eng[-100:].mean()-eng[:100].mean():+.2f}",
    ]
    return "\n".join(lines)


def plot_cycling_features(
    h5_path: str,
    out_path: str | None = None,
    smooth_window_s: float = 60.0,
    show_raw: bool = True,
    annotations_path: str | None = None,
    figsize: tuple[float, float] = (12, 9),
    dpi: int = 140,
) -> str:
    """
    Generate a 3-panel time-series plot of ToF, amplitude, and energy from an
    A-scan HDF5 cycling session, with optional rolling-mean smoothing and
    annotation markers.

    Parameters
    ----------
    h5_path : str
        Path to the A-scan HDF5 file.
    out_path : str, optional
        Where to save the PNG. Defaults to <stem>_cycling.png next to the source.
    smooth_window_s : float
        Rolling-mean window in seconds. Set to 0 to disable smoothing.
    show_raw : bool
        If True, plot the raw per-snapshot trace as a faint line behind the
        smoothed line.
    annotations_path : str, optional
        Path to the SOC/event annotations JSON. If None, looks for the standard
        sibling file <stem>_annotations.json.
    figsize, dpi
        Matplotlib figure size and resolution.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    data = _load_session(h5_path)
    anns = _load_annotations(h5_path, annotations_path)

    ts  = data["timestamps"]
    if len(ts) < 2:
        raise ValueError(f"{h5_path} has fewer than 2 snapshots — nothing to plot.")
    t_h = (ts - ts[0]) / 3600.0   # elapsed hours from session start

    # X-axis unit choice: hours for long runs, minutes for short
    use_hours = t_h[-1] >= 1.0
    x = t_h if use_hours else t_h * 60.0
    x_label = "elapsed time (hours)" if use_hours else "elapsed time (minutes)"

    # Smoothing: convert seconds to sample count given the median snapshot interval
    dt_med = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    win = max(1, int(round(smooth_window_s / max(dt_med, 1e-6))))
    do_smooth = smooth_window_s > 0 and win > 1

    series = [
        ("ToF (us)",       data["tof_us"],    "tab:red"),
        ("Amplitude (V)",  data["amplitude"], "tab:green"),
        ("Energy",         data["energy"],    "tab:purple"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)
    title = f"{Path(h5_path).name}  -  {len(ts):,} snapshots, {t_h[-1]:.2f} h"
    if do_smooth:
        title += f"  (smoothed {smooth_window_s:.0f}s)"
    fig.suptitle(title, fontsize=11)

    for ax, (name, y, color) in zip(axes, series):
        if show_raw and do_smooth:
            ax.plot(x, y, color=color, linewidth=0.4, alpha=0.25, label="raw")
        if do_smooth:
            y_smooth = _rolling_mean(y, win)
            ax.plot(x, y_smooth, color=color, linewidth=1.4, label="smoothed")
        else:
            ax.plot(x, y, color=color, linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)

        # Annotation markers — vertical dashed lines + small SOC label
        for a in anns:
            t_ann_s = a.get("elapsed_s")
            if t_ann_s is None:
                continue
            t_ann = (t_ann_s / 3600.0) if use_hours else (t_ann_s / 60.0)
            ax.axvline(t_ann, color="orange", linestyle="--", linewidth=0.7, alpha=0.7)

        if do_smooth and show_raw and ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    # Annotation labels on the top axis only (avoids visual clutter)
    if anns:
        ymin, ymax = axes[0].get_ylim()
        for a in anns:
            t_ann_s = a.get("elapsed_s")
            if t_ann_s is None:
                continue
            t_ann = (t_ann_s / 3600.0) if use_hours else (t_ann_s / 60.0)
            soc = a.get("soc_pct")
            label = a.get("label", "")
            tag = f"{int(soc)}% {label}" if soc is not None else label
            axes[0].text(t_ann, ymax, f" {tag}", rotation=90, fontsize=7,
                         color="darkorange", va="top", ha="left", alpha=0.9)

    axes[-1].set_xlabel(x_label)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is None:
        out_path = str(Path(h5_path).with_suffix("")) + "_cycling.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ToF / amplitude / energy trends from a cycling A-scan HDF5."
    )
    parser.add_argument("path", help="A-scan HDF5 file path")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--smooth", type=float, default=60.0,
                        help="Rolling-mean window in seconds (0 disables)")
    parser.add_argument("--no-raw", action="store_true",
                        help="Hide the faint raw trace under the smoothed line")
    parser.add_argument("--annotations", default=None,
                        help="Override path to annotations JSON")
    args = parser.parse_args()

    data = _load_session(args.path)
    print(f"=== {args.path} ===")
    print(_format_summary(data))
    print()

    out = plot_cycling_features(
        args.path,
        out_path=args.out,
        smooth_window_s=args.smooth,
        show_raw=not args.no_raw,
        annotations_path=args.annotations,
    )
    print(f"saved plot: {out}")


if __name__ == "__main__":
    main()
