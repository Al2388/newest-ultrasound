"""
Plot waveform data stored in A-scan / gauging HDF5 archives.

Usage
-----
  python plot_h5_waveforms.py data/ascan/ascan_1778240169.h5
  python plot_h5_waveforms.py data/ascan/ascan_new.h5 --dataset raw_waveforms
  python plot_h5_waveforms.py data/ascan/ascan_new.h5 --out-dir plots --max-traces 80

Outputs
-------
  <stem>_<dataset>_overlay.png   Selected waveform traces overlaid
  <stem>_<dataset>_heatmap.png   Waveform amplitude heatmap
  <stem>_features.png            ToF / amplitude / energy trends, if available

Notes
-----
New A-scan files can contain /raw_waveforms, one row per individual pulse.
Older A-scan files contain /waveforms only, one row per averaged snapshot.
Both are binary HDF5 datasets; this script converts them back to viewable PNGs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _choose_dataset(f: h5py.File, requested: str) -> str:
    if requested != "auto":
        if requested not in f:
            raise KeyError(f"Dataset /{requested} not found in {f.filename}")
        return requested
    if "raw_waveforms" in f:
        return "raw_waveforms"
    if "waveforms" in f:
        return "waveforms"
    raise KeyError("No /raw_waveforms or /waveforms dataset found.")


def _sample_indices(n_rows: int, max_rows: int, start: int = 0) -> np.ndarray:
    if n_rows <= 0:
        return np.array([], dtype=np.int64)
    start = max(0, min(int(start), n_rows - 1))
    count = max(1, min(int(max_rows), n_rows - start))
    return np.unique(np.linspace(start, n_rows - 1, count).astype(np.int64))


def _time_axis_us(f: h5py.File, n_samples: int, dataset_name: str) -> np.ndarray:
    fs = float(f.attrs.get("fs_hz", 20_000_000.0))
    if dataset_name in {"raw_waveforms", "waveforms"}:
        start_us = float(f.attrs.get("gate_us_start", 0.0))
    else:
        start_us = 0.0
    return start_us + np.arange(n_samples, dtype=np.float64) / fs * 1e6


def _feature_names(dataset_name: str) -> tuple[str, str, str, str]:
    if dataset_name == "raw_waveforms":
        return "raw_timestamps", "raw_tof_us", "raw_amplitude", "raw_energy"
    return "timestamps", "tof_us", "amplitude", "energy"


def plot_overlay(f: h5py.File, dataset_name: str, out_path: Path,
                 max_traces: int, start: int) -> None:
    ds = f[dataset_name]
    idx = _sample_indices(ds.shape[0], max_traces, start)
    if idx.size == 0:
        raise ValueError(f"/{dataset_name} is empty")

    waves = ds[idx, :]
    x_us = _time_axis_us(f, ds.shape[1], dataset_name)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    for row_idx, y in zip(idx, waves):
        ax.plot(x_us, y, linewidth=0.8, alpha=0.45, label=str(row_idx))

    gate_start = float(f.attrs.get("gate_us_start", x_us[0]))
    gate_end = float(f.attrs.get("gate_us_end", x_us[-1]))
    ax.axvspan(gate_start, gate_end, color="tab:blue", alpha=0.10, label="gate")
    ax.set_title(f"{Path(f.filename).name} /{dataset_name} overlay ({idx.size} traces)")
    ax.set_xlabel("time after sync (us)")
    ax.set_ylabel("voltage (V)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap(f: h5py.File, dataset_name: str, out_path: Path,
                 max_rows: int, start: int) -> None:
    ds = f[dataset_name]
    idx = _sample_indices(ds.shape[0], max_rows, start)
    if idx.size == 0:
        raise ValueError(f"/{dataset_name} is empty")

    waves = ds[idx, :]
    x_us = _time_axis_us(f, ds.shape[1], dataset_name)
    vlim = np.nanpercentile(np.abs(waves), 98)
    if not np.isfinite(vlim) or vlim <= 0:
        vlim = 1.0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    im = ax.imshow(
        waves,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[x_us[0], x_us[-1], int(idx[0]), int(idx[-1])],
        cmap="seismic",
        vmin=-vlim,
        vmax=vlim,
    )
    gate_start = float(f.attrs.get("gate_us_start", x_us[0]))
    gate_end = float(f.attrs.get("gate_us_end", x_us[-1]))
    ax.axvline(gate_start, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(gate_end, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(f"{Path(f.filename).name} /{dataset_name} heatmap ({idx.size} rows)")
    ax.set_xlabel("time after sync (us)")
    ax.set_ylabel("dataset row index")
    fig.colorbar(im, ax=ax, label="voltage (V)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_features(f: h5py.File, dataset_name: str, out_path: Path,
                  max_points: int, start: int) -> bool:
    ts_name, tof_name, amp_name, eng_name = _feature_names(dataset_name)
    needed = [tof_name, amp_name, eng_name]
    if not all(name in f for name in needed):
        return False

    n = f[tof_name].shape[0]
    idx = _sample_indices(n, max_points, start)
    if idx.size == 0:
        return False

    if ts_name in f:
        x = f[ts_name][idx]
        x = x - float(x[0])
        x_label = "elapsed time (s)"
    else:
        x = idx
        x_label = "dataset row index"

    tof = f[tof_name][idx]
    amp = f[amp_name][idx]
    eng = f[eng_name][idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=140, sharex=True)
    for ax, y, label, color in [
        (axes[0], tof, "ToF (us)", "tab:red"),
        (axes[1], amp, "Amplitude (V)", "tab:green"),
        (axes[2], eng, "Energy", "tab:purple"),
    ]:
        ax.plot(x, y, color=color, linewidth=1.0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(x_label)
    fig.suptitle(f"{Path(f.filename).name} features from /{dataset_name}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HDF5 waveform datasets back into viewable plots."
    )
    parser.add_argument("path", help="Path to .h5 file")
    parser.add_argument(
        "--dataset",
        choices=["auto", "raw_waveforms", "waveforms"],
        default="auto",
        help="Dataset to plot. auto prefers /raw_waveforms, then /waveforms.",
    )
    parser.add_argument("--out-dir", default="", help="Output directory for PNG plots")
    parser.add_argument("--max-traces", type=int, default=40,
                        help="Number of traces to overlay")
    parser.add_argument("--max-heatmap-rows", type=int, default=1000,
                        help="Maximum rows to include in the heatmap")
    parser.add_argument("--max-feature-points", type=int, default=5000,
                        help="Maximum points to include in feature trend plots")
    parser.add_argument("--start", type=int, default=0,
                        help="Start plotting from this dataset row index")
    args = parser.parse_args()

    h5_path = Path(args.path)
    out_dir = Path(args.out_dir) if args.out_dir else h5_path.with_suffix("").with_name(
        f"{h5_path.stem}_plots"
    )
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        dataset_name = _choose_dataset(f, args.dataset)
        stem = h5_path.stem

        overlay_path = out_dir / f"{stem}_{dataset_name}_overlay.png"
        heatmap_path = out_dir / f"{stem}_{dataset_name}_heatmap.png"
        features_path = out_dir / f"{stem}_features.png"

        plot_overlay(f, dataset_name, overlay_path, args.max_traces, args.start)
        plot_heatmap(f, dataset_name, heatmap_path, args.max_heatmap_rows, args.start)
        made_features = plot_features(
            f, dataset_name, features_path, args.max_feature_points, args.start
        )

        print(f"Input:    {h5_path}")
        print(f"Dataset:  /{dataset_name} shape={f[dataset_name].shape}")
        print(f"Overlay:  {overlay_path}")
        print(f"Heatmap:  {heatmap_path}")
        if made_features:
            print(f"Features: {features_path}")
        else:
            print("Features: skipped (feature datasets not found)")


if __name__ == "__main__":
    main()
