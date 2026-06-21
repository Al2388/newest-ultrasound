#!/usr/bin/env python
"""Re-render a C-scan from per-line pulse timestamps.

This is a diagnostic renderer. It leaves the original scan files untouched and
adds *_time.png / scan_time_mapping.npz outputs in the scan session folder.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURES = {
    "amplitude": ("scan_amp_{suffix}.png", "Amplitude, {label}"),
    "tof_us": ("scan_tof_{suffix}.png", "ToF, {label}"),
    "energy": ("scan_eng_{suffix}.png", "Energy, {label}"),
}


def _latest_completed(root: Path) -> Path:
    sessions = sorted(
        [p for p in root.glob("cscan_scan_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for session in sessions:
        if list(session.glob("scan_*.npz")) and list(session.glob("scan_*_meta.json")):
            return session
    raise FileNotFoundError(f"No completed C-scan session found under {root}")


def _load_meta(session: Path) -> dict:
    metas = sorted(session.glob("scan_*_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if metas:
        return json.loads(metas[0].read_text())
    manifest = session / "session_manifest.json"
    return json.loads(manifest.read_text()) if manifest.exists() else {}


def _line_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _motion_profile(distance_mm: float, speed_mm_s: float,
                    accel_mm_s2: float) -> tuple[float, float, float, float, float]:
    distance = float(abs(distance_mm))
    speed = float(abs(speed_mm_s))
    accel = float(abs(accel_mm_s2))
    if distance <= 0 or speed <= 0:
        return 0.0, 0.0, 0.0, 0.0, distance
    if accel <= 0:
        total = distance / speed
        return 0.0, total, 0.0, total, distance

    t_acc = speed / accel
    d_acc = 0.5 * accel * t_acc * t_acc
    if 2.0 * d_acc >= distance:
        t_acc = math.sqrt(distance / accel)
        t_cruise = 0.0
        v_peak = accel * t_acc
        total = 2.0 * t_acc
    else:
        t_cruise = (distance - 2.0 * d_acc) / speed
        v_peak = speed
        total = 2.0 * t_acc + t_cruise
    return t_acc, t_cruise, v_peak, total, distance


def _accel_fraction(
    timestamps: np.ndarray,
    t_start: float,
    t_end: float,
    distance_mm: float,
    speed_mm_s: float,
    accel_mm_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_acc, t_cruise, v_peak, total, distance = _motion_profile(
        distance_mm, speed_mm_s, accel_mm_s2
    )
    frac = np.full(timestamps.shape, np.nan, dtype=np.float64)
    if total <= 0 or distance <= 0:
        return frac, np.zeros(timestamps.shape, dtype=bool)

    capture_duration = max(0.0, float(t_end - t_start))
    motion_start = float(t_start) + max(0.0, 0.5 * (capture_duration - total))
    t_rel = timestamps - motion_start
    valid = (t_rel >= 0.0) & (t_rel <= total)
    if not np.any(valid):
        return frac, valid

    t = t_rel[valid]
    s = np.empty_like(t)
    if t_acc <= 0.0:
        s[:] = v_peak * t
    else:
        accel = v_peak / t_acc
        d_acc = 0.5 * accel * t_acc * t_acc
        m_acc = t <= t_acc
        m_cruise = (t > t_acc) & (t <= t_acc + t_cruise)
        m_dec = t > t_acc + t_cruise
        s[m_acc] = 0.5 * accel * t[m_acc] ** 2
        s[m_cruise] = d_acc + v_peak * (t[m_cruise] - t_acc)
        t_left = total - t[m_dec]
        s[m_dec] = distance - 0.5 * accel * t_left ** 2

    frac[valid] = np.clip(s / distance, 0.0, 1.0)
    return frac, valid


def _grid_by_time(
    values: np.ndarray,
    timestamps: np.ndarray,
    t_start: float,
    t_end: float,
    ncols: int,
    reverse: bool,
    model: str,
    distance_mm: float,
    speed_mm_s: float,
    accel_mm_s2: float,
) -> np.ndarray:
    out = np.full(ncols, np.nan, dtype=np.float32)
    if ncols <= 0 or values.size == 0 or not np.isfinite(t_start) or not np.isfinite(t_end):
        return out
    duration = float(t_end - t_start)
    if duration <= 0:
        return out

    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if model == "accel":
        frac_all, motion_valid = _accel_fraction(
            timestamps, t_start, t_end, distance_mm, speed_mm_s, accel_mm_s2
        )
    else:
        frac_all = (timestamps - t_start) / duration
        motion_valid = (timestamps >= t_start) & (timestamps <= t_end)

    valid = np.isfinite(values) & np.isfinite(timestamps) & motion_valid & np.isfinite(frac_all)
    if not np.any(valid):
        return out

    frac = frac_all[valid]
    frac = np.clip(frac, 0.0, 1.0)
    if reverse:
        frac = 1.0 - frac

    x = frac * (ncols - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, ncols - 1)
    w1 = x - i0
    w0 = 1.0 - w1

    acc = np.zeros(ncols, dtype=np.float64)
    sumw = np.zeros(ncols, dtype=np.float64)
    v = values[valid]
    np.add.at(acc, i0, w0 * v)
    np.add.at(sumw, i0, w0)
    np.add.at(acc, i1, w1 * v)
    np.add.at(sumw, i1, w1)

    filled = sumw > 1e-12
    out[filled] = (acc[filled] / sumw[filled]).astype(np.float32)

    # Time binning can leave tiny gaps near the ends; interpolate only inside
    # the span with real samples so missing columns remain visible if severe.
    good = np.flatnonzero(np.isfinite(out))
    if good.size >= 2:
        lo, hi = int(good[0]), int(good[-1])
        gaps = np.arange(lo, hi + 1)
        bad = ~np.isfinite(out[gaps])
        if np.any(bad):
            out[gaps[bad]] = np.interp(gaps[bad], good, out[good]).astype(np.float32)
    return out


def _save_image(arr: np.ndarray, out_path: Path, title: str, cmap: str, xlabel: str) -> None:
    finite = np.isfinite(arr)
    if not np.any(finite):
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.nanpercentile(arr, [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    plt.figure(figsize=(7.5, 5.0), dpi=150)
    plt.imshow(arr, cmap=cmap, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Scan line")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _save_compare(original: np.ndarray, remapped: np.ndarray, out_path: Path,
                  title: str, cmap: str, remapped_label: str) -> None:
    both = np.concatenate([original[np.isfinite(original)], remapped[np.isfinite(remapped)]])
    if both.size:
        vmin, vmax = np.percentile(both, [1, 99])
    else:
        vmin, vmax = 0.0, 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharey=True)
    for ax, arr, label in zip(axes, [original, remapped], ["original", remapped_label]):
        im = ax.imshow(arr, cmap=cmap, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("X column")
    axes[0].set_ylabel("Scan line")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render(session: Path, ncols: int | None = None, cmap: str = "turbo",
           model: str = "accel", accel_mm_s2: float | None = None) -> dict:
    meta = _load_meta(session)
    line_files = sorted((session / "lines_raw").glob("line_*.npz"), key=_line_index)
    if not line_files:
        raise FileNotFoundError(f"No line_*.npz files found in {session / 'lines_raw'}")

    max_line = max(_line_index(p) for p in line_files)
    nlines = int(meta.get("nlines") or max_line + 1)
    if max_line + 1 > nlines:
        nlines = max_line + 1
    ncols = int(ncols or meta.get("ncols") or 500)
    roi_w = float(meta.get("roi_w_mm") or meta.get("config", {}).get("roi_w") or 0.0)
    speed = float(meta.get("speed_mm_s") or meta.get("config", {}).get("speed") or 0.0)
    accel = float(accel_mm_s2 if accel_mm_s2 is not None else meta.get("accel_mm_s2", 500.0))
    suffix = "accel" if model == "accel" else "time"
    label = "acceleration-mapped" if model == "accel" else "time-mapped"

    images = {name: np.full((nlines, ncols), np.nan, dtype=np.float32) for name in FEATURES}
    samples_per_line = np.zeros(nlines, dtype=np.int32)
    line_duration_s = np.full(nlines, np.nan, dtype=np.float64)

    for path in line_files:
        i = _line_index(path)
        with np.load(path) as z:
            direction = int(np.asarray(z["direction"]).item()) if "direction" in z.files else (i % 2)
            reverse = direction == 1
            timestamps = np.asarray(z["timestamps"], dtype=np.float64)
            t_start = float(np.asarray(z["line_perf_start_s"]).item())
            t_end = float(np.asarray(z["line_perf_end_s"]).item())
            line_duration_s[i] = t_end - t_start
            valid_t = (timestamps >= t_start) & (timestamps <= t_end)
            samples_per_line[i] = int(np.count_nonzero(valid_t))
            for feature in FEATURES:
                images[feature][i] = _grid_by_time(
                    z[feature], timestamps, t_start, t_end, ncols, reverse,
                    model, roi_w, speed, accel,
                )

    out_npz = session / f"scan_{suffix}_mapping.npz"
    np.savez_compressed(
        out_npz,
        amplitude=images["amplitude"],
        tof=images["tof_us"],
        energy=images["energy"],
        samples_per_line=samples_per_line,
        line_duration_s=line_duration_s,
        model=np.array(suffix),
        speed_mm_s=np.float64(speed),
        accel_mm_s2=np.float64(accel),
    )

    outputs = {"npz": str(out_npz), "images": {}}
    for feature, (filename, title) in FEATURES.items():
        out_png = session / filename.format(suffix=suffix)
        _save_image(
            images[feature], out_png, title.format(label=label), cmap,
            f"X column, {label}",
        )
        outputs["images"][feature] = str(out_png)

    originals = sorted(
        [
            p for p in session.glob("scan_*.npz")
            if p.name not in {"scan_time_mapping.npz", "scan_accel_mapping.npz"}
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if originals:
        with np.load(originals[0]) as z:
            compare_map = {
                "amplitude": "amplitude",
                "tof_us": "tof",
                "energy": "energy",
            }
            for feature, original_key in compare_map.items():
                if original_key not in z.files:
                    continue
                original = z[original_key]
                if original.shape != images[feature].shape:
                    continue
                compare_path = session / FEATURES[feature][0].format(suffix=f"original_vs_{suffix}")
                _save_compare(
                    original, images[feature], compare_path,
                    FEATURES[feature][1].format(label=label), cmap, label,
                )
                outputs["images"][feature + "_compare"] = str(compare_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", help="C-scan session folder, or omit for latest completed.")
    parser.add_argument("--root", default="data/raw/cscan", help="Root folder containing cscan_scan_* sessions.")
    parser.add_argument("--ncols", type=int, default=None, help="Override output column count.")
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--model", choices=["accel", "time"], default="accel")
    parser.add_argument("--accel-mm-s2", type=float, default=None,
                        help="Acceleration for model=accel. Defaults to scan meta or 500.")
    args = parser.parse_args()

    session = Path(args.session) if args.session else _latest_completed(Path(args.root))
    outputs = render(
        session, ncols=args.ncols, cmap=args.cmap,
        model=args.model, accel_mm_s2=args.accel_mm_s2,
    )
    print(json.dumps({"session": str(session), **outputs}, indent=2))


if __name__ == "__main__":
    main()
