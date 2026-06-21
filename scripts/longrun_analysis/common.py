"""Shared loader for the 22h longrun cycling analyses.

Loads all 147 scans, applies the canonical 50x40 mm ROI mask, merges with
the cycler-tagged metadata, and caches as a single NPZ for fast re-use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "reports" / "experiments" / "longrun_cycling_3p238start_2026-05-29_18-54-23"
CHECKPOINT = BATCH_DIR / "checkpoint.json"
TAGGED_CSV = BATCH_DIR / "overview" / "scans_tagged.csv"
ROI_MASK_PATH = ROOT / "reports" / "experiments" / "roi_clean_figs_2026-05-29_18-29-16_50x40_grayamp" / "roi_mask.npy"

OUT_ROOT = ROOT / "reports" / "longrun_cycling_22h_analysis"
CACHE_PATH = OUT_ROOT / "_cache" / "stack.npz"
META_PATH = OUT_ROOT / "_cache" / "meta.csv"

NOISE_FLOOR = {
    "amp_sigma_roi_mv": 2.22,
    "tof_sigma_roi_ns": 4.70,
    "energy_sigma_roi": 0.0814,
    "amp_sigma_pixp95_mv": 23.7,
    "tof_sigma_pixp95_ns": 15.9,
    "energy_sigma_pixp95": 0.738,
}


@dataclass
class Stack:
    amplitude: np.ndarray
    tof: np.ndarray
    energy: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    roi_mask: np.ndarray
    meta: pd.DataFrame

    @property
    def n_frames(self) -> int:
        return self.amplitude.shape[0]

    def roi_mean(self, modality: str) -> np.ndarray:
        arr = getattr(self, modality)
        mask = self.roi_mask
        flat = arr.reshape(arr.shape[0], -1)
        m = mask.reshape(-1)
        return np.nanmean(flat[:, m], axis=1)


def _load_checkpoint_index() -> pd.DataFrame:
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {
                "run_idx": c["run_idx"],
                "scan_id": c["scan_id"],
                "session_dir": c["session_dir"].replace("\\", "/"),
            }
            for c in cp["completed"]
        ]
    )


def _build_cache() -> Stack:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cp_df = _load_checkpoint_index()
    tagged = pd.read_csv(TAGGED_CSV)
    merged = tagged.merge(cp_df, on="scan_id", how="inner").sort_values("run_idx").reset_index(drop=True)

    roi_mask = np.load(ROI_MASK_PATH)

    amp_list, tof_list, eng_list = [], [], []
    x_mm = y_mm = None
    for _, row in merged.iterrows():
        sess_dir = ROOT / row["session_dir"]
        npz = next(sess_dir.glob("scan_*.npz"))
        d = np.load(npz)
        amp_list.append(d["amplitude"].astype(np.float32))
        tof_list.append(d["tof"].astype(np.float32))
        eng_list.append(d["energy"].astype(np.float32))
        if x_mm is None:
            x_mm = d["x_mm"].astype(np.float32)
            y_mm = d["y_mm"].astype(np.float32)

    stack = Stack(
        amplitude=np.stack(amp_list, axis=0),
        tof=np.stack(tof_list, axis=0),
        energy=np.stack(eng_list, axis=0),
        x_mm=x_mm,
        y_mm=y_mm,
        roi_mask=roi_mask,
        meta=merged,
    )

    np.savez_compressed(
        CACHE_PATH,
        amplitude=stack.amplitude,
        tof=stack.tof,
        energy=stack.energy,
        x_mm=stack.x_mm,
        y_mm=stack.y_mm,
        roi_mask=stack.roi_mask,
    )
    merged.to_csv(META_PATH, index=False)
    return stack


def load(rebuild: bool = False) -> Stack:
    if not rebuild and CACHE_PATH.exists() and META_PATH.exists():
        d = np.load(CACHE_PATH)
        meta = pd.read_csv(META_PATH)
        return Stack(
            amplitude=d["amplitude"],
            tof=d["tof"],
            energy=d["energy"],
            x_mm=d["x_mm"],
            y_mm=d["y_mm"],
            roi_mask=d["roi_mask"],
            meta=meta,
        )
    return _build_cache()


def roi_bounds_mm(stack: Stack) -> tuple[float, float, float, float]:
    rows = np.where(stack.roi_mask.any(axis=1))[0]
    cols = np.where(stack.roi_mask.any(axis=0))[0]
    return (
        float(stack.x_mm[cols.min()]),
        float(stack.x_mm[cols.max()]),
        float(stack.y_mm[rows.min()]),
        float(stack.y_mm[rows.max()]),
    )


def rest_segments(meta: pd.DataFrame) -> list[pd.DataFrame]:
    is_rest = (meta.step_tag == "rest").values
    breaks = np.flatnonzero(np.diff(is_rest.astype(int)) != 0) + 1
    segs = np.split(np.arange(len(meta)), breaks)
    return [meta.iloc[s].reset_index(drop=True) for s in segs if is_rest[s[0]]]


if __name__ == "__main__":
    s = load(rebuild=True)
    print(f"Built cache: amp={s.amplitude.shape} tof={s.tof.shape} energy={s.energy.shape}")
    print(f"x_mm range: {s.x_mm.min():.2f} .. {s.x_mm.max():.2f}, y_mm: {s.y_mm.min():.2f} .. {s.y_mm.max():.2f}")
    print(f"ROI true: {int(s.roi_mask.sum())} pixels")
    print(f"ROI bounds: {roi_bounds_mm(s)}")
    print(f"meta rows: {len(s.meta)}, columns: {list(s.meta.columns)}")
