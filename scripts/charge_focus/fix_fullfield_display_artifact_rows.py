"""Patch full-field display artifact rows in the 35C charge-focus cache.

These rows are outside the analysis ROI, so earlier ROI-only dropout checks did
not catch them. They are visible in full-field GIF/PDF renders, and they occur
simultaneously in tof, amplitude, and energy, which is consistent with a
single acquisition/display row artifact rather than real modality-specific
structure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJ / "reports" / "longrun_cycling_35c_charge_focus" / "_cache"
STACK = CACHE_DIR / "stack.npz"
BACKUP = CACHE_DIR / "stack_pre_fullfield_display_artifact_patch.npz"
LOG = CACHE_DIR.parent / "fullfield_display_artifact_patch_log.md"

MODS = ("tof", "amplitude", "energy")
ARTIFACT_ROWS = [
    (27, 23),
    (33, 119),
    (74, 128),
]


def _replacement_rows(row_idx: int, n_rows: int) -> list[int]:
    rows: list[int] = []
    for delta in (-1, 1, -2, 2, -3, 3):
        rr = row_idx + delta
        if 0 <= rr < n_rows and rr not in rows:
            rows.append(rr)
        if len(rows) >= 4:
            break
    return rows


def main() -> None:
    print(f"loading {STACK}")
    loaded = np.load(STACK)
    keys = list(loaded.keys())
    data = {k: loaded[k].copy() for k in keys}

    if not BACKUP.exists():
        np.savez_compressed(BACKUP, **{k: loaded[k] for k in keys})
        print(f"saved backup -> {BACKUP}")

    n_rows = data["amplitude"].shape[1]
    log_lines = [
        "# Full-field display artifact row patch\n\n",
        "Patched rows that were visible in full-field plots but outside the analysis ROI.\n\n",
        f"Backup: `_cache/{BACKUP.name}`.\n\n",
        "| scan_idx | row_idx | Y (mm) | replacement rows |\n",
        "|---:|---:|---:|---|\n",
    ]

    for scan_idx, row_idx in ARTIFACT_ROWS:
        replacement = _replacement_rows(row_idx, n_rows)
        print(f"patching scan {scan_idx}, row {row_idx} using rows {replacement}")
        for mod in MODS:
            arr = data[mod]
            arr[scan_idx, row_idx, :] = np.nanmedian(arr[scan_idx, replacement, :], axis=0)
        y_mm = float(data["y_mm"][row_idx])
        log_lines.append(
            f"| {scan_idx} | {row_idx} | {y_mm:.2f} | "
            f"{', '.join(str(r) for r in replacement)} |\n"
        )

    np.savez_compressed(STACK, **data)
    LOG.write_text("".join(log_lines), encoding="utf-8")
    print(f"wrote patched stack -> {STACK}")
    print(f"wrote log -> {LOG}")


if __name__ == "__main__":
    main()
