"""
Quick verification utility for gauging / A-scan HDF5 files.

Usage
-----
  python inspect_h5.py                          (auto-finds latest gauging .h5)
  python inspect_h5.py path/to/file.h5          (inspect a specific file)

Prints session attributes, dataset shapes, and the first/last 5 values of
each scalar feature so you can confirm real data was captured.
"""
import glob
import os
import sys

import h5py


def _fmt_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def _row_summary(f: h5py.File, i: int, count_name: str, count_label: str) -> str:
    tof_name = "tof_us_absolute" if "tof_us_absolute" in f else "tof_us"
    tof_label = "tof_abs" if tof_name == "tof_us_absolute" else "tof"
    parts = [
        f"row {i:<3}",
        f"t={f['timestamps'][i]:.3f}",
        f"tof={f['tof_us'][i]:.4f} µs",
        f"amp={f['amplitude'][i]:.4f} V",
        f"eng={f['energy'][i]:.4f}",
        f"{count_label}={f[count_name][i]}",
    ]
    if "tof_us_absolute" in f:
        parts.append(f"tof_abs={float(f['tof_us_absolute'][i]):.4f} us")
    if "tof_us_envelope" in f and "tof_us_absolute" in f:
        gate_start = float(f.attrs.get("gate_us_start", 0.0))
        parts.append(f"env_abs={gate_start + float(f['tof_us_envelope'][i]):.4f} us")
    if "tracking_corr" in f:
        corr = float(f["tracking_corr"][i])
        if corr == corr:
            parts.append(f"xcorr={corr:.3f}")
    # Schema 2.0+ quality metadata, when present
    if "n_rejected" in f:
        parts.append(f"rej={int(f['n_rejected'][i])}")
    if "prf_actual" in f:
        parts.append(f"prf={float(f['prf_actual'][i]):.0f}Hz")
    return "  ".join(parts)


def inspect(path: str):
    print(f"\n=== {path} ===")
    print(f"size on disk: {_fmt_size(os.path.getsize(path))}\n")

    with h5py.File(path, "r") as f:
        # Highlight schema/provenance first so older v1.x files are obvious
        schema = f.attrs.get("schema_version", "1.x (legacy)")
        print(f"schema_version: {schema}")

        print("\n--- attributes ---")
        for k, v in f.attrs.items():
            print(f"  {k:<20} {v}")

        print("\n--- datasets ---")
        for name in f.keys():
            ds = f[name]
            print(f"  /{name:<14} shape={ds.shape}  dtype={ds.dtype}  "
                  f"chunks={ds.chunks}  compression={ds.compression}")

        print("\n--- first/last 3 rows of scalar features ---")
        n = f["timestamps"].shape[0]
        if n == 0:
            print("  (file is empty — no rows written)")
            return

        count_name  = "n_pulses" if "n_pulses" in f else "n_averaged"
        count_label = "n_p" if count_name == "n_pulses" else "n_avg"

        for i in range(min(3, n)):
            print("  " + _row_summary(f, i, count_name, count_label))
        if n > 6:
            print("  ...")
        for i in range(max(3, n - 3), n):
            print("  " + _row_summary(f, i, count_name, count_label))

        print("\n--- waveform sanity check ---")
        wf = f["waveforms"]
        print(f"  rows × samples  = {wf.shape[0]} × {wf.shape[1]}")
        print(f"  total samples   = {wf.size:,}  (~{wf.size*4/1024:.1f} KB raw, "
              f"{os.path.getsize(path)/wf.size:.2f} bytes/sample after gzip)")
        print(f"  first row: min={wf[0].min():.4f}V  "
              f"max={wf[0].max():.4f}V  "
              f"std={wf[0].std():.4f}V")
        print(f"  last row:  min={wf[-1].min():.4f}V  "
              f"max={wf[-1].max():.4f}V  "
              f"std={wf[-1].std():.4f}V")

        # Schema 2.0+: per-snapshot std waveform — quick read on whether
        # pulse-to-pulse noise is comparable to the echo amplitude.
        if "waveforms_std" in f and f["waveforms_std"].shape[0] > 0:
            sw = f["waveforms_std"]
            print("\n--- pulse-to-pulse std (schema 2.0+) ---")
            print(f"  first row mean(std) = {float(sw[0].mean()):.5f} V")
            print(f"  last  row mean(std) = {float(sw[-1].mean()):.5f} V")

        if "raw_waveforms" in f:
            raw = f["raw_waveforms"]
            print("\n--- raw pulse archive ---")
            print(f"  raw pulses × samples = {raw.shape[0]} × {raw.shape[1]}")
            if "raw_snapshot_index" in f and raw.shape[0] > 0:
                print(f"  first snapshot idx   = {f['raw_snapshot_index'][0]}")
                print(f"  last snapshot idx    = {f['raw_snapshot_index'][-1]}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob("data/gauging/gauge_*.h5") +
                       glob.glob("data/ascan/ascan_*.h5"))
        if not files:
            print("No gauging or A-scan .h5 files found in data/")
            sys.exit(1)
        path = files[-1]
        print(f"(auto-picked latest: {path})")

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    inspect(path)


if __name__ == "__main__":
    main()
