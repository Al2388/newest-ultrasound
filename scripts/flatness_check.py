"""
Bed-flatness gauging at the 4 corners of a cell.

Workflow
--------
Manual probe placement. The operator moves the transducer to each of the
4 corners of the cell in turn (TL, TR, BL, BR), presses Enter, and the
script acquires ~1 s of pulses, coherently averages them, and reports the
envelope-peak ToF at that corner. After all 4 are captured it prints the
flatness summary (per-corner ToF, peak-to-peak spread, per-axis tilt, the
equivalent corner heights in mm, and which corner to raise). Loops so the
bed can be shimmed and re-measured without restarting.

No printer is used — this is a setup utility to be run before a C-scan.

Notes
-----
- The "height" conversion uses round-trip ToF: dh = c * dt / 2. Default
  c = 1480 m/s (water); pass --sound-speed to override. The conversion is
  only used for the human-readable mm column — the tilt decision uses ToF
  directly and is sound-speed-independent.
- We use envelope-peak ToF (parabolic-refined argmax) at each corner, not
  NCC-vs-reference, because the packet shape can differ between corners
  (coupling variation) and NCC against a corner-A reference would alias
  shape change into spurious lag at corners B/C/D.
"""

import argparse
import sys
import time

import numpy as np

from ultrasound_battery.hardware.hs5 import HS5StreamPeaks, envelope_hilbert


CORNERS = [
    ("TL", "top-left"),
    ("TR", "top-right"),
    ("BL", "bottom-left"),
    ("BR", "bottom-right"),
]


def _parabolic_peak(y: np.ndarray, k: int) -> float:
    n = len(y)
    if k <= 0 or k >= n - 1:
        return float(k)
    denom = float(y[k - 1] - 2 * y[k] + y[k + 1])
    if abs(denom) < 1e-12:
        return float(k)
    return float(k) + 0.5 * float(y[k - 1] - y[k + 1]) / denom


def measure_corner(hs: HS5StreamPeaks, dwell_s: float) -> dict:
    """Acquire dwell_s of pulses, return ToF (us, sub-sample), amp, n_pulses."""
    _, aa, _, _, wf = hs.acquire_peaks(
        duration_s=dwell_s, save_waveforms=True, save_full_waveforms=False
    )
    if wf is None or wf.shape[0] == 0:
        return {"tof_us": None, "amp_v": None, "n_pulses": 0}

    avg_wf = np.mean(wf, axis=0).astype(np.float64)
    avg_wf -= float(avg_wf.mean())
    env = envelope_hilbert(avg_wf)
    k0 = int(np.argmax(env))
    k_ref = _parabolic_peak(env, k0)

    dt_us = 1e6 / float(hs.fs)
    gate_start_us = float(hs.g0) / float(hs.fs) * 1e6
    tof_us = gate_start_us + k_ref * dt_us

    return {
        "tof_us": float(tof_us),
        "amp_v": float(np.median(aa)),
        "n_pulses": int(wf.shape[0]),
    }


def prompt_corner(label: str, descr: str) -> bool:
    """Return True to capture, False to skip this corner."""
    while True:
        s = input(f"  [{label}]  Place probe at {descr:>12s} corner, "
                  f"press Enter to capture (s=skip, q=quit): ").strip().lower()
        if s == "" :
            return True
        if s == "s":
            return False
        if s == "q":
            raise KeyboardInterrupt
        print("    (Enter / s / q)")


def fit_plane(xs, ys, zs):
    """Least-squares plane z = a*x + b*y + c through N>=3 points."""
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
    a, b, c = coeffs
    return float(a), float(b), float(c)


def print_report(results: dict, width_mm: float, height_mm: float,
                 sound_speed_m_s: float) -> None:
    # Corner coordinates in mm — TL is origin.
    corner_xy = {
        "TL": (0.0,        0.0),
        "TR": (width_mm,   0.0),
        "BL": (0.0,        height_mm),
        "BR": (width_mm,   height_mm),
    }

    captured = [(lab, results[lab]["tof_us"]) for lab, _ in CORNERS
                if results.get(lab) and results[lab]["tof_us"] is not None]
    if len(captured) < 3:
        print(f"\n  Need at least 3 captured corners for tilt analysis. "
              f"Got {len(captured)}.\n")
        return

    print("\n  " + "=" * 60)
    print("  FLATNESS REPORT")
    print("  " + "=" * 60)
    print(f"  {'Corner':<8s} {'X (mm)':>8s} {'Y (mm)':>8s} "
          f"{'ToF (us)':>10s} {'Amp (V)':>9s} {'N pulses':>9s}")
    for lab, descr in CORNERS:
        r = results.get(lab)
        if not r or r["tof_us"] is None:
            print(f"  {lab:<8s} {'-':>8s} {'-':>8s} {'(skipped)':>10s}")
            continue
        x, y = corner_xy[lab]
        print(f"  {lab:<8s} {x:>8.1f} {y:>8.1f} "
              f"{r['tof_us']:>10.3f} {r['amp_v']:>9.3f} {r['n_pulses']:>9d}")

    tofs = np.array([t for _, t in captured])
    spread_us = float(tofs.max() - tofs.min())
    spread_mm = sound_speed_m_s * spread_us * 1e-6 / 2.0 * 1e3   # round-trip

    print("\n  Peak-to-peak ToF spread : "
          f"{spread_us:7.3f} us  ({spread_mm*1e3:7.1f} um equivalent "
          f"@ c={sound_speed_m_s:.0f} m/s)")

    xs = np.array([corner_xy[lab][0] for lab, _ in captured])
    ys = np.array([corner_xy[lab][1] for lab, _ in captured])
    a, b, c = fit_plane(xs, ys, tofs)

    span_x_us = a * width_mm
    span_y_us = b * height_mm
    span_x_um = sound_speed_m_s * span_x_us * 1e-6 / 2.0 * 1e6
    span_y_um = sound_speed_m_s * span_y_us * 1e-6 / 2.0 * 1e6

    print(f"  Tilt across X span      : {span_x_us:+7.3f} us "
          f"({span_x_um:+7.1f} um)  "
          f"slope = {a*1e3:+7.3f} ns/mm")
    print(f"  Tilt across Y span      : {span_y_us:+7.3f} us "
          f"({span_y_um:+7.1f} um)  "
          f"slope = {b*1e3:+7.3f} ns/mm")

    # Heights relative to the lowest (largest-ToF) corner.
    # Larger ToF = echo took longer = surface is further from transducer
    # = bed at that corner sits LOWER than the others.
    print("\n  Corner heights (relative to lowest corner):")
    tof_max = tofs.max()
    for lab, descr in CORNERS:
        r = results.get(lab)
        if not r or r["tof_us"] is None:
            continue
        dh_mm = sound_speed_m_s * (tof_max - r["tof_us"]) * 1e-6 / 2.0 * 1e3
        marker = "  <-- LOWEST (raise this corner)" if r["tof_us"] == tof_max else ""
        print(f"    {lab}: {dh_mm:+7.3f} mm above lowest{marker}")

    print("  " + "=" * 60)
    print("  Goal: minimise peak-to-peak spread. Shim the LOWEST corner up "
          "in small steps,\n        re-measure, repeat.\n")


def main():
    ap = argparse.ArgumentParser(
        description="Manual 4-corner bed-flatness gauging "
                    "(no printer — operator places probe at each corner)."
    )
    ap.add_argument("--width", type=float, default=70.0,
                    help="Cell width in mm (X span between L and R corners). "
                         "Used for tilt slope only. Default 70.")
    ap.add_argument("--height", type=float, default=60.0,
                    help="Cell height in mm (Y span between T and B corners). "
                         "Used for tilt slope only. Default 60.")
    ap.add_argument("--dwell", type=float, default=1.0,
                    help="Seconds of pulses to acquire per corner. Default 1.0.")
    ap.add_argument("--fs-hz", type=float, default=20e6,
                    help="HS5 sample rate. Default 20 MHz.")
    ap.add_argument("--gate-us-start", type=float, default=25.0,
                    help="Gate window start (us from sync edge). Default 25.")
    ap.add_argument("--gate-us-end", type=float, default=50.0,
                    help="Gate window end (us from sync edge). Default 50.")
    ap.add_argument("--ch1-range", type=float, default=2.0,
                    help="HS5 CH1 voltage range (V). Default 2.0.")
    ap.add_argument("--sound-speed", type=float, default=1480.0,
                    help="Sound speed (m/s) used only for human-readable mm "
                         "conversions. Default 1480 (water).")
    args = ap.parse_args()

    print(f"\n  Opening HS5 (fs={args.fs_hz/1e6:.0f} MHz, "
          f"gate=[{args.gate_us_start:.1f}, {args.gate_us_end:.1f}] us, "
          f"range=+/-{args.ch1_range:.1f} V) ...")

    hs = HS5StreamPeaks(
        fs_hz=args.fs_hz,
        gate_us=(args.gate_us_start, args.gate_us_end),
        ch1_range=args.ch1_range,
        feature_mode="envelope",
    ).open()
    try:
        print("  Calibrating sync ...")
        hs.calibrate_sync(seconds=1.0, verbose=True)

        run = 1
        while True:
            print(f"\n  ===== Run #{run}: {args.width:.0f} mm x "
                  f"{args.height:.0f} mm cell =====")
            results = {}
            try:
                for lab, descr in CORNERS:
                    if not prompt_corner(lab, descr):
                        results[lab] = {"tof_us": None, "amp_v": None,
                                        "n_pulses": 0}
                        continue
                    print(f"    Acquiring {args.dwell:.1f} s ...", end="",
                          flush=True)
                    t0 = time.perf_counter()
                    r = measure_corner(hs, args.dwell)
                    dt = time.perf_counter() - t0
                    results[lab] = r
                    if r["tof_us"] is None:
                        print(f"  no pulses received in {dt:.1f} s "
                              "(check coupling / pulser)")
                    else:
                        print(f"  ToF = {r['tof_us']:.3f} us  "
                              f"amp = {r['amp_v']:.3f} V  "
                              f"({r['n_pulses']} pulses in {dt:.2f} s)")
            except KeyboardInterrupt:
                print("\n  Aborted.")
                break

            print_report(results, args.width, args.height, args.sound_speed)

            s = input("  Press Enter to re-measure, q to quit: ").strip().lower()
            if s == "q":
                break
            run += 1
    finally:
        hs.close()
        print("  HS5 closed.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(130)
