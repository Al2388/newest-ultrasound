"""
Rebuild the C-scan amplitude map using peak-of-Hilbert-envelope from saved raw
waveforms in lines_raw/, then compare scan A (3.209 V) vs scan B (3.366 V) on
the new amplitude alongside the existing (correct) ToF and Energy maps.

We reuse the original gridding function from the scanner service so the new
amplitude grid lines up exactly with the saved ToF/Energy grids.
"""
import os
import glob
import json
import numpy as np

from ultrasound_battery.services.scanner import row_from_pulses_nosmooth

SESS_A = r"data/raw/cscan/cscan_scan_2026-05-14_14-46-55"   # 3.209 V
SESS_B = r"data/raw/cscan/cscan_scan_2026-05-14_22-22-19"   # 3.366 V
X_LO, X_HI = 12.0, 65.0
Y_LO, Y_HI = 10.0, 60.0


def envelope_peak(wf: np.ndarray) -> np.ndarray:
    """Peak of Hilbert envelope per pulse. wf shape (N, S) -> (N,)."""
    v = wf - wf.mean(axis=1, keepdims=True)
    n = v.shape[-1]
    X = np.fft.fft(v, axis=-1)
    H = np.zeros(n)
    if n % 2 == 0:
        H[0] = 1; H[n // 2] = 1; H[1:n // 2] = 2
    else:
        H[0] = 1; H[1:(n + 1) // 2] = 2
    env = np.abs(np.fft.ifft(X * H, axis=-1))
    return env.max(axis=1).astype(np.float32)


def rebuild_amp_grid(session: str, ncols: int, expected_cycles: int):
    """Rebuild a (nlines, ncols) envelope-peak amplitude grid for one session."""
    files = sorted(glob.glob(os.path.join(session, "lines_raw", "line_*.npz")))
    nlines = len(files)
    grid = np.full((nlines, ncols), np.nan, dtype=np.float32)
    for fp in files:
        # line index from filename
        i = int(os.path.basename(fp).replace("line_", "").replace(".npz", ""))
        z = np.load(fp)
        wf = z["waveforms"]
        ltr = bool(int(z["direction"]) == 0) if "direction" in z.keys() else (i % 2 == 0)
        amp_pulse = envelope_peak(wf)
        # match scanner_service: RTL lines are reversed so the image grid reads left-to-right
        if not ltr:
            amp_pulse = amp_pulse[::-1]
        grid[i] = row_from_pulses_nosmooth(amp_pulse, ncols, expected_cycles)
    return grid


def interior_stats(arr: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray):
    mx = (x_mm >= X_LO) & (x_mm <= X_HI)
    my = (y_mm >= Y_LO) & (y_mm <= Y_HI)
    sub = arr[np.ix_(my, mx)]
    return float(np.nanmean(sub)), float(np.nanstd(sub)), sub


def compare(name, A, B):
    a = A.ravel(); b = B.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
    da = b[ok] - a[ok]
    print(f"{name:<14}  A mean={np.nanmean(A):9.4f}  B mean={np.nanmean(B):9.4f}  "
          f"diff={np.nanmean(B)-np.nanmean(A):+9.4f} ({(np.nanmean(B)-np.nanmean(A))/abs(np.nanmean(A))*100:+6.2f}%)  "
          f"spatial r={corr:.4f}  pixel-diff std={np.std(da):.4f}")


def main():
    # Read scan metadata
    metaA = json.load(open(glob.glob(os.path.join(SESS_A, "scan_*_meta.json"))[0]))
    metaB = json.load(open(glob.glob(os.path.join(SESS_B, "scan_*_meta.json"))[0]))
    ncols = metaA["ncols"]
    # expected_cycles = detected_prf * theo_time = detected_prf * roi_w / speed
    prf = metaA["detected_prf_hz"]
    speed = metaA["speed_mm_s"]
    roi_w = metaA["roi_w_mm"]
    expected_cycles = int(prf * roi_w / speed)
    print(f"ncols={ncols}, expected_cycles={expected_cycles}")

    print("Rebuilding envelope-amplitude grid for A...")
    ampA_env = rebuild_amp_grid(SESS_A, ncols, expected_cycles)
    print("Rebuilding envelope-amplitude grid for B...")
    ampB_env = rebuild_amp_grid(SESS_B, ncols, expected_cycles)

    # Load the original saved feature maps for ToF, Energy, and the OLD (p2p_robust) amplitude
    A = np.load(glob.glob(os.path.join(SESS_A, "scan_*.npz"))[0])
    B = np.load(glob.glob(os.path.join(SESS_B, "scan_*.npz"))[0])
    x_mm, y_mm = A["x_mm"], A["y_mm"]
    mx = (x_mm >= X_LO) & (x_mm <= X_HI)
    my = (y_mm >= Y_LO) & (y_mm <= Y_HI)
    box = lambda M: M[np.ix_(my, mx)]

    print("\nInterior cell comparison (x=[12,65] mm, y=[10,60] mm):")
    print("-" * 100)
    compare("ToF",          box(A["tof"]),       box(B["tof"]))
    compare("Energy",       box(A["energy"]),    box(B["energy"]))
    compare("Amp(p2p_rob)", box(A["amplitude"]), box(B["amplitude"]))
    compare("Amp(env)",     box(ampA_env),       box(ampB_env))

    # Save rebuilt amp maps for the next step (PNG render)
    out_dir = "analysis/rebuilt"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "amp_envelope.npz"),
                        ampA_env=ampA_env, ampB_env=ampB_env, x_mm=x_mm, y_mm=y_mm)
    print(f"\nSaved rebuilt amplitude maps to {out_dir}/amp_envelope.npz")


if __name__ == "__main__":
    main()
