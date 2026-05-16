"""
Compare amplitude metrics between two C-scans to diagnose the +30% amp anomaly.

We re-derive amplitude five different ways from the raw gated waveforms, aggregate
to the cell interior, and compare scan A (3.209 V, low SOC) vs scan B (3.366 V, high SOC).
Energy and ToF stayed coherent between the two scans (r=0.98). Only amplitude jumped.
This script finds out whether the metric is the culprit.
"""
import glob
import os
import sys
import numpy as np

SESS_A = r"data/raw/cscan/cscan_scan_2026-05-14_14-46-55"   # 3.209 V
SESS_B = r"data/raw/cscan/cscan_scan_2026-05-14_22-22-19"   # 3.366 V

# Cell-interior bounds (mm) — same as earlier analysis
X_LO, X_HI = 12.0, 65.0
Y_LO, Y_HI = 10.0, 60.0

# Subsample for speed: every Nth pulse, every Mth line
PULSE_STRIDE = 20
LINE_STRIDE  = 5


def envelope(x: np.ndarray) -> np.ndarray:
    """FFT-based Hilbert envelope along last axis."""
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    H = np.zeros(n)
    if n % 2 == 0:
        H[0] = 1; H[n // 2] = 1; H[1:n // 2] = 2
    else:
        H[0] = 1; H[1:(n + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * H, axis=-1))


def per_pulse_metrics(wf: np.ndarray):
    """wf shape (N, S). Returns dict of N-length arrays per metric."""
    v = wf - wf.mean(axis=1, keepdims=True)
    env = envelope(v)
    return {
        "env_peak":  env.max(axis=1),                 # current method
        "env_rms":   np.sqrt((env ** 2).mean(axis=1)),# RMS of envelope across gate
        "env_int":   env.sum(axis=1),                 # integrated envelope (area)
        "p2p_raw":   v.max(axis=1) - v.min(axis=1),   # naive p2p
        "abs_max":   np.abs(v).max(axis=1),           # max |sample|
        "env_p95":   np.percentile(env, 95, axis=1),  # robust envelope peak
    }


def collect(session_dir: str):
    files = sorted(glob.glob(os.path.join(session_dir, "lines_raw", "line_*.npz")))
    files = files[::LINE_STRIDE]
    agg = {k: [] for k in ["env_peak", "env_rms", "env_int", "p2p_raw", "abs_max", "env_p95"]}
    n_pulses_used = 0
    for fp in files:
        z = np.load(fp)
        y_mm = float(z["y_mm"])
        if not (Y_LO <= y_mm <= Y_HI):
            continue
        x_mm = z["x_mm"]
        wf   = z["waveforms"]
        # interior pulses on this line
        mask = (x_mm >= X_LO) & (x_mm <= X_HI)
        if not mask.any():
            continue
        idx = np.where(mask)[0][::PULSE_STRIDE]
        if idx.size == 0:
            continue
        m = per_pulse_metrics(wf[idx])
        for k, v in m.items():
            agg[k].append(v)
        n_pulses_used += idx.size
    print(f"  {session_dir}: used {n_pulses_used} pulses from {len(files)} lines")
    return {k: np.concatenate(v) for k, v in agg.items()}


def main():
    print("Collecting scan A (3.209 V, low SOC)...")
    A = collect(SESS_A)
    print("Collecting scan B (3.366 V, high SOC)...")
    B = collect(SESS_B)

    print()
    print(f"{'metric':<12} {'A median':>12} {'B median':>12} {'B-A':>10} {'% change':>10} {'A IQR':>10} {'B IQR':>10}")
    print("-" * 80)
    for k in A.keys():
        a, b = A[k], B[k]
        ma, mb = np.median(a), np.median(b)
        ia = np.percentile(a, 75) - np.percentile(a, 25)
        ib = np.percentile(b, 75) - np.percentile(b, 25)
        pct = (mb - ma) / ma * 100 if ma != 0 else float("nan")
        print(f"{k:<12} {ma:>12.4f} {mb:>12.4f} {mb - ma:>+10.4f} {pct:>+9.2f}% {ia:>10.4f} {ib:>10.4f}")


if __name__ == "__main__":
    main()
