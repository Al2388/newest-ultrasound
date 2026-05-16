"""Peak-hopping / clipping diagnostic for the 21h LFP cycling session.

Reconstructs four diagnostics from the raw waveforms (the older v2.0 schema
of this file did not save tracking_corr / tracking_lag_samples / tof_us_envelope):

  1. Acoustic colourmap of the gated waveform across the whole 21 h
  2. Three independent ToF estimators overlaid (stored / envelope argmax / xcorr)
  3. Cross-correlation confidence vs time, with reference taken at t=0
  4. Clipping fraction per snapshot (samples within 1% of +/- 2 V rail)

Outputs:
  data/raw/ascan/ascan_session_2026-05-08_17-28-46/diagnostic_*.png
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

H5 = r"d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/ascan/ascan_session_2026-05-12_15-42-53/ascan_session_2026-05-12_15-42-53.h5"
OUT = r"d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/ascan/ascan_session_2026-05-12_15-42-53"

CLIP_V = 2.0          # observed saturation rail
CLIP_TOL = 0.01       # within 1% of rail counts as clipped

def hilbert_envelope(x):
    """FFT-based analytic-signal envelope, axis=-1."""
    N = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    Xa = X * h
    return np.abs(np.fft.ifft(Xa, axis=-1))

def parabolic_subsample(y, i):
    """Return sub-sample peak location given index i of integer max."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = (a - 2 * b + c)
    if denom == 0:
        return float(i)
    return i + 0.5 * (a - c) / denom

def main():
    print(f"Reading {H5} ...")
    with h5py.File(H5, "r") as f:
        fs = float(f.attrs["fs_hz"])
        gate_start_us = float(f.attrs["gate_us_start"])
        gate_end_us = float(f.attrs["gate_us_end"])
        n = f["waveforms"].shape[0]
        gate_samples = f["waveforms"].shape[1]
        ts = f["timestamps"][:]
        tof_stored = f["tof_us"][:]
        amp_stored = f["amplitude"][:]
        # waveforms is large (~143 MB float32); load all at once is fine here
        wf = f["waveforms"][:]
        print(f"  fs = {fs/1e6:.1f} MHz, gate = {gate_start_us}-{gate_end_us} us, "
              f"N = {n}, gate_samples = {gate_samples}")

    t_h = (ts - ts[0]) / 3600.0
    dt_us = 1e6 / fs               # 0.05 us per sample at 20 MHz
    gate_t = gate_start_us + np.arange(gate_samples) * dt_us  # absolute us within gate

    # --- 1) Envelope and envelope-based ToF for every snapshot --------------
    print("Computing Hilbert envelopes (in chunks)...")
    env = np.empty_like(wf, dtype=np.float32)
    chunk = 8192
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        env[i0:i1] = hilbert_envelope(wf[i0:i1].astype(np.float64)).astype(np.float32)
    env_argmax = np.argmax(env, axis=1)
    # sub-sample refinement
    print("Sub-sample refining envelope ToF...")
    tof_env_us = np.empty(n, dtype=np.float64)
    for k in range(n):
        i = int(env_argmax[k])
        sub = parabolic_subsample(env[k], i)
        tof_env_us[k] = gate_start_us + sub * dt_us

    # --- 2) Cross-correlation against reference at t=0 ----------------------
    print("Computing xcorr against reference waveform 0...")
    ref = wf[0].astype(np.float64)
    ref = ref - ref.mean()
    ref_norm = np.linalg.norm(ref) + 1e-12
    max_lag = gate_samples - 1
    # we want lag in samples such that argmax of correlation between wf[k] and ref shifted
    # use np.correlate(full)
    xcorr_lag = np.empty(n, dtype=np.float64)
    xcorr_peak = np.empty(n, dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = wf[i0:i1].astype(np.float64)
        block -= block.mean(axis=1, keepdims=True)
        bn = np.linalg.norm(block, axis=1) + 1e-12
        # correlate each row with ref using FFT
        L = 2 * gate_samples - 1
        Nfft = 1
        while Nfft < L:
            Nfft <<= 1
            # compute fft of ref once per chunk
        REF = np.fft.rfft(ref[::-1], n=Nfft)
        BLK = np.fft.rfft(block, n=Nfft, axis=1)
        corr = np.fft.irfft(BLK * REF[None, :], n=Nfft, axis=1)[:, :L]
        # lag axis: lag = idx - (gate_samples - 1)
        idx = np.argmax(corr, axis=1)
        # sub-sample refinement on the peak
        for j, ii in enumerate(idx):
            if ii > 0 and ii < L - 1:
                a, b, c = corr[j, ii - 1], corr[j, ii], corr[j, ii + 1]
                denom = (a - 2 * b + c)
                sub = ii + (0.5 * (a - c) / denom if denom != 0 else 0.0)
            else:
                sub = float(ii)
            lag = sub - (gate_samples - 1)
            xcorr_lag[i0 + j] = lag
            xcorr_peak[i0 + j] = corr[j, ii] / (ref_norm * bn[j])

    # Convert xcorr lag -> absolute ToF in us assuming the reference envelope-peak
    # defines ToF_ref. We anchor ToF_xcorr at ToF_env(t=0) and add lag * dt.
    tof_xcorr_us = tof_env_us[0] + xcorr_lag * dt_us

    # --- 3) Clipping diagnostic --------------------------------------------
    print("Computing clipping fraction per snapshot...")
    clip_mask = np.abs(wf) >= (CLIP_V * (1.0 - CLIP_TOL))
    clip_frac = clip_mask.mean(axis=1)

    # --- Plot 1: Acoustic colourmap ----------------------------------------
    print("Plotting acoustic colourmap...")
    # downsample along time to make a viewable image
    target_cols = 1800
    step = max(1, n // target_cols)
    wf_ds = np.abs(wf[::step]).T   # rectified, like Owen's figures
    t_ds = t_h[::step]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    # stored tof_us is gate-relative; convert to absolute for the overlay
    tof_stored_abs = tof_stored + gate_start_us
    extent = [t_ds[0], t_ds[-1], gate_t[-1], gate_t[0]]
    im = ax.imshow(wf_ds, aspect="auto", extent=extent, origin="upper",
                   cmap="viridis", vmin=0, vmax=2.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, label="|amplitude| (V, clip at 2 V)")
    ax.set_ylim(gate_t[-1], gate_t[0])
    # overlay the three ToF estimators
    ax.plot(t_h, tof_stored_abs, color="white", lw=0.6, alpha=0.7, label="ToF stored")
    ax.plot(t_h, tof_env_us, color="orange", lw=0.6, alpha=0.7, label="ToF envelope (recomputed)")
    ax.plot(t_h, tof_xcorr_us, color="red", lw=0.6, alpha=0.7, label="ToF xcorr (recomputed)")
    ax.set_xlabel("Elapsed time (h)")
    ax.set_ylabel("Absolute time-of-flight (us)")
    ax.set_title("Acoustic colourmap of gated waveform — full 21 h cycle\n"
                 "ToF estimators overlaid; rectified amplitudes")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diagnostic_1_colourmap.png"), dpi=140)
    plt.close(fig)

    # --- Plot 2: ToF estimators side-by-side --------------------------------
    print("Plotting ToF estimator comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t_h, (tof_stored - tof_stored[0]) * 1000, color="#1f77b4", lw=0.7, label="stored tof_us")
    axes[0].plot(t_h, (tof_env_us - tof_env_us[0]) * 1000, color="#ff7f0e", lw=0.7, label="envelope-argmax")
    axes[0].plot(t_h, (tof_xcorr_us - tof_xcorr_us[0]) * 1000, color="#d62728", lw=0.7, label="xcorr vs ref0")
    axes[0].set_ylabel("ToF shift vs start (ns)")
    axes[0].set_title("Three independent ToF estimators — same waveforms")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(alpha=0.3)

    # stored is gate-relative; bring it to absolute us before comparing
    tof_stored_abs = tof_stored + gate_start_us
    axes[1].plot(t_h, (tof_env_us - tof_stored_abs) * 1000, color="#ff7f0e", lw=0.5,
                 label="envelope - stored")
    axes[1].plot(t_h, (tof_xcorr_us - tof_stored_abs) * 1000, color="#d62728", lw=0.5,
                 label="xcorr - stored")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axhline(200, color="grey", lw=0.5, ls="--", alpha=0.5)
    axes[1].axhline(-200, color="grey", lw=0.5, ls="--", alpha=0.5)
    axes[1].set_ylabel("Disagreement (ns)\n(+/- 200 ns = 1 cycle @ 5 MHz)")
    axes[1].set_title("Disagreement between estimators — non-zero implies peak-hopping or lobe-slip")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_h, xcorr_peak, color="purple", lw=0.4)
    axes[2].axhline(0.95, color="grey", lw=0.5, ls="--")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("Normalised xcorr peak\n(vs reference at t=0)")
    axes[2].set_xlabel("Elapsed time (h)")
    axes[2].set_title("Reference-template confidence — drops mean the waveform shape diverged from t=0")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diagnostic_2_tof_estimators.png"), dpi=140)
    plt.close(fig)

    # --- Plot 3: Clipping fraction & amplitude diagnostics ------------------
    print("Plotting clipping diagnostic...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    axes[0].plot(t_h, clip_frac * 100, color="crimson", lw=0.5)
    axes[0].set_ylabel("Clipped samples per snapshot (%)")
    axes[0].set_title("Hard clipping at +/- 2 V rail — fraction of waveform samples saturated")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_h, amp_stored, color="green", lw=0.4, label="stored amplitude")
    axes[1].plot(t_h, np.max(np.abs(wf), axis=1), color="black", lw=0.4, alpha=0.5,
                 label="peak |V| (recomputed)")
    axes[1].axhline(CLIP_V, color="red", lw=0.5, ls="--", label="rail")
    axes[1].set_ylabel("Amplitude (V)")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(alpha=0.3)

    # peak position (integer argmax of envelope) -- watch for step jumps
    axes[2].plot(t_h, env_argmax, color="navy", lw=0.4)
    axes[2].set_ylabel("Envelope argmax index\n(integer sample in gate)")
    axes[2].set_xlabel("Elapsed time (h)")
    axes[2].set_title("Integer-sample peak position — visible step jumps would indicate packet hops")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diagnostic_3_clipping.png"), dpi=140)
    plt.close(fig)

    # --- Plot 4: Sample waveforms across the cycle --------------------------
    print("Plotting sample waveforms...")
    sample_hours = [0.0, 3.0, 7.0, 9.5, 12.0, 16.0, 20.5]
    idxs = [int(np.argmin(np.abs(t_h - h_target))) for h_target in sample_hours]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    cmap = plt.cm.viridis
    for j, idx in enumerate(idxs):
        c = cmap(j / max(len(idxs) - 1, 1))
        axes[0].plot(gate_t, wf[idx], color=c, lw=0.7, label=f"t={t_h[idx]:.1f} h")
        axes[1].plot(gate_t, env[idx], color=c, lw=0.9)
    axes[0].axhline(CLIP_V, color="red", lw=0.5, ls="--", alpha=0.5)
    axes[0].axhline(-CLIP_V, color="red", lw=0.5, ls="--", alpha=0.5)
    axes[0].set_ylabel("Raw amplitude (V)")
    axes[0].set_title("Sample waveforms across the 21 h cycle — red dashed lines = clipping rails")
    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Hilbert envelope")
    axes[1].set_xlabel("Time within gate (us, absolute)")
    axes[1].set_title("Envelopes — watch for change in dominant peak position")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diagnostic_4_sample_waveforms.png"), dpi=140)
    plt.close(fig)

    # --- Print summary statistics ------------------------------------------
    print("\n=== Diagnostic summary ===")
    print(f"Clipping: snapshots with any clipping = {(clip_frac > 0).mean()*100:.1f}%; "
          f"median clipping fraction = {np.median(clip_frac)*100:.2f}% of samples")
    print(f"Stored ToF range: {tof_stored.min():.4f} - {tof_stored.max():.4f} us "
          f"({(tof_stored.max()-tof_stored.min())*1000:.1f} ns)")
    print(f"Envelope ToF range: {tof_env_us.min():.4f} - {tof_env_us.max():.4f} us "
          f"({(tof_env_us.max()-tof_env_us.min())*1000:.1f} ns)")
    print(f"xcorr ToF range: {tof_xcorr_us.min():.4f} - {tof_xcorr_us.max():.4f} us "
          f"({(tof_xcorr_us.max()-tof_xcorr_us.min())*1000:.1f} ns)")
    tof_stored_abs = tof_stored + gate_start_us
    print(f"Envelope vs stored disagreement: median = {np.median((tof_env_us-tof_stored_abs)*1000):.1f} ns, "
          f"max abs = {np.max(np.abs((tof_env_us-tof_stored_abs)*1000)):.1f} ns")
    print(f"xcorr vs stored disagreement: median = {np.median((tof_xcorr_us-tof_stored_abs)*1000):.1f} ns, "
          f"max abs = {np.max(np.abs((tof_xcorr_us-tof_stored_abs)*1000)):.1f} ns")
    print(f"xcorr confidence: median = {np.median(xcorr_peak):.3f}, min = {xcorr_peak.min():.3f}, "
          f"snapshots below 0.95 = {(xcorr_peak < 0.95).sum()} ({(xcorr_peak < 0.95).mean()*100:.1f}%)")
    print(f"Integer envelope argmax: unique values = {np.unique(env_argmax).size}, "
          f"min = {env_argmax.min()}, max = {env_argmax.max()}")
    # Detect step jumps in env_argmax
    jumps = np.abs(np.diff(env_argmax.astype(int)))
    big_jumps = (jumps >= 5).sum()
    print(f"Snapshot-to-snapshot jumps >= 5 samples (>= 250 ns): {big_jumps}")
    print("\nWrote 4 PNGs to", OUT)

if __name__ == "__main__":
    main()
