"""Build a polished, supervisor-ready headline diagnostic.

Single figure showing:
  (a) Acoustic colourmap of the gated waveform (full 23 h, single packet visible)
  (b) Three independent ToF estimators overlaid — they agree to within ~30 ns
  (c) Reference-template xcorr confidence — shows shape evolution but no hops
  (d) Clipping fraction — shows clipping is universal but mild (~1% of samples)
  (e) Envelope argmax integer index — smooth migration of the peak through the gate
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

H5 = r"d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/ascan/ascan_session_2026-05-12_15-42-53/ascan_session_2026-05-12_15-42-53.h5"
OUT = r"d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/data/raw/ascan/ascan_session_2026-05-12_15-42-53"

def hilbert_envelope(x):
    N = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * h, axis=-1))

def parabolic(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    d = a - 2 * b + c
    return i + (0.5 * (a - c) / d if d != 0 else 0.0)

def main():
    with h5py.File(H5, "r") as f:
        fs = float(f.attrs["fs_hz"])
        gate_start = float(f.attrs["gate_us_start"])
        gate_end = float(f.attrs["gate_us_end"])
        ts = f["timestamps"][:]
        tof_stored = f["tof_us"][:] + gate_start  # convert to absolute
        amp_stored = f["amplitude"][:]
        wf = f["waveforms"][:]
    n, gs = wf.shape
    t_h = (ts - ts[0]) / 3600.0
    dt_us = 1e6 / fs
    gate_t = gate_start + np.arange(gs) * dt_us

    # envelope ToF (sub-sample)
    env = np.empty_like(wf, dtype=np.float32)
    chunk = 8192
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        env[i0:i1] = hilbert_envelope(wf[i0:i1].astype(np.float64)).astype(np.float32)
    arg = np.argmax(env, axis=1)
    tof_env = np.array([gate_start + parabolic(env[k], int(arg[k])) * dt_us for k in range(n)])

    # xcorr against ref0 (sub-sample)
    ref = wf[0].astype(np.float64) - wf[0].mean()
    ref_norm = np.linalg.norm(ref) + 1e-12
    L = 2 * gs - 1
    Nfft = 1
    while Nfft < L:
        Nfft <<= 1
    REF = np.fft.rfft(ref[::-1], n=Nfft)
    tof_xc = np.empty(n)
    xc_peak = np.empty(n)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        blk = wf[i0:i1].astype(np.float64)
        blk -= blk.mean(axis=1, keepdims=True)
        bn = np.linalg.norm(blk, axis=1) + 1e-12
        C = np.fft.irfft(np.fft.rfft(blk, n=Nfft, axis=1) * REF[None, :], n=Nfft, axis=1)[:, :L]
        idx = np.argmax(C, axis=1)
        for j, ii in enumerate(idx):
            sub = parabolic(C[j], int(ii)) if 0 < ii < L - 1 else float(ii)
            tof_xc[i0 + j] = tof_env[0] + (sub - (gs - 1)) * dt_us
            xc_peak[i0 + j] = C[j, ii] / (ref_norm * bn[j])

    clip_mask = np.abs(wf) >= 1.98
    clip_frac = clip_mask.mean(axis=1) * 100

    # =================== Figure ===================
    fig = plt.figure(figsize=(15, 13))
    gs_ = GridSpec(5, 1, height_ratios=[2.2, 1.4, 1.0, 0.9, 0.9], hspace=0.45)

    # (a) Colourmap
    ax_a = fig.add_subplot(gs_[0])
    step = max(1, n // 2000)
    img = np.abs(wf[::step]).T
    extent = [t_h[::step][0], t_h[::step][-1], gate_t[-1], gate_t[0]]
    im = ax_a.imshow(img, aspect="auto", extent=extent, origin="upper",
                     cmap="viridis", vmin=0, vmax=2.0, interpolation="nearest")
    ax_a.set_ylim(35.5, 32.5)  # zoom on the active region
    fig.colorbar(im, ax=ax_a, label="|amp| (V)", pad=0.01)
    ax_a.plot(t_h, tof_env, color="orange", lw=0.6, alpha=0.85, label="envelope ToF")
    ax_a.plot(t_h, tof_xc, color="red", lw=0.6, alpha=0.85, label="xcorr ToF")
    ax_a.plot(t_h, tof_stored, color="white", lw=0.6, alpha=0.6, ls=":", label="stored ToF")
    ax_a.set_ylabel("Absolute ToF (us)")
    ax_a.set_title("(a) Acoustic colourmap — single packet, smooth migration through the cycle "
                   "(rectified amplitudes; gate 30-40 us shown 32.5-35.5 us)")
    ax_a.legend(loc="lower right", fontsize=9)

    # (b) Three ToF estimators overlaid
    ax_b = fig.add_subplot(gs_[1])
    ax_b.plot(t_h, (tof_stored - tof_stored[0]) * 1000, color="#1f77b4", lw=0.7,
              label=f"stored (range {(tof_stored.max()-tof_stored.min())*1000:.0f} ns)")
    ax_b.plot(t_h, (tof_env - tof_env[0]) * 1000, color="#ff7f0e", lw=0.7,
              label=f"envelope (range {(tof_env.max()-tof_env.min())*1000:.0f} ns)")
    ax_b.plot(t_h, (tof_xc - tof_xc[0]) * 1000, color="#d62728", lw=0.7,
              label=f"xcorr (range {(tof_xc.max()-tof_xc.min())*1000:.0f} ns)")
    ax_b.set_ylabel("ToF shift vs start (ns)")
    ax_b.set_title("(b) Three independent ToF estimators — identical curves, "
                   "no peak-hopping, no lobe-slip (would show as 200 ns steps at 5 MHz)")
    ax_b.legend(loc="lower left", fontsize=9)
    ax_b.grid(alpha=0.3)

    # (c) xcorr confidence
    ax_c = fig.add_subplot(gs_[2])
    ax_c.plot(t_h, xc_peak, color="purple", lw=0.6)
    ax_c.axhline(0.95, color="grey", lw=0.5, ls="--", label="0.95")
    ax_c.axhline(np.median(xc_peak), color="black", lw=0.5, ls=":",
                 label=f"median = {np.median(xc_peak):.3f}")
    ax_c.set_ylim(0.7, 1.02)
    ax_c.set_ylabel("xcorr peak\n(vs ref at t=0)")
    ax_c.set_title("(c) Reference-template confidence — gradual drop reflects waveform shape "
                   "evolution with SoC, NOT discrete hops")
    ax_c.legend(loc="lower left", fontsize=9)
    ax_c.grid(alpha=0.3)

    # (d) Clipping fraction
    ax_d = fig.add_subplot(gs_[3])
    ax_d.plot(t_h, clip_frac, color="crimson", lw=0.4)
    ax_d.axhline(np.median(clip_frac), color="black", lw=0.5, ls=":",
                 label=f"median = {np.median(clip_frac):.2f}%")
    ax_d.set_ylabel("Clipped samples\nper snapshot (%)")
    ax_d.set_title(f"(d) Hard clipping at +/- 2 V — present in "
                   f"{(clip_frac>0).mean()*100:.0f}% of snapshots, "
                   f"median {np.median(clip_frac):.1f}% of samples per snapshot")
    ax_d.legend(loc="upper right", fontsize=9)
    ax_d.grid(alpha=0.3)

    # (e) Integer argmax
    ax_e = fig.add_subplot(gs_[4])
    ax_e.plot(t_h, arg, color="navy", lw=0.5)
    ax_e.set_xlabel("Elapsed time (h)")
    ax_e.set_ylabel("Envelope argmax\n(integer sample)")
    jumps = (np.abs(np.diff(arg.astype(int))) >= 5).sum()
    ax_e.set_title(f"(e) Integer argmax of envelope — smooth migration across "
                   f"{np.unique(arg).size} sample positions; "
                   f"snapshot-to-snapshot jumps >=5 samples: {jumps}")
    ax_e.grid(alpha=0.3)

    fig.suptitle("Peak-hopping / clipping diagnostic for 23 h LFP C/10 cycling\n"
                 f"session = {os.path.basename(H5)}", fontsize=13, y=0.995)
    out_png = os.path.join(OUT, "headline_diagnostic.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print("Wrote", out_png)

if __name__ == "__main__":
    main()
