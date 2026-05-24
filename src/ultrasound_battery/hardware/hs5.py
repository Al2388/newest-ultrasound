# -*- coding: utf-8 -*-
"""
TiePie HS5 Oscilloscope Interface
===================================
Controls a TiePie HS5 USB oscilloscope in stream mode to continuously capture
ultrasonic pulse waveforms for both C-scan and A-scan acquisition modes.

Hardware wiring
---------------
  CH1 (AC-coupled) — ultrasonic transducer receive signal
  CH2 (DC-coupled) — sync/trigger pulse from the ultrasound pulser (TTL-level)

Acquisition strategy
--------------------
The HS5 runs in MM_STREAM mode, pushing blocks of samples into a software
buffer continuously. CH2 acts as the synchronisation channel: each rising edge
of CH2 marks the start of a new ultrasonic pulse. For every detected pulse the
gate window [gate_us_start, gate_us_end] is extracted from CH1 and three
features are computed:

  Amplitude  — peak of the Hilbert envelope  (phase-insensitive, unit: V)
  ToF        — sample index of that peak converted to µs  (time within gate)
  Energy     — sum of squared DC-removed samples  (proportional to signal power)

A carry-buffer mechanism prevents pulses from being missed when a sync edge
falls right at the boundary between two consecutive hardware data blocks.

Dependencies
------------
  python-libtiepie — TiePie SDK Python bindings
  numpy            — array maths, FFT

Usage
-----
  hs = HS5StreamPeaks(fs_hz=20e6, gate_us=(30.0, 40.0)).open()
  hs.calibrate_sync(seconds=1.0)
  tt, aa, tf, ee, wf = hs.acquire_peaks(duration_s=1.0, save_waveforms=True)
  hs.close()
"""

import time
import numpy as np

try:
    import libtiepie
except ImportError as e:
    raise SystemExit("Missing dependency: pip install python-libtiepie") from e


# =============================================================================
# Public helper — used by both this module and ascan_service.py
# =============================================================================

def envelope_hilbert(x: np.ndarray) -> np.ndarray:
    """
    Compute the analytic-signal envelope of a real-valued 1-D array.

    This is a pure-NumPy implementation that matches ``scipy.signal.hilbert``
    bit-for-bit on real input, but avoids the SciPy dependency.

    Algorithm
    ---------
    1. Compute the full complex FFT of ``x``.
    2. Apply the analytic-signal one-sided weight vector H of length n:
         H[0]        = 1   (DC, unchanged)
         H[1 : n//2] = 2   (positive freqs, doubled to compensate for the
                            zeroed negative-frequency mirror)
         H[n//2]     = 1   (Nyquist bin, only present for even n)
         H[n//2+1 :] = 0   (negative freqs, zeroed)
    3. Inverse-FFT — yields the complex analytic signal x_a(t) = x(t) + j·Ĥ{x(t)}.
    4. Take |x_a(t)| sample-by-sample to obtain the real-valued envelope.

    Why full FFT, not rfft/irfft
    ----------------------------
    A natural-looking optimisation is to replace steps 1–3 with
    ``irfft(H · rfft(x))`` since x is real. This is mathematically WRONG for
    envelope extraction and worth documenting because the inherited prototype
    used exactly that form:

      * ``rfft(x)`` returns only the positive-frequency half of the spectrum.
      * ``irfft`` interprets its input as the positive half of a
        Hermitian-symmetric spectrum and mirror-conjugates the negative half
        before transforming. The output is therefore the unique *real* signal
        whose half-spectrum is ``H·rfft(x)`` — equivalently, with the positive
        weighting [1, 2, 2, …, 2, 1], the signal 2·x(t).
      * The analytic signal x_a(t) is *complex* — its negative-frequency
        components are zero, not the conjugate mirror of its positive ones.
        That asymmetry cannot be represented in the rfft compact form.

    Consequence: ``|irfft(H·rfft(x))|`` = 2·|x(t)|, i.e. a full-wave rectified
    copy of the carrier. Its peak per gate happens to track the envelope peak
    on clean tone bursts (which is why the prototype appeared to "work"), but
    its *shape* is the rectified RF, not the smooth envelope — so any
    measurement derived from the envelope's shape (most importantly the ToF
    index ``argmax(env)`` consumed by :meth:`_compute_features`) would be
    corrupted by zero-crossings of the carrier. The full FFT here is therefore
    a deliberate choice, not an oversight, and matches the formulation in
    ``scipy.signal.hilbert`` for the same reason.

    Parameters
    ----------
    x : np.ndarray
        1-D float array — typically DC-removed samples from the gate window.

    Returns
    -------
    np.ndarray (float32, same length as x)
        Non-negative envelope values.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n == 0:
        return x

    X = np.fft.fft(x)
    H = np.zeros(n, dtype=np.float32)

    # One-sided weighting: double positive-frequency bins, keep DC and Nyquist at ×1
    if n % 2 == 0:
        H[0] = 1.0; H[1:n // 2] = 2.0; H[n // 2] = 1.0
    else:
        H[0] = 1.0; H[1:(n + 1) // 2] = 2.0

    return np.abs(np.fft.ifft(X * H)).astype(np.float32, copy=False)


# =============================================================================
# Main oscilloscope class
# =============================================================================

class HS5StreamPeaks:
    """
    Stream-mode driver for the TiePie HS5 oscilloscope.

    Configured with two channels:
      CH1 — AC-coupled, user-selectable voltage range — receives the ultrasonic
            echo signal from the transducer.
      CH2 — DC-coupled, ±5 V — receives the TTL sync pulse from the pulser.
            Every rising edge on CH2 marks the arrival of a new transmit pulse.

    Parameters
    ----------
    fs_hz : float
        Requested sample rate in Hz (default 20 MHz). The hardware may clamp
        this slightly; self.fs is updated to the actual rate after open().
    win_us : float
        Capture window length in µs — used to set the HS5 record length.
    gate_us : (float, float)
        (start, end) of the analysis gate in µs, measured from the sync edge.
        Only the CH1 samples within this window are used for feature extraction.
    ch1_range : float
        CH1 full-scale voltage range in volts (default ±1 V, i.e. range = 1.0).
    ch1_coupling : str
        "AC" (default) blocks DC offset from the transducer cable.
        "DC" keeps the absolute voltage.
    sync_thresh_guess : float
        Initial guess for the CH2 trigger voltage (V). Refined by calibrate_sync().
    min_hyst_v : float
        Minimum comparator hysteresis half-width (V). Prevents chattering on
        noisy sync edges.
    prf_hint_hz : float
        Expected pulse repetition frequency in Hz. Used only to set the initial
        refractory period before calibration runs.
    feature_mode : str
        "envelope"   — peak of Hilbert envelope (default, most robust)
        "p2p"        — raw peak-to-peak amplitude
        "p2p_robust" — percentile-based peak-to-peak (rejects single-sample spikes)
    p2p_percentiles : (int, int)
        (lo, hi) percentile pair for "p2p_robust" mode.
    """

    def __init__(self,
                 fs_hz: float = 20_000_000,
                 win_us: float = 60.0,
                 gate_us: tuple[float, float] = (30.0, 40.0),
                 ch1_range: float = 1.0,
                 ch1_coupling: str = "AC",
                 sync_thresh_guess: float = 0.0,
                 min_hyst_v: float = 0.02,
                 prf_hint_hz: float = 5000.0,
                 feature_mode: str = "envelope",
                 p2p_percentiles: tuple[int, int] = (5, 95)):

        self.fs  = float(fs_hz)
        self.dt  = 1.0 / self.fs
        self.win_us = float(win_us)
        self.gate_us = (float(gate_us[0]), float(gate_us[1]))

        # Gate boundaries: convert µs → integer sample indices
        self.g0       = max(0, int(round(self.gate_us[0] * 1e-6 * self.fs)))
        self.g1       = max(self.g0 + 1, int(round(self.gate_us[1] * 1e-6 * self.fs)))
        self.gate_len = self.g1 - self.g0   # number of samples in the analysis gate

        # Minimum HS5 record length: 0.5 ms worth of samples, at least 10 000
        self.rec_len = max(10_000, int(0.0005 * self.fs))

        self.range    = float(ch1_range)
        self.coupling = ch1_coupling.upper()

        self.sync_thresh_guess = float(sync_thresh_guess)
        self.min_hyst_v        = float(min_hyst_v)
        self.prf_hz            = float(prf_hint_hz)

        # Refractory period: half the expected inter-pulse interval.
        # Prevents a single TTL pulse from registering as multiple trigger edges
        # if the sync signal has ringing after the rising edge.
        T = 1.0 / self.prf_hz if self.prf_hz > 0 else 2e-4
        self.refractory_samp = max(1, int(round(0.5 * T * self.fs)))

        # Comparator thresholds — will be refined by calibrate_sync()
        self.ch2_lo = self.sync_thresh_guess - self.min_hyst_v
        self.ch2_hi = self.sync_thresh_guess + self.min_hyst_v
        self.detected_prf = self.prf_hz

        self.feature_mode    = (feature_mode or "envelope").lower()
        self.p2p_percentiles = p2p_percentiles
        self.scp             = None   # libtiepie oscilloscope handle (set in open())

    # -------------------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------------------

    def _compute_features(self, x: np.ndarray) -> tuple[float, float, float]:
        """
        Extract amplitude, time-of-flight, and energy from a single gate window.

        Pre-processing
        --------------
        The DC offset is removed first (``v - mean(v)``) so that:
          * a non-zero baseline cannot inflate the energy value (which is
            scale-quadratic and unforgiving of bias), and
          * the analytic-signal envelope used for ToF is computed on a signal
            centred on zero, where ``argmax(|x_a|)`` is dominated by the echo
            and not by any slow drift in the cable / coupling.

        Why the Hilbert envelope is computed unconditionally
        ----------------------------------------------------
        The ``feature_mode`` selector only controls the AMPLITUDE estimator
        (envelope-peak, raw peak-to-peak, or percentile peak-to-peak). The
        envelope itself is computed for every pulse regardless of mode,
        because the ToF measurement is defined as the sample index of the
        envelope peak — there is no "p2p ToF" or "percentile ToF" that would
        be physically meaningful for a damped echo.

        This is the reason ``envelope_hilbert`` is implemented with the full
        complex FFT rather than the rfft/irfft shortcut (see that function's
        docstring): a rectified-carrier surrogate would have peak indices
        sitting at zero-crossings of the RF rather than at the true envelope
        peak, biasing the ToF map.

        ToF reference
        -------------
        ``tof_us`` is measured from the START of the gate window, not from the
        sync edge. Add the gate's ``gate_us[0]`` if absolute time-of-flight
        from transmit is needed.

        Parameters
        ----------
        x : np.ndarray
            Raw gate-length float32 samples from CH1 for one pulse.

        Returns
        -------
        (amplitude, tof_us, energy) : tuple[float, float, float]
            amplitude — signal amplitude in volts (mode-dependent, see __init__)
            tof_us    — time from gate start to envelope peak in µs
            energy    — Σ(v²) — sum of squared DC-removed samples (V² · samples)
        """
        if x is None or x.size == 0:
            return np.nan, np.nan, np.nan

        v      = x.astype(np.float32, copy=False)
        v      = v - np.mean(v, dtype=np.float64)   # remove DC baseline

        env    = envelope_hilbert(v)
        pk_idx = int(np.argmax(env))                 # envelope peak sample index

        mode = self.feature_mode
        if mode == "p2p":
            amp = float(np.max(v) - np.min(v))
        elif mode == "p2p_robust":
            lo, hi = self.p2p_percentiles
            amp = float(np.percentile(v, hi) - np.percentile(v, lo))
        else:  # "envelope" — default
            amp = float(env[pk_idx])

        tof_us = (pk_idx / self.fs) * 1e6   # convert sample index → µs within gate
        energy = float(np.dot(v, v))         # proportional to received acoustic energy
        return amp, tof_us, energy

    # -------------------------------------------------------------------------
    # Hardware lifecycle
    # -------------------------------------------------------------------------

    def open(self) -> "HS5StreamPeaks":
        """
        Discover the first available HS5 oscilloscope on the USB bus, configure
        both channels, and prepare for stream-mode acquisition.

        Returns self so the constructor and open() can be chained:
            hs = HS5StreamPeaks(...).open()

        Raises RuntimeError if no compatible device is found.
        """
        libtiepie.network.auto_detect_enabled = True
        libtiepie.device_list.update()
        self.scp = None

        for it in libtiepie.device_list:
            if it.can_open(libtiepie.DEVICETYPE_OSCILLOSCOPE):
                s = it.open_oscilloscope()
                if s.measure_modes & libtiepie.MM_STREAM:
                    self.scp = s
                    break

        if self.scp is None:
            raise RuntimeError("HS5 not found — check USB connection")

        s = self.scp
        s.measure_mode = libtiepie.MM_STREAM

        # Use the highest available sample rate up to the requested value
        s.sample_rate = min(self.fs, s.sample_rate_max)
        self.fs = s.sample_rate
        self.dt = 1.0 / self.fs

        # Record length: large enough to always contain the full gate window
        target          = max(self.rec_len, 5 * self.g1, 50_000)
        s.record_length = min(int(target), s.record_length_max)
        self.rec_len    = s.record_length

        # Disable all channels, then enable only CH1 (signal) and CH2 (sync)
        for ch in s.channels:
            ch.enabled = False

        ch1          = s.channels[0]
        ch1.enabled  = True
        ch1.range    = float(self.range)
        ch1.coupling = libtiepie.CK_ACV if self.coupling == "AC" else libtiepie.CK_DCV

        ch2          = s.channels[1]
        ch2.enabled  = True
        ch2.range    = 5.0              # CH2 carries a TTL-level sync pulse (~3–5 V)
        ch2.coupling = libtiepie.CK_DCV

        # Recompute gate sample indices at the actual (hardware-clamped) sample rate
        self.g0       = max(0, int(round(self.gate_us[0] * 1e-6 * self.fs)))
        self.g1       = max(self.g0 + 1, int(round(self.gate_us[1] * 1e-6 * self.fs)))
        self.gate_len = self.g1 - self.g0

        print(f"[HS5] Opened.  fs={self.fs/1e6:.1f} MHz  "
              f"Gate=[{self.g0}:{self.g1}]  "
              f"({self.gate_len} samples, {self.gate_len/self.fs*1e6:.1f} µs)")
        return self

    def close(self):
        """Stop any running acquisition and release the oscilloscope handle."""
        if self.scp:
            if self.scp.is_running:
                self.scp.stop()
            self.scp = None

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    def calibrate_sync(self, seconds: float = 1.0, verbose: bool = True):
        """
        Auto-calibrate the CH2 comparator thresholds by observing the sync pulse
        for `seconds` seconds.

        Method
        ------
        1. Stream CH2 data for `seconds` seconds.
        2. Compute the 5th and 95th percentiles of the CH2 voltage — these
           represent the LOW and HIGH levels of the TTL sync pulse.
        3. Set the comparator window to ±20 % of the observed swing around the
           midpoint. This is robust against slow amplitude drift in the sync signal.
        4. Estimate the actual PRF from the median falling-edge interval and
           update the refractory period accordingly.

        After this call:
          self.ch2_lo, self.ch2_hi  — low/high thresholds for edge detection
          self.detected_prf         — measured PRF in Hz
          self.refractory_samp      — half-period in samples (anti-double-trigger)
        """
        s = self.scp
        if s.is_running:
            s.stop()
        s.start()

        t0      = time.perf_counter()
        ch2_all = []
        while (time.perf_counter() - t0) < float(seconds):
            if not s.is_data_ready:
                time.sleep(0.0005)
                continue
            data = s.get_data()
            if data and len(data) >= 2:
                ch2_all.append(np.asarray(data[1], dtype=np.float32))
        s.stop()

        if not ch2_all:
            raise RuntimeError("Calibration failed: no data received from CH2")

        ch2 = np.concatenate(ch2_all)

        # Fit thresholds to ±20 % of the observed CH2 swing
        vmin, vmax = np.percentile(ch2, 5), np.percentile(ch2, 95)
        span       = max(vmax - vmin, self.min_hyst_v * 2)
        vmid       = 0.5 * (vmin + vmax)
        self.ch2_lo = vmid - 0.2 * span
        self.ch2_hi = vmid + 0.2 * span

        # Estimate PRF from intervals between rising edges on CH2. A Schmitt
        # trigger (hi/lo hysteresis) is applied vectorised: the latched state
        # flips high only when ch2 ≥ ch2_hi, and stays high until ch2 ≤ ch2_lo.
        # This rejects ringing right after the TTL edge — single-threshold
        # detection would otherwise register one pulse as several.
        idx     = np.arange(ch2.size, dtype=np.int64)
        last_hi = np.maximum.accumulate(np.where(ch2 >= self.ch2_hi, idx, -1))
        last_lo = np.maximum.accumulate(np.where(ch2 <= self.ch2_lo, idx, -1))
        state   = last_hi > last_lo                             # latched comparator output
        edges   = np.flatnonzero(state[1:] & ~state[:-1])       # rising-edge sample indices
        if edges.size >= 2:
            self.detected_prf = self.fs / np.median(np.diff(edges))

        T = 1.0 / max(self.detected_prf, 1.0)
        self.refractory_samp = max(1, int(round(0.5 * T * self.fs)))

        if verbose:
            print(f"[CAL] CH2 thresholds: {self.ch2_lo:.3f} / {self.ch2_hi:.3f} V  "
                  f"PRF: {self.detected_prf:.1f} Hz  "
                  f"Refractory: {self.refractory_samp} samples")

    # -------------------------------------------------------------------------
    # Pulse acquisition
    # -------------------------------------------------------------------------

    def acquire_peaks(self,
                      duration_s: float = 1.0,
                      save_waveforms: bool = True,
                      save_full_waveforms: bool = False
                      ) -> tuple:
        """
        Collect pulse features continuously for `duration_s` seconds.

        The oscilloscope streams data in hardware blocks. Each block is scanned
        for CH2 rising edges. For each valid edge (subject to the refractory
        period) the corresponding CH1 gate window is extracted and features are
        computed. A carry buffer handles pulses that straddle two consecutive
        hardware blocks.

        Parameters
        ----------
        duration_s : float
            Collection window length in seconds.
        save_waveforms : bool
            If True, the raw float32 gate window is stored for every pulse and
            returned as a 2-D array. Set False to reduce memory usage for fast
            C-scan lines where only features are needed.
        save_full_waveforms : bool
            If True, also return a 2-D array of full post-sync windows from
            0 to win_us. This is intended for live display / gate visualisation.

        Returns
        -------
        timestamps : np.ndarray [float64]
            Wall-clock time of each pulse (time.perf_counter() origin).
        amplitude  : np.ndarray [float32]   envelope peak in volts.
        tof_us     : np.ndarray [float32]   time-of-flight within gate (µs).
        energy     : np.ndarray [float32]   pulse energy (V² · samples).
        waveforms  : np.ndarray [float32, shape (N, gate_len)] or None
            Raw DC-removed gate windows, one row per accepted pulse.
            None if save_waveforms=False.

        All returned arrays are sorted in ascending timestamp order and share
        the same index (timestamps[i] corresponds to waveforms[i], etc.).
        """
        s    = self.scp
        if s.is_running:
            s.stop()
        s.start()

        thr  = float(self.ch2_hi)       # rising-edge detection threshold (V)
        refr = int(self.refractory_samp)
        g0, g1 = self.g0, self.g1
        full_len = max(g1, int(round(self.win_us * 1e-6 * self.fs)))
        t0     = time.perf_counter()

        cum_abs       = 0       # cumulative sample count across all blocks received
        last_edge_abs = -1e12   # absolute sample index of last accepted sync edge

        # Carry buffers: keep the last g1 samples of each block so that a gate
        # window starting near the end of block N can be completed using the
        # beginning of block N+1.
        carry_ch1 = np.empty(0, dtype=np.float32)
        carry_ch2 = np.empty(0, dtype=np.float32)

        p_amp, p_tof, p_eng, p_tt = [], [], [], []
        p_wf = [] if save_waveforms else None
        p_full = [] if save_full_waveforms else None

        try:
            while (time.perf_counter() - t0) < duration_s:
                if not s.is_data_ready:
                    time.sleep(0.0005)
                    continue
                data  = s.get_data()
                if not data:
                    continue

                c1    = np.asarray(data[0], dtype=np.float32)
                c2    = np.asarray(data[1], dtype=np.float32)
                n_new = min(c1.size, c2.size)
                if n_new == 0:
                    continue

                c1        = c1[:n_new]
                c2        = c2[:n_new]
                blk_start = cum_abs
                cum_abs  += n_new

                # Prepend carry buffer so inter-block gate windows are intact
                if carry_ch1.size:
                    c1          = np.concatenate((carry_ch1, c1))
                    c2          = np.concatenate((carry_ch2, c2))
                    combo_start = blk_start - carry_ch1.size
                else:
                    combo_start = blk_start

                # Detect all CH2 rising edges: sample transitions from below thr to above thr
                edges = np.flatnonzero((c2[:-1] < thr) & (c2[1:] >= thr))

                for e in edges:
                    e_abs = combo_start + e

                    # Refractory check: ignore pulses too close to the previous one
                    if (e_abs - last_edge_abs) < refr:
                        continue

                    s0, s1 = e + g0, e + g1
                    f1 = e + full_len
                    # Skip if the gate extends beyond the current combined buffer
                    if s1 > c1.size or (save_full_waveforms and f1 > c1.size):
                        continue

                    gate_slice       = c1[s0:s1].copy()
                    amp, tof, eng    = self._compute_features(gate_slice)
                    p_amp.append(amp)
                    p_tof.append(tof)
                    p_eng.append(eng)
                    p_tt.append(t0 + e_abs / self.fs)
                    if save_waveforms:
                        p_wf.append(gate_slice)
                    if save_full_waveforms:
                        p_full.append(c1[e:f1].copy())
                    last_edge_abs = e_abs

                # Keep enough samples as carry for gates/full display windows
                # that straddle the next hardware block.
                rem       = min(full_len if save_full_waveforms else g1, c1.size)
                carry_ch1 = c1[-rem:]
                carry_ch2 = c2[-rem:]

        finally:
            if s.is_running:
                s.stop()

        # Return empty arrays if no pulses were detected in this window
        if not p_tt:
            empty = np.array([], dtype=np.float32)
            result = (np.array([], dtype=np.float64), empty, empty, empty, None)
            return (*result, None) if save_full_waveforms else result

        # Sort by wall-clock timestamp (almost always already sorted, but guard anyway)
        idx = np.argsort(p_tt)
        tt  = np.array(p_tt,  dtype=np.float64)[idx]
        aa  = np.array(p_amp, dtype=np.float32)[idx]
        tf  = np.array(p_tof, dtype=np.float32)[idx]
        ee  = np.array(p_eng, dtype=np.float32)[idx]
        wf  = np.stack(p_wf, axis=0)[idx] if save_waveforms else None
        full_wf = np.stack(p_full, axis=0)[idx] if save_full_waveforms else None

        if save_full_waveforms:
            return tt, aa, tf, ee, wf, full_wf
        return tt, aa, tf, ee, wf
