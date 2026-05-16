"""
Live ToF Gauging Service (with HDF5 archival)
==============================================
Streams a single time-of-flight reading to the dashboard at ~3 Hz so the
operator can watch the value change as they manually move the probe across
the sample. Used as a quick "is the coupling good?" check, or to eyeball
ToF stability before starting a longer A-scan or C-scan acquisition.

Hardware
--------
TiePie HS5 oscilloscope only (no printer connection). The HS5 is opened on
start, calibrated to the sync pulse, and acquires pulses in fixed windows
of `sample_window_s` seconds.

Persistence
-----------
Every acquisition window is appended to an HDF5 file as it happens, so a
crash, power loss, or unclean stop does not lose any data already collected.
The file is flushed on every append (each ~0.3 s), so worst-case data loss
is one acquisition window.

HDF5 file layout  (data/raw/gauging/gauge_<session_id>.h5)
------------------------------------------------------
Attributes (session-level metadata):
  session_id        str      unique identifier
  fs_hz             float    oscilloscope sample rate (Hz)
  gate_us_start     float    gate window start (µs from sync edge)
  gate_us_end       float    gate window end   (µs from sync edge)
  gate_samples      int      number of samples per averaged waveform
  detected_prf_hz   float    measured pulse repetition frequency
  sample_window_s   float    acquisition window duration per row
  timestamp_iso     str      ISO-8601 UTC session start

Datasets (resizable, chunk-compressed, one row appended per window):
  /waveforms          [N, gate_samples]  float32  coherent average of pulses in window
  /waveforms_std      [N, gate_samples]  float32  per-sample std across the window
  /timestamps         [N]                float64  Unix wall-clock time per window
  /tof_us             [N]                float32  TRACKED ToF — NCC lag vs locked reference (primary)
  /tof_us_peak_median [N]                float32  legacy median of per-pulse argmax ToF (peak-hops)
  /tracking_corr      [N]                float32  NCC peak correlation [-1, 1]
  /tracking_lag_us    [N]                float32  lag from the reference packet (µs)
  /amplitude          [N]                float32  median envelope amplitude
  /energy             [N]                float32  Σ(v²) of the averaged waveform
  /n_pulses           [N]                int32    pulses contributing to each window
  /n_rejected         [N]                int32    pulses missed vs calibrated PRF
  /prf_actual         [N]                float32  kept-pulses-per-second

Why the tracked ToF — and how to use it
---------------------------------------
The legacy /tof_us_peak_median is the median of per-pulse argmax-of-envelope.
When the echo packet contains multiple near-equal peaks (front-wall + internal
reflectors + back-wall), the argmax flips between peaks for tiny coupling /
pressure / SoC changes, so the reported ToF jumps even when the physical echo
barely moved. The tracked ToF locks a reference envelope from the first
`tracking_warmup_n` windows, then reports `ref_tof + NCC_lag` thereafter:
the lag measures global packet-shape motion, which can't peak-hop.

`/tracking_corr` is the NCC peak correlation; values close to 1 mean the
echo packet still matches the locked reference (good coupling), values
falling below ~0.5 mean the packet has changed shape (probe moved to a new
spot, coupling lost) and the tracked ToF should be treated as suspect.
"""

import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import threading
import traceback
from collections import deque
from datetime import datetime, timezone

import h5py
import numpy as np

from ultrasound_battery.hardware.hs5 import HS5StreamPeaks
from ultrasound_battery.cloud.manager import CloudManager
from ultrasound_battery.utils import session_timestamp


SCHEMA_VERSION  = "2.1"
SERVICE_VERSION = "gauging_service v2.1"


def _safe_session_name(name: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("._-")
    return safe or fallback


# ---------------------------------------------------------------------------
# Envelope-tracking helpers (mirror ascan_service.py)
# ---------------------------------------------------------------------------
# Why these exist
# ---------------
# A pulse-echo packet usually has several near-equal peaks (front-wall, several
# internal reflectors, back-wall). Picking ToF as "argmax of the envelope"
# therefore peak-hops: if reflector A is 0.99x the height of reflector B, a
# tiny SoC / pressure / coupling change can flip which peak wins, causing the
# reported ToF to jump by hundreds of nanoseconds even though the physical
# echo barely moved.
#
# Cross-correlating the *whole* envelope against a locked reference packet is
# the standard fix: NCC measures global shape similarity, so the lag tracks
# the entire packet's motion rather than any single peak. Parabolic peak
# refinement gives sub-sample precision (well below the 50 ns sample period
# at fs = 20 MHz).

def _envelope_fft(wf: np.ndarray) -> np.ndarray:
    """Hilbert envelope of a 1-D waveform via numpy FFT (avoids scipy dep)."""
    wf = np.asarray(wf, dtype=np.float64)
    n  = wf.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    fft = np.fft.fft(wf)
    h   = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(fft * h)).astype(np.float32)


def _parabolic_peak(y: np.ndarray, k: int) -> float:
    """3-point parabolic refinement of a peak index k → sub-sample float index."""
    n = len(y)
    if k <= 0 or k >= n - 1:
        return float(k)
    denom = float(y[k - 1] - 2 * y[k] + y[k + 1])
    if abs(denom) < 1e-12:
        return float(k)
    return float(k) + 0.5 * float(y[k - 1] - y[k + 1]) / denom


def _ncc_lag(env: np.ndarray, ref: np.ndarray,
             max_lag_samples: int) -> tuple[float, float]:
    """
    Normalised cross-correlation between two equal-length envelopes.

    Search is restricted to ±max_lag_samples around zero lag so a low-quality
    correlation cannot wrap to a spurious match far from the locked reference.
    Returns (lag_in_samples (sub-sample refined), peak_correlation).
    """
    s = env.astype(np.float64); s -= float(s.mean())
    r = ref.astype(np.float64); r -= float(r.mean())
    norm = float(np.sqrt(np.sum(s * s) * np.sum(r * r)))
    if norm < 1e-12 or s.size == 0 or r.size == 0:
        return 0.0, 0.0
    nc     = np.correlate(s, r, mode="full") / norm
    center = len(r) - 1
    lo     = max(0, center - max_lag_samples)
    hi     = min(len(nc), center + max_lag_samples + 1)
    k      = lo + int(np.argmax(nc[lo:hi]))
    k_ref  = _parabolic_peak(nc, k)
    return float(k_ref - center), float(nc[k])


def _provenance_attrs() -> dict:
    """Code + runtime identity stamped into every HDF5 archive — see ascan_service."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        commit = ""
    return {
        "schema_version": SCHEMA_VERSION,
        "code_version":   SERVICE_VERSION,
        "git_commit":     commit,
        "host":           socket.gethostname(),
        "python_version": sys.version.split()[0],
        "platform":       platform.platform(),
    }


class GaugingService:
    """
    Continuous live ToF stream with on-disk HDF5 archival.

    Lifecycle:
      start(cfg)  — opens HS5, creates HDF5 file, spawns acquisition worker.
      stop()      — signals the worker to exit; HS5 is released and the
                    HDF5 is flushed/closed in the worker's finally block.
                    If cloud archival is enabled, the file is uploaded on stop.
      get_status()— thread-safe snapshot for the API (live ToF + history).
    """

    def __init__(self, cloud: CloudManager | None = None):
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self.running     = False
        self.status      = "IDLE"
        self.cloud       = cloud   # may be None if no cloud archival wanted

        # Default config — overridden per session via start(config)
        self.config = {
            "base_out_dir":     "data/raw/gauging",
            "out_dir":          "data/raw/gauging",
            "session_name":     "",              # used in the HDF5 filename
            "gate_us_start":    25.0,            # gate window start (µs from sync edge)
            "gate_us_end":      50.0,            # gate window end   (µs from sync edge)
            "fs_hz":            20_000_000,      # oscilloscope sample rate (Hz)
            "ch1_range":        2.0,             # CH1 voltage range (V)
            "sample_window_s":  0.3,             # acquisition window per row
            "tracking_warmup_n": 5,              # windows averaged into the locked reference envelope
            "tracking_max_lag_us": 2.0,          # NCC search bound; >> physical ToF drift per window
        }

        # Live readout — written by the worker, read by get_status()
        self._latest_tof  = None
        self._latest_amp  = None
        self._latest_n    = 0
        self._latest_wf   = None
        self._latest_corr = 0.0
        self._tof_history = deque(maxlen=120)

        # Envelope-tracking state — set on first call to start(), then mutated by worker
        self._ref_env             = None   # locked reference envelope (gate_len,)
        self._ref_tof_us          = None   # absolute ToF (µs from sync edge) of reference peak
        self._warmup_envelopes    = []     # accumulates first N envelopes before locking

        # Archival state
        self._h5_path     = None
        self._session_dir = None
        self._session_id  = None
        self._n_windows   = 0      # number of rows written to the HDF5

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Thread-safe snapshot used by the /api/gauge/status endpoint."""
        with self._lock:
            wf = self._latest_wf.tolist() if self._latest_wf is not None else []
            return {
                "running":     self.running,
                "status":      self.status,
                "session_id":  self._session_id,
                "file":        os.path.basename(self._h5_path) if self._h5_path else None,
                "folder":      self._session_dir,
                "tof_us":      self._latest_tof,
                "amp_v":       self._latest_amp,
                "n_pulses":    self._latest_n,
                "n_windows":   self._n_windows,
                "waveform":    wf,
                "tof_history": list(self._tof_history),
                "tracking_corr":   self._latest_corr,
                "tracking_locked": self._ref_env is not None,
                "gate_us":     [self.config["gate_us_start"],
                                self.config["gate_us_end"]],
                "fs_hz":       self.config["fs_hz"],
                "waveform_start_us": 0.0,
            }

    def start(self, new_config: dict | None = None) -> tuple[bool, str]:
        """
        Begin live ToF streaming with HDF5 archival.

        Builds the output filename from session_name (if provided) and a Unix
        timestamp, creates data/raw/gauging/ if needed, and spawns the worker.
        Returns (success, message).
        """
        if self.running:
            return False, "Gauging already active"
        if new_config:
            self.config.update(new_config)

        ts   = session_timestamp()
        name = _safe_session_name(self.config.get("session_name", ""), "session")
        stem = f"gauge_{name}_{ts}"
        base_out = self.config.get("base_out_dir", "data/raw/gauging")
        out = os.path.join(base_out, stem)
        os.makedirs(out, exist_ok=True)
        self.config["out_dir"] = out

        self._stop_event.clear()
        with self._lock:
            self._tof_history.clear()
            self._latest_tof  = None
            self._latest_amp = None
            self._latest_n   = 0
            self._latest_wf  = None
            self._latest_corr = 0.0
            self._n_windows  = 0
            self._session_id = stem
            self._session_dir = out
            self._h5_path    = os.path.join(out, f"{stem}.h5")
            self.status      = "STARTING"
            # Reset tracking state for the new session
            self._ref_env          = None
            self._ref_tof_us       = None
            self._warmup_envelopes = []

        self._write_session_manifest("starting")

        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()
        return True, f"Gauging started: {stem}"

    def stop(self) -> tuple[bool, str]:
        """Signal the worker to exit after the current sample window."""
        if not self.running:
            return False, "Gauging not active"
        self._stop_event.set()
        with self._lock:
            self.status = "STOPPING"
        return True, "Stopping..."

    def _write_session_manifest(self, state: str):
        """Write a small manifest so each gauging folder explains itself."""
        if not self._session_dir:
            return
        manifest = {
            "mode": "gauging",
            "state": state,
            "session_id": self._session_id,
            "session_dir": self._session_dir,
            "h5_file": os.path.basename(self._h5_path) if self._h5_path else None,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "n_windows": self._n_windows,
            "config": dict(self.config),
        }
        manifest_path = os.path.join(self._session_dir, "session_manifest.json")
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            if self.cloud:
                self.cloud.upload_path_async(manifest_path)
        except Exception as e:
            print(f"[GAUGE] Manifest write error: {e}")

    # -------------------------------------------------------------------------
    # Worker thread
    # -------------------------------------------------------------------------

    def _worker(self):
        """
        Continuous ToF acquisition loop with HDF5 append-on-the-fly.

        Each iteration:
          1. acquire `sample_window_s` of pulses (raw waveforms + features)
          2. coherently average the pulse waveforms — preserves phase, gives a
             clean reference-grade waveform per row
          3. take median ToF/amplitude across pulses for outlier rejection
          4. append one row (waveform + scalars) to all HDF5 datasets
          5. flush the file so a crash here loses at most one window
          6. update the in-memory live readout
        """
        hs  = None
        h5f = None
        try:
            hs = HS5StreamPeaks(
                fs_hz        = float(self.config["fs_hz"]),
                gate_us      = (float(self.config["gate_us_start"]),
                                float(self.config["gate_us_end"])),
                ch1_range    = float(self.config["ch1_range"]),
                feature_mode = "envelope",
            ).open()

            with self._lock:
                self.status = "CALIBRATING"
            hs.calibrate_sync(seconds=1.0, verbose=True)

            gate_len = hs.gate_len
            window   = float(self.config["sample_window_s"])

            # Sample-rate-aware parameters for envelope tracking
            dt_us            = 1e6 / float(hs.fs)
            gate_start_us    = float(hs.g0) / float(hs.fs) * 1e6
            warmup_n         = max(1, int(self.config.get("tracking_warmup_n", 5)))
            max_lag_us       = float(self.config.get("tracking_max_lag_us", 2.0))
            max_lag_samples  = max(1, int(round(max_lag_us / dt_us)))

            # ---- Create HDF5 file -----------------------------------------
            h5f = h5py.File(self._h5_path, "w")
            h5f.attrs["session_id"]      = self._session_id
            h5f.attrs["session_dir"]     = self._session_dir
            h5f.attrs["fs_hz"]           = hs.fs
            h5f.attrs["gate_us_start"]   = self.config["gate_us_start"]
            h5f.attrs["gate_us_end"]     = self.config["gate_us_end"]
            h5f.attrs["gate_samples"]    = gate_len
            h5f.attrs["detected_prf_hz"] = hs.detected_prf
            h5f.attrs["sample_window_s"] = window
            h5f.attrs["timestamp_iso"]   = datetime.now(timezone.utc).isoformat()
            h5f.attrs["tracking_method"]    = "ncc_envelope"
            h5f.attrs["tracking_warmup_n"]  = int(self.config.get("tracking_warmup_n", 5))
            h5f.attrs["tracking_max_lag_us"] = float(self.config.get("tracking_max_lag_us", 2.0))
            for k, v in _provenance_attrs().items():
                h5f.attrs[k] = v

            # Coherent-average waveform per acquisition window
            h5f.create_dataset(
                "waveforms",
                shape=(0, gate_len), maxshape=(None, gate_len),
                dtype="f4", chunks=(64, gate_len),
                compression="gzip", compression_opts=4,
            )
            # Per-sample std across the window's pulses (same shape as /waveforms).
            # Doubles waveform storage (~MB/hour) but enables noise-aware re-analysis
            # without keeping individual raw pulses.
            h5f.create_dataset(
                "waveforms_std",
                shape=(0, gate_len), maxshape=(None, gate_len),
                dtype="f4", chunks=(64, gate_len),
                compression="gzip", compression_opts=4,
            )
            # Scalar feature time-series. n_rejected estimates pulses missed
            # vs the calibrated PRF; prf_actual is kept-pulses-per-second.
            #   /tof_us            tracked ToF — NCC lag against locked reference (primary)
            #   /tof_us_peak_median raw median of per-pulse argmax-of-envelope (legacy,
            #                       kept for transparency; can peak-hop)
            #   /tracking_corr     NCC peak correlation [-1, 1] — quality indicator
            #   /tracking_lag_us   lag from the reference packet (µs)
            for ds_name, dtype in [
                ("timestamps",         "f8"),
                ("tof_us",             "f4"),
                ("tof_us_peak_median", "f4"),
                ("tracking_corr",      "f4"),
                ("tracking_lag_us",    "f4"),
                ("amplitude",          "f4"),
                ("energy",             "f4"),
                ("n_pulses",           "i4"),
                ("n_rejected",         "i4"),
                ("prf_actual",         "f4"),
            ]:
                h5f.create_dataset(ds_name, shape=(0,), maxshape=(None,),
                                   dtype=dtype, chunks=(2048,))

            with self._lock:
                self.status = "GAUGING"

            # ---- Acquisition loop -----------------------------------------
            while not self._stop_event.is_set():
                _, aa, tf, _, wf, full_wf = hs.acquire_peaks(
                    duration_s=window,
                    save_waveforms=True,
                    save_full_waveforms=True,
                )
                if wf is None or wf.shape[0] == 0:
                    continue   # no pulses this window — keep last live value

                # Coherent average preserves phase across pulses → clean reference.
                # Std across pulses is the noise-floor reference for each window.
                avg_wf = np.mean(wf, axis=0).astype(np.float32)
                std_wf = np.std(wf, axis=0).astype(np.float32)
                live_wf = (np.mean(full_wf, axis=0).astype(np.float32)
                           if full_wf is not None and full_wf.shape[0] > 0
                           else avg_wf)

                # Legacy raw value: median of per-pulse argmax-of-envelope.
                # Kept for transparency, but it is the one that peak-hops.
                peak_median_tof = float(np.median(tf))
                amp_med         = float(np.median(aa))

                # ---- Envelope-tracked ToF (primary) ----------------------
                # Envelope of the coherent average — phase-stable, low-noise.
                v   = (avg_wf - float(np.mean(avg_wf))).astype(np.float32)
                env = _envelope_fft(v)

                if self._ref_env is None:
                    # Warm-up: accumulate envelopes, lock reference once we have N.
                    self._warmup_envelopes.append(env)
                    if len(self._warmup_envelopes) >= warmup_n:
                        ref_env_locked = np.mean(
                            np.stack(self._warmup_envelopes, axis=0), axis=0
                        ).astype(np.float32)
                        k0          = int(np.argmax(ref_env_locked))
                        k0_refined  = _parabolic_peak(ref_env_locked, k0)
                        self._ref_env    = ref_env_locked
                        self._ref_tof_us = gate_start_us + k0_refined * dt_us
                        self._warmup_envelopes = []   # release memory
                    # During warm-up, expose the median value so the user sees
                    # *something* — but flag tracking as unlocked via corr=0.
                    tracked_tof  = peak_median_tof
                    tracked_corr = 0.0
                    lag_us       = 0.0
                else:
                    # Locked: lag tracks the whole packet shape, not a single peak.
                    lag_samples, tracked_corr = _ncc_lag(
                        env, self._ref_env, max_lag_samples
                    )
                    lag_us      = lag_samples * dt_us
                    tracked_tof = self._ref_tof_us + lag_us

                tof_med = tracked_tof   # /tof_us = the tracked, peak-hop-immune ToF

                # Energy of the DC-removed averaged waveform
                eng = float(np.dot(v, v))

                ts  = time.time()
                n_p = int(wf.shape[0])

                # Acquisition-health metrics: deficit vs calibrated PRF
                expected   = int(round(hs.detected_prf * window))
                n_rejected = max(0, expected - n_p)
                prf_actual = float(n_p / max(window, 1e-9))

                # Append one row to all datasets
                n = h5f["waveforms"].shape[0]
                h5f["waveforms"].resize((n + 1, gate_len))
                h5f["waveforms"][n] = avg_wf
                h5f["waveforms_std"].resize((n + 1, gate_len))
                h5f["waveforms_std"][n] = std_wf
                for ds_name, val in [
                    ("timestamps",         ts),
                    ("tof_us",             tof_med),
                    ("tof_us_peak_median", peak_median_tof),
                    ("tracking_corr",      tracked_corr),
                    ("tracking_lag_us",    lag_us),
                    ("amplitude",          amp_med),
                    ("energy",             eng),
                    ("n_pulses",           n_p),
                    ("n_rejected",         n_rejected),
                    ("prf_actual",         prf_actual),
                ]:
                    h5f[ds_name].resize((n + 1,))
                    h5f[ds_name][n] = val

                # Flush every window — at 0.3 s/window, worst-case data loss
                # on a hard crash is one row (~600 µs of waveform data)
                h5f.flush()

                with self._lock:
                    self._latest_tof  = tof_med
                    self._latest_amp  = amp_med
                    self._latest_n    = n_p
                    self._latest_wf   = live_wf
                    self._latest_corr = float(tracked_corr)
                    self._n_windows   = n + 1
                    self._tof_history.append(round(tof_med, 4))

            with self._lock:
                self.status = "IDLE"
            self._write_session_manifest("stopped")

        except Exception as e:
            with self._lock:
                self.status = f"ERROR: {e}"
            self._write_session_manifest("error")
            print(f"[GAUGE ERROR] {e}")
            traceback.print_exc()
        finally:
            # Best-effort close: flush + close HDF5, then release HS5
            if h5f:
                try: h5f.flush(); h5f.close()
                except Exception: pass
            if hs:
                hs.close()

            # Async upload to Box if cloud is enabled (no-op otherwise)
            if self.cloud and self._h5_path and os.path.exists(self._h5_path):
                self.cloud.upload_path_async(self._h5_path)

            self.running = False
