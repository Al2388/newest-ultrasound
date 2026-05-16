"""
A-Scan Monitor Service
=======================
Continuous single-point ultrasonic monitoring for battery cycling experiments.

Purpose
-------
The probe is clamped over a fixed location on the battery while the battery
is cycled (charged and discharged) through one or more SOC levels. This service
captures and archives the received waveform at regular intervals so that slow
changes in the acoustic path — caused by electrode swelling, electrolyte
redistribution, and SEI growth — can be tracked over time.

Key quantities monitored
------------------------
  Amplitude — envelope value at the tracked echo position (V).
              Decreases as acoustic attenuation increases (e.g. gas evolution,
              delamination, or electrolyte drying at high SOC).
  ToF       — time-of-flight of the *tracked* echo packet, with two reports:
                tof_us            position within the gate (µs)
                tof_us_absolute   = gate_us_start + tof_us, from sync edge (µs)
              Tracking follows the reference echo packet established from the
              first `tracking_ref_n` snapshots; new waveforms are aligned by
              constrained cross-correlation, so the same wave packet is
              followed even when SOC/temperature drift moves it or when a
              competing reflector becomes momentarily larger.
              `tof_us_envelope` is the legacy argmax-of-envelope ToF, kept
              as a diagnostic so peak-hopping artifacts remain visible.
  Energy    — sum of squared DC-removed samples (V²·samples). Correlates with
              total transmitted acoustic power.

Coherent averaging
------------------
All raw pulses received within one snapshot interval are coherently averaged
(np.mean over the pulse axis). Coherent averaging preserves the signal phase,
which is essential for accurate sub-sample ToF tracking. It also improves SNR
by √N where N is the number of pulses averaged (~PRF × interval_s).

HDF5 file layout  (data/raw/ascan/<session_id>/<session_id>.h5)
-----------------------------------------------------------
Root attributes (session-level metadata):
  session_id, session_dir, fs_hz, gate_us_start, gate_us_end, gate_samples,
  detected_prf_hz, interval_s, raw_mode, raw_window_s, raw_decimate_k,
  tracking_ref_n, tracking_max_lag_us, tracking_reference_ready_at_snapshot,
  tracking_reference_peak_us, schema_version, code_version, git_commit, host,
  python_version, platform, timestamp_iso, and (when triggered)
  coupling_warning, coupling_warning_at_s, coupling_baseline_v.

Per-snapshot datasets (all resizable, chunk-compressed, one row per snapshot):
  /waveforms              [N, gate_samples]  float32  coherent average
  /waveforms_std          [N, gate_samples]  float32  per-sample std across pulses
  /timestamps             [N]                float64  Unix wall-clock time
  /amplitude              [N]                float32  envelope @ tracked peak (V)
  /amplitude_envelope     [N]                float32  envelope argmax peak (V) — legacy
  /tof_us                 [N]                float32  tracked ToF, gate-relative (µs)
  /tof_us_absolute        [N]                float32  tracked ToF from sync edge (µs)
  /tof_us_envelope        [N]                float32  legacy argmax-envelope ToF (µs)
  /tracking_lag_samples   [N]                float32  xcorr lag from reference (samples)
  /tracking_corr          [N]                float32  normalized xcorr coefficient
  /tracking_method        [N]                int8     0=envelope fallback, 1=xcorr
  /energy                 [N]                float32  Σ(v²) of averaged waveform
  /n_averaged             [N]                int32    pulses averaged
  /n_rejected             [N]                int32    pulses expected but not received
  /prf_actual             [N]                float32  kept-pulses-per-second

Optional /raw_* datasets (only present when raw_mode != 'off'):
  /raw_waveforms          [M, gate_samples]  float32  individual pulse waveforms
  /raw_timestamps         [M]                float64
  /raw_snapshot_index     [M]                int32
  /raw_amplitude          [M]                float32
  /raw_tof_us             [M]                float32
  /raw_energy             [M]                float32

SOC/event annotations
---------------------
Stored in data/raw/ascan/<session_id>/<session_id>_annotations.json — a JSON array,
one object per event. Each object has: timestamp, timestamp_iso, elapsed_s,
soc_pct, label, snapshot_idx.
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

from ultrasound_battery.hardware.hs5 import HS5StreamPeaks, envelope_hilbert
from ultrasound_battery.cloud.manager import CloudManager
from ultrasound_battery.utils import session_timestamp


# Schema version — bump when on-disk layout changes in a way readers must branch on.
# 1.x = legacy (save_raw_waveforms boolean, no /waveforms_std, no quality metadata).
# 2.0 = raw_mode tiered retention, /waveforms_std present, n_rejected/prf_actual scalars,
#       provenance attributes (git_commit, schema_version, code_version, host).
# 2.1 = live reference-envelope tracking; tof_us is now the *tracked* echo position,
#       tof_us_absolute and tof_us_envelope are added; tracking_lag/corr/method captured.
# 2.2 = CH1 range metadata plus clipping diagnostics per snapshot.
SCHEMA_VERSION  = "2.2"
SERVICE_VERSION = "ascan_service v2.2"


# =============================================================================
# Helpers — tracking primitives
# =============================================================================

def _parabolic_peak(y: np.ndarray, k: int) -> float:
    """Sub-sample peak location from a three-point quadratic fit around index k.

    Standard interpolation trick: fit y = ax² + bx + c through (k-1, k, k+1)
    and return the vertex x-coordinate. The fractional offset delta is clamped
    to [-1, 1] to suppress occasional outliers when the curvature is tiny.
    """
    if y.size < 3 or k <= 0 or k >= y.size - 1:
        return float(k)
    y0, y1, y2 = float(y[k - 1]), float(y[k]), float(y[k + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(k)
    delta = 0.5 * (y0 - y2) / denom
    return float(k) + max(-1.0, min(1.0, delta))


def _sample_at_fractional(y: np.ndarray, x: float) -> float:
    """Linear interpolation of y at fractional sample index x, edge-clamped."""
    if y.size == 0:
        return float("nan")
    if x <= 0:
        return float(y[0])
    if x >= y.size - 1:
        return float(y[-1])
    i = int(np.floor(x))
    frac = float(x - i)
    return float((1.0 - frac) * y[i] + frac * y[i + 1])


def _track_envelope_xcorr(
    env: np.ndarray,
    ref_env: np.ndarray,
    ref_peak_sample: float,
    max_lag_samples: int,
) -> tuple[float, float, float]:
    """Track the reference echo packet by constrained envelope cross-correlation.

    Returns (tracked_peak_sample, lag_samples, normalized_correlation).
    The search is restricted to |lag| <= max_lag_samples so the xcorr cannot
    lock onto a spurious correlation peak far from where the echo physically is.
    NaN-tuple is returned only when both arrays are degenerate; otherwise the
    function falls back to lag=0 with NaN correlation.
    """
    if env.size == 0 or ref_env.size == 0 or env.size != ref_env.size:
        return float("nan"), float("nan"), float("nan")

    max_lag = max(1, min(int(max_lag_samples), env.size // 2))
    env_zm = env.astype(np.float32, copy=False) - float(np.mean(env))
    ref_zm = ref_env.astype(np.float32, copy=False) - float(np.mean(ref_env))

    denom = float(np.linalg.norm(env_zm) * np.linalg.norm(ref_zm))
    if denom <= 1e-12:
        return ref_peak_sample, 0.0, float("nan")

    xc = np.correlate(env_zm, ref_zm, mode="full")
    center = env.size - 1
    lo = center - max_lag
    hi = center + max_lag + 1
    window = xc[lo:hi]
    k_local = int(np.argmax(window))
    k_abs = lo + k_local
    k_refined = _parabolic_peak(xc, k_abs)
    lag_samples = float(k_refined - center)
    corr = float(xc[k_abs] / denom)
    return float(ref_peak_sample + lag_samples), lag_samples, corr


# =============================================================================
# Helpers — file naming, provenance, config translation
# =============================================================================

def _safe_session_name(name: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("._-")
    return safe or fallback


def _provenance_attrs() -> dict:
    """Snapshot of code + runtime identity, embedded in every HDF5 archive.

    Stored as root attributes so a file from any past session can be matched
    back to the exact code that wrote it (git_commit) and the host that ran
    the experiment. Falls back to empty strings if git or system calls fail —
    never raises, since this is metadata, not data.
    """
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


def _normalize_raw_mode(cfg: dict) -> dict:
    """Translate legacy `save_raw_waveforms: bool` into the new `raw_mode` field.

    Accepts either form so older clients keep working. If both are present,
    `raw_mode` wins. Mutates and returns cfg.
    """
    if "save_raw_waveforms" in cfg and "raw_mode" not in cfg:
        cfg["raw_mode"] = "full" if cfg.pop("save_raw_waveforms") else "off"
    elif "save_raw_waveforms" in cfg:
        cfg.pop("save_raw_waveforms")
    return cfg


# =============================================================================
# Service
# =============================================================================

class AScanService:
    """
    Manages the full lifecycle of one continuous A-scan monitoring session.

    One session = one battery cycling run, one HDF5 file, one annotations JSON.
    The HDF5 file is opened once and held open for the entire session so that
    each snapshot can be appended with a single resize + write operation,
    without the overhead of opening and closing the file thousands of times.

    Thread safety
    -------------
    The acquisition loop runs in a daemon worker thread. All shared state
    (status, progress, _history, _latest_wf, _annotations, _n_snapshots)
    is protected by self._lock. self._stop_event is set by stop_session()
    and polled at the top of each snapshot loop iteration.
    """

    def __init__(self, cloud: CloudManager):
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self.running      = False
        self.status       = "IDLE"
        self.progress     = {"msg": "Idle", "n_snapshots": 0, "duration_s": 0}
        self.cloud        = cloud

        # Default acquisition parameters — overridden per session via start_session()
        self.config = {
            "interval_s":    1.0,           # seconds between averaged snapshots
            "base_out_dir":  "data/raw/ascan",  # parent directory for session folders
            "out_dir":       "data/raw/ascan",  # output directory (relative to CWD)
            "session_name":  "",            # label for the HDF5 filename (optional)

            # Acquisition gate (CH1 window around each sync edge). Intentionally
            # wider than the analysis window: must contain the echo across the
            # full range of expected SOC-driven and thermal ToF drift (~±2 µs in
            # practice). Reference-envelope tracking picks the right packet
            # within this wider waveform even when reflectors compete.
            "gate_us_start": 25.0,
            "gate_us_end":   50.0,

            "fs_hz":         20_000_000,
            # Use extra headroom by default. If the echo repeatedly touches the
            # ADC rails, peak amplitude/energy are no longer quantitative.
            "ch1_range":     2.0,

            # Reference-envelope tracking settings
            "tracking_ref_n":      60,      # opening snapshots used as the reference
            "tracking_max_lag_us": 2.0,     # max allowed drift from reference (µs)

            # Raw pulse retention. Coherent averaging means /waveforms already
            # captures the science at √N better SNR; default to discarding raw.
            #   off       — average + features only (smallest, ~kB/hour)
            #   window    — keep every raw pulse for the first raw_window_s seconds
            #               (verifies averaging worked), then averaged-only
            #   decimated — keep every raw_decimate_k-th pulse for the whole session
            #   full      — keep every raw pulse (largest, ~GB/hour at PRF≈1 kHz)
            "raw_mode":       "off",
            "raw_window_s":   60.0,
            "raw_decimate_k": 100,
        }

        # Rolling deque of the last 500 feature-point dicts — feeds the live charts.
        self._history     = deque(maxlen=500)
        self._latest_wf   = None
        self._annotations = []
        self._h5_path     = None
        self._ann_path    = None
        self._session_dir = None
        self._session_id  = None
        self._session_t0  = 0.0
        self._n_snapshots = 0
        self._n_raw_pulses = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Thread-safe snapshot used by /api/ascan/status."""
        with self._lock:
            wf = self._latest_wf.tolist() if self._latest_wf is not None else []
            return {
                "status":      self.status,
                "progress":    dict(self.progress),
                "waveform":    wf,
                "history":     list(self._history)[-200:],
                "annotations": list(self._annotations),
                "session_id":  self._session_id,
                "file":        os.path.basename(self._h5_path) if self._h5_path else None,
                "folder":      self._session_dir,
                "gate_us":     [self.config["gate_us_start"], self.config["gate_us_end"]],
                "fs_hz":       self.config["fs_hz"],
                "ch1_range":   float(self.config.get("ch1_range", 2.0)),
                "raw_mode":       str(self.config.get("raw_mode", "off")),
                "raw_window_s":   float(self.config.get("raw_window_s", 60.0)),
                "raw_decimate_k": int(self.config.get("raw_decimate_k", 100)),
                "tracking_ref_n":      int(self.config.get("tracking_ref_n", 60)),
                "tracking_max_lag_us": float(self.config.get("tracking_max_lag_us", 2.0)),
                "n_raw_pulses": self._n_raw_pulses,
            }

    def start_session(self, new_config: dict | None = None) -> tuple[bool, str]:
        """Start a new A-scan monitoring session. Returns (success, message)."""
        if self.running:
            return False, "Session already active"
        if new_config:
            self.config.update(_normalize_raw_mode(dict(new_config)))

        ts   = session_timestamp()
        name = _safe_session_name(self.config.get("session_name", ""), "session")
        stem = f"ascan_{name}_{ts}" if name else f"ascan_{ts}"
        base_out = self.config.get("base_out_dir", "data/raw/ascan")
        out = os.path.join(base_out, stem)
        os.makedirs(out, exist_ok=True)
        self.config["out_dir"] = out

        self._stop_event.clear()
        with self._lock:
            self._history.clear()
            self._annotations.clear()
            self._latest_wf    = None
            self._n_snapshots  = 0
            self._n_raw_pulses = 0
            self._session_id   = stem
            self._session_dir  = out
            self._h5_path      = os.path.join(out, f"{stem}.h5")
            self._ann_path     = os.path.join(out, f"{stem}_annotations.json")
            self._session_t0   = time.time()
            self.status        = "RECORDING"
            self.progress      = {"msg": "Starting...", "n_snapshots": 0, "duration_s": 0}
        self._write_session_manifest("starting")

        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()
        return True, f"Session started: {stem}"

    def stop_session(self) -> tuple[bool, str]:
        """Signal the worker to finish the current snapshot interval and exit cleanly."""
        if not self.running:
            return False, "No active session"
        self._stop_event.set()
        with self._lock:
            self.status = "STOPPING"
        return True, "Stopping..."

    def mark_event(self, soc_pct: float | None, label: str = "") -> tuple[bool, str]:
        """Create a timestamped SOC/event annotation."""
        if not self.running:
            return False, "No active session"

        with self._lock:
            idx = self._n_snapshots
            t0  = self._session_t0

        ann = {
            "timestamp":     time.time(),
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "elapsed_s":     round(time.time() - t0, 2),
            "soc_pct":       float(soc_pct) if soc_pct is not None else None,
            "label":         label,
            "snapshot_idx":  idx,
        }

        with self._lock:
            self._annotations.append(ann)
            ann_copy = list(self._annotations)

        if self._ann_path:
            try:
                with open(self._ann_path, "w") as f:
                    json.dump(ann_copy, f, indent=2)
                if self.cloud:
                    self.cloud.upload_path_async(self._ann_path)
            except Exception as e:
                print(f"[ASCAN] Annotation write error: {e}")

        soc_str = f"SOC={soc_pct:.1f}%" if soc_pct is not None else "no SOC"
        return True, f"Marked {soc_str} '{label}' at snapshot {idx}"

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _set(self, status: str | None = None, progress: dict | None = None):
        """Update status and/or progress fields under the lock."""
        with self._lock:
            if status   is not None: self.status = status
            if progress is not None: self.progress.update(progress)

    def _write_session_manifest(self, state: str):
        if not self._session_dir:
            return
        manifest = {
            "mode": "ascan",
            "state": state,
            "session_id": self._session_id,
            "session_dir": self._session_dir,
            "h5_file": os.path.basename(self._h5_path) if self._h5_path else None,
            "annotations_file": os.path.basename(self._ann_path) if self._ann_path else None,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "progress": dict(self.progress),
            "n_snapshots": self._n_snapshots,
            "n_raw_pulses": self._n_raw_pulses,
            "config": dict(self.config),
        }
        manifest_path = os.path.join(self._session_dir, "session_manifest.json")
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            if self.cloud:
                self.cloud.upload_path_async(manifest_path)
        except Exception as e:
            print(f"[ASCAN] Manifest write error: {e}")

    # -------------------------------------------------------------------------
    # Worker thread
    # -------------------------------------------------------------------------

    def _worker(self):
        """
        A-scan monitoring acquisition loop — runs in a daemon thread.

        Execution flow
        --------------
        1. Open HS5 and run PRF auto-calibration.
        2. Create the HDF5 file with resizable, chunked, gzip-compressed datasets.
        3. Loop until stop_session() is called:
             a. Acquire raw pulses for interval_s seconds.
             b. Coherently average all pulses → one averaged waveform.
             c. Compute envelope; argmax gives a legacy diagnostic ToF.
             d. While snapshots < tracking_ref_n: accumulate envelopes into the
                reference; once ready, freeze it and switch to xcorr tracking.
             e. After reference is ready: cross-correlate current envelope
                against reference (constrained to ±max_lag_samples), report
                tracked peak position + lag + correlation.
             f. Append one row to every per-snapshot dataset.
             g. (raw_mode != off) Archive selected raw pulses.
             h. Coupling watchdog: detect amplitude collapse early.
             i. Flush the HDF5 file every ~10 s.
        4. Final flush + close HDF5.
        5. Upload HDF5 to S3 if cloud is enabled.
        """
        cfg = self.config
        hs  = None
        h5f = None

        try:
            hs = HS5StreamPeaks(
                fs_hz        = float(cfg["fs_hz"]),
                gate_us      = (float(cfg["gate_us_start"]), float(cfg["gate_us_end"])),
                ch1_range    = float(cfg["ch1_range"]),
                feature_mode = "envelope",
            ).open()
            hs.calibrate_sync(seconds=1.0, verbose=True)

            gate_len = hs.gate_len
            interval = float(cfg["interval_s"])

            # ---- Create HDF5 file -------------------------------------------
            h5f = h5py.File(self._h5_path, "w")

            # Root-level attributes — session params + provenance + tracking config
            h5f.attrs["session_id"]      = self._session_id
            h5f.attrs["session_dir"]     = self._session_dir
            h5f.attrs["fs_hz"]           = hs.fs
            h5f.attrs["gate_us_start"]   = cfg["gate_us_start"]
            h5f.attrs["gate_us_end"]     = cfg["gate_us_end"]
            h5f.attrs["gate_samples"]    = gate_len
            h5f.attrs["ch1_range_requested_v"] = float(cfg["ch1_range"])
            h5f.attrs["ch1_range_actual_v"]    = float(hs.range)
            clip_threshold_v = 0.495 * float(hs.range)
            h5f.attrs["clip_threshold_v"]      = float(clip_threshold_v)
            h5f.attrs["detected_prf_hz"] = hs.detected_prf
            h5f.attrs["interval_s"]      = interval
            h5f.attrs["raw_mode"]        = str(cfg.get("raw_mode", "off"))
            h5f.attrs["raw_window_s"]    = float(cfg.get("raw_window_s", 60.0))
            h5f.attrs["raw_decimate_k"]  = int(cfg.get("raw_decimate_k", 100))
            h5f.attrs["tracking_ref_n"]      = int(cfg.get("tracking_ref_n", 60))
            h5f.attrs["tracking_max_lag_us"] = float(cfg.get("tracking_max_lag_us", 2.0))
            h5f.attrs["timestamp_iso"]   = datetime.now(timezone.utc).isoformat()
            for k, v in _provenance_attrs().items():
                h5f.attrs[k] = v

            # Coherent average per snapshot — the primary scientific signal
            h5f.create_dataset(
                "waveforms",
                shape=(0, gate_len), maxshape=(None, gate_len),
                dtype="f4", chunks=(100, gate_len),
                compression="gzip", compression_opts=4,
            )
            # Per-sample std across the pulses in each snapshot — noise reference
            h5f.create_dataset(
                "waveforms_std",
                shape=(0, gate_len), maxshape=(None, gate_len),
                dtype="f4", chunks=(100, gate_len),
                compression="gzip", compression_opts=4,
            )

            # Scalar feature time-series
            for ds_name, dtype in [
                ("timestamps",            "f8"),
                ("amplitude",             "f4"),
                ("amplitude_envelope",    "f4"),
                ("tof_us",                "f4"),
                ("tof_us_absolute",       "f4"),
                ("tof_us_envelope",       "f4"),
                ("tracking_lag_samples",  "f4"),
                ("tracking_corr",         "f4"),
                ("tracking_method",       "i1"),
                ("energy",                "f4"),
                ("clip_fraction",         "f4"),
                ("raw_clip_fraction",     "f4"),
                ("waveform_min_v",        "f4"),
                ("waveform_max_v",        "f4"),
                ("n_averaged",            "i4"),
                ("n_rejected",            "i4"),
                ("prf_actual",            "f4"),
            ]:
                h5f.create_dataset(ds_name, shape=(0,), maxshape=(None,),
                                   dtype=dtype, chunks=(5000,))

            # Optional raw pulse archive — controlled by raw_mode
            raw_mode       = str(cfg.get("raw_mode", "off"))
            raw_window_s   = float(cfg.get("raw_window_s", 60.0))
            raw_decimate_k = max(1, int(cfg.get("raw_decimate_k", 100)))
            if raw_mode != "off":
                chunk_rows = 1024 if raw_mode == "full" else 256
                h5f.create_dataset(
                    "raw_waveforms",
                    shape=(0, gate_len), maxshape=(None, gate_len),
                    dtype="f4", chunks=(chunk_rows, gate_len),
                    compression="gzip", compression_opts=6,
                )
                for ds_name, dtype in [
                    ("raw_timestamps",     "f8"),
                    ("raw_snapshot_index", "i4"),
                    ("raw_amplitude",      "f4"),
                    ("raw_tof_us",         "f4"),
                    ("raw_energy",         "f4"),
                ]:
                    h5f.create_dataset(ds_name, shape=(0,), maxshape=(None,),
                                       dtype=dtype, chunks=(10000,))

            self._set(status="RECORDING", progress={"msg": "Recording..."})

            # Flush every ~10 s — bounds data loss on hard crash to one window
            flush_every = max(1, int(10 / interval))
            flush_count = 0
            wall_clock_offset = time.time() - time.perf_counter()

            # Coupling-loss watchdog — see comments where it fires.
            COUPLING_BASELINE_N      = 60
            COUPLING_DROP_FRAC       = 0.30
            COUPLING_STREAK_REQUIRED = 60
            coupling_baseline   = None
            coupling_window     = []
            coupling_streak     = 0
            coupling_warned     = False
            clipping_warned     = False
            CLIP_WARN_FRAC      = 0.001   # warn if >=0.1% of samples hit rails

            # Reference-envelope tracker
            tracking_ref_n           = max(1, int(cfg.get("tracking_ref_n", 60)))
            tracking_max_lag_us      = max(0.05, float(cfg.get("tracking_max_lag_us", 2.0)))
            tracking_max_lag_samples = max(1, int(round(tracking_max_lag_us * 1e-6 * hs.fs)))
            ref_env_accum  = []
            ref_env        = None
            ref_peak_sample = float("nan")

            # ---- Main acquisition loop -------------------------------------
            while not self._stop_event.is_set():
                tt, aa, tf, ee, wf = hs.acquire_peaks(duration_s=interval,
                                                       save_waveforms=True)
                if wf is None or wf.shape[0] == 0:
                    continue   # no pulses detected this interval

                # Coherent average + per-sample std
                avg_wf = np.mean(wf, axis=0).astype(np.float32)
                std_wf = np.std(wf, axis=0).astype(np.float32)
                n_avg  = int(wf.shape[0])
                wf_min = float(np.min(avg_wf))
                wf_max = float(np.max(avg_wf))
                clip_fraction = float(np.mean(np.abs(avg_wf) >= clip_threshold_v))
                raw_clip_fraction = float(np.mean(np.abs(wf) >= clip_threshold_v))

                # Acquisition-health metrics
                expected   = int(round(hs.detected_prf * interval))
                n_rejected = max(0, expected - n_avg)
                prf_actual = float(n_avg / max(interval, 1e-9))

                # Envelope of the averaged waveform — used for both the legacy
                # diagnostic ToF and as input to the cross-correlation tracker.
                v   = (avg_wf - float(np.mean(avg_wf))).astype(np.float32)
                env = envelope_hilbert(v)
                pk_env         = int(np.argmax(env))
                pk_env_refined = _parabolic_peak(env, pk_env)
                amp_env = float(env[pk_env])
                tof_env = float(pk_env_refined / hs.fs * 1e6)   # legacy argmax ToF
                eng     = float(np.dot(v, v))

                # Reference build / tracked feature extraction
                if ref_env is None:
                    # Accumulate opening envelopes; while accumulating, report
                    # the envelope-argmax as the tracked peak so dashboard charts
                    # still show something sensible.
                    ref_env_accum.append(env.copy())
                    tracked_peak    = pk_env_refined
                    tracking_lag    = 0.0
                    tracking_corr   = float("nan")
                    tracking_method = 0   # envelope fallback while reference builds
                    if len(ref_env_accum) >= tracking_ref_n:
                        ref_env = np.mean(np.stack(ref_env_accum, axis=0),
                                          axis=0).astype(np.float32)
                        ref_peak_sample = _parabolic_peak(ref_env, int(np.argmax(ref_env)))
                        h5f.attrs["tracking_reference_ready_at_snapshot"] = int(h5f["waveforms"].shape[0])
                        h5f.attrs["tracking_reference_peak_us"] = float(ref_peak_sample / hs.fs * 1e6)
                        ref_env_accum.clear()
                        print(f"[ASCAN] tracking reference ready after {tracking_ref_n} snapshots "
                              f"(ref peak @ {h5f.attrs['tracking_reference_peak_us']:.3f} µs in gate)")
                else:
                    tracked_peak, tracking_lag, tracking_corr = _track_envelope_xcorr(
                        env, ref_env, ref_peak_sample, tracking_max_lag_samples,
                    )
                    tracking_method = 1   # reference-envelope cross-correlation
                    if not np.isfinite(tracked_peak):
                        # Degenerate input (e.g. zero waveform) — fall back gracefully
                        tracked_peak    = pk_env_refined
                        tracking_lag    = 0.0
                        tracking_corr   = float("nan")
                        tracking_method = 0

                tracked_peak = float(np.clip(tracked_peak, 0.0, gate_len - 1.0))
                amp     = _sample_at_fractional(env, tracked_peak)
                tof     = float(tracked_peak / hs.fs * 1e6)
                tof_abs = float(float(cfg["gate_us_start"]) + tof)
                ts      = time.time()
                elapsed = ts - self._session_t0

                # Append one row to each per-snapshot dataset
                n = h5f["waveforms"].shape[0]
                h5f["waveforms"].resize((n + 1, gate_len))
                h5f["waveforms"][n] = avg_wf
                h5f["waveforms_std"].resize((n + 1, gate_len))
                h5f["waveforms_std"][n] = std_wf

                for ds_name, val in [
                    ("timestamps",            ts),
                    ("amplitude",             amp),
                    ("amplitude_envelope",    amp_env),
                    ("tof_us",                tof),
                    ("tof_us_absolute",       tof_abs),
                    ("tof_us_envelope",       tof_env),
                    ("tracking_lag_samples",  tracking_lag),
                    ("tracking_corr",         tracking_corr),
                    ("tracking_method",       tracking_method),
                    ("energy",                eng),
                    ("clip_fraction",         clip_fraction),
                    ("raw_clip_fraction",     raw_clip_fraction),
                    ("waveform_min_v",        wf_min),
                    ("waveform_max_v",        wf_max),
                    ("n_averaged",            n_avg),
                    ("n_rejected",            n_rejected),
                    ("prf_actual",            prf_actual),
                ]:
                    h5f[ds_name].resize((n + 1,))
                    h5f[ds_name][n] = val

                # Optional raw pulse archive based on raw_mode
                raw_idx = None
                if raw_mode == "full":
                    raw_idx = np.arange(n_avg, dtype=np.int64)
                elif raw_mode == "window" and elapsed < raw_window_s:
                    raw_idx = np.arange(n_avg, dtype=np.int64)
                elif raw_mode == "decimated":
                    raw_idx = np.arange(0, n_avg, raw_decimate_k, dtype=np.int64)

                if raw_idx is not None and raw_idx.size > 0:
                    m  = int(raw_idx.size)
                    r0 = h5f["raw_waveforms"].shape[0]
                    r1 = r0 + m
                    h5f["raw_waveforms"].resize((r1, gate_len))
                    h5f["raw_waveforms"][r0:r1] = wf[raw_idx].astype(np.float32, copy=False)
                    for ds_name, val in [
                        ("raw_timestamps",     tt[raw_idx] + wall_clock_offset),
                        ("raw_snapshot_index", np.full(m, n, dtype=np.int32)),
                        ("raw_amplitude",      aa[raw_idx]),
                        ("raw_tof_us",         tf[raw_idx]),
                        ("raw_energy",         ee[raw_idx]),
                    ]:
                        h5f[ds_name].resize((r1,))
                        h5f[ds_name][r0:r1] = val

                flush_count += 1
                if flush_count >= flush_every:
                    h5f.flush()
                    flush_count = 0

                # Clipping watchdog. Any sustained rail hits mean amplitude and
                # energy are no longer fully quantitative, even though ToF may
                # remain usable. The fix is to increase CH1 range or reduce gain.
                if clip_fraction >= CLIP_WARN_FRAC or raw_clip_fraction >= CLIP_WARN_FRAC:
                    if not clipping_warned:
                        msg = (f"WARNING: CH1 clipping detected "
                               f"(avg={clip_fraction*100:.2f}%, "
                               f"raw={raw_clip_fraction*100:.2f}% samples above "
                               f"{clip_threshold_v:.3f} V threshold). "
                               f"Increase ch1_range or reduce receiver gain.")
                        print(f"[ASCAN] {msg}")
                        h5f.attrs["clipping_warning"] = True
                        h5f.attrs["clipping_warning_at_s"] = float(elapsed)
                        h5f.attrs["clipping_warning_threshold_v"] = float(clip_threshold_v)
                        clipping_warned = True
                    with self._lock:
                        self.progress["clipping_warning"] = True
                        self.progress["clip_fraction_pct"] = round(clip_fraction * 100.0, 3)
                        self.progress["raw_clip_fraction_pct"] = round(raw_clip_fraction * 100.0, 3)

                # Coupling-loss watchdog — uses the tracked amplitude, which is
                # already sampling the envelope at the same physical position the
                # reference echo lives at. A genuine coupling failure will drop
                # this number sharply, even if some other reflection still rings.
                if coupling_baseline is None:
                    coupling_window.append(amp)
                    if len(coupling_window) >= COUPLING_BASELINE_N:
                        coupling_baseline = float(np.mean(coupling_window))
                        print(f"[ASCAN] coupling baseline established: "
                              f"{coupling_baseline:.3f} V over first "
                              f"{COUPLING_BASELINE_N} snapshots")
                else:
                    if amp < COUPLING_DROP_FRAC * coupling_baseline:
                        coupling_streak += 1
                    else:
                        coupling_streak = 0
                    if coupling_streak >= COUPLING_STREAK_REQUIRED and not coupling_warned:
                        msg = (f"WARNING: amplitude has been below "
                               f"{int(COUPLING_DROP_FRAC*100)}% of baseline "
                               f"({coupling_baseline:.3f} V) for "
                               f"{COUPLING_STREAK_REQUIRED} consecutive snapshots — "
                               f"likely coupling loss. Session continues; "
                               f"intervene if needed.")
                        print(f"[ASCAN] {msg}")
                        coupling_warned = True
                        h5f.attrs["coupling_warning"]      = True
                        h5f.attrs["coupling_warning_at_s"] = float(elapsed)
                        h5f.attrs["coupling_baseline_v"]   = float(coupling_baseline)
                        with self._lock:
                            self.progress["coupling_warning"] = True
                            self.progress["coupling_warning_at_s"] = round(float(elapsed), 1)

                # Update shared state for the dashboard
                with self._lock:
                    self._n_snapshots += 1
                    if raw_idx is not None:
                        self._n_raw_pulses += int(raw_idx.size)
                    self._latest_wf = avg_wf
                    self._history.append({
                        "t":   round(elapsed, 2),
                        "amp": round(amp, 6),
                        "tof":        round(tof_abs, 4),     # absolute ToF for charts
                        "tof_rel":    round(tof, 4),
                        "tof_env":    round(float(cfg["gate_us_start"]) + tof_env, 4),
                        "track_corr": None if not np.isfinite(tracking_corr)
                                      else round(tracking_corr, 4),
                        "eng": round(eng, 2),
                        "clip_pct": round(clip_fraction * 100.0, 3),
                        "raw_clip_pct": round(raw_clip_fraction * 100.0, 3),
                        "n":   n_avg,
                    })
                    self.progress = {
                        "msg":         f"{self._n_snapshots} snapshots  |  {elapsed:.0f}s",
                        "n_snapshots":  self._n_snapshots,
                        "duration_s":   round(elapsed, 1),
                        "n_raw_pulses": self._n_raw_pulses,
                        "tof_us_absolute": round(tof_abs, 4),
                        "tof_us_relative": round(tof, 4),
                        "tracking_method": "xcorr" if tracking_method == 1 else "envelope",
                        "clip_fraction_pct": round(clip_fraction * 100.0, 3),
                        "raw_clip_fraction_pct": round(raw_clip_fraction * 100.0, 3),
                        "clipping_warning": clipping_warned,
                    }
                    if coupling_warned:
                        self.progress["coupling_warning"] = True

            # ---- Session complete ------------------------------------------
            h5f.flush()
            with self._lock:
                n_done = self._n_snapshots

            self._set(
                status="IDLE",
                progress={"msg": f"Done — {n_done} snapshots saved.",
                          "n_snapshots": n_done},
            )

            self._write_session_manifest("stopped")

            if self._h5_path:
                self.cloud.upload_path_async(self._h5_path)

        except Exception as e:
            self._set(status="ERROR", progress={"msg": f"Error: {e}"})
            self._write_session_manifest("error")
            print(f"[ASCAN ERROR] {e}")
            traceback.print_exc()
        finally:
            if h5f:
                try: h5f.flush(); h5f.close()
                except Exception: pass
            if hs:
                hs.close()
            self.running = False
