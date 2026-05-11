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

HDF5 file layout  (data/gauging/gauge_<session_id>.h5)
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
  /waveforms  [N, gate_samples]  float32  coherent average of pulses in window
  /timestamps [N]                float64  Unix wall-clock time per window
  /tof_us     [N]                float32  median ToF across the window's pulses
  /amplitude  [N]                float32  median envelope amplitude
  /energy     [N]                float32  Σ(v²) of the averaged waveform
  /n_pulses   [N]                int32    pulses contributing to each window
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

from hs5_control import HS5StreamPeaks
from cloud_manager import CloudManager


SCHEMA_VERSION  = "2.0"
SERVICE_VERSION = "gauging_service v2.0"


def _safe_session_name(name: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("._-")
    return safe or fallback


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
            "base_out_dir":    "data/gauging",
            "out_dir":         "data/gauging",
            "session_name":    "",              # used in the HDF5 filename
            "gate_us_start":   30.0,            # gate window start (µs)
            "gate_us_end":     40.0,            # gate window end   (µs)
            "fs_hz":           20_000_000,      # oscilloscope sample rate (Hz)
            "ch1_range":       1.0,             # CH1 voltage range (V)
            "sample_window_s": 0.3,             # acquisition window per row
        }

        # Live readout — written by the worker, read by get_status()
        self._latest_tof  = None
        self._latest_amp  = None
        self._latest_n    = 0
        self._latest_wf   = None
        self._tof_history = deque(maxlen=120)

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
                "gate_us":     [self.config["gate_us_start"],
                                self.config["gate_us_end"]],
                "fs_hz":       self.config["fs_hz"],
                "waveform_start_us": 0.0,
            }

    def start(self, new_config: dict | None = None) -> tuple[bool, str]:
        """
        Begin live ToF streaming with HDF5 archival.

        Builds the output filename from session_name (if provided) and a Unix
        timestamp, creates data/gauging/ if needed, and spawns the worker.
        Returns (success, message).
        """
        if self.running:
            return False, "Gauging already active"
        if new_config:
            self.config.update(new_config)

        ts   = int(time.time())
        name = _safe_session_name(self.config.get("session_name", ""), "session")
        stem = f"gauge_{name}_{ts}"
        base_out = self.config.get("base_out_dir", "data/gauging")
        out = os.path.join(base_out, stem)
        os.makedirs(out, exist_ok=True)
        self.config["out_dir"] = out

        self._stop_event.clear()
        with self._lock:
            self._tof_history.clear()
            self._latest_tof = None
            self._latest_amp = None
            self._latest_n   = 0
            self._latest_wf  = None
            self._n_windows  = 0
            self._session_id = stem
            self._session_dir = out
            self._h5_path    = os.path.join(out, f"{stem}.h5")
            self.status      = "STARTING"

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
        try:
            with open(os.path.join(self._session_dir, "session_manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
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
            for ds_name, dtype in [
                ("timestamps", "f8"),
                ("tof_us",     "f4"),
                ("amplitude",  "f4"),
                ("energy",     "f4"),
                ("n_pulses",   "i4"),
                ("n_rejected", "i4"),
                ("prf_actual", "f4"),
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

                # Median features across the window's pulses (outlier-robust)
                tof_med = float(np.median(tf))
                amp_med = float(np.median(aa))

                # Energy of the *averaged* waveform (after DC removal) —
                # consistent with how A-scan computes per-snapshot energy
                v   = (avg_wf - float(np.mean(avg_wf))).astype(np.float32)
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
                    ("timestamps", ts),
                    ("tof_us",     tof_med),
                    ("amplitude",  amp_med),
                    ("energy",     eng),
                    ("n_pulses",   n_p),
                    ("n_rejected", n_rejected),
                    ("prf_actual", prf_actual),
                ]:
                    h5f[ds_name].resize((n + 1,))
                    h5f[ds_name][n] = val

                # Flush every window — at 0.3 s/window, worst-case data loss
                # on a hard crash is one row (~600 µs of waveform data)
                h5f.flush()

                with self._lock:
                    self._latest_tof = tof_med
                    self._latest_amp = amp_med
                    self._latest_n   = n_p
                    self._latest_wf  = live_wf
                    self._n_windows  = n + 1
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

            # Async upload to S3 if cloud is enabled (no-op otherwise)
            if self.cloud and self.cloud.enabled and self._h5_path \
                    and os.path.exists(self._h5_path):
                self.cloud.upload_async(
                    self._h5_path,
                    os.path.basename(self._h5_path),
                    "application/octet-stream",
                )

            self.running = False
