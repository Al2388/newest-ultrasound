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
  Amplitude — envelope peak of the averaged waveform.
              Decreases as acoustic attenuation increases (e.g. gas evolution,
              delamination, or electrolyte drying at high SOC).
  ToF       — time-of-flight of the echo peak within the gate (µs).
              Shifts as electrode thickness or sound velocity changes with SOC.
  Energy    — integral of squared waveform.
              Correlates with total transmitted acoustic power.

Coherent averaging
------------------
All raw pulses received within one snapshot interval are coherently averaged
(np.mean over the pulse axis). Coherent averaging preserves the signal phase,
which is essential for accurate sub-sample ToF tracking. It also improves SNR
by √N where N is the number of pulses averaged (~PRF × interval_s).

HDF5 file layout  (data/ascan/<session_id>.h5)
----------------------------------------------
Attributes (session-level metadata):
  session_id        str      unique identifier, e.g. "ascan_cell_A1_1715000000"
  fs_hz             float    oscilloscope sample rate (Hz)
  gate_us_start     float    gate window start (µs from sync edge)
  gate_us_end       float    gate window end   (µs from sync edge)
  gate_samples      int      number of samples per averaged waveform
  detected_prf_hz   float    measured pulse repetition frequency
  interval_s        float    snapshot interval (seconds)
  timestamp_iso     str      ISO-8601 UTC session start time

Datasets (all resizable, chunk-compressed, appended one row per snapshot):
  /waveforms  [N, gate_samples]  float32   coherently-averaged gate windows
  /timestamps [N]                float64   Unix wall-clock time of each snapshot
  /amplitude  [N]                float32   envelope peak of averaged waveform (V)
  /tof_us     [N]                float32   envelope peak time within gate (µs)
  /energy     [N]                float32   sum-of-squares of averaged waveform
  /n_averaged [N]                int32     number of raw pulses averaged

SOC/event annotations
---------------------
Stored in data/ascan/<session_id>_annotations.json — a JSON array, one object
per event. Each object has: timestamp, timestamp_iso, elapsed_s, soc_pct,
label, snapshot_idx. The file is human-readable and can be edited after the
experiment to add notes or correct SOC values.
"""

import json
import os
import time
import threading
import traceback
from collections import deque
from datetime import datetime, timezone

import h5py
import numpy as np

from hs5_control import HS5StreamPeaks, envelope_hilbert
from cloud_manager import CloudManager


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
            "out_dir":       "data/ascan",  # output directory (relative to CWD)
            "session_name":  "",            # label for the HDF5 filename (optional)
            "gate_us_start": 30.0,          # gate window start (µs from sync edge)
            "gate_us_end":   40.0,          # gate window end   (µs from sync edge)
            "fs_hz":         20_000_000,    # oscilloscope sample rate (Hz)
            "ch1_range":     1.0,           # CH1 voltage range (V)
        }

        # Rolling deque of the last 500 feature-point dicts — feeds the live charts.
        # Each entry: {"t": elapsed_s, "amp": float, "tof": float,
        #              "eng": float, "n": int}
        self._history     = deque(maxlen=500)
        self._latest_wf   = None   # most recently computed averaged waveform (float32)
        self._annotations = []     # list of event annotation dicts
        self._h5_path     = None   # absolute path to the open HDF5 file
        self._ann_path    = None   # absolute path to the annotations JSON sidecar
        self._session_id  = None   # unique session identifier string
        self._session_t0  = 0.0   # wall-clock time at session start
        self._n_snapshots = 0      # number of snapshots written so far

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """
        Return a thread-safe snapshot of the current session state.

        This is the only method called from the FastAPI event-loop thread
        (the /api/ascan/status endpoint). It returns everything the dashboard
        needs to update all charts in one HTTP round-trip:
          - latest averaged waveform (for the waveform canvas)
          - rolling feature history (for ToF / Amplitude time-series charts)
          - annotation list (for event log and chart markers)
          - gate config and fs (so the waveform chart can label the x-axis in µs)
        """
        with self._lock:
            wf = self._latest_wf.tolist() if self._latest_wf is not None else []
            return {
                "status":      self.status,
                "progress":    dict(self.progress),
                "waveform":    wf,
                "history":     list(self._history)[-200:],   # last 200 points for chart
                "annotations": list(self._annotations),
                "session_id":  self._session_id,
                "file":        os.path.basename(self._h5_path) if self._h5_path else None,
                "gate_us":     [self.config["gate_us_start"], self.config["gate_us_end"]],
                "fs_hz":       self.config["fs_hz"],
            }

    def start_session(self, new_config: dict | None = None) -> tuple[bool, str]:
        """
        Start a new A-scan monitoring session.

        Creates the output directory, builds the HDF5 filename from the session
        name and a Unix timestamp, resets all in-memory state, and spawns the
        worker daemon thread.

        The HDF5 filename format is:
            ascan_<session_name>_<unix_ts>.h5   (if session_name provided)
            ascan_<unix_ts>.h5                  (otherwise)

        Returns (success: bool, message: str).
        """
        if self.running:
            return False, "Session already active"
        if new_config:
            self.config.update(new_config)

        ts   = int(time.time())
        name = self.config.get("session_name", "").strip().replace(" ", "_")
        stem = f"ascan_{name}_{ts}" if name else f"ascan_{ts}"
        out  = self.config["out_dir"]
        os.makedirs(out, exist_ok=True)

        self._stop_event.clear()
        with self._lock:
            self._history.clear()
            self._annotations.clear()
            self._latest_wf    = None
            self._n_snapshots  = 0
            self._session_id   = stem
            self._h5_path      = os.path.join(out, f"{stem}.h5")
            self._ann_path     = os.path.join(out, f"{stem}_annotations.json")
            self._session_t0   = time.time()
            self.status        = "RECORDING"
            self.progress      = {"msg": "Starting...", "n_snapshots": 0, "duration_s": 0}

        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()
        return True, f"Session started: {stem}"

    def stop_session(self) -> tuple[bool, str]:
        """
        Signal the worker to finish the current snapshot interval and exit cleanly.

        The worker will flush and close the HDF5 file, then upload it to S3
        if cloud archival is enabled, before setting self.running = False.
        """
        if not self.running:
            return False, "No active session"
        self._stop_event.set()
        with self._lock:
            self.status = "STOPPING"
        return True, "Stopping..."

    def mark_event(self, soc_pct: float | None, label: str = "") -> tuple[bool, str]:
        """
        Create a timestamped annotation at the current moment in the session.

        Annotations are written to both the in-memory list (for live dashboard
        display) and the JSON sidecar on disk (for permanent archival). They
        appear as vertical dashed amber lines on the ToF and Amplitude charts.

        Typical usage during a cycling experiment:
          mark_event(80.0, "CC charge start")    — when charging begins at 80 % SOC
          mark_event(None, "equilibrating")      — during OCV rest (SOC unknown)
          mark_event(50.0, "CC discharge start") — halfway through discharge

        Parameters
        ----------
        soc_pct : float or None
            Battery state-of-charge in percent. Pass None if SOC is unknown
            or the cycler has not reported it yet.
        label : str
            Short description of the event. Shown on the chart and in the log.

        Returns (success: bool, message: str).
        """
        if not self.running:
            return False, "No active session"

        with self._lock:
            idx = self._n_snapshots     # snapshot index at the time of the mark
            t0  = self._session_t0

        ann = {
            "timestamp":     time.time(),
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "elapsed_s":     round(time.time() - t0, 2),
            "soc_pct":       float(soc_pct) if soc_pct is not None else None,
            "label":         label,
            "snapshot_idx":  idx,   # index into /timestamps dataset at mark time
        }

        with self._lock:
            self._annotations.append(ann)
            ann_copy = list(self._annotations)

        if self._ann_path:
            try:
                with open(self._ann_path, "w") as f:
                    json.dump(ann_copy, f, indent=2)
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
             c. Compute amplitude, ToF, energy from the averaged waveform.
             d. Append one row to every HDF5 dataset via resize().
             e. Flush the HDF5 file every ~10 seconds.
             f. Update the in-memory rolling history and latest_wf.
        4. Final flush + close HDF5.
        5. Upload HDF5 to S3 if cloud is enabled.

        Coherent averaging detail
        -------------------------
        np.mean(wf, axis=0) averages the N pulse waveforms along the pulse axis,
        preserving the phase of the signal. This is equivalent to coherent
        addition (beamforming in the time domain). SNR improves by √N relative
        to a single pulse, while the phase — and therefore the ToF — is preserved
        to sub-sample accuracy. Incoherent (envelope) averaging would smear the
        phase and reduce ToF sensitivity.
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

            # ---- Create HDF5 file ------------------------------------------------
            # Keep the file open for the whole session. Datasets are created with
            # maxshape=(None, ...) so resize() can extend them one row at a time.
            # gzip-4 is a good balance of compression ratio vs. write latency.
            h5f = h5py.File(self._h5_path, "w")

            # Root-level attributes capture all session parameters for later analysis
            h5f.attrs["session_id"]      = self._session_id
            h5f.attrs["fs_hz"]           = hs.fs
            h5f.attrs["gate_us_start"]   = cfg["gate_us_start"]
            h5f.attrs["gate_us_end"]     = cfg["gate_us_end"]
            h5f.attrs["gate_samples"]    = gate_len
            h5f.attrs["detected_prf_hz"] = hs.detected_prf
            h5f.attrs["interval_s"]      = interval
            h5f.attrs["timestamp_iso"]   = datetime.now(timezone.utc).isoformat()

            # Waveform dataset: 2-D, chunks of 100 rows × gate_len columns
            h5f.create_dataset(
                "waveforms",
                shape=(0, gate_len), maxshape=(None, gate_len),
                dtype="f4", chunks=(100, gate_len),
                compression="gzip", compression_opts=4,
            )

            # Scalar feature time-series: 1-D, large chunks (5000) minimise
            # seek overhead when reading a long session sequentially later.
            for ds_name, dtype in [
                ("timestamps", "f8"),   # float64 Unix time
                ("amplitude",  "f4"),
                ("tof_us",     "f4"),
                ("energy",     "f4"),
                ("n_averaged", "i4"),   # int32 pulse count per snapshot
            ]:
                h5f.create_dataset(ds_name, shape=(0,), maxshape=(None,),
                                   dtype=dtype, chunks=(5000,))

            self._set(status="RECORDING", progress={"msg": "Recording..."})

            # Flush the HDF5 file every ~10 s to bound data loss on crash
            flush_every = max(1, int(10 / interval))
            flush_count = 0

            # ---- Main acquisition loop -------------------------------------------
            while not self._stop_event.is_set():
                _, _, _, _, wf = hs.acquire_peaks(duration_s=interval,
                                                   save_waveforms=True)
                if wf is None or wf.shape[0] == 0:
                    continue   # no pulses detected — skip this interval

                # Coherent average preserves phase — required for accurate ToF tracking
                avg_wf = np.mean(wf, axis=0).astype(np.float32)
                n_avg  = int(wf.shape[0])   # number of raw pulses averaged

                # Feature extraction on the averaged waveform (with fresh DC removal)
                v   = (avg_wf - float(np.mean(avg_wf))).astype(np.float32)
                env = envelope_hilbert(v)
                pk  = int(np.argmax(env))
                amp = float(env[pk])
                tof = float(pk / hs.fs * 1e6)   # peak sample index → µs within gate
                eng = float(np.dot(v, v))
                ts  = time.time()

                # Append one row to each HDF5 dataset by growing them by 1
                n = h5f["waveforms"].shape[0]
                h5f["waveforms"].resize((n + 1, gate_len))
                h5f["waveforms"][n] = avg_wf

                for ds_name, val in [
                    ("timestamps", ts),
                    ("amplitude",  amp),
                    ("tof_us",     tof),
                    ("energy",     eng),
                    ("n_averaged", n_avg),
                ]:
                    h5f[ds_name].resize((n + 1,))
                    h5f[ds_name][n] = val

                flush_count += 1
                if flush_count >= flush_every:
                    h5f.flush()
                    flush_count = 0

                # Update shared state for the dashboard
                elapsed = round(ts - self._session_t0, 1)
                with self._lock:
                    self._n_snapshots += 1
                    self._latest_wf    = avg_wf
                    self._history.append({
                        "t":   round(elapsed, 2),
                        "amp": round(amp, 6),
                        "tof": round(tof, 4),
                        "eng": round(eng, 2),
                        "n":   n_avg,
                    })
                    self.progress = {
                        "msg":         f"{self._n_snapshots} snapshots  |  {elapsed:.0f}s",
                        "n_snapshots":  self._n_snapshots,
                        "duration_s":   elapsed,
                    }

            # ---- Session complete ------------------------------------------------
            h5f.flush()
            with self._lock:
                n_done = self._n_snapshots

            self._set(
                status="IDLE",
                progress={"msg": f"Done — {n_done} snapshots saved.",
                          "n_snapshots": n_done},
            )

            # Upload the completed HDF5 file to S3 for archival
            if self.cloud.enabled and self._h5_path:
                self.cloud.upload_async(
                    self._h5_path,
                    os.path.basename(self._h5_path),
                    "application/octet-stream",
                )

        except Exception as e:
            self._set(status="ERROR", progress={"msg": f"Error: {e}"})
            print(f"[ASCAN ERROR] {e}")
            traceback.print_exc()
        finally:
            if h5f:
                try: h5f.flush(); h5f.close()
                except Exception: pass
            if hs:
                hs.close()
            self.running = False
