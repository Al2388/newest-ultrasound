"""
C-Scan Service
===============
Orchestrates a complete 2-D ultrasonic C-scan of a battery cell by coordinating:

  PrecisionEnder  (printer_control.py)  — moves the transducer in a bidi raster
  HS5StreamPeaks  (hs5_control.py)      — acquires pulse waveforms and features
  CloudManager    (cloud_manager.py)    — async S3 upload of results when done

Scan pattern (bidirectional serpentine)
---------------------------------------
  Even-numbered lines scan left  → right (LTR, direction=0).
  Odd-numbered  lines scan right → left  (RTL, direction=1).
  Bidi halves the total scan time: the probe does not return to the left edge
  before each line. Each RTL row is reversed in software before being written
  into the image grid so the output is always left-to-right.

Feature maps produced
----------------------
  Amplitude — peak of the Hilbert envelope (unit: V). Tracks signal attenuation
              through the electrode stack; low amplitude → delamination / voids.
  ToF       — time-of-flight of the echo peak within the gate (unit: µs).
              Shifts with electrode thickness / density changes during cycling.
  Energy    — sum of squared samples (unit: V²·samples). Correlates with the
              total acoustic energy transmitted through the cell.

Data outputs (data/raw/cscan/cscan_<name>_<YYYY-MM-DD_HH-MM-SS>/)
---------------------------
  scan_amp.png / scan_tof.png / scan_eng.png   live-updating PNG images
  lines_raw/line_XXXX.npz                      per-line raw pulse archive
  scan_<YYYY-MM-DD_HH-MM-SS>.npz               final feature maps (2-D arrays)
  scan_<YYYY-MM-DD_HH-MM-SS>_meta.json         acquisition parameters

Thread safety
-------------
The scan runs in a daemon worker thread. The FastAPI server accesses shared
state (status, progress, images) only through get_status() / _set(), which
both acquire self._lock. The worker signals completion or stop via
self._stop_event (threading.Event).
"""

import json
import math
import os
import re
import time
import threading
import traceback
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")   # non-interactive Matplotlib backend — safe for threads
import matplotlib.pyplot as plt
import numpy as np

from ultrasound_battery.hardware.printer import setup_precision_printer
from ultrasound_battery.hardware.hs5 import HS5StreamPeaks
from ultrasound_battery.cloud.manager import CloudManager
from ultrasound_battery.utils import session_timestamp


# =============================================================================
# Alignment tuning constants
# =============================================================================
# Maximum lateral shift (image columns) that align_row_minblur will ever apply.
# Set just above the largest mechanical offset we've measured on the Ender
# (~5 px for tight pitch, ~8 px after long warm-up drift).
ALIGN_MAX_SHIFT_PX = 8.0

# Peak-to-Sidelobe Ratio gate. Below this, the cross-correlation peak isn't
# significantly above the background, so we fall back to the previous shift
# rather than chase a noise lobe. Empirically tuned on real C-scans where the
# carrier-to-noise ratio gives PSR ≈ 10–20 for clean rows and < 5 for dropouts.
ALIGN_PSR_THRESHOLD = 6.0

# Per-row step gate. A real mechanical offset evolves smoothly between adjacent
# rows; a jump larger than this is almost always a correlation artefact (e.g.
# row dominated by a single bright pixel pulling the peak off-centre).
ALIGN_MAX_STEP_PX = 1.8

# PSR sidelobe exclusion half-width — samples either side of the main peak to
# omit from the background statistics.
ALIGN_PSR_EXCLUDE_PX = 5


# =============================================================================
# Signal-processing helpers
# =============================================================================

def row_from_pulses_nosmooth(aa: np.ndarray, ncols: int,
                              expected_cycles: int,
                              center_window: bool = True) -> np.ndarray:
    """
    Resample a variable-length pulse-feature array onto a fixed ncols pixel grid.

    The number of ultrasonic pulses acquired per scan line varies slightly with
    scan speed, PRF stability, and timing jitter. This function maps whatever
    number of pulses were actually collected onto the fixed image grid width
    using a scatter-accumulate with bilinear weights.

    Centre-window crop
    ------------------
    If more pulses than expected are present (e.g. because acquisition started
    slightly before the printer motion began), only the central `expected_cycles`
    pulses are used. This removes edge artefacts from the acceleration ramp at
    the start and deceleration ramp at the end of each scan line.

    Parameters
    ----------
    aa : np.ndarray
        1-D float32 array of pulse feature values (amplitude, ToF, or energy).
    ncols : int
        Number of output image columns.
    expected_cycles : int
        Approximate pulses-per-line (PRF × line_duration). Used as the crop
        window width.
    center_window : bool
        If True (default), crop to the central expected_cycles pulses.

    Returns
    -------
    np.ndarray (float32, length ncols)
        Gridded feature values. NaN where no pulse contributed to that column.
    """
    out = np.full(ncols, np.nan, dtype=np.float32)
    m   = int(aa.size)
    if m == 0 or ncols <= 0:
        return out

    # Centre crop: take at most expected_cycles pulses from the middle of the line
    W     = int(max(8, min(expected_cycles, m)))
    start = max(0, (m - W) // 2) if center_window else 0
    core  = aa[start:start + W].astype(np.float32)
    valid = np.isfinite(core)
    if not np.any(valid):
        return out
    core = core[valid]
    W    = core.size
    if W == 0:
        return out

    # Map each pulse at fractional position [0, ncols) to its two nearest grid
    # columns, then scatter-accumulate with bilinear weights.
    x  = (np.arange(W, dtype=np.float64) + 0.5) * (ncols / W)
    i0 = np.floor(x).astype(np.int64)
    w1 = x - i0
    w0 = 1.0 - w1
    i1 = i0 + 1

    sumw = np.zeros(ncols, dtype=np.float64)
    acc  = np.zeros(ncols, dtype=np.float64)

    m0 = (i0 >= 0) & (i0 < ncols)
    if np.any(m0):
        np.add.at(sumw, i0[m0], w0[m0])
        np.add.at(acc,  i0[m0], w0[m0] * core[m0])

    m1 = (i1 >= 0) & (i1 < ncols)
    if np.any(m1):
        np.add.at(sumw, i1[m1], w1[m1])
        np.add.at(acc,  i1[m1], w1[m1] * core[m1])

    nz      = sumw > 1e-12
    out[nz] = (acc[nz] / sumw[nz]).astype(np.float32)

    # Forward-fill isolated NaN columns caused by sparse pulse coverage
    for k in range(1, ncols):
        if np.isnan(out[k]) and not np.isnan(out[k - 1]):
            out[k] = out[k - 1]

    return out


def _phase_corr_psr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Phase-only cross-correlation between two equal-length 1-D signals.

    Returns the sub-pixel shift (in samples) that best aligns ``a`` onto ``b``,
    together with the Peak-to-Sidelobe Ratio (PSR) of the correlation surface
    — a confidence metric for the shift estimate.

    Caller is responsible for any DC removal / NaN handling on ``a`` and ``b``.

    Returns
    -------
    shift : float
        Estimated shift in samples, in the wrapped range ``[-n/2, n/2)``.
    psr   : float
        Peak height above the sidelobe-region mean, in units of sidelobe σ.
        Values > ~6 indicate a well-defined peak; values near 0 indicate the
        peak is indistinguishable from background noise.
    """
    n = a.size

    # Whitened cross-correlation: divide out spectral magnitude so only the
    # phase ramp (= shift) remains. This prevents a single bright row pixel
    # from biasing the peak location.
    R  = np.fft.rfft(a) * np.conj(np.fft.rfft(b))
    R /= np.maximum(np.abs(R), 1e-12)
    c  = np.fft.irfft(R, n=n)

    k0 = int(np.argmax(c))

    # Parabolic sub-sample refinement around the integer-shift peak
    cm1, cp1 = c[(k0 - 1) % n], c[(k0 + 1) % n]
    denom    = cm1 - 2 * c[k0] + cp1
    delta    = 0.5 * (cm1 - cp1) / denom if abs(denom) > 1e-12 else 0.0
    shift    = float(k0 + delta)
    if shift > n / 2:
        shift -= n   # unwrap from circular [0, n) to symmetric [-n/2, n/2)

    # PSR: peak height in units of background σ, excluding ±N samples around
    # the peak so the metric reflects true background, not the peak's skirt.
    mask = np.ones_like(c, dtype=bool)
    lo, hi = (k0 - ALIGN_PSR_EXCLUDE_PX) % n, (k0 + ALIGN_PSR_EXCLUDE_PX) % n
    if lo <= hi:
        mask[lo:hi + 1] = False
    else:
        mask[:hi + 1] = False; mask[lo:] = False

    bg     = c[mask]
    std_bg = float(np.std(bg))
    psr    = float((c[k0] - np.mean(bg)) / (std_bg + 1e-12)) if std_bg > 0 else 0.0
    return shift, psr


def align_row_minblur(row: np.ndarray, ref: np.ndarray,
                      max_shift: float = ALIGN_MAX_SHIFT_PX,
                      prev_shift: float = 0.0
                      ) -> tuple[np.ndarray, float]:
    """
    Sub-pixel lateral alignment of a scan row against a reference row.

    Background
    ----------
    In a bidi scan, LTR and RTL lines often have a systematic lateral offset
    caused by belt backlash, motor inertia, and timing jitter. If left
    uncorrected this produces a "comb" artefact in the image. This function
    estimates the offset via phase-only cross-correlation, gates the estimate
    against two confidence checks, and applies it to sub-pixel precision.

    Confidence gates (both must pass for the new estimate to be accepted)
    ---------------------------------------------------------------------
    * PSR ≥ ``ALIGN_PSR_THRESHOLD``  — the correlation peak is statistically
      above the sidelobe background.
    * ``|shift − prev_shift| ≤ ALIGN_MAX_STEP_PX``  — the shift evolves
      smoothly between adjacent rows of the same scan direction.

    If either gate fails, ``prev_shift`` is reused (the row is still shifted,
    but by the last trusted value rather than a noisy new estimate).

    Parameters
    ----------
    row        : np.ndarray   Newly acquired row to align (float32, length ncols).
    ref        : np.ndarray   Reference row for this scan direction (frozen at
                              the first row by the caller).
    max_shift  : float        Hard clip on the estimate, in image columns.
    prev_shift : float        Last accepted shift for this scan direction.

    Returns
    -------
    (aligned_row, shift_used) : tuple[np.ndarray, float]
    """
    if ref is None or not np.any(np.isfinite(ref)):
        return row, 0.0

    a = np.nan_to_num(row, nan=0.0); a -= np.mean(a)
    b = np.nan_to_num(ref, nan=0.0); b -= np.mean(b)

    shift_est, psr = _phase_corr_psr(a, b)
    shift_est      = float(np.clip(shift_est, -max_shift, max_shift))

    # Accept the new estimate only if the correlation peak is strong AND the
    # shift doesn't jump unreasonably from the previous row of this direction
    accept    = (psr >= ALIGN_PSR_THRESHOLD
                 and abs(shift_est - prev_shift) <= ALIGN_MAX_STEP_PX)
    shift_use = shift_est if accept else prev_shift

    return apply_fractional_shift(row, shift_use), shift_use


def apply_fractional_shift(row: np.ndarray, shift: float) -> np.ndarray:
    """Apply a sub-pixel lateral shift using the same FFT phase-ramp method."""
    n = len(row)
    if n == 0 or abs(shift) < 1e-12:
        return row.astype(np.float32, copy=False)
    k = np.fft.rfftfreq(n)
    shifted = np.fft.irfft(
        np.fft.rfft(np.nan_to_num(row, nan=0.0)) * np.exp(-2j * np.pi * k * shift),
        n=n,
    )
    return shifted.astype(np.float32)


def _safe_session_name(name: str, fallback: str) -> str:
    """Return a filesystem-safe run label for a scan/session folder."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("._-")
    return safe or fallback


def _json_safe(value):
    """Convert NumPy values into plain Python values before json.dump()."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# =============================================================================
# Service class
# =============================================================================

class CScanService:
    """
    Manages the full lifecycle of one 2-D C-scan acquisition session.

    Typical call sequence from the FastAPI server layer:
      1. start_scan(config)   — validate config, spawn worker daemon thread
      2. get_status()         — polled at ~1 Hz by the dashboard frontend
      3. stop_scan()          — optional early termination
      4. return_to_start()    — drive probe back to origin after scan ends

    The worker thread exclusively owns the hardware objects (HS5 and printer).
    The FastAPI event-loop thread communicates via get_status() and _set(),
    both of which acquire self._lock for a brief critical section.
    """

    def __init__(self):
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self.running     = False
        self.thread      = None
        self.status      = "IDLE"
        self.progress    = {"line": 0, "total": 0, "msg": "Ready"}
        self.images      = {"Amplitude": None, "ToF": None, "Energy": None}

        # Default configuration — overridden per-request via start_scan(config)
        self.config = {
            "com_port":       "COM6",        # serial port for the Ender
            "roi_w":          80.0,          # scan region width  (mm)
            "roi_h":          70.0,          # scan region height (mm)
            "pitch":          0.1,           # line spacing (mm) — nlines = roi_h / pitch
            "speed":          10.0,          # scan speed (mm/s)
            "ch1_range":      5.0,           # CH1 voltage range (V) — fixed; sized for typical ±4 V echoes
            "cols":           500,           # output image width in pixels
            "scan_name":      "",            # label for the run folder
            "base_out_dir":   "data/raw/cscan",  # parent directory for run folders
            "out_dir":        "data/raw/cscan",  # output directory (relative to CWD)
            "cmap":           "turbo",       # matplotlib colormap for PNG exports
            "save_waveforms": True,          # False → skip raw waveform archive
        }
        self._session_id = None
        self._session_dir = None
        self._session_ts = None        # YYYY-MM-DD_HH-MM-SS string set at start_scan()
        self.cloud = CloudManager()

    # -------------------------------------------------------------------------
    # Thread-safe state access
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a locked snapshot of scan state for the API status endpoint."""
        with self._lock:
            return {
                "status":   self.status,
                "progress": dict(self.progress),
                "images":   dict(self.images),
                "session_id": self._session_id,
                "folder": self._session_dir,
            }

    def _set(self, status=None, progress=None, images=None):
        """Update one or more shared-state fields under the lock."""
        with self._lock:
            if status   is not None: self.status = status
            if progress is not None: self.progress.update(progress)
            if images   is not None: self.images.update(images)

    def _write_session_manifest(self, state: str):
        """Write a human-readable manifest into the current run folder."""
        if not self._session_dir:
            return
        manifest = {
            "mode": "cscan",
            "state": state,
            "session_id": self._session_id,
            "session_dir": self._session_dir,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "progress": dict(self.progress),
            "images": dict(self.images),
            "config": dict(self.config),
        }
        manifest_path = os.path.join(self._session_dir, "session_manifest.json")
        try:
            with open(manifest_path, "w") as f:
                json.dump(_json_safe(manifest), f, indent=2)
            self.cloud.upload_path_async(manifest_path)
        except Exception as e:
            print(f"[DATA] Manifest write error: {e}")

    # -------------------------------------------------------------------------
    # Control API (called from FastAPI handlers)
    # -------------------------------------------------------------------------

    def start_scan(self, new_config: dict | None = None) -> tuple[bool, str]:
        """
        Start a new C-scan. Merges new_config over self.config, clears the stop
        event, sets running=True, and spawns the worker thread.

        Returns (success, message).
        """
        if self.running:
            return False, "Already Running"
        if new_config:
            self.config.update(new_config)

        ts = session_timestamp()
        name = _safe_session_name(self.config.get("scan_name", ""), "scan")
        stem = f"cscan_{name}_{ts}"
        base_out = self.config.get("base_out_dir", "data/raw/cscan")
        out_dir = os.path.join(base_out, stem)
        os.makedirs(out_dir, exist_ok=True)
        self.config["out_dir"] = out_dir
        self._session_id = stem
        self._session_dir = out_dir
        self._session_ts = ts

        self._stop_event.clear()
        self.running = True
        self._set(
            status="RUNNING",
            progress={"msg": f"Started {stem}", "session_id": stem},
            images={"Amplitude": None, "ToF": None, "Energy": None},
        )
        self._write_session_manifest("starting")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return True, f"Started: {stem}"

    def stop_scan(self) -> bool:
        """
        Request an early stop. The worker checks _stop_event after each line
        then returns to the origin and saves the partial feature maps.
        """
        if self.running:
            self._stop_event.set()
            self._set(status="STOPPING")
            return True
        return False

    def return_to_start(self) -> tuple[bool, str]:
        """
        Open a fresh printer connection and drive it back to (0, 0).
        Safe to call after scan completion or after stop_scan().
        """
        if self.running:
            return False, "Stop scan first."
        try:
            self._set(status="MOVING")
            pr, _, _, _, _ = setup_precision_printer(
                self.config["com_port"], 115200,
                self.config["roi_w"], self.config["roi_h"],
                reset_origin=False,   # don't redefine the origin — just move to it
            )
            pr.move_to_position(0.0, 0.0, fast=True)
            pr.wait_for_completion()
            pr.close()
            self._set(status="IDLE", progress={"msg": "Returned to Start."})
            return True, "Returned"
        except Exception as e:
            self._set(status="ERROR")
            return False, str(e)

    # -------------------------------------------------------------------------
    # Per-line raw data archive
    # -------------------------------------------------------------------------

    def _save_line_worker(self, path: str, **arrays):
        """Background thread target: compress and write one scan-line NPZ file."""
        try:
            np.savez_compressed(path, **arrays)
            self.cloud.upload_path_async(path)
        except Exception as e:
            print(f"[DATA] Line save error ({os.path.basename(path)}): {e}")

    def _save_line(self, line_idx: int, y_mm: float, ltr: bool,
                   tt, aa, tf, ee, wf, x_mm,
                   fs_hz: float, g0_samp: int, g1_samp: int):
        """
        Archive all raw pulse data for one scan line as a compressed NPZ file.

        The file is written in a background thread so it does not delay the
        scan timing. Files go to data/raw/cscan/<run>/lines_raw/line_XXXX.npz.

        NPZ layout
        ----------
        waveforms  [n_pulses, gate_samples]  float32  DC-removed gate windows
        amplitude  [n_pulses]                float32  envelope peak (V)
        tof_us     [n_pulses]                float32  time-of-flight within gate (µs)
        energy     [n_pulses]                float32  pulse energy (V²·samples)
        timestamps [n_pulses]                float64  perf_counter wall-clock time
        x_mm       [n_pulses]                float32  estimated probe X position (mm)
        y_mm       scalar                    float32  scan line Y position (mm)
        direction  scalar uint8              0=LTR, 1=RTL
        fs_hz      scalar float64            actual oscilloscope sample rate
        gate_us    [2]                       float32  [gate_start, gate_end] in µs
        """
        lines_dir = os.path.join(self.config["out_dir"], "lines_raw")
        os.makedirs(lines_dir, exist_ok=True)
        path = os.path.join(lines_dir, f"line_{line_idx:04d}.npz")

        arrays = dict(
            amplitude  = aa,
            tof_us     = tf,
            energy     = ee,
            timestamps = tt,
            x_mm       = x_mm,
            y_mm       = np.float32(y_mm),
            direction  = np.uint8(0 if ltr else 1),
            fs_hz      = np.float64(fs_hz),
            gate_us    = np.array([g0_samp / fs_hz * 1e6,
                                   g1_samp / fs_hz * 1e6], dtype=np.float32),
        )
        if wf is not None:
            arrays["waveforms"] = wf

        threading.Thread(
            target=self._save_line_worker, kwargs={"path": path, **arrays}, daemon=True
        ).start()

    # -------------------------------------------------------------------------
    # Final scan archive
    # -------------------------------------------------------------------------

    def _save_final_npz(self, amp_arr: np.ndarray, tof_arr: np.ndarray,
                        eng_arr: np.ndarray, hs_info: dict):
        """
        Save the completed scan as scan_<YYYY-MM-DD_HH-MM-SS>.npz + scan_<YYYY-MM-DD_HH-MM-SS>_meta.json.

        The NPZ file contains all three feature maps as 2-D arrays [nlines, ncols]
        plus the physical spatial axes in mm. Load with:
            d = np.load("scan_<YYYY-MM-DD_HH-MM-SS>.npz")
            amp = d["amplitude"]   # shape (nlines, ncols)
            x   = d["x_mm"]        # physical X axis
            y   = d["y_mm"]        # physical Y axis

        The JSON sidecar records every hardware and acquisition parameter so
        that the scan can be fully understood and reproduced without the source
        code. It should always be kept alongside the NPZ file.

        Both files are uploaded asynchronously to S3 if cloud archival is enabled.
        """
        cfg  = self.config
        # Match the session folder's timestamp so a glance at any file inside
        # `cscan_<name>_<ts>/` shows the same `<ts>` — easier to track than
        # mixing the start time (folder) with the save time (file).
        ts   = self._session_ts or session_timestamp()
        rows, cols = amp_arr.shape

        npz_name   = f"scan_{ts}.npz"
        meta_name  = f"scan_{ts}_meta.json"
        npz_path   = os.path.join(cfg["out_dir"], npz_name)
        meta_path  = os.path.join(cfg["out_dir"], meta_name)

        np.savez_compressed(
            npz_path,
            amplitude = amp_arr,
            tof       = tof_arr,
            energy    = eng_arr,
            x_mm      = np.linspace(0, cfg["roi_w"], cols, dtype=np.float32),
            y_mm      = np.linspace(0, cfg["roi_h"], rows, dtype=np.float32),
        )

        meta = {
            "scan_id":           f"scan_{ts}",
            "session_id":        self._session_id,
            "scan_name":         cfg.get("scan_name", ""),
            "session_dir":       cfg["out_dir"],
            "timestamp_iso":     datetime.now(timezone.utc).isoformat(),
            "instrument":        "TiePie HS5",
            "fs_hz":             hs_info.get("fs_hz", 20_000_000),
            "detected_prf_hz":   hs_info.get("detected_prf_hz"),
            "gate_us":           hs_info.get("gate_us"),
            "gate_samples":      hs_info.get("gate_samples"),
            "sync_thresholds_v": hs_info.get("sync_thresholds_v"),
            "ch1_range_requested_v": hs_info.get("ch1_range_requested_v"),
            "ch1_range_actual_v":    hs_info.get("ch1_range_actual_v"),
            "partial":           bool(hs_info.get("partial", False)),
            "roi_w_mm":          cfg["roi_w"],
            "roi_h_mm":          cfg["roi_h"],
            "pitch_mm":          cfg["pitch"],
            "speed_mm_s":        cfg["speed"],
            "nlines":            rows,
            "ncols":             cols,
            "save_waveforms":    cfg.get("save_waveforms", True),
            "feature_map_file":  npz_name,
            "line_files_dir":    "lines_raw/",
        }
        with open(meta_path, "w") as f:
            json.dump(_json_safe(meta), f, indent=2)
        self._write_session_manifest("saved")

        print(f"[DATA] Saved: {npz_path}  ({rows}×{cols})")
        print(f"[DATA] Meta:  {meta_path}")

        self.cloud.upload_path_async(npz_path)
        self.cloud.upload_path_async(meta_path)

    # -------------------------------------------------------------------------
    # Live image rendering
    # -------------------------------------------------------------------------

    def _save_plot(self, img: np.ndarray, filename: str, label: str):
        """
        Render a feature map to a PNG file and update the dashboard image URL.

        Colour scale uses the 5th–95th percentile of finite values to suppress
        hot/dead pixel artefacts. NaN cells (unscanned rows) render as white.
        Uses OO Matplotlib (fig, ax) rather than the global pyplot state so it
        is safe to call concurrently from the worker thread.
        """
        if not np.any(np.isfinite(img)):
            return

        local_path = os.path.join(self.config["out_dir"], filename)
        cm         = matplotlib.colormaps.get_cmap(self.config["cmap"]).copy()
        cm.set_bad("white")   # NaN → white (rows not yet scanned)

        finite     = img[np.isfinite(img)]
        vmin, vmax = (np.percentile(finite, [5, 95])
                      if finite.size >= 16 else (0, 1))

        with self._lock:
            line, total = self.progress["line"], self.progress["total"]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        im = ax.imshow(
            img, cmap=cm, origin="upper", aspect="equal",
            extent=[0, self.config["roi_w"], self.config["roi_h"], 0],
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"{label}  (Line {line}/{total})")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(local_path, bbox_inches="tight")
        plt.close(fig)

        self.cloud.upload_path_async(local_path)

        rel_path = os.path.relpath(local_path, self.config.get("base_out_dir", "data/raw/cscan"))
        self._set(images={label: f"/local/{rel_path.replace(os.sep, '/')}"})

    # -------------------------------------------------------------------------
    # Worker thread
    # -------------------------------------------------------------------------

    def _worker(self):
        """
        Main scan worker — runs entirely in a daemon thread.

        Execution flow:
          1. Open the printer and HS5; run PRF calibration.
          2. Pre-allocate three NaN-filled feature map arrays [nlines, ncols].
          3. For each scan line i:
             a. Rapid-move the probe to the line start position.
             b. Issue the G1 scan move (printer starts moving immediately).
             c. Acquire HS5 pulses for (line_time + 0.3 s) — the overlap guards
                against the printer finishing before the HS5 does.
             d. Crop acquired pulses to the actual motion window [t0, t1].
             e. Estimate physical X position of each pulse from its timestamp.
             f. Archive raw line data in a background NPZ-write thread.
             g. Reverse RTL lines to left-to-right order.
             h. Resample pulses onto the fixed image grid.
             i. Align row to the rolling per-direction reference.
             j. Every 5 lines: update the live PNG images.
          4. Return probe to (0, 0); save final NPZ + metadata; upload to S3.
        """
        cfg = self.config
        os.makedirs(cfg["out_dir"], exist_ok=True)

        nlines = max(2, int(math.ceil(cfg["roi_h"] / cfg["pitch"])))
        self._set(progress={"line": 0, "total": nlines, "msg": "Initializing Hardware..."})

        # Pre-allocate feature maps with NaN — unscanned rows show as white in the PNG
        img_amp = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)
        img_tof = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)
        img_eng = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)

        pr, hs = None, None
        try:
            pr, xl, xr, ys, _ = setup_precision_printer(
                cfg["com_port"], 115200, cfg["roi_w"], cfg["roi_h"], reset_origin=True
            )
            requested_ch1_range = float(cfg.get("ch1_range", 1.0))
            hs = HS5StreamPeaks(
                fs_hz=20_000_000,
                gate_us=(25.0, 50.0),
                ch1_range=requested_ch1_range,
                feature_mode="envelope",
            ).open()
            hs.calibrate_sync(seconds=1.0, verbose=True)

            theo_time      = abs(xr - xl) / cfg["speed"]          # theoretical line duration (s)
            detected_prf   = getattr(hs, "detected_prf", 5000.0) or 5000.0
            expected_cycles = int(detected_prf * theo_time)        # expected pulses/line
            save_wf        = cfg.get("save_waveforms", True)

            # Separate rolling alignment references for LTR and RTL lines.
            # Using per-direction references prevents a spatially-reversed RTL row
            # from polluting the reference used for the next LTR row.
            ref_e = ref_o = None   # LTR (even) and RTL (odd) reference rows
            sh_e  = sh_o  = 0.0   # last accepted alignment shift per direction

            for i in range(nlines):
                if self._stop_event.is_set():
                    break

                self._set(progress={"line": i + 1, "msg": f"Scanning Line {i + 1}/{nlines}..."})

                y   = ys + i * cfg["pitch"]
                ltr = (i % 2 == 0)
                x0, x1 = (xl, xr) if ltr else (xr, xl)

                # Rapid-move to line start, wait for motion to complete, then start scan
                pr.move_to_position(x0, y, fast=True)
                pr.wait_for_completion()
                time.sleep(0.05)   # brief settle before issuing the scan stroke
                pr.send_command(f"G1 X{x1:.3f} F{int(cfg['speed'] * 60)}")

                # Acquire pulses while the printer traverses the line.
                # 0.2 s buffer on both sides absorbs printer accel/decel ramps
                # and serial-port latency; the downstream center-window crop in
                # row_from_pulses_nosmooth discards what doesn't fit.
                t0 = time.perf_counter()
                tt, aa, tf, ee, wf = hs.acquire_peaks(
                    duration_s=theo_time + 0.4, save_waveforms=save_wf
                )
                pr.wait_for_completion()
                t1 = time.perf_counter()

                # Lenient ±50 ms slack on the time-window crop — guards against
                # perf_counter / printer wall-clock skew at line boundaries.
                sel  = (tt >= t0 - 0.05) & (tt <= t1 + 0.05)
                tt_s = tt[sel]
                aa_s, tf_s, ee_s = aa[sel], tf[sel], ee[sel]
                wf_s = wf[sel] if wf is not None else None

                # Estimate physical X position of each pulse via linear interpolation.
                # The probe moves from x0 to x1 in theo_time seconds; each pulse's
                # timestamp maps to a fractional position along that line.
                t_rel = tt_s - t0
                x_pos = (x0 + (x1 - x0) *
                         np.clip(t_rel / theo_time, 0.0, 1.0)).astype(np.float32)

                # Archive raw line data (in natural acquisition order, real x positions)
                self._save_line(
                    i, y, ltr, tt_s, aa_s, tf_s, ee_s, wf_s, x_pos,
                    hs.fs, hs.g0, hs.g1,
                )

                # Reverse RTL lines so the image grid is always left-to-right
                if not ltr:
                    aa_s, tf_s, ee_s = aa_s[::-1], tf_s[::-1], ee_s[::-1]

                # Resample variable-count pulse arrays onto the fixed image grid
                ra = row_from_pulses_nosmooth(aa_s, cfg["cols"], expected_cycles)
                rf = row_from_pulses_nosmooth(tf_s, cfg["cols"], expected_cycles)
                re = row_from_pulses_nosmooth(ee_s, cfg["cols"], expected_cycles)

                # Align each row to the first row in the same scan direction.
                # Reference is frozen at line 0 (per direction) so per-row alignment
                # errors don't compound into long-scan drift.
                if ltr:
                    ra, sh_e = align_row_minblur(ra, ref_e, prev_shift=sh_e)
                    rf       = apply_fractional_shift(rf, sh_e)
                    re       = apply_fractional_shift(re, sh_e)
                    if ref_e is None:
                        ref_e = ra.copy()   # freeze LTR reference at first line
                else:
                    ra, sh_o = align_row_minblur(ra, ref_o, prev_shift=sh_o)
                    rf       = apply_fractional_shift(rf, sh_o)
                    re       = apply_fractional_shift(re, sh_o)
                    if ref_o is None:
                        ref_o = ra.copy()   # freeze RTL reference at first line

                img_amp[i] = ra
                img_tof[i] = rf
                img_eng[i] = re

                # Refresh live PNG images every 5 lines (balance I/O vs latency)
                if i % 5 == 0 or i == nlines - 1:
                    self._save_plot(img_amp, "scan_amp.png", "Amplitude")
                    self._save_plot(img_tof, "scan_tof.png", "ToF")
                    self._save_plot(img_eng, "scan_eng.png", "Energy")

            stopped = self._stop_event.is_set()
            self._set(progress={"msg": "Returning to Start..."})
            pr.move_to_position(0.0, 0.0, fast=True)
            pr.wait_for_completion()

            hs_info = {
                "fs_hz":             hs.fs,
                "detected_prf_hz":   round(hs.detected_prf, 2),
                "gate_us":           [round(hs.g0 / hs.fs * 1e6, 3),
                                      round(hs.g1 / hs.fs * 1e6, 3)],
                "gate_samples":      hs.gate_len,
                "sync_thresholds_v": [round(hs.ch2_lo, 4), round(hs.ch2_hi, 4)],
                "ch1_range_requested_v": requested_ch1_range,
                "ch1_range_actual_v":    float(getattr(hs, "range", requested_ch1_range)),
                "partial":           stopped,
            }
            self._save_final_npz(img_amp, img_tof, img_eng, hs_info)

            if stopped:
                self._set(status="STOPPED", progress={"msg": "Scan stopped; partial data saved."})
                self._write_session_manifest("stopped")
            else:
                self._set(status="COMPLETED", progress={"msg": "Scan Finished."})
                self._write_session_manifest("completed")

        except Exception as e:
            self._set(status="ERROR", progress={"msg": f"Error: {e}"})
            self._write_session_manifest("error")
            print(f"[ERROR] {e}")
            traceback.print_exc()
        finally:
            if hs: hs.close()
            if pr: pr.close()
            self.running = False
