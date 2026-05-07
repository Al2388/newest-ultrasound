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

Data outputs (data/cscan/)
---------------------------
  scan_amp.png / scan_tof.png / scan_eng.png   live-updating PNG images
  lines_raw/line_XXXX.npz                      per-line raw pulse archive
  scan_TIMESTAMP.npz                           final feature maps (2-D arrays)
  scan_TIMESTAMP_meta.json                     acquisition parameters

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
import time
import threading
import traceback
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")   # non-interactive Matplotlib backend — safe for threads
import matplotlib.pyplot as plt
import numpy as np

from printer_control import setup_precision_printer
from hs5_control import HS5StreamPeaks
from cloud_manager import CloudManager


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


def align_row_minblur(row: np.ndarray, ref: np.ndarray,
                      max_shift: float = 8.0,
                      prev_shift: float = 0.0
                      ) -> tuple[np.ndarray, float]:
    """
    Sub-pixel lateral alignment of a scan row against a reference row using
    phase-only cross-correlation (whitened cross-correlation in the frequency
    domain).

    Background
    ----------
    In a bidi scan, LTR and RTL lines often have a systematic lateral offset
    caused by belt backlash, motor inertia, and timing jitter. If left
    uncorrected this produces a "comb" artefact in the image. This function
    estimates and corrects that offset to sub-pixel precision.

    Algorithm
    ---------
    1. Compute the cross-correlation of `row` with `ref` in the frequency domain.
    2. Whiten the cross-correlation (divide by its amplitude) so that the phase
       shift dominates — this is the "phase-only" cross-correlation.
    3. Find the peak of the correlation and refine its position to sub-pixel
       accuracy with parabolic interpolation.
    4. Compute the Peak-to-Sidelobe Ratio (PSR) to assess confidence:
       if PSR < 6 or the shift is very different from the previous row, fall
       back to prev_shift (the last accepted shift for this scan direction).
    5. Apply the chosen shift via frequency-domain phase multiplication.

    Parameters
    ----------
    row        : np.ndarray   Newly acquired row to align (float32, length ncols).
    ref        : np.ndarray   Reference row — updated to the latest row each call.
    max_shift  : float        Maximum allowed shift magnitude in columns (default 8).
    prev_shift : float        Shift used for the previous row in this direction
                              (used as fallback if the PSR gate rejects the estimate).

    Returns
    -------
    (aligned_row, shift_used) : tuple[np.ndarray, float]
        aligned_row — row shifted by shift_used columns (float32).
        shift_used  — the sub-pixel shift that was applied (columns).
    """
    if ref is None or not np.any(np.isfinite(ref)):
        return row, 0.0

    a = np.nan_to_num(row, nan=0.0); a -= np.mean(a)
    b = np.nan_to_num(ref, nan=0.0); b -= np.mean(b)
    n = len(a)

    # Phase-only cross-correlation: divide out amplitude so only phase (shift)
    # information remains — prevents high-amplitude regions from biasing the peak.
    R  = np.fft.rfft(a) * np.conj(np.fft.rfft(b))
    R /= np.maximum(np.abs(R), 1e-12)
    c  = np.fft.irfft(R, n=n)

    k0 = int(np.argmax(c))

    # Sub-pixel refinement: fit a parabola through the peak and its two neighbours
    denom     = c[(k0 - 1) % n] - 2 * c[k0] + c[(k0 + 1) % n]
    delta     = (0.5 * (c[(k0 - 1) % n] - c[(k0 + 1) % n]) / denom
                 if abs(denom) > 1e-12 else 0.0)
    shift_est = float(k0 + delta)

    # Wrap from the circular correlation range [0, n) to the symmetric range [-n/2, n/2)
    if shift_est > n / 2:
        shift_est -= n
    shift_est = float(np.clip(shift_est, -max_shift, max_shift))

    # Peak-to-Sidelobe Ratio: exclude ±5 samples around the peak from the background
    mask = np.ones_like(c, dtype=bool)
    lo, hi = (k0 - 5) % n, (k0 + 5) % n
    if lo <= hi:
        mask[lo:hi + 1] = False
    else:
        mask[:hi + 1] = False; mask[lo:] = False

    std_bg = np.std(c[mask])
    psr    = ((c[k0] - np.mean(c[mask])) / (std_bg + 1e-12)
              if std_bg > 0 else 0.0)

    # Accept the new estimate only if the correlation peak is strong and
    # the shift doesn't jump unreasonably from the previous row
    shift_use = (shift_est
                 if (psr >= 6.0 and abs(shift_est - prev_shift) <= 1.8)
                 else prev_shift)

    # Apply fractional shift via phase ramp in the frequency domain (sinc interpolation)
    k       = np.fft.rfftfreq(n)
    row_out = np.fft.irfft(
        np.fft.rfft(np.nan_to_num(row, nan=0.0)) * np.exp(-2j * np.pi * k * shift_use),
        n=n,
    )
    return row_out.astype(np.float32), shift_use


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
            "roi_w":          50.0,          # scan region width  (mm)
            "roi_h":          50.0,          # scan region height (mm)
            "pitch":          0.1,           # line spacing (mm) — nlines = roi_h / pitch
            "speed":          10.0,          # scan speed (mm/s)
            "cols":           500,           # output image width in pixels
            "out_dir":        "data/cscan",  # output directory (relative to CWD)
            "cmap":           "turbo",       # matplotlib colormap for PNG exports
            "save_waveforms": True,          # False → skip raw waveform archive
        }
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
            }

    def _set(self, status=None, progress=None, images=None):
        """Update one or more shared-state fields under the lock."""
        with self._lock:
            if status   is not None: self.status = status
            if progress is not None: self.progress.update(progress)
            if images   is not None: self.images.update(images)

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
        self._stop_event.clear()
        self.running = True
        self._set(status="RUNNING")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return True, "Started"

    def stop_scan(self) -> bool:
        """
        Request an early stop. The worker checks _stop_event after each line
        and exits cleanly without saving the final NPZ.
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

    def jog_z_axis(self, z_distance: float) -> tuple[bool, str]:
        """
        Move the Z axis by z_distance mm relative to current position.

        Used to adjust the water-coupling gap between the transducer face and
        the battery surface. The G91/G90 switch ensures only this single move
        is relative; all subsequent moves remain absolute.
        """
        if self.running:
            return False, "Cannot move Z while scanning."
        try:
            pr, _, _, _, _ = setup_precision_printer(
                self.config["com_port"], 115200,
                self.config["roi_w"], self.config["roi_h"],
                reset_origin=False,
            )
            pr.send_command("G91")                              # relative mode
            pr.send_command(f"G1 Z{z_distance:.3f} F300")
            pr.wait_for_completion()
            pr.send_command("G90")                              # back to absolute
            pr.close()
            return True, f"Moved Z by {z_distance:.3f} mm"
        except Exception as e:
            return False, str(e)

    # -------------------------------------------------------------------------
    # Per-line raw data archive
    # -------------------------------------------------------------------------

    def _save_line_worker(self, path: str, **arrays):
        """Background thread target: compress and write one scan-line NPZ file."""
        try:
            np.savez_compressed(path, **arrays)
        except Exception as e:
            print(f"[DATA] Line save error ({os.path.basename(path)}): {e}")

    def _save_line(self, line_idx: int, y_mm: float, ltr: bool,
                   tt, aa, tf, ee, wf, x_mm,
                   fs_hz: float, g0_samp: int, g1_samp: int):
        """
        Archive all raw pulse data for one scan line as a compressed NPZ file.

        The file is written in a background thread so it does not delay the
        scan timing. Files go to data/cscan/lines_raw/line_XXXX.npz.

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
        Save the completed scan as scan_TIMESTAMP.npz + scan_TIMESTAMP_meta.json.

        The NPZ file contains all three feature maps as 2-D arrays [nlines, ncols]
        plus the physical spatial axes in mm. Load with:
            d = np.load("scan_TIMESTAMP.npz")
            amp = d["amplitude"]   # shape (nlines, ncols)
            x   = d["x_mm"]        # physical X axis
            y   = d["y_mm"]        # physical Y axis

        The JSON sidecar records every hardware and acquisition parameter so
        that the scan can be fully understood and reproduced without the source
        code. It should always be kept alongside the NPZ file.

        Both files are uploaded asynchronously to S3 if cloud archival is enabled.
        """
        cfg  = self.config
        ts   = int(time.time())
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
            "timestamp_iso":     datetime.now(timezone.utc).isoformat(),
            "instrument":        "TiePie HS5",
            "fs_hz":             hs_info.get("fs_hz", 20_000_000),
            "detected_prf_hz":   hs_info.get("detected_prf_hz"),
            "gate_us":           hs_info.get("gate_us"),
            "gate_samples":      hs_info.get("gate_samples"),
            "sync_thresholds_v": hs_info.get("sync_thresholds_v"),
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
            json.dump(meta, f, indent=2)

        print(f"[DATA] Saved: {npz_path}  ({rows}×{cols})")
        print(f"[DATA] Meta:  {meta_path}")

        if self.cloud.enabled:
            self.cloud.upload_async(npz_path,  npz_name,  "application/octet-stream")
            self.cloud.upload_async(meta_path, meta_name, "application/json")

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

        if self.cloud.enabled:
            self.cloud.upload_async(
                local_path,
                f"{label.lower()}_{int(time.time())}.png",
                "image/png",
            )

        self._set(images={label: f"/local/{filename}"})

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
            hs = HS5StreamPeaks(fs_hz=20_000_000, feature_mode="envelope").open()
            hs.calibrate_sync(seconds=1.0, verbose=False)

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
                # The +0.3 s margin ensures we don't cut off pulses near the end.
                t0 = time.perf_counter()
                tt, aa, tf, ee, wf = hs.acquire_peaks(
                    duration_s=theo_time + 0.3, save_waveforms=save_wf
                )
                pr.wait_for_completion()
                t1 = time.perf_counter()

                # Crop to pulses detected during the actual printer motion window
                sel  = (tt >= t0) & (tt <= t1)
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

                # Align each row to the most recent row in the same scan direction.
                # This corrects LTR/RTL lateral offset (backlash) line by line.
                if ltr:
                    ra, sh_e = align_row_minblur(ra, ref_e, prev_shift=sh_e)
                    rf, _    = align_row_minblur(rf, ref_e, prev_shift=sh_e)
                    re, _    = align_row_minblur(re, ref_e, prev_shift=sh_e)
                    ref_e    = ra.copy()   # update rolling LTR reference
                else:
                    ra, sh_o = align_row_minblur(ra, ref_o, prev_shift=sh_o)
                    rf, _    = align_row_minblur(rf, ref_o, prev_shift=sh_o)
                    re, _    = align_row_minblur(re, ref_o, prev_shift=sh_o)
                    ref_o    = ra.copy()   # update rolling RTL reference

                img_amp[i] = ra
                img_tof[i] = rf
                img_eng[i] = re

                # Refresh live PNG images every 5 lines (balance I/O vs latency)
                if i % 5 == 0 or i == nlines - 1:
                    self._save_plot(img_amp, "scan_amp.png", "Amplitude")
                    self._save_plot(img_tof, "scan_tof.png", "ToF")
                    self._save_plot(img_eng, "scan_eng.png", "Energy")

            if not self._stop_event.is_set():
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
                }
                self._save_final_npz(img_amp, img_tof, img_eng, hs_info)

            self._set(status="COMPLETED", progress={"msg": "Scan Finished."})

        except Exception as e:
            self._set(status="ERROR", progress={"msg": f"Error: {e}"})
            print(f"[ERROR] {e}")
            traceback.print_exc()
        finally:
            if hs: hs.close()
            if pr: pr.close()
            self.running = False
