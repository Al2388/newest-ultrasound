import os
import time
import math
import threading
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from printer_control_v2 import setup_precision_printer
from hs5_control import HS5StreamPeaks


# ---------------- Helper functions ----------------
def row_from_pulses_nosmooth(arr, ncols, expected_cycles, center_window=True):
    """
    Resample an irregular-length 1D pulse-feature stream into a fixed-length row (ncols).
    Uses weighted binning / linear mixing. Returns float32 array length ncols.
    """
    out = np.full(ncols, np.nan, dtype=np.float32)
    m = int(arr.size)
    if m == 0 or ncols <= 0:
        return out

    W = int(max(8, min(int(expected_cycles), m)))
    start = max(0, (m - W) // 2) if center_window else 0

    core = arr[start:start + W].astype(np.float32)
    valid = np.isfinite(core)
    if not np.any(valid):
        return out

    core = core[valid]
    W = int(core.size)
    if W == 0:
        return out

    # map W samples -> ncols bins
    x = (np.arange(W, dtype=np.float64) + 0.5) * (ncols / W)
    i0 = np.floor(x).astype(np.int64)
    w1 = x - i0
    w0 = 1.0 - w1
    i1 = i0 + 1

    sumw = np.zeros(ncols, dtype=np.float64)
    acc = np.zeros(ncols, dtype=np.float64)

    m0 = (i0 >= 0) & (i0 < ncols)
    if np.any(m0):
        np.add.at(sumw, i0[m0], w0[m0])
        np.add.at(acc, i0[m0], w0[m0] * core[m0])

    m1 = (i1 >= 0) & (i1 < ncols)
    if np.any(m1):
        np.add.at(sumw, i1[m1], w1[m1])
        np.add.at(acc, i1[m1], w1[m1] * core[m1])

    nz = sumw > 1e-12
    out[nz] = (acc[nz] / sumw[nz]).astype(np.float32)

    # fill short NaN gaps left-to-right
    for k in range(1, ncols):
        if np.isnan(out[k]) and not np.isnan(out[k - 1]):
            out[k] = out[k - 1]

    return out


def align_row_minblur(row, ref, max_shift=8.0, prev_shift=0.0):
    """
    Phase-correlation alignment between row and ref.
    Uses a sanity check (PSR + jump limit) to avoid jitter.
    Returns (shifted_row, used_shift).
    """
    if ref is None or not np.any(np.isfinite(ref)):
        return row, 0.0

    a = np.nan_to_num(row, nan=0.0).astype(np.float64)
    a -= np.mean(a)
    b = np.nan_to_num(ref, nan=0.0).astype(np.float64)
    b -= np.mean(b)

    n = len(a)
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    R = fa * np.conj(fb)
    R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.irfft(R, n=n)

    k0 = int(np.argmax(c))

    # Sub-sample peak estimate via parabola fit
    denom = (c[(k0 - 1) % n] - 2 * c[k0] + c[(k0 + 1) % n])
    if abs(denom) > 1e-12:
        delta = 0.5 * (c[(k0 - 1) % n] - c[(k0 + 1) % n]) / denom
    else:
        delta = 0.0

    shift_est = float(k0 + delta)
    if shift_est > n / 2:
        shift_est -= n
    shift_est = float(np.clip(shift_est, -max_shift, max_shift))

    # Peak-to-sidelobe ratio check
    mask = np.ones_like(c, dtype=bool)
    lo, hi = (k0 - 5) % n, (k0 + 5) % n
    if lo <= hi:
        mask[lo:hi + 1] = False
    else:
        mask[:hi + 1] = False
        mask[lo:] = False

    if np.std(c[mask]) == 0:
        psr = 0.0
    else:
        psr = float((c[k0] - np.mean(c[mask])) / (np.std(c[mask]) + 1e-12))

    # Reject unstable shifts
    if psr < 6.0 or abs(shift_est - prev_shift) > 1.8:
        shift_use = float(prev_shift)
    else:
        shift_use = float(shift_est)

    # Apply shift in frequency domain
    k = np.fft.rfftfreq(n)
    row_fft = np.fft.rfft(np.nan_to_num(row, nan=0.0).astype(np.float64))
    shifted = np.fft.irfft(row_fft * np.exp(-2j * np.pi * k * shift_use), n=n)

    return shifted.astype(np.float32), shift_use


# ---------------- Main service ----------------
class CScanService:
    def __init__(self):
        self.running = False
        self.stop_signal = False
        self.thread = None

        self.status = "IDLE"
        self.progress = {"line": 0, "total": 0, "msg": "Ready"}

        self.config = {
            "roi_w": 50.0,
            "roi_h": 50.0,
            "pitch": 0.1,
            "speed": 10.0,
            "cols": 500,
            "out_dir": "cscan_out",
            "cmap": "turbo",
        }

        # served by FastAPI mount: /local -> cscan_out
        self.images = {"Amplitude": None, "ToF": None, "Energy": None}

    def start_scan(self, new_config=None):
        if self.running:
            return False, "Already Running"

        if new_config:
            self.config.update(new_config)

        self.stop_signal = False
        self.running = True
        self.status = "RUNNING"

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return True, "Started"

    def stop_scan(self):
        if self.running:
            self.stop_signal = True
            self.status = "STOPPING"
            return True
        return False

    def return_to_start(self):
        if self.running:
            return False, "Stop scan first."
        try:
            self.status = "MOVING"
            pr, _, _, _, _ = setup_precision_printer(
                "COM6", 115200,
                self.config["roi_w"], self.config["roi_h"],
                reset_origin=False
            )
            pr.move_to_position(0.0, 0.0, fast=True)
            pr.wait_for_completion()
            pr.close()

            self.status = "IDLE"
            self.progress["msg"] = "Returned to Start."
            return True, "Returned"
        except Exception as e:
            self.status = "ERROR"
            return False, str(e)

    def jog_z_axis(self, z_distance):
        if self.running:
            return False, "Cannot move Z while scanning."
        try:
            pr, _, _, _, _ = setup_precision_printer(
                "COM6", 115200,
                self.config["roi_w"], self.config["roi_h"],
                reset_origin=False
            )
            pr.send_command("G91")
            pr.send_command(f"G1 Z{float(z_distance):.3f} F300")
            pr.wait_for_completion()
            pr.send_command("G90")
            pr.close()
            return True, f"Moved Z by {z_distance}mm"
        except Exception as e:
            return False, str(e)

    # -------- local-only save methods (NO AWS) --------
    def _save_full_data_package_local(self, amp_arr, tof_arr, eng_arr):
        ts = int(time.time())
        filename = f"scan_data_package_{ts}.json"
        out_dir = self.config["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        local_path = os.path.join(out_dir, filename)

        rows, cols = amp_arr.shape
        y_axis = np.linspace(0, self.config["roi_h"], rows).tolist()
        x_axis = np.linspace(0, self.config["roi_w"], cols).tolist()

        data_package = {
            "experiment_info": {
                "id": f"scan_{ts}",
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "operator": "IoT_System",
            },
            "acquisition_settings": {
                "instrument": "TiePie HS5",
                "sampling_rate_hz": 20_000_000,
                "gate_window_us": {"start": 30.0, "end": 40.0},
            },
            "spatial_axes": {"x_mm": x_axis, "y_mm": y_axis, "units": "millimeters"},
            "features": {
                "amplitude": np.nan_to_num(amp_arr).tolist(),
                "time_of_flight": np.nan_to_num(tof_arr).tolist(),
                "pulse_energy": np.nan_to_num(eng_arr).tolist(),
            },
        }

        with open(local_path, "w") as f:
            json.dump(data_package, f)

        print(f"[DATA] Saved Package: {filename}")

    def _save_plot(self, img, name, label):
        if not np.any(np.isfinite(img)):
            return

        out_dir = self.config["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        local_path = os.path.join(out_dir, name)

        cm = matplotlib.colormaps.get_cmap(self.config["cmap"]).copy()
        cm.set_bad("white")

        finite = img[np.isfinite(img)]
        if finite.size >= 16:
            vmin, vmax = np.percentile(finite, [5, 95])
        else:
            vmin, vmax = 0, 1

        plt.figure(figsize=(8, 6), dpi=100)
        plt.imshow(
            img,
            cmap=cm,
            origin="upper",
            aspect="equal",
            extent=[0, self.config["roi_w"], self.config["roi_h"], 0],
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.title(f"{label} (Line {self.progress['line']}/{self.progress['total']})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(local_path, bbox_inches="tight")
        plt.close()

        self.images[label] = f"/local/{name}"

    # -------- scan worker --------
    def _worker(self):
        cfg = self.config
        os.makedirs(cfg["out_dir"], exist_ok=True)

        nlines = max(2, int(math.ceil(cfg["roi_h"] / cfg["pitch"])))
        self.progress["total"] = nlines

        img_amp = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)
        img_tof = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)
        img_eng = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)

        pr = None
        hs = None

        try:
            self.progress["msg"] = "Initializing Hardware..."

            pr, xl, xr, ys, ye = setup_precision_printer(
                "COM6", 115200, cfg["roi_w"], cfg["roi_h"], reset_origin=True
            )

            hs = HS5StreamPeaks(fs_hz=20_000_000, feature_mode="envelope").open()
            hs.calibrate_sync(seconds=1.0, verbose=False)

            theo_time = abs(xr - xl) / float(cfg["speed"])
            detected_prf = float(getattr(hs, "detected_prf", 5000.0) or 5000.0)
            expected_cycles = int(detected_prf * theo_time)

            ref_e, ref_o = None, None
            sh_e, sh_o = 0.0, 0.0

            for i in range(nlines):
                if self.stop_signal:
                    break

                self.progress["line"] = i + 1
                self.progress["msg"] = f"Scanning Line {i + 1}..."

                y = ys + i * float(cfg["pitch"])
                ltr = (i % 2 == 0)  # serpentine
                x0, x1 = (xl, xr) if ltr else (xr, xl)

                # Move to line start
                pr.move_to_position(x0, y, fast=True)
                pr.wait_for_completion()
                time.sleep(0.05)

                # Start traverse (non-blocking)
                pr.send_command(f"G1 X{x1:.3f} F{int(float(cfg['speed']) * 60)}")

                # Acquire features while moving
                t0 = time.perf_counter()
                tt, aa, tf, ee = hs.acquire_peaks(duration_s=theo_time + 0.3)
                pr.wait_for_completion()
                t1 = time.perf_counter()

                # Keep only motion window
                sel = (tt >= t0) & (tt <= t1)
                aa, tf, ee = aa[sel], tf[sel], ee[sel]

                # If right->left, reverse stream so it maps left->right in image space
                if not ltr:
                    aa, tf, ee = aa[::-1], tf[::-1], ee[::-1]

                # Resample streams to a fixed-width row
                ra = row_from_pulses_nosmooth(aa, cfg["cols"], expected_cycles)
                rf = row_from_pulses_nosmooth(tf, cfg["cols"], expected_cycles)
                re = row_from_pulses_nosmooth(ee, cfg["cols"], expected_cycles)

                # Align even and odd lines independently
                if ltr:
                    ra, sh_e = align_row_minblur(ra, ref_e, prev_shift=sh_e)
                    rf, _ = align_row_minblur(rf, ref_e, prev_shift=sh_e)
                    re, _ = align_row_minblur(re, ref_e, prev_shift=sh_e)
                    if ref_e is None:
                        ref_e = ra
                else:
                    ra, sh_o = align_row_minblur(ra, ref_o, prev_shift=sh_o)
                    rf, _ = align_row_minblur(rf, ref_o, prev_shift=sh_o)
                    re, _ = align_row_minblur(re, ref_o, prev_shift=sh_o)
                    if ref_o is None:
                        ref_o = ra

                # Write row into maps
                img_amp[i, :] = ra
                img_tof[i, :] = rf
                img_eng[i, :] = re

                # Save periodically
                if i % 5 == 0 or i == nlines - 1:
                    self._save_plot(img_amp, "scan_amp.png", "Amplitude")
                    self._save_plot(img_tof, "scan_tof.png", "ToF")
                    self._save_plot(img_eng, "scan_eng.png", "Energy")
                    self._save_full_data_package_local(img_amp, img_tof, img_eng)

            # Return home if finished normally
            if not self.stop_signal and pr is not None:
                self.progress["msg"] = "Returning to Start..."
                pr.move_to_position(0.0, 0.0, fast=True)
                pr.wait_for_completion()

            self.status = "COMPLETED" if not self.stop_signal else "STOPPED"
            self.progress["msg"] = "Scan Finished." if not self.stop_signal else "Scan Stopped."

        except Exception as e:
            self.status = "ERROR"
            self.progress["msg"] = f"Error: {str(e)}"
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

        finally:
            if hs is not None:
                try:
                    hs.close()
                except Exception:
                    pass
            if pr is not None:
                try:
                    pr.close()
                except Exception:
                    pass
            self.running = False