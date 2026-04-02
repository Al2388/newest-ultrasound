import os
import csv
import json
import time
import threading
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hs5_control import HS5StreamPeaks


def evaluate_acceptance(quality, min_snr=5.0):
    ok = (
        quality["valid"]
        and not quality["clipped"]
        and np.isfinite(quality["snr_est"])
        and quality["snr_est"] >= min_snr
    )
    return {
        "ok": ok,
        "reason": "ok" if ok else "Signal failed quality threshold",
    }


def compare_to_reference(waveform, ref_waveform):
    if waveform is None or ref_waveform is None:
        return {"corr": np.nan, "rmse": np.nan}
    if len(waveform) == 0 or len(ref_waveform) == 0:
        return {"corr": np.nan, "rmse": np.nan}

    n = min(len(waveform), len(ref_waveform))
    a = waveform[:n].astype(np.float32)
    b = ref_waveform[:n].astype(np.float32)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    corr = float(np.dot(a, b) / denom)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))

    return {"corr": corr, "rmse": rmse}


class AScanMonitorService:
    def __init__(self):
        self.running = False
        self.stop_signal = False
        self.thread = None

        self.status = "IDLE"
        self.progress = {
            "sample": 0,
            "msg": "Ready",
            "run_folder": None,
            "last_quality": None,
            "last_features": None,
            "last_reference_compare": None,
            "plots": {
                "amplitude": None,
                "tof": None,
                "energy": None,
                "last_waveform": None,
            },
        }

        self.config = {
            "cell_id": "Cell01",
            "run_index": "01",
            "sample_interval_s": 2.0,
            "baseline_duration_s": 180.0,
            "out_root": "ascan_runs",
            "fs_hz": 20_000_000,
            "gate_us": (30.0, 40.0),
            "feature_mode": "envelope",
            "average_pulses": 8,
            "ch1_range": 1.0,
            "phase": "monitoring",
        }

        self.reference_waveform = None

    def start_monitor(self, new_config=None):
        if self.running:
            return False, "Already running"

        if new_config:
            self.config.update(new_config)

        try:
            if float(self.config["sample_interval_s"]) <= 0:
                return False, "Sample interval must be > 0"
            if float(self.config["baseline_duration_s"]) < 0:
                return False, "Baseline duration must be >= 0"
            if int(self.config["average_pulses"]) <= 0:
                return False, "average_pulses must be > 0"
        except Exception as e:
            return False, f"Invalid config: {e}"

        self.stop_signal = False
        self.running = True
        self.status = "RUNNING"
        self.reference_waveform = None

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return True, "A-scan monitoring started"

    def stop_monitor(self):
        if not self.running:
            return False
        self.stop_signal = True
        self.status = "STOPPING"
        return True

    def set_phase(self, phase: str):
        self.config["phase"] = phase
        return True, f"Phase set to {phase}"

    def mark_event(self, label, note=""):
        run_folder = self.progress.get("run_folder")
        if not run_folder:
            return False, "No active run"

        event_path = os.path.join(run_folder, "events.csv")
        file_exists = os.path.exists(event_path)
        with open(event_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["timestamp_iso", "label", "note"])
            w.writerow([datetime.now(timezone.utc).isoformat(), label, note])

        self.append_run_note(f"EVENT: {label} | {note}")
        return True, f"Marked event: {label}"

    def append_run_note(self, note):
        run_folder = self.progress.get("run_folder")
        if not run_folder:
            return

        path = os.path.join(run_folder, "run_notes.txt")
        with open(path, "a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).isoformat()
            f.write(f"[{ts}] {note}\n")

    def append_temperature_row(self, cell_temp_c=None, oil_temp_c=None):
        run_folder = self.progress.get("run_folder")
        if not run_folder:
            return False, "No active run"

        path = os.path.join(run_folder, "temp.csv")
        file_exists = os.path.exists(path)
        row = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "cell_temp_c": cell_temp_c,
            "oil_temp_c": oil_temp_c,
        }
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)
        return True, "Temperature row added"

    def append_cycler_row(self, voltage_v=None, current_a=None, capacity_ah=None, step_id=None, flags=None):
        run_folder = self.progress.get("run_folder")
        if not run_folder:
            return False, "No active run"

        path = os.path.join(run_folder, "cycler.csv")
        file_exists = os.path.exists(path)
        row = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "voltage_v": voltage_v,
            "current_a": current_a,
            "capacity_ah": capacity_ah,
            "step_id": step_id,
            "flags": flags,
        }
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)
        return True, "Cycler row added"

    def _make_run_folder(self):
        ts = datetime.now().strftime("%Y%m%d")
        folder_name = f"{ts}_{self.config['cell_id']}_Run{self.config['run_index']}"
        run_folder = os.path.join(self.config["out_root"], folder_name)
        os.makedirs(run_folder, exist_ok=True)
        os.makedirs(os.path.join(run_folder, "waveforms"), exist_ok=True)
        os.makedirs(os.path.join(run_folder, "plots"), exist_ok=True)
        return run_folder

    def _save_settings(self, run_folder):
        settings = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "instrument": "TiePie HS5",
            "logged_channels": [
                "timestamp",
                "raw_waveform",
                "amplitude",
                "tof_us",
                "energy",
                "quality",
                "acceptance",
            ],
        }
        with open(os.path.join(run_folder, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)

    def _append_feature_row(self, run_folder, row):
        path = os.path.join(run_folder, "ultrasound_features.csv")
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)

    def _save_waveform_npz(self, run_folder, sample_idx, waveform, meta, quality, feats, compare, acceptance):
        path = os.path.join(run_folder, "waveforms", f"ascan_{sample_idx:06d}.npz")
        np.savez_compressed(
            path,
            waveform=waveform.astype(np.float32),
            meta=json.dumps(meta),
            quality=json.dumps(quality),
            features=json.dumps(feats),
            compare=json.dumps(compare),
            acceptance=json.dumps(acceptance),
        )

    def _save_waveform_plot(self, run_folder, sample_idx, waveform, fs_hz):
        if waveform is None or waveform.size == 0:
            return

        t_us = np.arange(len(waveform)) / fs_hz * 1e6
        path = os.path.join(run_folder, "plots", f"waveform_{sample_idx:06d}.png")

        plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(t_us, waveform)
        plt.xlabel("Time (us)")
        plt.ylabel("Amplitude (V)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        self.progress["plots"]["last_waveform"] = f"/local_ascan/{os.path.relpath(path, self.config['out_root']).replace(os.sep, '/')}"

    def _update_trend_plot(self, run_folder):
        csv_path = os.path.join(run_folder, "ultrasound_features.csv")
        if not os.path.exists(csv_path):
            return

        times, amps, tofs, engs = [], [], [], []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row["elapsed_s"]))
                amps.append(float(row["amplitude"]))
                tofs.append(float(row["tof_us"]))
                engs.append(float(row["energy"]))

        if not times:
            return

        def save_plot(y, ylabel, filename):
            path = os.path.join(run_folder, "plots", filename)
            plt.figure(figsize=(8, 4), dpi=120)
            plt.plot(times, y)
            plt.xlabel("Elapsed time (s)")
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            return path

        p_amp = save_plot(amps, "Amplitude", "trend_amplitude.png")
        p_tof = save_plot(tofs, "ToF (us)", "trend_tof.png")
        p_eng = save_plot(engs, "Energy", "trend_energy.png")

        self.progress["plots"]["amplitude"] = f"/local_ascan/{os.path.relpath(p_amp, self.config['out_root']).replace(os.sep, '/')}"
        self.progress["plots"]["tof"] = f"/local_ascan/{os.path.relpath(p_tof, self.config['out_root']).replace(os.sep, '/')}"
        self.progress["plots"]["energy"] = f"/local_ascan/{os.path.relpath(p_eng, self.config['out_root']).replace(os.sep, '/')}"

    def _worker(self):
        hs = None
        t_run0 = time.time()

        try:
            run_folder = self._make_run_folder()
            self.progress["run_folder"] = run_folder
            self._save_settings(run_folder)
            self.append_run_note("Run started")

            hs = HS5StreamPeaks(
                fs_hz=self.config["fs_hz"],
                gate_us=self.config["gate_us"],
                feature_mode=self.config["feature_mode"],
                ch1_range=self.config["ch1_range"],
            ).open()

            hs.calibrate_sync(seconds=1.0, verbose=False)
            self.append_run_note("Sync calibrated")

            baseline_duration = float(self.config["baseline_duration_s"])
            sample_interval_s = float(self.config["sample_interval_s"])
            average_pulses = int(self.config["average_pulses"])

            sample_idx = 0

            if baseline_duration > 0:
                self.progress["msg"] = "Collecting baseline A-scans"
                baseline_end = time.time() + baseline_duration

                while time.time() < baseline_end and not self.stop_signal:
                    loop_start = time.time()

                    waveform, meta = hs.acquire_single_ascan(
                        duration_s=0.1,
                        average_pulses=average_pulses,
                    )
                    quality = hs.compute_ascan_quality(waveform)
                    feats = hs.extract_ascan_features(waveform)

                    if self.reference_waveform is None and waveform.size > 0:
                        self.reference_waveform = waveform.copy()
                        self.append_run_note("Reference waveform captured")

                    compare = compare_to_reference(waveform, self.reference_waveform)
                    acceptance = evaluate_acceptance(quality)

                    row = {
                        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                        "elapsed_s": round(time.time() - t_run0, 3),
                        "sample_index": sample_idx,
                        "phase": "baseline",
                        "amplitude": feats["amplitude"],
                        "tof_us": feats["tof_us"],
                        "energy": feats["energy"],
                        "peak": quality["peak"],
                        "rms": quality["rms"],
                        "snr_est": quality["snr_est"],
                        "clipped": quality["clipped"],
                        "ref_corr": compare["corr"],
                        "ref_rmse": compare["rmse"],
                        "accepted": acceptance["ok"],
                        "accept_reason": acceptance["reason"],
                    }

                    self._append_feature_row(run_folder, row)
                    self._save_waveform_npz(run_folder, sample_idx, waveform, meta, quality, feats, compare, acceptance)

                    if sample_idx % 20 == 0:
                        self._save_waveform_plot(run_folder, sample_idx, waveform, meta["fs_hz"])
                    if sample_idx % 10 == 0:
                        self._update_trend_plot(run_folder)

                    if not acceptance["ok"]:
                        self.append_run_note(f"Quality warning at baseline sample {sample_idx}: {acceptance['reason']}")

                    self.progress["sample"] = sample_idx
                    self.progress["last_quality"] = quality
                    self.progress["last_features"] = feats
                    self.progress["last_reference_compare"] = compare

                    sample_idx += 1
                    elapsed = time.time() - loop_start
                    time.sleep(max(0.0, sample_interval_s - elapsed))

                self.mark_event("baseline_complete")

            self.progress["msg"] = "Continuous monitoring"

            while not self.stop_signal:
                loop_start = time.time()

                waveform, meta = hs.acquire_single_ascan(
                    duration_s=0.1,
                    average_pulses=average_pulses,
                )
                quality = hs.compute_ascan_quality(waveform)
                feats = hs.extract_ascan_features(waveform)
                compare = compare_to_reference(waveform, self.reference_waveform)
                acceptance = evaluate_acceptance(quality)

                row = {
                    "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": round(time.time() - t_run0, 3),
                    "sample_index": sample_idx,
                    "phase": self.config.get("phase", "monitoring"),
                    "amplitude": feats["amplitude"],
                    "tof_us": feats["tof_us"],
                    "energy": feats["energy"],
                    "peak": quality["peak"],
                    "rms": quality["rms"],
                    "snr_est": quality["snr_est"],
                    "clipped": quality["clipped"],
                    "ref_corr": compare["corr"],
                    "ref_rmse": compare["rmse"],
                    "accepted": acceptance["ok"],
                    "accept_reason": acceptance["reason"],
                }

                self._append_feature_row(run_folder, row)
                self._save_waveform_npz(run_folder, sample_idx, waveform, meta, quality, feats, compare, acceptance)

                if sample_idx % 20 == 0:
                    self._save_waveform_plot(run_folder, sample_idx, waveform, meta["fs_hz"])
                if sample_idx % 10 == 0:
                    self._update_trend_plot(run_folder)

                if not acceptance["ok"]:
                    self.append_run_note(f"Quality warning at sample {sample_idx}: {acceptance['reason']}")

                self.progress["sample"] = sample_idx
                self.progress["last_quality"] = quality
                self.progress["last_features"] = feats
                self.progress["last_reference_compare"] = compare

                sample_idx += 1
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, sample_interval_s - elapsed))

            self.status = "STOPPED"
            self.progress["msg"] = "Monitoring stopped"
            self.append_run_note("Run stopped")

        except Exception as e:
            self.status = "ERROR"
            self.progress["msg"] = f"Error: {e}"
            self.append_run_note(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if hs is not None:
                try:
                    hs.close()
                except Exception:
                    pass
            self.running = False