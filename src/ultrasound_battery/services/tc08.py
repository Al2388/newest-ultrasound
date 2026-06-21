"""
USB TC-08 temperature service.

Provides a small ctypes wrapper around Pico's USB TC-08 driver plus a threaded
sampler used by the FastAPI dashboard and C-scan line timing.
"""

from __future__ import annotations

import csv
import ctypes
import ctypes.util
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ultrasound_battery.utils import session_timestamp


class TC08DriverError(RuntimeError):
    """Raised when the USB TC-08 driver or device cannot be opened."""


def _load_tc08_library():
    candidates = [
        os.getenv("USBTC08_DLL"),
        r"D:\SIOT\PicoLog\usbtc08.dll",
        r"C:\Program Files\Pico Technology\PicoLog\usbtc08.dll",
        r"C:\Program Files (x86)\Pico Technology\PicoLog\usbtc08.dll",
        ctypes.util.find_library("usbtc08"),
        ctypes.util.find_library("usb_tc08"),
        "usbtc08.dll",
        "usb_tc08.dll",
        "libusbtc08.so",
    ]
    last_error = None
    for name in [c for c in candidates if c]:
        try:
            return ctypes.WinDLL(name) if os.name == "nt" else ctypes.CDLL(name)
        except OSError as exc:
            last_error = exc
    raise TC08DriverError(
        "USB TC-08 driver DLL not found. Install PicoSDK/PicoLog drivers or set USBTC08_DLL."
    ) from last_error


class USBTC08:
    """Minimal blocking reader for one USB TC-08 unit."""

    UNIT_CENTIGRADE = 0

    def __init__(self):
        self._dll = _load_tc08_library()
        self._configure_signatures()
        self.handle: int | None = None

    def _configure_signatures(self):
        self._dll.usb_tc08_open_unit.argtypes = []
        self._dll.usb_tc08_open_unit.restype = ctypes.c_int16

        self._dll.usb_tc08_close_unit.argtypes = [ctypes.c_int16]
        self._dll.usb_tc08_close_unit.restype = ctypes.c_int16

        self._dll.usb_tc08_set_channel.argtypes = [
            ctypes.c_int16,
            ctypes.c_int16,
            ctypes.c_int8,
        ]
        self._dll.usb_tc08_set_channel.restype = ctypes.c_int16

        self._dll.usb_tc08_get_single.argtypes = [
            ctypes.c_int16,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int16,
        ]
        self._dll.usb_tc08_get_single.restype = ctypes.c_int16

    def open(self, channels: list[int], thermocouple_type: str = "K") -> "USBTC08":
        handle = int(self._dll.usb_tc08_open_unit())
        if handle <= 0:
            raise TC08DriverError(f"usb_tc08_open_unit failed with code {handle}")
        self.handle = handle

        tc_code = (thermocouple_type or "K").strip().upper()[:1].encode("ascii", "ignore")
        if not tc_code:
            tc_code = b"K"

        enabled = set(int(ch) for ch in channels if 1 <= int(ch) <= 8)
        for ch in range(1, 9):
            code = tc_code[0] if ch in enabled else ord(" ")
            ok = int(self._dll.usb_tc08_set_channel(handle, ch, code))
            if ok == 0:
                self.close()
                raise TC08DriverError(f"usb_tc08_set_channel failed for channel {ch}")
        return self

    def close(self):
        if self.handle is not None:
            try:
                self._dll.usb_tc08_close_unit(self.handle)
            finally:
                self.handle = None

    def read_single(self) -> tuple[dict[int, float], int]:
        if self.handle is None:
            raise TC08DriverError("USB TC-08 is not open")
        temps = (ctypes.c_float * 9)()
        overflow = ctypes.c_int16()
        ok = int(
            self._dll.usb_tc08_get_single(
                self.handle,
                temps,
                ctypes.byref(overflow),
                self.UNIT_CENTIGRADE,
            )
        )
        if ok == 0:
            raise TC08DriverError("usb_tc08_get_single failed")
        # Index 0 is cold junction; thermocouple channels are 1..8.
        return {ch: float(temps[ch]) for ch in range(1, 9)}, int(overflow.value)


class TC08Service:
    """Threaded TC-08 logger shared by the webapp and C-scan service."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._device: USBTC08 | None = None
        self._samples = deque(maxlen=200_000)
        self.running = False
        self.status = "IDLE"
        self.progress = {"msg": "Temperature logger idle", "n_samples": 0}
        self.latest: dict | None = None
        self.config = {
            "session_name": "",
            "base_out_dir": "data/raw/temperature",
            "out_dir": "data/raw/temperature",
            "interval_s": 0.1,
            "channels": [1],
            "thermocouple_type": "T",
        }
        self._csv_path: str | None = None

    def start(self, cfg: dict | None = None) -> tuple[bool, str]:
        if self.running:
            return False, "Temperature logger already running"
        if cfg:
            merged = dict(self.config)
            merged.update(cfg)
            self.config = merged

        ts = session_timestamp()
        name = str(self.config.get("session_name", "") or "").strip()
        stem = f"tc08_{name}_{ts}" if name else f"tc08_{ts}"
        out_dir = Path(self.config.get("base_out_dir", "data/raw/temperature")) / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        self.config["out_dir"] = str(out_dir)
        self._csv_path = str(out_dir / "temperature_log.csv")

        self._stop_event.clear()
        self.running = True
        self.status = "STARTING"
        self.progress = {"msg": f"Starting {stem}", "n_samples": 0}
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return True, f"Started: {stem}"

    def stop(self) -> tuple[bool, str]:
        if not self.running:
            return False, "Temperature logger is not running"
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return True, "Stopping temperature logger"

    def _worker(self):
        channels = [int(ch) for ch in self.config.get("channels", [1]) if 1 <= int(ch) <= 8]
        if not channels:
            channels = [1]
        interval_s = max(0.1, float(self.config.get("interval_s", 0.1)))
        tc_type = str(self.config.get("thermocouple_type", "T") or "T")

        try:
            self._device = USBTC08().open(channels, tc_type)
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["unix_s", "timestamp_iso", "overflow"]
                    + [f"ch{ch}_c" for ch in channels]
                    + ["temperature_mean_c"]
                )
                self.status = "LOGGING"
                self.progress["msg"] = "Temperature logging"

                while not self._stop_event.is_set():
                    t_unix = time.time()
                    temps, overflow = self._device.read_single()
                    selected = {ch: temps.get(ch, np.nan) for ch in channels}
                    values = np.asarray(list(selected.values()), dtype=np.float64)
                    mean_c = float(np.nanmean(values)) if values.size else np.nan
                    row = {
                        "unix_s": float(t_unix),
                        "timestamp_iso": datetime.fromtimestamp(t_unix, timezone.utc).isoformat(),
                        "overflow": int(overflow),
                        "channels_c": selected,
                        "temperature_mean_c": mean_c,
                    }
                    writer.writerow(
                        [f"{t_unix:.6f}", row["timestamp_iso"], overflow]
                        + [f"{selected[ch]:.6f}" for ch in channels]
                        + [f"{mean_c:.6f}"]
                    )
                    f.flush()
                    with self._lock:
                        self._samples.append(row)
                        self.latest = row
                        self.progress["n_samples"] = int(self.progress.get("n_samples", 0)) + 1
                    time.sleep(interval_s)
        except Exception as exc:
            self.status = "ERROR"
            self.progress["msg"] = str(exc)
        finally:
            if self._device:
                self._device.close()
                self._device = None
            self.running = False
            if self.status != "ERROR":
                self.status = "IDLE"
                self.progress["msg"] = "Temperature logger stopped"

    def get_status(self) -> dict:
        with self._lock:
            latest = dict(self.latest) if self.latest else None
            if latest and "channels_c" in latest:
                latest["channels_c"] = dict(latest["channels_c"])
            return {
                "running": self.running,
                "status": self.status,
                "progress": dict(self.progress),
                "latest": latest,
                "config": dict(self.config),
                "file": self._csv_path,
            }

    def samples_between(self, start_unix_s: float, end_unix_s: float) -> list[dict]:
        with self._lock:
            return [
                dict(sample)
                for sample in self._samples
                if start_unix_s <= float(sample["unix_s"]) <= end_unix_s
            ]

    def stats_between(self, start_unix_s: float, end_unix_s: float) -> dict:
        samples = self.samples_between(start_unix_s, end_unix_s)
        values = np.asarray(
            [sample.get("temperature_mean_c", np.nan) for sample in samples],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {
                "temperature_mean_c": np.nan,
                "temperature_min_c": np.nan,
                "temperature_max_c": np.nan,
                "n_temperature_samples": 0,
            }
        return {
            "temperature_mean_c": float(np.mean(finite)),
            "temperature_min_c": float(np.min(finite)),
            "temperature_max_c": float(np.max(finite)),
            "n_temperature_samples": int(finite.size),
        }
