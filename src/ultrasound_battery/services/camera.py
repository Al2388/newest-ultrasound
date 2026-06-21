"""
USB camera live-stream service
==============================
A single capture worker reads frames from the camera and publishes the most
recent JPEG-encoded frame. Multiple HTTP clients can subscribe to the MJPEG
stream without each opening the camera (OpenCV's VideoCapture is not
thread-safe and most webcams refuse two concurrent opens).

Used to put a live view of the scan rig into the dashboard so the operator
can see the probe position from a phone over ngrok.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraService:
    """Publish-subscribe webcam capture with on-demand worker lifecycle.

    The capture thread starts on the first subscriber and stops after the
    last one disconnects — so when nobody is watching the camera is free
    for other applications.
    """

    def __init__(self, device_index: int = 0,
                 width: int = 640, height: int = 480,
                 fps: float = 10.0,
                 jpeg_quality: int = 70):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_period_s = 1.0 / fps
        self.jpeg_quality = int(jpeg_quality)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._latest_ts: float = 0.0
        self._frame_index: int = 0

        self._subscribers: int = 0
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_error: Optional[str] = None
        self._opened = False

    # ------------------------------------------------------------------ status
    def get_status(self) -> dict:
        with self._lock:
            return {
                "device_index": self.device_index,
                "running": self._worker is not None and self._worker.is_alive(),
                "subscribers": self._subscribers,
                "frame_index": self._frame_index,
                "frame_age_s": (time.time() - self._latest_ts) if self._latest_ts else None,
                "width": self.width,
                "height": self.height,
                "last_error": self._last_error,
                "have_frame": self._latest_jpeg is not None,
            }

    # ----------------------------------------------------------- subscribe API
    def subscribe(self) -> None:
        """Register a stream client; start the worker if first subscriber."""
        with self._lock:
            self._subscribers += 1
            if self._worker is None or not self._worker.is_alive():
                self._stop_event.clear()
                self._worker = threading.Thread(target=self._capture_loop,
                                                daemon=True, name="camera-capture")
                self._worker.start()

    def unsubscribe(self) -> None:
        """Deregister a stream client; stop the worker when count hits zero."""
        with self._lock:
            self._subscribers = max(0, self._subscribers - 1)
            if self._subscribers == 0:
                self._stop_event.set()

    def get_frame(self, last_index: int, timeout_s: float = 2.0) -> tuple[bytes, int]:
        """Block until a frame newer than `last_index` is available, or timeout.

        Returns (jpeg_bytes, frame_index). On timeout returns the last frame we
        have (which may be the same as last_index) so the stream doesn't stall.
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._latest_jpeg is not None and self._frame_index > last_index,
                timeout=timeout_s,
            )
            return self._latest_jpeg or b"", self._frame_index

    def snapshot(self) -> Optional[bytes]:
        """Return the latest JPEG without subscribing. Briefly subscribes if
        no frame is cached yet so a snapshot call also works cold."""
        with self._lock:
            if self._latest_jpeg is not None:
                return self._latest_jpeg
        # Cold start — subscribe briefly to get one frame.
        self.subscribe()
        try:
            jpeg, _ = self.get_frame(last_index=self._frame_index, timeout_s=3.0)
            return jpeg if jpeg else None
        finally:
            self.unsubscribe()

    # ------------------------------------------------------------- capture loop
    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        # On Windows the DirectShow backend is usually the most reliable for
        # consumer USB webcams (avoids the MSMF probe delay).
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            self._last_error = f"failed to open camera at index {self.device_index}"
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Read back actual size — drivers often clamp to nearest supported
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
        with self._lock:
            self.width = actual_w
            self.height = actual_h
        return cap

    def _capture_loop(self) -> None:
        cap = self._open_capture()
        if cap is None:
            return
        self._opened = True
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        try:
            while not self._stop_event.is_set():
                t_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._last_error = "frame read failed"
                    time.sleep(0.1)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue
                with self._cond:
                    self._latest_jpeg = buf.tobytes()
                    self._latest_ts = time.time()
                    self._frame_index += 1
                    self._cond.notify_all()
                # Frame-rate throttle: wait the remainder of the target period
                elapsed = time.perf_counter() - t_start
                sleep_s = self.target_period_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            cap.release()
            self._opened = False
