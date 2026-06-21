"""Probe camera device indices and save a sample frame from each."""
import cv2
import numpy as np
from pathlib import Path

OUT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main/reports/camera_probe")
OUT.mkdir(parents=True, exist_ok=True)

for idx in range(6):
    for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # Grab a few frames — first ones are often black/green
        for _ in range(5):
            ok, frame = cap.read()
        if ok and frame is not None:
            mean = frame.mean(axis=(0, 1))  # B, G, R
            sigma = frame.std()
            w, h = int(cap.get(3)), int(cap.get(4))
            verdict = "FLAT" if sigma < 5 else "OK"
            out = OUT / f"cam_{idx}_{backend_name}.jpg"
            cv2.imwrite(str(out), frame)
            print(f"idx={idx} backend={backend_name:5s} {w}x{h} BGR_mean={mean} std={sigma:.1f} -> {verdict}  saved {out.name}")
        cap.release()
        break  # one working backend per index is enough

print("done")
