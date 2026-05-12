"""
FastAPI Web Server — Ultrasound Battery Lab
============================================
Entry point for the IoT acquisition system. Provides the HTTP API that the
web dashboard uses to control hardware and stream live data.

Two acquisition modes share this server:

  C-Scan  (CScanService)    — 2-D raster scan over the battery surface.
                              Requires: TiePie HS5 oscilloscope + Ender printer.
                              Outputs:  feature map PNGs + NPZ archive.

  A-Scan  (AScanService)    — Continuous single-point monitoring during cycling.
                              Requires: TiePie HS5 oscilloscope only.
                              Outputs:  HDF5 time-series + annotations JSON.

Both modes use the same HS5 hardware. The _hardware_free() dependency enforces
mutual exclusion — if one mode is running, attempting to start the other returns
HTTP 409 Conflict.

Authentication
--------------
All state-changing endpoints require an X-API-Key header.
  - Set API_KEY in .env for a persistent key.
  - If not set, a random key is auto-generated on each server start and
    printed to the console (useful for first-run testing).
  - Read-only endpoints (status, images, metadata) are unauthenticated so
    the dashboard can poll them freely without storing the key client-side.

Running locally
---------------
  python server.py
  Open http://localhost:8000 in a browser.

Remote access (via ngrok)
--------------------------
  tools/ngrok.exe http 8000
  The ngrok URL is printed to the console; share it to access the dashboard
  from another machine or mobile device.

API endpoint summary
---------------------
  GET  /                        Web dashboard (HTML)

  C-scan:
  POST /api/start               Start scan  (key required, hardware must be free)
  POST /api/stop                Stop scan   (key required)
  POST /api/return              Return probe to origin  (key required)
  GET  /api/status              Live scan status, progress, image URLs
  GET  /api/data/latest         Download latest scan NPZ  (key required)
  GET  /api/data/meta           Metadata JSON for latest scan

  A-scan:
  POST /api/ascan/start         Start monitoring session  (key + hardware free)
  POST /api/ascan/stop          Stop session  (key required)
  POST /api/ascan/mark          Add SOC/event annotation  (key required)
  GET  /api/ascan/status        Live session status, waveform, history
  GET  /api/ascan/download      Download active/latest .h5 file  (key required)

  Gauging (live ToF readout, HDF5 archived):
  POST /api/gauge/start         Start live ToF stream  (key + hardware free)
  POST /api/gauge/stop          Stop the ToF stream  (key required)
  GET  /api/gauge/status        Latest ToF / amplitude / history
  GET  /api/gauge/download      Download active/latest .h5 file  (key required)
"""

import glob
import json
import os
import secrets

from fastapi import FastAPI, Depends, HTTPException, Request, Security
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Literal

from pydantic import BaseModel, Field, model_validator
import uvicorn

from scanner_service import CScanService
from ascan_service import AScanService
from gauging_service import GaugingService


# =============================================================================
# Authentication
# =============================================================================

# Prefer an explicit API_KEY from .env (or the shell environment).
# If absent, generate a fresh random key — printed once at startup so the
# operator can use the dashboard immediately without any configuration.
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    API_KEY = secrets.token_urlsafe(16)
    print(f"[WARN] API_KEY not set. Using generated key: {API_KEY}")
    print(f"[WARN] To persist it, add  API_KEY={API_KEY}  to your .env file.")

_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_key(key: str = Security(_key_header)):
    """
    FastAPI dependency — raises HTTP 403 if the X-API-Key header is missing
    or does not match API_KEY.
    Applied to all mutating endpoints (start, stop, jog, mark, download).
    """
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


def _hardware_free():
    """
    FastAPI dependency — raises HTTP 409 if any acquisition mode is using the
    TiePie HS5 oscilloscope or the printer.

    All three services (C-scan, A-scan, Gauging) share the same physical
    oscilloscope and the same printer serial port. This guard prevents them
    from being started concurrently, which would cause libtiepie errors,
    serial conflicts, or corrupt data.
    """
    if scanner.running:
        raise HTTPException(status_code=409, detail="C-scan is currently active.")
    if ascan.running:
        raise HTTPException(status_code=409, detail="A-scan session is currently active.")
    if gauge.running:
        raise HTTPException(status_code=409, detail="Gauging is currently active.")


def _latest_matching_file(base_dir: str, pattern: str) -> str | None:
    """Return the newest file matching pattern under base_dir, including run folders."""
    files = set(glob.glob(os.path.join(base_dir, pattern)))
    files.update(glob.glob(os.path.join(base_dir, "**", pattern), recursive=True))
    files = [p for p in files if os.path.isfile(p)]
    return max(files, key=os.path.getmtime) if files else None


# =============================================================================
# Request / response models  (Pydantic v2)
# =============================================================================

class ScanConfig(BaseModel):
    """Configuration for starting a 2-D C-scan."""
    scan_name:      str   = Field("",
                                  description="Label included in the scan folder")
    roi_w:          float = Field(50.0, gt=0, le=500,
                                  description="Scan region width (mm)")
    roi_h:          float = Field(50.0, gt=0, le=500,
                                  description="Scan region height (mm)")
    pitch:          float = Field(0.1,  gt=0, le=10,
                                  description="Spacing between scan lines (mm)")
    speed:          float = Field(10.0, gt=0, le=200,
                                  description="Transducer scan speed (mm/s)")
    cols:           int   = Field(500,  gt=0, le=5000,
                                  description="Output image width in pixels")
    cmap:           str   = "turbo"    # matplotlib colormap name for PNG exports
    save_waveforms: bool  = True       # False = skip raw waveform NPZ (faster scan)


class AScanConfig(BaseModel):
    """Configuration for starting a continuous A-scan monitoring session."""
    interval_s:    float = Field(1.0,  gt=0.05, le=60.0,
                                 description="Seconds between averaged snapshots")
    session_name:  str   = Field("",
                                 description="Label included in the output folder and filename")
    gate_us_start: float = Field(25.0, gt=0, le=200.0,
                                 description="Gate window start (µs from sync edge)")
    gate_us_end:   float = Field(50.0, gt=0, le=200.0,
                                 description="Gate window end (µs from sync edge)")
    ch1_range:     float = Field(1.0,  gt=0, le=10.0,
                                 description="CH1 voltage range (V)")
    tracking_ref_n: int = Field(60, ge=1, le=1000,
                                description="Opening snapshots used as the A-scan tracking reference")
    tracking_max_lag_us: float = Field(2.0, gt=0, le=25.0,
                                       description="Maximum ToF drift allowed from the reference echo")
    raw_mode: Literal["off", "window", "decimated", "full"] = Field(
        "off",
        description=("Raw pulse retention strategy. 'off' (default) saves only the "
                     "coherent average per snapshot — ~1000× smaller than 'full'. "
                     "'window' archives every raw pulse for the first raw_window_s "
                     "seconds. 'decimated' keeps every raw_decimate_k-th pulse for "
                     "the whole session."))
    raw_window_s: float = Field(60.0, gt=0, le=86400,
                                 description="raw_mode='window' duration in seconds")
    raw_decimate_k: int = Field(100, ge=2, le=10000,
                                 description="raw_mode='decimated' keeps every K-th pulse")

    @model_validator(mode="before")
    @classmethod
    def _legacy_save_raw_waveforms(cls, data):
        # Accept the deprecated `save_raw_waveforms: bool` and translate it
        # before normal validation, so existing clients keep working without
        # the legacy field leaking into the validated model.
        if isinstance(data, dict) and "save_raw_waveforms" in data:
            data = dict(data)
            legacy = data.pop("save_raw_waveforms")
            if "raw_mode" not in data:
                data["raw_mode"] = "full" if legacy else "off"
        return data


class EventMark(BaseModel):
    """SOC/event annotation for the A-scan session event log."""
    soc_pct: float | None = Field(None, ge=0, le=100,
                                   description="Battery SOC in percent, or null")
    label:   str          = Field("",
                                   description="Short description, e.g. 'CC charge'")


class GaugeConfig(BaseModel):
    """Configuration for starting a live ToF gauging session."""
    session_name:  str   = Field("",   description="Label included in the output folder and HDF5 filename")
    gate_us_start: float = Field(25.0, gt=0, le=200.0,
                                 description="Gate window start (µs from sync edge)")
    gate_us_end:   float = Field(50.0, gt=0, le=200.0,
                                 description="Gate window end (µs from sync edge)")
    ch1_range:     float = Field(1.0,  gt=0, le=10.0,
                                 description="CH1 voltage range (V)")


# =============================================================================
# Application setup
# =============================================================================

app     = FastAPI(title="Ultrasound Battery Lab", version="2.6")
scanner = CScanService()
ascan   = AScanService(cloud=scanner.cloud)   # share the CloudManager instance
gauge   = GaugingService(cloud=scanner.cloud) # HS5-only; HDF5 archival to data/gauging/

templates = Jinja2Templates(directory="templates")

# Static file mounts let the browser load scan PNGs directly from the server
# filesystem rather than having them embedded in JSON responses.
# The URLs /local/... and /aslocal/... map to the data output directories.
for _static_dir in ("data/cscan", "data/ascan", "data/gauging"):
    os.makedirs(_static_dir, exist_ok=True)
app.mount("/local",   StaticFiles(directory="data/cscan"),   name="local")
app.mount("/aslocal", StaticFiles(directory="data/ascan"),   name="aslocal")
app.mount("/glocal",  StaticFiles(directory="data/gauging"), name="glocal")


# =============================================================================
# Dashboard
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Serve the single-page web dashboard.

    The API key is injected into the local dashboard so browser actions can send
    the X-API-Key header on protected endpoints.
    """
    return templates.TemplateResponse("index.html",
                                      {"request": request, "api_key": API_KEY})


# =============================================================================
# C-scan endpoints
# =============================================================================

@app.post("/api/start", dependencies=[Depends(_require_key), Depends(_hardware_free)])
async def start_scan(cfg: ScanConfig):
    """
    Start a new 2-D C-scan with the given configuration.

    Requires: valid API key + neither C-scan nor A-scan already running.
    The scan runs asynchronously in a worker thread; poll /api/status for progress.
    """
    success, msg = scanner.start_scan(cfg.model_dump())
    return {"success": success, "msg": msg}


@app.post("/api/stop", dependencies=[Depends(_require_key)])
async def stop_scan():
    """
    Request an early stop of the running C-scan.

    The worker finishes the current scan line, returns the probe to the origin,
    and saves whatever feature maps have been collected so far.
    """
    if scanner.stop_scan():
        return {"msg": "Stopping..."}
    return {"msg": "Not running"}


@app.post("/api/return", dependencies=[Depends(_require_key)])
async def return_home():
    """
    Drive the probe back to the scan origin (0, 0).

    Safe to call after the scan has finished or been stopped. Requires the
    printer to be idle (not scanning).
    """
    success, msg = scanner.return_to_start()
    return {"success": success, "msg": msg}


@app.get("/api/status")
async def status():
    """
    Return the current C-scan state, progress, and live image URLs.

    Unauthenticated so the dashboard can poll it at 1 Hz without sending the
    API key on every request.
    """
    return scanner.get_status()


@app.get("/api/data/latest", dependencies=[Depends(_require_key)])
async def download_latest_scan():
    """
    Download the most recently completed C-scan as a compressed NPZ file.

    The NPZ contains: amplitude, tof, energy [nlines × ncols float32 arrays]
    plus x_mm and y_mm spatial axis arrays. Load with np.load().
    """
    base_dir = scanner.config.get("base_out_dir", scanner.config["out_dir"])
    path = _latest_matching_file(base_dir, "scan_*.npz")
    if not path:
        raise HTTPException(status_code=404, detail="No scan data available yet.")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=os.path.basename(path))


@app.get("/api/data/meta")
async def get_latest_meta():
    """
    Return the acquisition metadata JSON for the most recently completed scan.

    Contains all hardware parameters (fs, PRF, gate, thresholds) and scan
    geometry (ROI, pitch, speed) needed to interpret or reproduce the scan.
    Unauthenticated — metadata does not contain credentials or raw data.
    """
    base_dir = scanner.config.get("base_out_dir", scanner.config["out_dir"])
    path = _latest_matching_file(base_dir, "scan_*_meta.json")
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


# =============================================================================
# A-scan endpoints
# =============================================================================

@app.post("/api/ascan/start", dependencies=[Depends(_require_key), Depends(_hardware_free)])
async def ascan_start(cfg: AScanConfig):
    """
    Start a new A-scan monitoring session.

    Requires: valid API key + neither C-scan nor A-scan already running.
    The session runs asynchronously; poll /api/ascan/status for live data.
    """
    success, msg = ascan.start_session(cfg.model_dump())
    return {"success": success, "msg": msg}


@app.post("/api/ascan/stop", dependencies=[Depends(_require_key)])
async def ascan_stop():
    """
    Stop the active A-scan session.

    The worker finishes the current snapshot interval, flushes the HDF5 file,
    and uploads it to S3 if cloud archival is enabled.
    """
    success, msg = ascan.stop_session()
    return {"success": success, "msg": msg}


@app.post("/api/ascan/mark", dependencies=[Depends(_require_key)])
async def ascan_mark(req: EventMark):
    """
    Add a timestamped SOC/event annotation to the current session.

    The annotation is written to the in-memory list (for dashboard display)
    and to the JSON sidecar file on disk (for permanent archival).
    Call this each time the battery SOC changes or a notable event occurs.
    """
    success, msg = ascan.mark_event(req.soc_pct, req.label)
    return {"success": success, "msg": msg}


@app.get("/api/ascan/status")
async def ascan_status():
    """
    Return the current A-scan session state including the latest waveform,
    rolling feature history, and annotation list.

    Unauthenticated so the dashboard can poll it at 1 Hz.
    """
    return ascan.get_status()


@app.get("/api/ascan/download", dependencies=[Depends(_require_key)])
async def ascan_download():
    """
    Download the active or most recently completed A-scan HDF5 file.

    If a session is currently running, the HDF5 file is returned as-is
    (it is safely readable while open because h5py uses internal locking).
    Otherwise the most recent file in the output directory is returned.
    """
    if ascan._h5_path and os.path.exists(ascan._h5_path):
        path = ascan._h5_path
    else:
        base_dir = ascan.config.get("base_out_dir", ascan.config["out_dir"])
        path = _latest_matching_file(base_dir, "ascan_*.h5")
        if not path:
            raise HTTPException(status_code=404, detail="No A-scan data available.")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=os.path.basename(path))


# =============================================================================
# Gauging endpoints (live ToF readout)
# =============================================================================

@app.post("/api/gauge/start", dependencies=[Depends(_require_key), Depends(_hardware_free)])
async def gauge_start(cfg: GaugeConfig):
    """
    Start a live ToF gauging session.

    Requires: valid API key + no other acquisition mode running.
    Opens the HS5 and begins continuous ToF streaming.
    """
    success, msg = gauge.start(cfg.model_dump())
    return {"success": success, "msg": msg}


@app.post("/api/gauge/stop", dependencies=[Depends(_require_key)])
async def gauge_stop():
    """Stop the active gauging session and release the HS5."""
    success, msg = gauge.stop()
    return {"success": success, "msg": msg}


@app.get("/api/gauge/status")
async def gauge_status():
    """Live gauging status: latest ToF, amplitude, pulse count, ToF history."""
    return gauge.get_status()


@app.get("/api/gauge/download", dependencies=[Depends(_require_key)])
async def gauge_download():
    """
    Download the active or most recently completed gauging HDF5 file.

    Safe to call while a session is still running — h5py uses internal locking
    so the file is readable at the last flush boundary (every ~0.3 s).
    """
    if gauge._h5_path and os.path.exists(gauge._h5_path):
        path = gauge._h5_path
    else:
        base_dir = gauge.config.get("base_out_dir", gauge.config["out_dir"])
        path = _latest_matching_file(base_dir, "gauge_*.h5")
        if not path:
            raise HTTPException(status_code=404, detail="No gauging data available.")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=os.path.basename(path))


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # Run with uvicorn on all interfaces so the dashboard is reachable on the LAN.
    # For remote access outside the lab network, start ngrok separately:
    #   tools/ngrok.exe http 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
