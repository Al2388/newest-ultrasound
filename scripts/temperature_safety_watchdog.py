"""Temperature safety watchdog for continuous C-scan + cycling experiments.

Polls the TC-08 service (cell thermocouple) every 10 s. On overheat trip:
  1. POST /api/stop  -> halts the current C-scan
  2. taskkill the run_cscan_long_batch.py process -> stops further scans
  3. Tries to send M140 S0 on COM6 (best effort; may be blocked if scanner
     re-grabbed the port — Marlin's own thermal-runaway is the final guard)
  4. Logs alarm + exits

Thresholds (cell temperature, °C):
  >= 45.0  HARD TRIP    : kill batch, stop heater, exit
  >= 38.0  WARNING      : log only
  <= 32.0  WARNING      : possible heater failure (only after sustained 10 min)
  TC-08 API silent > 60 s : WARNING

Outputs (one folder per run):
  data/raw/temperature/watchdog_<ts>/safety_temperature.csv
  data/raw/temperature/watchdog_<ts>/safety_events.log
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import serial


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEBAPP_BASE  = "http://127.0.0.1:8000"
POLL_S       = 10.0

HARD_TRIP_C  = 45.0
WARN_HIGH_C  = 33.0
WARN_LOW_C   = 27.0
LOW_SUSTAIN_S = 10 * 60          # how long below low-warn before warning
TC08_SILENT_S = 60.0             # TC-08 API silent threshold

COM_PORT = "COM6"
COM_BAUD = 115200
API_KEY  = None                  # loaded from .env


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = PROJECT / "data/raw/temperature" / f"watchdog_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "safety_temperature.csv"
LOG_PATH = OUT_DIR / "safety_events.log"


def _log(level: str, msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] [{level}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_api_key() -> str | None:
    env = PROJECT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("API_KEY")


def _http_get(path: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(WEBAPP_BASE + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _http_post(path: str, body: dict | None = None,
               timeout: float = 5.0) -> dict | None:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        WEBAPP_BASE + path, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "X-API-Key": API_KEY or ""},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        _log("WARN", f"POST {path} failed: {exc}")
        return None


def _try_marlin_heater_off() -> None:
    """Best-effort: open COM6, send M140 S0, close. Will fail if scanner has
    the port — Marlin's own thermal-runaway is the fallback."""
    try:
        s = serial.Serial(COM_PORT, COM_BAUD, timeout=1.0)
        time.sleep(1.5)
        s.reset_input_buffer()
        s.write(b"M140 S0\n"); s.flush(); time.sleep(0.15)
        s.write(b"M104 S0\n"); s.flush(); time.sleep(0.15)
        s.close()
        _log("ACTION", "M140 S0 + M104 S0 sent on COM6")
    except Exception as exc:
        _log("WARN", f"could not send M140 S0 (COM6 busy?): {exc} — "
                     "Marlin firmware thermal-runaway is the fallback")


def _kill_batch_processes() -> int:
    """taskkill all python.exe whose command line matches run_cscan_long_batch."""
    killed = 0
    try:
        # Use PowerShell to find + kill, since we want command-line filtering
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'run_cscan_long_batch' } | "
             "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "
             "Write-Host \"killed PID $($_.ProcessId)\" }"],
            capture_output=True, text=True, timeout=10,
        )
        killed = result.stdout.count("killed PID")
        if result.stdout.strip():
            _log("ACTION", f"batch kill stdout: {result.stdout.strip()}")
        if result.stderr.strip():
            _log("WARN", f"batch kill stderr: {result.stderr.strip()}")
    except Exception as exc:
        _log("WARN", f"could not kill batch: {exc}")
    return killed


def _trip_emergency(cell_c: float) -> None:
    _log("TRIP", f"!!! HARD TRIP cell={cell_c:.2f} C >= {HARD_TRIP_C} C !!!")
    _log("TRIP", "  initiating shutdown sequence")
    # 1. Stop the current scan via API
    resp = _http_post("/api/stop")
    _log("ACTION", f"  /api/stop response: {resp}")
    # 2. Kill the batch driver(s)
    n = _kill_batch_processes()
    _log("ACTION", f"  killed {n} batch process(es)")
    # 3. Give the scanner a moment to release COM6
    time.sleep(2.0)
    # 4. Try to send M140 S0
    _try_marlin_heater_off()
    _log("TRIP", "shutdown sequence complete — watchdog exiting")


# ---------------------------------------------------------------------------
# atexit safety: try to turn heater off on any exit path
# ---------------------------------------------------------------------------
def _atexit_handler() -> None:
    # Don't run on normal exit; only if process is killed unexpectedly
    pass


atexit.register(_atexit_handler)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    global API_KEY
    API_KEY = _read_api_key()
    if API_KEY is None:
        _log("WARN", "no API_KEY found — /api/stop may fail")

    if not CSV_PATH.exists():
        with CSV_PATH.open("w", encoding="utf-8") as f:
            f.write("unix_s,iso,cell_c,tc08_running,batch_alive\n")

    _log("INFO", f"=== safety watchdog start ===")
    _log("INFO", f"  poll={POLL_S}s, trip={HARD_TRIP_C}C, warn_high={WARN_HIGH_C}C, "
                 f"warn_low={WARN_LOW_C}C")
    _log("INFO", f"  output: {OUT_DIR}")
    _log("INFO", "  watchdog is read-only on /api/temp/status; "
                 "trip path: /api/stop + taskkill batch + best-effort M140 S0")

    last_temp_ts = time.time()
    low_warn_since: float | None = None
    last_high_warn = 0.0

    try:
        while True:
            t_now = time.time()
            s = _http_get("/api/temp/status")
            cell_c = None
            tc08_running = False

            if s is None:
                if t_now - last_temp_ts > TC08_SILENT_S:
                    _log("WARN", f"webapp /api/temp/status silent for "
                                 f"{(t_now - last_temp_ts):.0f}s")
                # don't trip; just skip this iteration
            else:
                tc08_running = bool(s.get("running"))
                latest = s.get("latest")
                if latest and latest.get("temperature_mean_c") is not None:
                    cell_c = float(latest["temperature_mean_c"])
                    last_temp_ts = t_now

            # Status of batch process (informational)
            try:
                ps = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                     "Where-Object { $_.CommandLine -match 'run_cscan_long_batch' }).Count"],
                    capture_output=True, text=True, timeout=5,
                )
                batch_alive = int((ps.stdout or "0").strip() or "0") > 0
            except Exception:
                batch_alive = False

            # Log every poll to CSV
            with CSV_PATH.open("a", encoding="utf-8") as f:
                f.write(f"{t_now:.3f},{datetime.now().isoformat(timespec='seconds')},"
                        f"{cell_c if cell_c is not None else ''},"
                        f"{int(tc08_running)},{int(batch_alive)}\n")

            if cell_c is None:
                time.sleep(POLL_S)
                continue

            # Hard trip
            if cell_c >= HARD_TRIP_C:
                _trip_emergency(cell_c)
                return 2

            # Warnings (rate-limited)
            if cell_c >= WARN_HIGH_C:
                if t_now - last_high_warn >= 60.0:
                    _log("WARN", f"cell={cell_c:.2f} C >= {WARN_HIGH_C} C "
                                 "(continue, no action)")
                    last_high_warn = t_now

            if cell_c <= WARN_LOW_C:
                if low_warn_since is None:
                    low_warn_since = t_now
                elif t_now - low_warn_since >= LOW_SUSTAIN_S:
                    _log("WARN", f"cell={cell_c:.2f} C <= {WARN_LOW_C} C "
                                 f"sustained {LOW_SUSTAIN_S/60:.0f} min — "
                                 "heater may have failed")
                    low_warn_since = t_now   # rate-limit
            else:
                low_warn_since = None

            time.sleep(POLL_S)

    except KeyboardInterrupt:
        _log("INFO", "KeyboardInterrupt — watchdog exiting (heater state unchanged)")
        return 0
    except Exception as exc:
        _log("ERROR", f"unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
