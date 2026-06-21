"""Wait for cell to cool to ~30 C, then auto-launch the 30 C continuous batch.

Same process tree: this script polls /api/temp/status; once the cell is
<= TARGET_MAX_C sustained for STABLE_S seconds, it execs the batch driver
via subprocess.run (blocking), so killing this python.exe will kill the
whole 30 C run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


WEBAPP = "http://127.0.0.1:8000"
TARGET_MAX_C = 30.5
STABLE_S = 5 * 60
POLL_S = 15.0
HEARTBEAT_S = 120.0

PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
TS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = PROJECT / f"data/raw/temperature/cooldown_30c_{TS}"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "cooldown.log"

BATCH_CMD = [
    sys.executable,
    "scripts/run_cscan_long_batch.py",
    "--runs", "9999",
    "--prefix", "longrun_cycling_30c",
    "--min-free-gb", "30",
    "--max-scan-minutes", "15",
]


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_cell_c() -> float | None:
    try:
        with urllib.request.urlopen(f"{WEBAPP}/api/temp/status", timeout=5) as r:
            s = json.loads(r.read())
            lat = s.get("latest") or {}
            v = lat.get("temperature_mean_c")
            return float(v) if v is not None else None
    except Exception:
        return None


def main() -> int:
    log("=== cooldown waiter -> 30 C batch ===")
    log(f"  target: cell <= {TARGET_MAX_C} C sustained {STABLE_S/60:.0f} min")
    log(f"  poll: {POLL_S}s; heartbeat: {HEARTBEAT_S}s")

    below_since: float | None = None
    last_hb = 0.0

    while True:
        t = time.time()
        c = get_cell_c()
        if c is None:
            log("WARN: /api/temp/status read failed; retry")
            time.sleep(POLL_S)
            continue

        if c <= TARGET_MAX_C:
            if below_since is None:
                below_since = t
                log(f"cell={c:.2f} entered <= {TARGET_MAX_C} band -- start stability timer")
            elif t - below_since >= STABLE_S:
                held = t - below_since
                log(f"STABLE: cell={c:.2f} for {held/60:.1f} min -- launching batch")
                break
        else:
            if below_since is not None:
                log(f"cell={c:.2f} left band -- reset timer")
            below_since = None

        if t - last_hb >= HEARTBEAT_S:
            state = "in band" if below_since is not None else "above target"
            log(f"  ... cell={c:.2f} C ({state})")
            last_hb = t

        time.sleep(POLL_S)

    log(f"command: {' '.join(BATCH_CMD)}")
    proc = subprocess.run(BATCH_CMD, cwd=str(PROJECT))
    log(f"batch exited with code {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
