"""Stepped heat-up of the oil bath to ~35 C using Ender-3 bed PID.

Strategy (per operator notes — thermal inertia is significant):
    30.0 -> 32.0 -> 33.5 -> 34.5    (let natural overshoot land near 35)

Each step waits for |B - target| < 0.3 C for 3 min, or 25 min hard cap.
Final step holds indefinitely; Ctrl+C cleanly sends M140 S0.

Safety
------
- B >= 38.0 C : auto-send M140 S0 and abort the script.
- B >= 40.0 C : same plus large warning — operator must cut 24 V PSU.
- atexit + try/finally guarantee M140 S0 is sent before exit.
"""
from __future__ import annotations

import atexit
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import serial


PORT = "COM6"
BAUD = 115200

STEPS = [30.0, 32.0, 33.5, 34.5]
TOL_C = 0.3
STABLE_S = 180           # 3 min stable in tolerance band
MAX_STEP_S = 25 * 60     # 25 min cap per step
POLL_S = 5
SAFE_LIMIT_C = 38.0
HARD_LIMIT_C = 40.0


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
TS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = PROJECT / "data/raw/temperature" / f"heatup_to_35c_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "heatup_log.csv"
LOG_PATH = OUT_DIR / "heatup_progress.log"


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def write_log(msg: str) -> None:
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------
_ser: serial.Serial | None = None


def send(cmd: str) -> None:
    if _ser is None:
        return
    try:
        _ser.write((cmd.rstrip() + "\n").encode())
        _ser.flush()
    except Exception as exc:
        write_log(f"send failed for '{cmd}': {exc}")


def read_drain(min_wait_s: float = 0.4) -> str:
    """Read all data currently available, with a short settle."""
    if _ser is None:
        return ""
    time.sleep(min_wait_s)
    buf = b""
    while _ser.in_waiting:
        buf += _ser.read(_ser.in_waiting)
        time.sleep(0.05)
    return buf.decode(errors="replace")


_M105_RE = re.compile(r"B:([\d.\-]+)\s*/([\d.\-]+).*?B@:(\d+)")


def parse_m105(text: str) -> tuple[float, float, int] | None:
    m = _M105_RE.search(text)
    if m:
        return float(m.group(1)), float(m.group(2)), int(m.group(3))
    return None


def safe_shutdown() -> None:
    """Always-on safety guard — registered with atexit + called by every error
    path. Sends M140 S0 and M104 S0 (hotend, just in case)."""
    if _ser is None:
        return
    try:
        for cmd in ("M140 S0", "M104 S0"):
            _ser.write((cmd + "\n").encode())
            _ser.flush()
            time.sleep(0.1)
        write_log("safe_shutdown: M140 S0 + M104 S0 sent")
    except Exception as exc:
        write_log(f"safe_shutdown ERROR: {exc}")


atexit.register(safe_shutdown)


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------
def csv_init() -> None:
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", encoding="utf-8") as f:
            f.write("unix_s,timestamp_iso,step_target_c,bed_c,bed_target_pid_c,bed_pwm\n")


def csv_append(step_target: float, bed_c: float, bed_target: float, pwm: int) -> None:
    with CSV_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f},{datetime.now().isoformat(timespec='seconds')},"
                f"{step_target:.2f},{bed_c:.3f},{bed_target:.2f},{pwm}\n")


def poll_once(step_target: float) -> tuple[float, float, int] | None:
    """Send M105, read response, parse, log, return (bed, bed_target_pid, pwm).
    Also performs safety checks; raises RuntimeError if a limit is hit."""
    send("M105")
    text = read_drain(min_wait_s=0.4)
    parsed = parse_m105(text)
    if parsed is None:
        return None
    bed_c, bed_target, pwm = parsed
    csv_append(step_target, bed_c, bed_target, pwm)

    if bed_c >= HARD_LIMIT_C:
        write_log(f"!!! HARD LIMIT B={bed_c:.2f} C >= {HARD_LIMIT_C} — "
                  "ABORT and PHYSICALLY CUT 24V PSU !!!")
        safe_shutdown()
        raise RuntimeError("hard temperature limit")
    if bed_c >= SAFE_LIMIT_C:
        write_log(f"!!! SAFE LIMIT B={bed_c:.2f} C >= {SAFE_LIMIT_C} — "
                  "auto-aborting via M140 S0")
        safe_shutdown()
        raise RuntimeError("safe temperature limit")

    return bed_c, bed_target, pwm


def run_step(step_target: float) -> bool:
    """Drive one step; return True if reached stable, False if hit time cap."""
    write_log(f"--- STEP: target {step_target:.1f} C ---")
    send(f"M140 S{step_target:.2f}")
    step_start = time.time()
    stable_start: float | None = None

    while True:
        time.sleep(POLL_S)
        result = poll_once(step_target)
        if result is None:
            continue
        bed_c, bed_target, pwm = result
        elapsed = time.time() - step_start

        # Stability check
        if abs(bed_c - step_target) <= TOL_C:
            if stable_start is None:
                stable_start = time.time()
                write_log(f"  B={bed_c:.2f} entered tolerance band — start stability timer")
            elif time.time() - stable_start >= STABLE_S:
                write_log(f"  STABLE at {bed_c:.2f} C for "
                          f"{(time.time()-stable_start)/60:.1f} min "
                          f"(target {step_target:.1f}, pwm={pwm})")
                return True
        else:
            if stable_start is not None:
                write_log(f"  B={bed_c:.2f} left tolerance band — reset stability timer")
            stable_start = None

        if elapsed >= MAX_STEP_S:
            write_log(f"  step {step_target:.1f} hit {MAX_STEP_S/60:.0f}-min cap "
                      f"(current B={bed_c:.2f}); advancing")
            return False

        # Heartbeat every 30s
        if int(elapsed) % 30 < POLL_S:
            write_log(f"  ... B={bed_c:.2f} (target {bed_target:.1f}, pwm={pwm}, "
                      f"elapsed {elapsed/60:.1f}min)")


def hold(final_target: float) -> None:
    write_log(f"=== HOLD at final step {final_target:.1f} C ===")
    write_log("Press Ctrl+C to end heating and send M140 S0 cleanly.")
    last_heartbeat = 0.0
    while True:
        time.sleep(POLL_S)
        result = poll_once(final_target)
        if result is None:
            continue
        bed_c, bed_target, pwm = result
        now = time.time()
        if now - last_heartbeat >= 60.0:
            write_log(f"  HOLD B={bed_c:.2f} (target {bed_target:.1f}, pwm={pwm})")
            last_heartbeat = now


def main() -> int:
    global _ser
    csv_init()
    write_log(f"=== stepped heat-up to ~35 C ===  out: {OUT_DIR}")
    write_log(f"steps: {STEPS}, tol={TOL_C}C, stable={STABLE_S}s, "
              f"step cap={MAX_STEP_S/60:.0f}min, safe={SAFE_LIMIT_C}C, hard={HARD_LIMIT_C}C")

    _ser = serial.Serial(PORT, BAUD, timeout=1.5)
    time.sleep(2.5)
    _ser.reset_input_buffer()

    # Sanity: confirm hotend is OFF, then probe one M105 to verify comms
    send("M104 S0")
    time.sleep(0.2)
    send("M105")
    text = read_drain(min_wait_s=0.6)
    parsed = parse_m105(text)
    if parsed is None:
        write_log(f"FATAL: no M105 response. Raw buffer: {text!r}")
        return 1
    bed_c, bed_target, pwm = parsed
    write_log(f"initial: B={bed_c:.2f} C (target {bed_target:.1f}, pwm={pwm})")

    try:
        for step in STEPS:
            run_step(step)
        hold(STEPS[-1])
    except KeyboardInterrupt:
        write_log("\nKeyboardInterrupt — sending M140 S0 and exiting cleanly")
        safe_shutdown()
        return 0
    except RuntimeError as exc:
        write_log(f"aborted by safety: {exc}")
        return 2
    except Exception as exc:
        write_log(f"unexpected error: {exc}")
        safe_shutdown()
        return 1
    finally:
        safe_shutdown()
        try:
            _ser.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
