"""Relaxed-criterion stepped heat-up: skip Step 1 (already past) and use a
one-way 'reach + 60s confirm' criterion instead of strict ±0.3 stable band.

Resume strategy after the initial run overshoot at 30 C:
    33.0  ->  34.5
Tolerance to advance:  B >= target - 0.3, hold 60 s, then go.
Safety still active: B >= 38 abort, B >= 40 hard abort.
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

STEPS = [35.0]
TOL_C = 0.3
HOLD_AT_TARGET_S = 60     # only 1 min one-way confirmation
MAX_STEP_S = 15 * 60
POLL_S = 5
SAFE_LIMIT_C = 38.0
HARD_LIMIT_C = 40.0


PROJECT = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
TS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = PROJECT / "data/raw/temperature" / f"heatup_quick_{TS}"
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


def csv_init() -> None:
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", encoding="utf-8") as f:
            f.write("unix_s,timestamp_iso,step_target_c,bed_c,bed_target_pid_c,bed_pwm\n")


def csv_append(step_target: float, bed_c: float, bed_target: float, pwm: int) -> None:
    with CSV_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f},{datetime.now().isoformat(timespec='seconds')},"
                f"{step_target:.2f},{bed_c:.3f},{bed_target:.2f},{pwm}\n")


def poll_once(step_target: float):
    send("M105")
    text = read_drain(min_wait_s=0.4)
    parsed = parse_m105(text)
    if parsed is None:
        return None
    bed_c, bed_target, pwm = parsed
    csv_append(step_target, bed_c, bed_target, pwm)
    if bed_c >= HARD_LIMIT_C:
        write_log(f"!!! HARD LIMIT B={bed_c:.2f}C  >= {HARD_LIMIT_C} — abort, CUT PSU NOW !!!")
        safe_shutdown()
        raise RuntimeError("hard temperature limit")
    if bed_c >= SAFE_LIMIT_C:
        write_log(f"!!! SAFE LIMIT B={bed_c:.2f}C  >= {SAFE_LIMIT_C} — auto M140 S0")
        safe_shutdown()
        raise RuntimeError("safe temperature limit")
    return bed_c, bed_target, pwm


def run_step(step_target: float) -> bool:
    """One-way reach: advance once B >= target - TOL and held HOLD_AT_TARGET_S."""
    write_log(f"--- STEP: target {step_target:.1f} C  (one-way reach, hold {HOLD_AT_TARGET_S}s) ---")
    send(f"M140 S{step_target:.2f}")
    step_start = time.time()
    above_start: float | None = None
    while True:
        time.sleep(POLL_S)
        res = poll_once(step_target)
        if res is None:
            continue
        bed_c, bed_target, pwm = res
        elapsed = time.time() - step_start

        if bed_c >= step_target - TOL_C:
            if above_start is None:
                above_start = time.time()
                write_log(f"  B={bed_c:.2f} reached target - {TOL_C}; hold timer start")
            elif time.time() - above_start >= HOLD_AT_TARGET_S:
                write_log(f"  step {step_target:.1f}C done — held >= {step_target - TOL_C:.1f}C "
                          f"for {HOLD_AT_TARGET_S}s  (final B={bed_c:.2f}, pwm={pwm})")
                return True
        else:
            if above_start is not None:
                write_log(f"  B={bed_c:.2f} dropped below {step_target-TOL_C:.1f}, reset hold timer")
            above_start = None

        if elapsed >= MAX_STEP_S:
            write_log(f"  step {step_target:.1f} hit {MAX_STEP_S/60:.0f}-min cap "
                      f"(B={bed_c:.2f}); advancing anyway")
            return False

        if int(elapsed) % 30 < POLL_S:
            write_log(f"  ... B={bed_c:.2f} (target {bed_target:.1f}, pwm={pwm}, "
                      f"elapsed {elapsed/60:.1f}min)")


def hold(final_target: float) -> None:
    write_log(f"=== HOLD at final target {final_target:.1f} C (Ctrl+C to stop) ===")
    last_heartbeat = 0.0
    while True:
        time.sleep(POLL_S)
        res = poll_once(final_target)
        if res is None:
            continue
        bed_c, bed_target, pwm = res
        now = time.time()
        if now - last_heartbeat >= 60.0:
            write_log(f"  HOLD B={bed_c:.2f} (target {bed_target:.1f}, pwm={pwm})")
            last_heartbeat = now


def main() -> int:
    global _ser
    csv_init()
    write_log(f"=== quick heat-up (relaxed criterion) ===  out: {OUT_DIR}")
    write_log(f"steps: {STEPS}, advance-when: B >= target - {TOL_C} for {HOLD_AT_TARGET_S}s")
    _ser = serial.Serial(PORT, BAUD, timeout=1.5)
    time.sleep(2.5)
    _ser.reset_input_buffer()
    send("M104 S0")
    time.sleep(0.2)
    send("M105")
    text = read_drain(min_wait_s=0.6)
    parsed = parse_m105(text)
    if parsed is None:
        write_log(f"FATAL: no M105 response. Raw: {text!r}")
        return 1
    bed_c, bed_target, pwm = parsed
    write_log(f"initial: B={bed_c:.2f} C (target {bed_target:.1f}, pwm={pwm})")

    try:
        for s in STEPS:
            run_step(s)
        hold(STEPS[-1])
    except KeyboardInterrupt:
        write_log("\nKeyboardInterrupt — safe shutdown")
        safe_shutdown()
        return 0
    except RuntimeError as exc:
        write_log(f"aborted: {exc}")
        return 2
    except Exception as exc:
        write_log(f"error: {exc}")
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
