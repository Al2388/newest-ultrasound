"""One-shot: send M140 S35 to the Marlin chiller on COM6 and read back M105.

Logs the action so we have a record of when the setpoint changed.
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

PORT = "COM6"
BAUD = 115200
TARGET_C = 35.0

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
LOG_DIR = PROJ / "data/raw/temperature" / f"bath_set_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / "action.log"


def write_log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def drain(ser, min_wait=0.4):
    time.sleep(min_wait)
    buf = b""
    while ser.in_waiting:
        buf += ser.read(ser.in_waiting)
        time.sleep(0.05)
    return buf.decode(errors="replace")


def parse_m105(text):
    # T:24.6 /0.0 B:24.6 /35.0 @:0 B@:127
    m = re.search(r"B:\s*([0-9.\-]+)\s*/\s*([0-9.\-]+)", text)
    pwm = re.search(r"B@:\s*([0-9]+)", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), int(pwm.group(1)) if pwm else None


def main():
    write_log(f"opening {PORT} @ {BAUD}")
    ser = serial.Serial(PORT, BAUD, timeout=1.5)
    time.sleep(2.5)
    ser.reset_input_buffer()

    # Make sure hotend is off (safety)
    ser.write(b"M104 S0\n"); ser.flush(); time.sleep(0.2)
    drain(ser)

    # Probe state before change
    ser.write(b"M105\n"); ser.flush()
    before = drain(ser, 0.6)
    write_log(f"BEFORE M105: {before.strip()!r}")
    parsed_before = parse_m105(before)
    if parsed_before is None:
        write_log("FATAL: cannot parse M105 response, aborting WITHOUT setting bath")
        ser.close()
        sys.exit(1)
    b_before, t_before, pwm_before = parsed_before
    write_log(f"BEFORE: bath={b_before:.2f} C, target={t_before:.1f} C, pwm={pwm_before}")

    # Send setpoint
    cmd = f"M140 S{TARGET_C:.2f}\n"
    write_log(f"SENDING: {cmd.strip()}")
    ser.write(cmd.encode()); ser.flush()
    drain(ser, 0.4)

    # Read back
    time.sleep(0.5)
    ser.write(b"M105\n"); ser.flush()
    after = drain(ser, 0.6)
    write_log(f"AFTER  M105: {after.strip()!r}")
    parsed_after = parse_m105(after)
    if parsed_after is None:
        write_log("FATAL: cannot parse post-set M105")
        ser.close()
        sys.exit(2)
    b_after, t_after, pwm_after = parsed_after
    write_log(f"AFTER : bath={b_after:.2f} C, target={t_after:.1f} C, pwm={pwm_after}")

    if abs(t_after - TARGET_C) > 0.5:
        write_log(f"WARNING: Marlin reported target {t_after} != requested {TARGET_C}")
    else:
        write_log(f"OK: setpoint confirmed at {t_after:.1f} C")

    ser.close()


if __name__ == "__main__":
    main()
