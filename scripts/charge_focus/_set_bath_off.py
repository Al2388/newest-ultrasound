"""Send M140 S0 to turn off the bath heater on COM6 (Marlin)."""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

PORT = "COM6"
BAUD = 115200

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
LOG_DIR = PROJ / "data/raw/temperature" / f"bath_off_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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

    ser.write(b"M105\n"); ser.flush()
    before = drain(ser, 0.6)
    write_log(f"BEFORE M105: {before.strip()!r}")
    parsed = parse_m105(before)
    if parsed:
        b, t, pwm = parsed
        write_log(f"BEFORE: bath={b:.2f}C  target={t:.1f}C  pwm={pwm}")

    # Hotend off too (safety)
    ser.write(b"M104 S0\n"); ser.flush(); time.sleep(0.2); drain(ser)
    # Bath off
    write_log("SENDING: M140 S0")
    ser.write(b"M140 S0\n"); ser.flush(); time.sleep(0.4); drain(ser)

    # Verify
    time.sleep(0.5)
    ser.write(b"M105\n"); ser.flush()
    after = drain(ser, 0.6)
    write_log(f"AFTER  M105: {after.strip()!r}")
    parsed = parse_m105(after)
    if parsed:
        b, t, pwm = parsed
        write_log(f"AFTER : bath={b:.2f}C  target={t:.1f}C  pwm={pwm}")
        if t == 0.0:
            write_log("OK: heater target is 0 (off)")
        else:
            write_log(f"WARNING: Marlin reports target={t} (expected 0)")

    ser.close()


if __name__ == "__main__":
    main()
