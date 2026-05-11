"""
Ender 3D Printer Control Module
==================================
Controls a modified Creality Ender printer over USB serial using the Marlin
firmware G-code protocol. The Ender is repurposed as a precision XY translation
stage that moves the ultrasonic transducer across the surface of the battery
under test.

Coordinate system
-----------------
  X axis — fast scan axis (left ↔ right during each scan line)
  Y axis — slow axis (advances by one pitch between scan lines)
  Z axis — fixed by the gantry; not commanded from software in this project.

All positional moves use absolute coordinates (G90). The XY origin is defined
at the start of each scan session by G92 X0 Y0.

Communication
-------------
A background reader thread continuously drains the serial receive buffer and
enqueues response lines so that _send_command_wait() can block until Marlin
acknowledges each command with an "ok" response without starving the caller.

Usage
-----
  pr, xl, xr, ys, ye = setup_precision_printer("COM6", 115200, 50.0, 50.0)
  pr.move_to_position(xl, ys)
  pr.wait_for_completion()
  pr.close()
"""

import serial
import time
import threading
import queue


class PrecisionEnder:
    """
    Serial interface to a Marlin-firmware 3D printer used as an XY scan stage.

    Parameters
    ----------
    port : str
        Serial port identifier, e.g. "COM6" on Windows or "/dev/ttyUSB0" on Linux.
    baud : int
        Baud rate — 115200 is the Marlin default.
    reset_time : float
        Seconds to wait after opening the port. The DTR signal resets the printer
        MCU on connection; this delay lets Marlin finish booting before sending
        any G-code commands.
    reset_origin : bool
        If True, send G92 X0 Y0 to define the current XY position as the scan
        origin. Set False when re-connecting mid-session so the existing
        coordinate system is preserved.
    """

    def __init__(self, port: str, baud: int = 115200,
                 reset_time: float = 2.0, reset_origin: bool = True):
        self.ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(reset_time)              # wait for Marlin MCU reset to complete
        self.ser.reset_input_buffer()

        self.lock           = threading.Lock()
        self.response_queue = queue.Queue()
        self._stop_reader   = False

        # Background thread drains the serial RX buffer continuously.
        # This prevents the OS serial buffer from overflowing during long waits.
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        # Motion parameters — override these before calling start_scan if needed
        self.scan_speed_mm_s      = 10.0
        self.scan_feedrate        = int(self.scan_speed_mm_s * 60)   # G1 F in mm/min
        self.positioning_feedrate = 1800    # mm/min for rapid between-line moves
        self.line_spacing_mm      = 0.1

        # Tracked current XY position in scan coordinates (mm).
        # Z is not software-controlled (gantry has no Z stage in this build).
        self.current_x            = 0.0
        self.current_y            = 0.0

        self._initialize_printer(reset_origin)

    # -------------------------------------------------------------------------
    # Serial I/O
    # -------------------------------------------------------------------------

    def _reader_loop(self):
        """
        Background thread: reads lines from the serial port and puts them in
        response_queue. Runs until self._stop_reader is set to True.

        Uses a string buffer to assemble complete lines from potentially
        partial serial reads — the OS may return any number of bytes per read().
        """
        buffer = ""
        while not self._stop_reader:
            try:
                if self.ser.in_waiting:
                    data    = self.ser.read(self.ser.in_waiting).decode("ascii", errors="ignore")
                    buffer += data
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            self.response_queue.put(line.strip())
                else:
                    time.sleep(0.005)   # brief yield when the buffer is empty
            except Exception:
                time.sleep(0.1)         # recover from transient read errors

    def _send_command_wait(self, command: str, timeout: float = 5.0) -> str:
        """
        Send one G-code command and block until Marlin responds with "ok".

        The response queue is flushed before sending so that stale "ok" tokens
        from previous commands cannot satisfy the wait for this one.

        Parameters
        ----------
        command : str  G-code string, with or without trailing newline.
        timeout : float  Maximum seconds to wait for the "ok" response.

        Returns
        -------
        str
            The Marlin response line containing "ok", or "timeout" if the
            deadline was reached before any "ok" was received.

        Raises
        ------
        RuntimeError if the serial write fails.
        """
        if not command.endswith("\n"):
            command += "\n"

        # Flush stale responses to prevent cross-command acknowledgement
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break

        try:
            with self.lock:
                self.ser.write(command.encode("ascii"))
        except Exception as e:
            raise RuntimeError(f"Serial send failed: {e}")

        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self.response_queue.get(timeout=0.1)
                if "ok" in resp.lower():
                    return resp
                elif "error" in resp.lower() or "!!" in resp:
                    print(f"[PRINTER] Marlin error: {resp}")
            except queue.Empty:
                continue
        raise TimeoutError(f"Timed out waiting for printer ok after: {command.strip()}")

    def send_command(self, command: str):
        """
        Fire-and-forget G-code send with no acknowledgement wait.

        Used for motion commands (G1) where wait_for_completion() / M400 is
        used separately to synchronise at the end of the move.
        """
        if not command.endswith("\n"):
            command += "\n"
        with self.lock:
            self.ser.write(command.encode("ascii"))

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    def _initialize_printer(self, reset_origin: bool):
        """
        Send startup G-code to configure Marlin for scan use.

        G90        — absolute positioning (all X/Y moves reference the origin)
        M83        — relative extrusion mode (extruder unused; safe to set)
        M211 S0    — disable software endstops so the probe can reach all edges
        G21        — metric (mm) units
        M92        — steps/mm: X80 Y80 matches Ender GT2 belt + 8× microstep
        M203        — max velocity limits (mm/s) to protect the belt and mechanics
        M204        — acceleration limits (mm/s²) for print, retract, travel
        M205        — jerk / junction deviation to smooth direction changes
        G92        — if reset_origin, define current XY position as (0, 0)
        M503        — echo current EEPROM settings to the serial console
        """
        print(f"[PRINTER] Initializing  (reset_origin={reset_origin}) ...")

        commands = [
            "G90",                       # absolute positioning
            "M83",                       # relative extrusion (unused, safe to set)
            "M211 S0",                   # disable software endstops
            "G21",                       # millimetre units
            "M92 X80 Y80",               # steps/mm calibration (XY only)
            "M203 X500 Y500 E50",        # max velocity (mm/s)
            "M204 P500 R500 T500",       # acceleration (mm/s²): print / retract / travel
            "M205 X5.0 Y5.0 E5.0",       # junction deviation / jerk (mm/s)
        ]

        # G92 redefines the current XY position as the coordinate origin.
        # Only do this at the very start of a fresh scan — not during return.
        if reset_origin:
            commands.append("G92 X0 Y0")

        commands.append("M503")   # request EEPROM echo for the log

        for cmd in commands:
            self._send_command_wait(cmd, timeout=3.0)
            time.sleep(0.05)

        print("[PRINTER] Ready.")

    # -------------------------------------------------------------------------
    # Motion
    # -------------------------------------------------------------------------

    def move_to_position(self, x: float, y: float, fast: bool = True):
        """
        Absolute XY move to (x, y) in mm.

        Parameters
        ----------
        x, y : float  Target coordinates in the scan coordinate system (mm).
        fast : bool
            True  — positioning feedrate (rapid, between-line repositioning).
            False — scan feedrate (slower, for pre-scan transducer alignment).
        """
        f = self.positioning_feedrate if fast else self.scan_feedrate
        self.send_command(f"G1 X{x:.3f} Y{y:.3f} F{f}")
        self.current_x = x
        self.current_y = y

    def wait_for_completion(self, timeout: float = 30.0):
        """
        Block until all buffered motion is complete.

        Sends M400 ("finish moves then respond ok"), which causes Marlin to
        wait until the motion planner queue is empty before sending "ok".
        Call this after a move command to guarantee the probe is stationary
        before starting ultrasonic acquisition.
        """
        self._send_command_wait("M400", timeout)

    def disable_motors(self):
        """
        Release stepper motor holding current (M84).

        Call at the end of a session to prevent the stepper drivers from
        overheating when the stage is idle for an extended period.
        """
        try:
            self._send_command_wait("M84", timeout=2.0)
        except Exception:
            pass

    def close(self):
        """Disable motors, stop the reader thread, and close the serial port."""
        self.disable_motors()
        self._stop_reader = True
        if hasattr(self, "ser") and self.ser.is_open:
            self.ser.close()


# =============================================================================
# Convenience factory used by CScanService
# =============================================================================

def setup_precision_printer(port: str, baud: int,
                             roi_w: float, roi_h: float,
                             safety_margin: float = 2.0,
                             reset_origin: bool = True
                             ) -> tuple:
    """
    Open the printer and compute the safe scan boundary coordinates.

    A safety margin is subtracted from each edge of the ROI to prevent the
    transducer mount from colliding with the sample holder walls.

    Parameters
    ----------
    port, baud      : serial connection parameters
    roi_w, roi_h    : region-of-interest width and height in mm
    safety_margin   : mm to shrink from each edge (default 2 mm on every side)
    reset_origin    : if True, define current XY position as scan origin (G92)

    Returns
    -------
    (printer, x_left, x_right, y_start, y_end) : tuple
        printer  — PrecisionEnder instance (caller must call .close() when done)
        x_left   — leftmost safe X coordinate (mm)
        x_right  — rightmost safe X coordinate (mm)
        y_start  — top-most safe Y coordinate (mm)  [first scan line]
        y_end    — bottom-most safe Y coordinate (mm) [last scan line]
    """
    pr      = PrecisionEnder(port, baud, reset_origin=reset_origin)
    x_left  = safety_margin
    x_right = roi_w - safety_margin
    y_start = safety_margin
    y_end   = roi_h - safety_margin

    if reset_origin:
        # Move to the scan start position before locking in the coordinate origin
        pr.move_to_position(x_left, y_start, fast=True)
        pr.wait_for_completion()

    return pr, x_left, x_right, y_start, y_end
