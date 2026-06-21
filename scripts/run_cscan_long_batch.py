"""Long-duration C-scan batch runner with retries, resume, and heartbeat.

Designed for overnight / multi-hour sessions that must survive a single bad
scan, a transient webapp crash, or a host reboot midway through. Reuses the
existing `/api/start` + `/api/status` endpoints — same as
`run_cscan_noise_floor_batch.py`, but without the noise-floor analysis bundled
on the end.

Usage
-----
    python scripts/run_cscan_long_batch.py --runs 133 --prefix overnight
    python scripts/run_cscan_long_batch.py --duration-hours 20 --prefix overnight
    python scripts/run_cscan_long_batch.py --resume reports/experiments/long_batch_<ts>

Stops gracefully on:
  * --runs reached
  * --duration-hours elapsed
  * free disk below --min-free-gb
  * KeyboardInterrupt (Ctrl+C)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


DEFAULT_CFG = {
    "roi_w": 80.0,
    "roi_h": 72.0,
    "pitch": 0.5,
    "speed": 25.0,
    "accel_mode": "auto",
    "accel_mm_s2": 800.0,
    "cols": 500,
    "cmap": "turbo",
    "save_waveforms": True,
}

CHECKPOINT_NAME = "checkpoint.json"
PROGRESS_LOG_NAME = "progress.log"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _read_api_key(root: Path) -> str:
    env_key = os.environ.get("API_KEY", "").strip()
    if env_key:
        return env_key
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("API_KEY not found in environment or .env")


def _request_json(base_url: str, endpoint: str, method: str = "GET",
                  body: dict | None = None, api_key: str | None = None,
                  timeout_s: float = 30.0) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=data, headers=headers, method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def _wait_for_webapp(base_url: str, max_wait_s: float, log) -> None:
    """Block until /api/status responds 200, up to max_wait_s. Raise if not."""
    deadline = time.time() + max_wait_s
    backoff = 5.0
    while time.time() < deadline:
        try:
            _request_json(base_url, "/api/status", timeout_s=5.0)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, OSError) as exc:
            log(f"  webapp unreachable: {exc.__class__.__name__}: {exc}; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
    raise RuntimeError(f"webapp at {base_url} did not come back within {max_wait_s:.0f}s")


def _wait_for_idle(base_url: str, poll_s: float, max_scan_minutes: float, log) -> dict:
    """Poll /api/status until the run terminates. Tolerates brief webapp blips."""
    deadline = time.time() + max_scan_minutes * 60.0
    consecutive_errors = 0
    while time.time() < deadline:
        try:
            status = _request_json(base_url, "/api/status", timeout_s=10.0)
            consecutive_errors = 0
            state = status.get("status")
            progress = status.get("progress", {})
            line = progress.get("line", 0)
            total = progress.get("total", 0)
            msg = progress.get("msg", "")
            log(f"    state={state} line={line}/{total} {msg}")
            if state in {"COMPLETED", "STOPPED", "ERROR", "IDLE"} and not status.get("running", False):
                return status
        except Exception as exc:
            consecutive_errors += 1
            log(f"    status poll error #{consecutive_errors}: {exc}")
            if consecutive_errors >= 3:
                # Try to reconnect; if successful, keep polling
                _wait_for_webapp(base_url, max_wait_s=300.0, log=log)
                consecutive_errors = 0
        time.sleep(poll_s)
    raise RuntimeError(f"scan exceeded max_scan_minutes={max_scan_minutes:.0f}")


def _latest_meta(base_url: str) -> dict:
    return _request_json(base_url, "/api/data/meta", timeout_s=10.0)


def _check_temp_logger(base_url: str, api_key: str, log,
                        restart_session_name: str = "longrun_auto_restart") -> bool:
    """Return True if TC-08 is running. If not, attempt to (re)start it once.

    The C-scan service expects the temperature logger to be active so that per
    scan-line temperature stats are recorded. If the logger died (webapp crash
    + restart, USB disconnect, manual stop), this re-launches it transparently.
    """
    try:
        s = _request_json(base_url, "/api/temp/status", timeout_s=5.0)
    except Exception as exc:
        log(f"  TC-08 status query failed: {exc}")
        return False
    if s.get("running"):
        return True
    log(f"  TC-08 was NOT running (status={s.get('status')}) — attempting restart")
    try:
        body = {"channels": [1], "interval_s": 1.0,
                "session_name": restart_session_name}
        resp = _request_json(base_url, "/api/temp/start", method="POST",
                              body=body, api_key=api_key, timeout_s=15.0)
        ok = bool(resp.get("success"))
        log(f"  TC-08 restart attempt: success={ok}, msg={resp.get('msg')}")
        return ok
    except Exception as exc:
        log(f"  TC-08 restart failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Disk / housekeeping
# ---------------------------------------------------------------------------
def _free_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def _load_checkpoint(batch_dir: Path) -> dict:
    cp = batch_dir / CHECKPOINT_NAME
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_checkpoint(batch_dir: Path, state: dict) -> None:
    cp = batch_dir / CHECKPOINT_NAME
    cp.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--runs", type=int, default=None,
                    help="Maximum scan count. Default: derived from --duration-hours / ~9 min if neither given, fallback 100.")
    ap.add_argument("--duration-hours", type=float, default=None,
                    help="Maximum elapsed wall-clock hours. Stop after this even if runs<max.")
    ap.add_argument("--prefix", default="long_batch")
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--out-dir", default="",
                    help="Override batch report dir. Default: reports/experiments/long_batch_<ts>")
    ap.add_argument("--resume", default="",
                    help="Resume from an existing batch dir (uses its checkpoint.json)")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="Stop if free space on the data drive drops below this many GB")
    ap.add_argument("--retries-per-scan", type=int, default=3)
    ap.add_argument("--max-scan-minutes", type=float, default=20.0,
                    help="Per-scan wall-clock ceiling (safety net for hung scans)")
    ap.add_argument("--data-dir", default="data/raw/cscan",
                    help="Used only for free-space checking")
    args = ap.parse_args()

    root = Path.cwd()
    api_key = _read_api_key(root)
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- resolve batch dir & checkpoint
    if args.resume:
        batch_dir = Path(args.resume).resolve()
        if not batch_dir.exists():
            raise SystemExit(f"resume path does not exist: {batch_dir}")
        state = _load_checkpoint(batch_dir)
        if not state:
            raise SystemExit(f"no checkpoint.json in {batch_dir}")
        batch_ts = state.get("batch_ts", "unknown")
        prefix = state.get("prefix", args.prefix)
    else:
        batch_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = args.prefix
        batch_dir = Path(args.out_dir) if args.out_dir else (
            root / "reports" / "experiments" / f"{prefix}_{batch_ts}"
        )
        batch_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "batch_ts": batch_ts,
            "prefix": prefix,
            "started_iso": datetime.now().isoformat(timespec="seconds"),
            "base_url": args.base_url,
            "config": DEFAULT_CFG,
            "completed": [],   # list of {run_idx, scan_id, session_dir, elapsed_s}
            "failed": [],      # list of {run_idx, error}
        }
        _save_checkpoint(batch_dir, state)

    # Determine run target
    if args.runs is not None:
        max_runs = args.runs
    elif args.duration_hours is not None:
        max_runs = int(args.duration_hours * 60 / 9) + 5   # crude estimate, +safety
    else:
        max_runs = 100

    duration_deadline = None
    if args.duration_hours is not None:
        duration_deadline = time.time() + args.duration_hours * 3600

    progress_log = batch_dir / PROGRESS_LOG_NAME

    def log(line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        msg = f"[{ts}] {line}"
        print(msg, flush=True)
        with progress_log.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"=== batch {prefix}_{batch_ts} ===")
    log(f"batch_dir: {batch_dir}")
    log(f"target: max_runs={max_runs}, duration_hours={args.duration_hours}")
    log(f"already completed (resume): {len(state['completed'])} scans")
    log(f"data drive free: {_free_gb(data_dir):.1f} GB (stop threshold: {args.min_free_gb} GB)")

    # Wait for webapp at startup
    try:
        _wait_for_webapp(args.base_url, max_wait_s=60.0, log=log)
    except RuntimeError as exc:
        log(f"FATAL: {exc}")
        return 2

    completed_count = len(state["completed"])
    stop_reason = ""

    try:
        while completed_count < max_runs:
            if duration_deadline is not None and time.time() >= duration_deadline:
                stop_reason = f"duration limit reached ({args.duration_hours} h)"
                break

            free = _free_gb(data_dir)
            if free < args.min_free_gb:
                stop_reason = f"free disk {free:.1f} GB < threshold {args.min_free_gb} GB"
                break

            run_idx = completed_count + 1
            scan_name = f"{prefix}_{batch_ts}_r{run_idx:03d}"

            log(f"-- run {run_idx}/{max_runs}: {scan_name} (free disk: {free:.1f} GB) --")

            # Watchdog: ensure TC-08 is still logging before launching the scan.
            # If it died (webapp crash, manual stop, USB blip), restart it so
            # line_temperature_* stats keep being recorded into each NPZ.
            if not _check_temp_logger(args.base_url, api_key, log,
                                      restart_session_name=f"{prefix}_auto_restart"):
                log("  WARNING: TC-08 not running and restart failed — continuing without temperature")

            t_run_start = time.time()

            success = False
            last_error = None
            for attempt in range(1, args.retries_per_scan + 1):
                try:
                    body = {**DEFAULT_CFG, "scan_name": scan_name}
                    resp = _request_json(args.base_url, "/api/start",
                                         method="POST", body=body,
                                         api_key=api_key, timeout_s=30.0)
                    log(f"    start[attempt {attempt}]: {resp}")
                    final = _wait_for_idle(args.base_url, args.poll_s,
                                           args.max_scan_minutes, log)
                    if final.get("status") != "COMPLETED":
                        raise RuntimeError(f"finished with status {final.get('status')}")
                    meta = _latest_meta(args.base_url)
                    elapsed = time.time() - t_run_start
                    state["completed"].append({
                        "run_idx": run_idx,
                        "scan_id": meta.get("scan_id"),
                        "session_dir": meta.get("session_dir"),
                        "elapsed_s": round(elapsed, 1),
                    })
                    _save_checkpoint(batch_dir, state)
                    log(f"    OK  {meta.get('scan_id')}  ({elapsed/60:.1f} min)")
                    success = True
                    break
                except Exception as exc:
                    last_error = f"{exc.__class__.__name__}: {exc}"
                    log(f"    attempt {attempt}/{args.retries_per_scan} FAILED: {last_error}")
                    # On non-trivial failure, give the webapp room to recover
                    try:
                        _wait_for_webapp(args.base_url, max_wait_s=300.0, log=log)
                    except Exception as wexc:
                        log(f"    webapp still down: {wexc}")
                    backoff = 10.0 * attempt
                    log(f"    sleeping {backoff:.0f}s before retry")
                    time.sleep(backoff)

            if not success:
                state["failed"].append({"run_idx": run_idx, "error": last_error})
                _save_checkpoint(batch_dir, state)
                log(f"    GIVE UP on run {run_idx}: {last_error}")

            completed_count = len(state["completed"])
            # small gap between scans
            time.sleep(2.0)

        if not stop_reason:
            stop_reason = f"reached max_runs={max_runs}"
    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt"

    state["finished_iso"] = datetime.now().isoformat(timespec="seconds")
    state["stop_reason"] = stop_reason
    _save_checkpoint(batch_dir, state)

    log(f"=== batch done ===  completed={len(state['completed'])} failed={len(state['failed'])}")
    log(f"reason: {stop_reason}")
    log(f"checkpoint: {batch_dir / CHECKPOINT_NAME}")
    log(f"progress log: {progress_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
