"""Poll webapp until the current scan completes, then run alignment check."""
import json, subprocess, sys, time, urllib.request

DEADLINE = time.time() + 15 * 60   # 15 min safety
while time.time() < DEADLINE:
    try:
        r = urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=5)
        s = json.loads(r.read())
        state = s.get("status", "")
        progress = s.get("progress", {})
        print(f"{time.strftime('%H:%M:%S')} state={state} "
              f"line={progress.get('line')}/{progress.get('total')} "
              f"{progress.get('msg','')}", flush=True)
        if state in ("COMPLETED", "STOPPED", "ERROR", "IDLE") \
                and not s.get("running", False):
            break
    except Exception as exc:
        print(f"poll error: {exc}", flush=True)
    time.sleep(20)

print("\n=== scan finished, running alignment check ===\n", flush=True)
subprocess.run([
    sys.executable,
    r"d:\SIOT\Ultrasound-Imaging-for-Li-ion-Battery-main\scripts\check_tank_alignment.py",
    "--new",       "post_tank_move_check",
    "--reference", "scan_2026-05-30_15-05-34",
], check=False)
