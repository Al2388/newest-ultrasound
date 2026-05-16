"""
One-shot backfill: upload every file under `data/` to Box, preserving structure.

Usage
-----
  python mirror_data_to_box.py                # mirror the whole data/ tree
  python mirror_data_to_box.py cscan          # mirror only data/raw/cscan/
  python mirror_data_to_box.py data/raw/cscan # equivalent explicit path

The folder hierarchy under `data/` is replicated under the Box folder
specified by BOX_FOLDER_ID. Existing Box files are overwritten with a new
version, so this script is safe to re-run — it will only do the work
needed to bring Box back in sync.

Concurrency
-----------
Uploads run through the CloudManager's thread pool (default 4 workers).
The script waits for the pool to drain before exiting.
"""

import os
import sys
import time

from ultrasound_battery.cloud.manager import CloudManager


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "data"

    # Allow short forms like `cscan` → `data/raw/cscan`, while still accepting
    # full paths like `data/raw/cscan` or absolute paths.
    if not os.path.isdir(target):
        for candidate in (os.path.join("data", "raw", target),
                          os.path.join("data", target)):
            if os.path.isdir(candidate):
                target = candidate
                break

    if not os.path.isdir(target):
        print(f"[BACKFILL] Not a directory: {target}")
        sys.exit(1)

    cm = CloudManager()
    if not cm.enabled:
        print("[BACKFILL] Cloud not enabled — check BOX_JWT_CONFIG in .env.")
        sys.exit(2)

    # Walk and submit
    n_submitted = 0
    total_bytes = 0
    t0 = time.time()
    for root, _dirs, files in os.walk(target):
        for f in files:
            local_path = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(local_path)
            except OSError:
                pass
            cm.upload_path_async(local_path)
            n_submitted += 1

    print(f"[BACKFILL] Submitted {n_submitted} files "
          f"({total_bytes / (1024 * 1024):.1f} MB). "
          f"Waiting for uploads to complete...")

    # Block until the executor drains
    cm._executor.shutdown(wait=True)
    print(f"[BACKFILL] Done in {time.time() - t0:.1f} s.")


if __name__ == "__main__":
    main()
