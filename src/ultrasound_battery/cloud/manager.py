"""
Box Cloud Archival Manager
===========================
Mirrors the local `data/` folder to a Box folder. Every file written under
`data/` can be uploaded into the corresponding subfolder on Box; the folder
hierarchy is created on demand.

Design principles
-----------------
  - Non-blocking: a thread pool executes uploads concurrently in the background.
  - Folder-preserving: a local file at `data/raw/cscan/sess1/scan.npz` becomes
    a Box file at `{BOX_FOLDER_ID}/cscan/sess1/scan.npz`.
  - Fail-safe: missing credentials silently disable uploads; the scan system
    operates fully offline without any code changes.
  - Cached: Box folder lookups/creates are cached so each subfolder is
    resolved (and created) at most once per session.

Configuration (via .env file or environment variables)
------------------------------------------------------
Pick ONE auth method:
  BOX_DEV_TOKEN     Developer token from Box App Console (60-minute expiry)
  BOX_JWT_CONFIG    Path to a JWT settings JSON file (production)

Folder targeting:
  BOX_FOLDER_ID     Numeric ID of the root destination folder. Default "0".

Public API
----------
  upload_path_async(local_path, base_dir="data")
      Upload `local_path` to Box, preserving its path under `base_dir`
      as the folder hierarchy. This is the preferred call site.

  upload_async(local_path, cloud_name, content_type)
      Legacy single-file API. `cloud_name` may include "/" separators —
      they will be turned into Box folder hierarchy.

  mirror_folder_async(local_folder, base_dir="data")
      Recursively upload every file under `local_folder`.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

# Load .env into os.environ before reading any BOX_* variables.
load_dotenv()

try:
    from box_sdk_gen import (
        BoxClient,
        BoxDeveloperTokenAuth,
        BoxJWTAuth,
        JWTConfig,
        UploadFileAttributes,
        UploadFileAttributesParentField,
        UploadFileVersionAttributes,
        CreateFolderParent,
    )
    _HAS_BOXSDK = True
except ImportError:
    _HAS_BOXSDK = False


# Number of concurrent uploads. Box's documented rate limits are generous,
# but 4 gives a good balance between throughput and not hammering the API.
_MAX_WORKERS = 4


class CloudManager:
    """
    Folder-preserving Box uploader with thread-pool concurrency.

    After __init__, check self.enabled before submitting uploads.
    Failures are logged and never re-raised — the acquisition pipeline
    will never be interrupted by a cloud problem.
    """

    def __init__(self):
        self.box           = None
        self.box_folder_id = "0"
        self.enabled       = False

        # rel_path ("cscan/sess1/lines_raw") → Box folder_id
        # Protected by self._cache_lock during read/write/Box-API calls.
        self._folder_cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()

        # Background uploader pool. Created only if Box is enabled.
        self._executor: ThreadPoolExecutor | None = None

        if not _HAS_BOXSDK:
            print("[CLOUD] Disabled: Box SDK not installed. "
                  "Run `pip install boxsdk` to enable.")
            return

        dev_token = os.getenv("BOX_DEV_TOKEN")
        jwt_path  = os.getenv("BOX_JWT_CONFIG")
        if not (dev_token or jwt_path):
            print("[CLOUD] Disabled: neither BOX_DEV_TOKEN nor BOX_JWT_CONFIG set.")
            return

        try:
            # Auth precedence: JWT (production) > DevToken (testing).
            if jwt_path:
                if not os.path.isfile(jwt_path):
                    print(f"[CLOUD] JWT config not found at {jwt_path}")
                    return
                config = JWTConfig.from_config_file(config_file_path=jwt_path)
                auth   = BoxJWTAuth(config=config)
                method = f"JWT (config={jwt_path})"
            else:
                auth   = BoxDeveloperTokenAuth(token=dev_token)
                method = "developer token"

            self.box           = BoxClient(auth=auth)
            self.box_folder_id = os.getenv("BOX_FOLDER_ID", "0")

            # Smoke-test: fetch the target folder's metadata.
            folder = self.box.folders.get_folder_by_id(self.box_folder_id)
            self.enabled  = True
            self._executor = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS,
                thread_name_prefix="cloud-upload",
            )
            print(
                f"[CLOUD] Connected via {method}; "
                f"uploading into '{folder.name}' (id={self.box_folder_id})"
            )
        except Exception as e:
            print(f"[CLOUD] Init failed: {e}")
            self.box = None

    # ------------------------------------------------------------------
    # Public upload API
    # ------------------------------------------------------------------

    def upload_path_async(self, local_path: str, base_dir: str = "data"):
        """
        Submit `local_path` for upload, preserving its path relative to
        `base_dir` as the Box folder hierarchy.

        Example: with default base_dir="data",
          local  : data/raw/cscan/cscan_xyz/scan.npz
          → cloud: {BOX_FOLDER_ID}/cscan/cscan_xyz/scan.npz

        Files outside base_dir are silently skipped.
        """
        if not self.enabled:
            return
        try:
            abs_local = os.path.abspath(local_path)
            abs_base  = os.path.abspath(base_dir)
            rel       = os.path.relpath(abs_local, abs_base)
        except ValueError:
            # Different drives on Windows
            return
        if rel.startswith("..") or os.path.isabs(rel):
            return
        cloud_rel = rel.replace(os.sep, "/")
        self._executor.submit(self._upload_worker, local_path, cloud_rel)

    def upload_async(self, local_path: str, cloud_name: str, content_type: str = ""):
        """
        Legacy API kept for compatibility. `cloud_name` may include "/" —
        it will be turned into Box folder hierarchy. Filename-only
        `cloud_name` uploads to BOX_FOLDER_ID root.

        `content_type` is unused (Box auto-detects MIME) but accepted for
        compatibility with previous call sites.
        """
        if not self.enabled:
            return
        cloud_rel = cloud_name.replace("\\", "/").lstrip("/")
        self._executor.submit(self._upload_worker, local_path, cloud_rel)

    def mirror_folder_async(self, local_folder: str, base_dir: str = "data"):
        """
        Recursively submit every file under `local_folder` for upload.
        Returns the number of files queued.
        """
        if not self.enabled:
            return 0
        count = 0
        for root, _dirs, files in os.walk(local_folder):
            for f in files:
                self.upload_path_async(os.path.join(root, f), base_dir=base_dir)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Upload worker
    # ------------------------------------------------------------------

    def _upload_worker(self, local_path: str, cloud_rel: str):
        """
        Upload one file. Resolves the destination folder (creating it if
        needed), then either uploads a fresh file or a new version of an
        existing one. Errors are logged and swallowed.
        """
        try:
            if not os.path.isfile(local_path):
                print(f"[CLOUD] Skip — missing local file: {local_path}")
                return

            parts        = [p for p in cloud_rel.split("/") if p]
            if not parts:
                print(f"[CLOUD] Skip — empty cloud path for {local_path}")
                return
            folder_parts = parts[:-1]
            file_name    = parts[-1]

            folder_id = (self._ensure_folder_path(folder_parts)
                         if folder_parts else self.box_folder_id)
            existing_id = self._find_box_file_id_in_folder(folder_id, file_name)

            with open(local_path, "rb") as fh:
                if existing_id is not None:
                    self.box.uploads.upload_file_version(
                        file_id    = existing_id,
                        attributes = UploadFileVersionAttributes(name=file_name),
                        file       = fh,
                    )
                    print(f"[CLOUD] Updated:  {cloud_rel}")
                else:
                    self.box.uploads.upload_file(
                        attributes = UploadFileAttributes(
                            name   = file_name,
                            parent = UploadFileAttributesParentField(id=folder_id),
                        ),
                        file = fh,
                    )
                    print(f"[CLOUD] Uploaded: {cloud_rel}")
        except Exception as e:
            print(f"[CLOUD] Upload error ({cloud_rel}): {e}")

    # ------------------------------------------------------------------
    # Folder hierarchy helpers
    # ------------------------------------------------------------------

    def _ensure_folder_path(self, folder_parts: list) -> str:
        """
        Walk the folder hierarchy under self.box_folder_id, creating any
        missing folders. Returns the folder_id of the deepest folder.

        Holds self._cache_lock for the duration of the walk so two upload
        threads can't race to create the same Box folder.
        """
        rel_key = "/".join(folder_parts)
        with self._cache_lock:
            cached = self._folder_cache.get(rel_key)
            if cached is not None:
                return cached

            current_id = self.box_folder_id
            for i, name in enumerate(folder_parts):
                partial_key = "/".join(folder_parts[:i + 1])
                hit = self._folder_cache.get(partial_key)
                if hit is not None:
                    current_id = hit
                    continue
                child_id = self._find_box_subfolder_id(current_id, name)
                if child_id is None:
                    child_id = self._create_box_subfolder(current_id, name)
                current_id = child_id
                self._folder_cache[partial_key] = current_id
            return current_id

    def _find_box_subfolder_id(self, parent_id: str, name: str):
        """Return the folder_id of `name` inside `parent_id`, or None."""
        try:
            items = self.box.folders.get_folder_items(parent_id, limit=1000)
            for entry in items.entries or []:
                if entry.type == "folder" and entry.name == name:
                    return entry.id
        except Exception as e:
            print(f"[CLOUD] list-folder error in {parent_id}: {e}")
        return None

    def _create_box_subfolder(self, parent_id: str, name: str) -> str:
        """Create folder `name` under `parent_id`. Returns the new folder_id."""
        result = self.box.folders.create_folder(
            name   = name,
            parent = CreateFolderParent(id=parent_id),
        )
        return result.id

    def _find_box_file_id_in_folder(self, folder_id: str, file_name: str):
        """Return the file_id of `file_name` inside `folder_id`, or None."""
        try:
            items = self.box.folders.get_folder_items(folder_id, limit=1000)
            for entry in items.entries or []:
                if entry.type == "file" and entry.name == file_name:
                    return entry.id
        except Exception:
            pass
        return None
