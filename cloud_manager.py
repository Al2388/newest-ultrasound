"""
AWS S3 Cloud Archival Manager
================================
Provides optional, non-blocking upload of scan data to an AWS S3 bucket for
remote access, long-term archival, and sharing between lab computers.

Design principles
-----------------
  - Non-blocking: all uploads run in daemon threads so they never delay the
    scan worker or introduce back-pressure on the acquisition loop.
  - Fail-safe: if AWS credentials are missing or the bucket is unreachable,
    CloudManager disables itself silently and all upload() calls become no-ops.
    The scan system operates fully offline without any code changes.
  - No presigned URLs: live dashboard images are served locally via FastAPI's
    StaticFiles mount. S3 is used for archival only — not for live display.

Configuration (via .env file or environment variables)
------------------------------------------------------
  AWS_ACCESS_KEY    IAM access key ID
  AWS_SECRET_KEY    IAM secret access key
  AWS_BUCKET_NAME   S3 bucket name (e.g. "cscan-lab-2025")
  AWS_REGION        AWS region code (e.g. "eu-north-1")

Security
--------
The .env file containing credentials must never be committed to version
control. See .env.example for the required variable names with placeholder
values safe to commit.
"""

import os
import threading

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load .env into os.environ before reading any AWS_* variables.
# This is a no-op if python-dotenv is not installed or .env is absent.
load_dotenv()


class CloudManager:
    """
    Thin, thread-safe wrapper around boto3 S3 client.

    After __init__, check self.enabled before calling upload_async().
    If self.enabled is False, upload_async() is a no-op — no exception raised.

    Typical usage
    -------------
      cloud = CloudManager()
      if cloud.enabled:
          print("S3 archival active")
      # Later, after saving a file locally:
      cloud.upload_async("/path/to/scan.npz", "scan_20250507.npz",
                         "application/octet-stream")
    """

    def __init__(self):
        self.bucket  = os.getenv("AWS_BUCKET_NAME")
        self.region  = os.getenv("AWS_REGION")
        self.s3      = None
        self.enabled = False

        if not self.bucket:
            # AWS_BUCKET_NAME not configured — operate in offline-only mode
            print("[CLOUD] Disabled: AWS_BUCKET_NAME not set.")
            return

        try:
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id     = os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key = os.getenv("AWS_SECRET_KEY"),
                region_name           = self.region,
            )
            # head_bucket() is a cheap call that verifies both the bucket name
            # and the IAM credentials without downloading any data
            self.s3.head_bucket(Bucket=self.bucket)
            self.enabled = True
            print(f"[CLOUD] Connected: s3://{self.bucket}  region={self.region}")
        except ClientError as e:
            # Raised if the bucket doesn't exist or the credentials are invalid
            print(f"[CLOUD] Access denied or bucket not found: {e}")
        except Exception as e:
            # Any other error (network, invalid region, etc.)
            print(f"[CLOUD] Init failed: {e}")

    def upload_async(self, local_path: str, cloud_name: str, content_type: str):
        """
        Upload a local file to S3 in a background daemon thread (non-blocking).

        If self.enabled is False this is a complete no-op — safe to call
        unconditionally at the end of every scan.

        Parameters
        ----------
        local_path   : str
            Path to the file on the local filesystem.
        cloud_name   : str
            S3 object key (filename in the bucket). Typically the basename of
            local_path, optionally prefixed with a folder (e.g. "scans/scan_42.npz").
        content_type : str
            MIME type for the S3 metadata, e.g.:
              "application/octet-stream"  — .npz, .h5 binary archives
              "application/json"          — metadata JSON
              "image/png"                 — feature map images
        """
        if not self.enabled:
            return
        threading.Thread(
            target = self._upload_worker,
            args   = (local_path, cloud_name, content_type),
            daemon = True,   # daemon=True so the thread doesn't block process exit
        ).start()

    def _upload_worker(self, local_path: str, cloud_name: str, content_type: str):
        """
        Worker function executed in a background thread by upload_async().

        Uploads the file using the multipart uploader built into boto3's
        upload_file(), which handles large files automatically. Errors are
        logged to stdout but never re-raised — a failed upload must not
        corrupt or abort the ongoing scan.
        """
        try:
            self.s3.upload_file(
                local_path, self.bucket, cloud_name,
                ExtraArgs={"ContentType": content_type},
            )
            print(f"[CLOUD] Uploaded: {cloud_name}")
        except Exception as e:
            print(f"[CLOUD] Upload error ({cloud_name}): {e}")
