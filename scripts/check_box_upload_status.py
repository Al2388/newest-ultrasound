"""Compare local data files against the configured Box archival folder.

This is a read-only diagnostic. It loads BOX_JWT_CONFIG/BOX_FOLDER_ID from
.env, lists the Box folder tree, and compares paths and sizes for files under
the local data directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth, BoxJWTAuth, JWTConfig


def box_client() -> tuple[BoxClient, str]:
    load_dotenv()
    folder_id = os.getenv("BOX_FOLDER_ID", "0")
    jwt_path = os.getenv("BOX_JWT_CONFIG")
    dev_token = os.getenv("BOX_DEV_TOKEN")
    if jwt_path:
        config = JWTConfig.from_config_file(config_file_path=jwt_path)
        return BoxClient(auth=BoxJWTAuth(config=config)), folder_id
    if dev_token:
        return BoxClient(auth=BoxDeveloperTokenAuth(token=dev_token)), folder_id
    raise SystemExit("No BOX_JWT_CONFIG or BOX_DEV_TOKEN is configured.")


def folder_items(client: BoxClient, folder_id: str):
    offset = 0
    while True:
        page = client.folders.get_folder_items(
            folder_id,
            limit=1000,
            offset=offset,
            fields=["id", "type", "name", "size"],
        )
        entries = list(page.entries or [])
        yield from entries
        if len(entries) < 1000:
            break
        offset += len(entries)


def find_folder(client: BoxClient, root_id: str, path: str) -> tuple[str, str] | None:
    current_id = root_id
    current_path = ""
    for part in [p for p in path.replace("\\", "/").split("/") if p]:
        match = None
        for item in folder_items(client, current_id):
            if item.type == "folder" and item.name == part:
                match = item
                break
        if match is None:
            return None
        current_id = match.id
        current_path = f"{current_path}/{part}" if current_path else part
    return current_id, current_path


def list_box_tree(client: BoxClient, root_id: str, prefix: str = "") -> dict[str, int | None]:
    files: dict[str, int | None] = {}
    if prefix:
        found = find_folder(client, root_id, prefix)
        if found is None:
            return files
        start_id, start_prefix = found
    else:
        start_id, start_prefix = root_id, ""
    stack: list[tuple[str, str]] = [(start_id, start_prefix)]
    while stack:
        folder_id, prefix = stack.pop()
        for item in folder_items(client, folder_id):
            name = item.name
            rel = f"{prefix}/{name}" if prefix else name
            if item.type == "folder":
                stack.append((item.id, rel))
            elif item.type == "file":
                files[rel.replace("\\", "/")] = getattr(item, "size", None)
    return files


def local_files(data_dir: Path) -> dict[str, int]:
    root = data_dir.resolve()
    files: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.resolve().relative_to(root).as_posix()
            files[rel] = path.stat().st_size
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--prefix", default="", help="Only compare relative paths with this prefix.")
    parser.add_argument("--show", type=int, default=30, help="Number of missing/mismatched examples.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    local = local_files(data_dir)
    if args.prefix:
        prefix = args.prefix.replace("\\", "/").strip("/")
        local = {path: size for path, size in local.items() if path.startswith(prefix + "/") or path == prefix}

    client, folder_id = box_client()
    root = client.folders.get_folder_by_id(folder_id)
    remote = list_box_tree(client, folder_id, args.prefix)

    missing = sorted(path for path in local if path not in remote)
    size_mismatch = sorted(
        path for path, size in local.items()
        if path in remote and remote[path] is not None and remote[path] != size
    )
    matched = len(local) - len(missing) - len(size_mismatch)

    print(f"Box folder: {root.name} (id={folder_id})")
    print(f"Compared local root: {data_dir.resolve()}")
    if args.prefix:
        print(f"Compared prefix: {args.prefix}")
    print(f"Local files compared: {len(local)}")
    print(f"Remote files listed: {len(remote)}")
    print(f"Matched files: {matched}")
    print(f"Missing on Box: {len(missing)}")
    print(f"Size mismatches: {len(size_mismatch)}")

    if missing:
        print("\nMissing examples:")
        for path in missing[: args.show]:
            print(f"  {path} ({local[path]} bytes)")
    if size_mismatch:
        print("\nSize mismatch examples:")
        for path in size_mismatch[: args.show]:
            print(f"  {path} local={local[path]} remote={remote[path]}")
    return 0 if not missing and not size_mismatch else 2


if __name__ == "__main__":
    raise SystemExit(main())
