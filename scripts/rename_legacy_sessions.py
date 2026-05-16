"""
Rename legacy unix-timestamp session folders/files under data/raw/ to the
human-readable YYYY-MM-DD_HH-MM-SS format introduced in v0.3.0.

What this fixes
---------------
Acquisitions made before v0.3.0 were named with raw unix integers, e.g.
  data/raw/cscan/cscan_scan_1778686236/scan_1778686291.npz
  data/raw/ascan/ascan_session_1778257726/ascan_session_1778257726.h5
  data/raw/gauging/gauge_session_1778257204/gauge_session_1778257204.h5

After running with --apply they become:
  data/raw/cscan/cscan_scan_2026-05-13_15-30-36/scan_2026-05-13_15-31-31.npz
  data/raw/ascan/ascan_session_2026-05-08_14-22-06/ascan_session_2026-05-08_14-22-06.h5
  data/raw/gauging/gauge_session_2026-05-08_14-20-04/gauge_session_2026-05-08_14-20-04.h5

How it works
------------
A single regex substitutes every occurrence of "_<9-10 digits>" (bounded by
separator or end-of-name) with "_<readable timestamp>". This handles both the
folder name and timestamped files inside (e.g. the C-scan save files
`scan_<save_ts>.npz` whose ts is independent from the folder's start ts).

The mtime of each file is NOT used — the timestamp is parsed from the name
itself, so the renames preserve the original acquisition time exactly.

Side effects beyond the data tree
---------------------------------
After folder renames, hardcoded session paths in `scripts/*.py` and writeups
in `reports/**/*.md` would break. To prevent silent breakage, --apply also
text-replaces every old session ID it renamed in those files. The diff is
visible via git so any unwanted change can be reverted.

JSON sidecars and HDF5 attributes inside each session folder are NOT modified;
their internal `session_id` remains a historical record of what the system
named the session at acquisition time.

Usage
-----
  python -m scripts.rename_legacy_sessions                  # dry-run
  python -m scripts.rename_legacy_sessions --apply          # actually rename
  python -m scripts.rename_legacy_sessions --root some/dir  # alternate root
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ultrasound_battery.utils import session_timestamp


# Match "_<9-or-10-digit-integer>" provided it's followed by a separator or
# end-of-string. The 9-10 digit window covers unix epoch seconds from
# year 2001 (1_000_000_000) through year 2286 (10_000_000_000 - 1), which
# safely brackets every acquisition this lab has ever made.
TS_RE = re.compile(r"_(\d{9,10})(?=[_./]|$)")


def convert_name(name: str) -> str:
    """Rewrite every legacy unix-ts substring in `name` to the readable form."""
    return TS_RE.sub(
        lambda m: "_" + session_timestamp(int(m.group(1))),
        name,
    )


def walk_session_root(root: Path):
    """Yield (folder_path, new_folder_name) for every legacy folder under root."""
    for sub in ("cscan", "ascan", "gauging"):
        cat = root / sub
        if not cat.is_dir():
            continue
        for entry in sorted(cat.iterdir()):
            if not entry.is_dir():
                continue
            new_name = convert_name(entry.name)
            if new_name == entry.name:
                continue  # already converted, or not a session folder
            yield entry, new_name


def plan_inner_renames(folder: Path) -> list[tuple[Path, Path]]:
    """Build a list of (old_path, new_path) for every timestamped file in `folder`."""
    renames: list[tuple[Path, Path]] = []
    for child in folder.iterdir():
        if child.is_file():
            new_name = convert_name(child.name)
            if new_name != child.name:
                renames.append((child, child.parent / new_name))
    return renames


def find_text_references(search_dirs: list[Path], session_ids: set[str]):
    """Yield (file_path, old_id, count) for each file containing any old session_id."""
    if not session_ids:
        return
    pattern = re.compile("|".join(re.escape(s) for s in session_ids))
    # The rename script's own docstring intentionally contains example IDs;
    # patching them would corrupt the "before/after" illustration.
    self_path = Path(__file__).resolve()
    for d in search_dirs:
        if not d.is_dir():
            continue
        for path in d.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".py", ".md", ".txt"):
                continue
            if path.resolve() == self_path:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            matches = pattern.findall(text)
            if matches:
                for m in set(matches):
                    yield path, m, matches.count(m)


def patch_text_files(file_paths: set[Path], mapping: dict[str, str]) -> int:
    """Replace every old session_id with its new one in each file. Returns count of files changed."""
    changed = 0
    for path in file_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text = text
        for old, new in mapping.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/raw"),
                        help="Root containing cscan/, ascan/, gauging/ subdirs")
    parser.add_argument("--apply", action="store_true",
                        help="Actually perform the renames (default is dry-run)")
    parser.add_argument("--search-dirs", nargs="*", type=Path,
                        default=[Path("scripts"), Path("reports")],
                        help="Where to look for stale session_id references in .py/.md/.txt")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"[error] {args.root} is not a directory", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== rename_legacy_sessions [{mode}] ===")
    print(f"root: {args.root.resolve()}")
    print()

    plan: list[tuple[Path, Path, list[tuple[Path, Path]]]] = []
    id_mapping: dict[str, str] = {}

    for folder, new_name in walk_session_root(args.root):
        new_folder = folder.parent / new_name
        if new_folder.exists():
            print(f"[skip] {folder} -> target exists: {new_folder}")
            continue
        inner = plan_inner_renames(folder)
        plan.append((folder, new_folder, inner))
        id_mapping[folder.name] = new_name

    if not plan:
        print("Nothing to rename — no legacy folders found.")
        return 0

    # ------------------------------------------------------------------
    # Print the rename plan
    # ------------------------------------------------------------------
    print(f"Found {len(plan)} legacy session folder(s).")
    print()
    for folder, new_folder, inner in plan:
        rel = folder.relative_to(args.root)
        new_rel = new_folder.relative_to(args.root)
        print(f"  {rel}/")
        print(f"  -> {new_rel}/")
        for old, new in inner:
            print(f"     {old.name}  ->  {new.name}")
        print()

    # ------------------------------------------------------------------
    # Find stale references in scripts/ and reports/
    # ------------------------------------------------------------------
    refs_by_file: dict[Path, set[str]] = {}
    for path, old_id, count in find_text_references(args.search_dirs, set(id_mapping)):
        refs_by_file.setdefault(path, set()).add(old_id)

    if refs_by_file:
        print(f"Stale session_id references in {len(refs_by_file)} file(s):")
        for path, ids in sorted(refs_by_file.items()):
            print(f"  {path}")
            for old in sorted(ids):
                print(f"     {old}  ->  {id_mapping[old]}")
        print()

    if not args.apply:
        print("[dry-run] No changes made. Re-run with --apply to perform the renames.")
        return 0

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    n_files = n_folders = 0
    for folder, new_folder, inner in plan:
        for old, new in inner:
            old.rename(new)
            n_files += 1
        folder.rename(new_folder)
        n_folders += 1

    n_patched = patch_text_files(set(refs_by_file), id_mapping) if refs_by_file else 0

    print(f"Renamed {n_folders} folder(s), {n_files} inner file(s); "
          f"patched {n_patched} script/report file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
