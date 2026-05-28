#!/usr/bin/env python3
"""
replace_hardcoded_paths.py

Replaces hardcoded absolute paths with paths relative to each source file,
using a configurable multi-root mapping.

Each --map entry defines:
    <absolute_prefix_in_source>:<real_path_on_disk>

The script finds every occurrence of a known absolute prefix in a file,
computes os.path.relpath from that file's directory to the real on-disk
equivalent, and substitutes it in.

Usage (dry run first, then apply):

    python replace_hardcoded_paths.py \\
        --map /blue/lic/huangzihang/repos/elion/src:/home/huangzihang/repos/elion/src \\
        --map /blue/lic/huangzihang/repos/common_utils:/home/huangzihang/repos/common_utils \\
        --map /blue/lic/huangzihang/repos/Elion-AGI-Ecosystem:/home/huangzihang/repos/Elion-AGI-Ecosystem \\
        --map /blue/lic/huangzihang/repos/AutoDock-Vina:/home/huangzihang/repos/AutoDock-Vina \\
        --map /blue/lic/huangzihang/repos/IAG933:/home/huangzihang/repos/IAG933 \\
        --map /blue/lic/huangzihang/raytmp:/home/huangzihang/raytmp \\
        --map /blue/lic/huangzihang/.cache:/home/huangzihang/.cache \\
        --search-dir /home/huangzihang/repos/elion/src \\
        --dry-run

    # Remove --dry-run to apply changes.

    # To also exclude specific files (e.g. this script itself):
        --exclude replace_hardcoded_paths.py

NOTE:
  - Prefixes are matched longest-first so more specific ones win.
  - The real paths in --map must exist on disk (used for relpath computation).
  - --search-dir defaults to cwd.
  - Files/dirs in .git are always skipped.
"""

import os
import re
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# File extensions treated as text. Everything else gets a binary heuristic.
# ---------------------------------------------------------------------------
TEXT_EXTENSIONS = {
    ".py", ".yml", ".yaml", ".json", ".sh", ".txt", ".md",
    ".cfg", ".ini", ".toml", ".env", ".html", ".j2", ".pdbqt",
}

# Always skip these filenames regardless of extension.
BUILTIN_SKIP_FILENAMES = {
    "AutoDock-Vina-GPU-2-1",  # known binary
}


def is_text_file(path: Path, skip_filenames: set) -> bool:
    if path.name in skip_filenames:
        return False
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            return b"\x00" not in f.read(1024)
    except OSError:
        return False


def build_prefix_map(map_args: list[str]) -> list[tuple[str, Path]]:
    """
    Parse --map entries of the form 'abs_prefix:real_path'.
    Returns a list sorted longest-prefix-first so more specific prefixes win.
    Validates that real_path exists on disk.
    """
    entries = []
    for entry in map_args:
        if ":" not in entry:
            print(f"Error: --map entry must be 'abs_prefix:real_path', got: {entry!r}",
                  file=sys.stderr)
            sys.exit(1)
        # Split on first colon only (real paths may contain colons on Windows, but
        # abs_prefix won't have one in practice on Linux).
        colon = entry.index(":")
        abs_prefix = entry[:colon].rstrip("/")
        real_path  = Path(entry[colon + 1:]).resolve()
        if not real_path.exists():
            print(f"Warning: real path does not exist on disk: {real_path}  "
                  f"(mapped from {abs_prefix!r})", file=sys.stderr)
        entries.append((abs_prefix, real_path))

    # Longest prefix first so '/blue/.../elion/src' matches before '/blue/.../repos'
    entries.sort(key=lambda t: len(t[0]), reverse=True)
    return entries


def make_regex(prefix_map: list[tuple[str, Path]]) -> re.Pattern:
    """Build a single regex matching any known prefix + optional path continuation."""
    alternation = "|".join(re.escape(prefix) for prefix, _ in prefix_map)
    # Path continuation: everything that isn't whitespace or a common delimiter.
    return re.compile(rf"(?:{alternation})(?:[^\s\"\'<>|,;{{}}()\[\]\\]*)")


def make_relative(
    abs_target: str,
    source_file: Path,
    prefix_map: list[tuple[str, Path]],
) -> str:
    """
    Convert abs_target (found verbatim in a source file) to a path relative
    to source_file's directory, using prefix_map to find the real on-disk root.
    """
    for abs_prefix, real_root in prefix_map:
        if abs_target.startswith(abs_prefix):
            suffix = abs_target[len(abs_prefix):].lstrip("/")
            target = (real_root / suffix) if suffix else real_root
            return os.path.relpath(target, start=source_file.parent)
    return abs_target  # no prefix matched — leave unchanged


def replace_in_content(
    content: str,
    source_file: Path,
    prefix_map: list[tuple[str, Path]],
    pattern: re.Pattern,
) -> tuple[str, list[str]]:
    changes = []
    parts = []
    offset = 0

    for m in pattern.finditer(content):
        abs_path = m.group(0)
        rel_path = make_relative(abs_path, source_file, prefix_map)
        if rel_path != abs_path:
            changes.append(f"  {abs_path!r}  →  {rel_path!r}")
            parts.append(content[offset:m.start()])
            parts.append(rel_path)
            offset = m.end()

    parts.append(content[offset:])
    return "".join(parts), changes


def process_file(
    path: Path,
    prefix_map: list[tuple[str, Path]],
    pattern: re.Pattern,
    skip_filenames: set,
    dry_run: bool,
) -> int:
    if not is_text_file(path, skip_filenames):
        return 0
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"[SKIP] {path}: {e}", file=sys.stderr)
        return 0

    new_content, changes = replace_in_content(original, path, prefix_map, pattern)
    if not changes:
        return 0

    tag = "DRY RUN" if dry_run else "CHANGED"
    print(f"\n[{tag}] {path}")
    for c in changes:
        print(c)

    if not dry_run:
        path.write_text(new_content, encoding="utf-8")

    return len(changes)


def main():
    parser = argparse.ArgumentParser(
        description="Replace hardcoded absolute paths with relative paths (multi-root).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--map", metavar="ABS_PREFIX:REAL_PATH",
        action="append", required=True,
        help=(
            "Map an absolute prefix used in source files to its real location "
            "on this machine. Repeat for each root. "
            "Example: --map /blue/lic/.../elion/src:/home/user/repos/elion/src"
        ),
    )
    parser.add_argument(
        "--search-dir", default=None,
        help="Directory to scan. Defaults to cwd.",
    )
    parser.add_argument(
        "--exclude", metavar="FILENAME", action="append", default=[],
        help="Filename(s) to skip (basename match). Repeat as needed. "
             "The script always excludes itself automatically.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without modifying files.",
    )
    args = parser.parse_args()

    # Always exclude this script itself.
    skip_filenames = BUILTIN_SKIP_FILENAMES | {Path(__file__).name} | set(args.exclude)

    prefix_map = build_prefix_map(args.map)
    pattern    = make_regex(prefix_map)

    search_dir = Path(args.search_dir).resolve() if args.search_dir else Path.cwd()
    if not search_dir.is_dir():
        print(f"Error: search directory does not exist: {search_dir}", file=sys.stderr)
        sys.exit(1)

    print("Prefix map (longest first):")
    for abs_prefix, real_root in prefix_map:
        print(f"  {abs_prefix!r}")
        print(f"      → {real_root}")
    print(f"\nSearching  : {search_dir}")
    print(f"Skipping   : {sorted(skip_filenames)}")
    print(f"Mode       : {'DRY RUN (no files changed)' if args.dry_run else 'APPLY CHANGES'}")
    print("=" * 70)

    total_files = total_replacements = 0
    for path in sorted(search_dir.rglob("*")):
        # Skip .git trees entirely.
        if ".git" in path.parts:
            continue
        if path.is_file():
            n = process_file(path, prefix_map, pattern, skip_filenames, args.dry_run)
            if n:
                total_files += 1
                total_replacements += n

    print("\n" + "=" * 70)
    print(f"Summary: {total_replacements} replacement(s) across {total_files} file(s).")
    if args.dry_run:
        print("Dry run complete — no files modified. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()