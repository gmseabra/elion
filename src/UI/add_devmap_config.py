#!/usr/bin/env python3
"""Add the `visualizer.devmap` block to the engine's input_TS.yml.

    python add_devmap_config.py                       # auto-locate input_TS.yml
    python add_devmap_config.py /path/to/input_TS.yml
    python add_devmap_config.py --print                # just show the block

The overlay's settings live beside `port` and `host` under `visualizer:`,
because that is already the one file that configures a run. Without this block
the defaults in uiapp/config.DEVMAP_DEFAULTS apply — the overlay is off and the
hotkey is Alt+Ctrl/⌘+Shift+D — so this script is a convenience, not a
requirement. Adding it is what makes those values *editable*.

Text insertion, not a YAML round-trip: ruamel/pyyaml would rewrite the whole
file and drop every comment in it, and input_TS.yml is mostly comments. This
appends to the end of the existing `visualizer:` block and leaves every other
byte alone.

Idempotent — a second run reports that the block is already there and exits 0.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BLOCK = '''
  # ── Dev source map (the hover "which file is this button in" overlay) ──────
  # Backed by uiapp/routes/devmap_routes.py + web/static/js/devmap.js. It is a
  # development aid, so it ships OFF: a tooltip quoting a source path over every
  # control is noise on a shared instance, and reads as a debug build that
  # escaped by accident.
  #
  # Read fresh on every page load, so editing this file and reloading the
  # browser is enough — no Flask restart.
  #
  # Precedence, per key:  $UI_DEVMAP_*  >  the values here  >  the built-in
  # defaults. Anything unparseable falls through to the next level and logs a
  # warning rather than raising; a typo here cannot stop the server starting.
  devmap:
    # State in a browser that has never pressed the shortcut. Once someone
    # toggles it, that browser remembers their choice and this value no longer
    # applies to them. `devmap.reset()` in the console clears that memory.
    enabled: false

    # Toggle combo. Tokens: ctrl | alt | shift | meta, plus one key, joined by
    # + - or spaces; case-insensitive. `ctrl` matches Ctrl **or** Cmd (the
    # Ctrl/⌘ convention used across this UI); `meta` is the strict Cmd form.
    #
    # Matched EXACTLY, so this binding is not fired by Ctrl+Shift+D — which
    # matters, because Chrome binds that to "bookmark all tabs".
    #
    # Matching is on KeyboardEvent.code, so layout and modifier remapping do
    # not break it: macOS reports Option+D as "∂", and a key-based match would
    # make any Alt binding unfireable.
    hotkey: "alt+ctrl+shift+d"

    # Hold this while hovering to pin the tooltip so its rows can be clicked
    # and copied. "" or "none" disables pin-on-hover.
    pin_modifier: "alt"

    # The ⌖ src map on/off badge in the bottom-left corner. false hides it; the
    # hotkey still works.
    show_badge: true

    # Hover dwell before a tooltip appears. Long enough that sweeping across a
    # toolbar does not strobe, short enough to feel like a tooltip.
    hover_delay_ms: 260
'''

CANDIDATES = [
    Path.home() / "repos" / "elion" / "src" / "elion" / "input_TS.yml",
    Path.cwd().parent / "elion" / "src" / "elion" / "input_TS.yml",
    Path.cwd().parent / "elion" / "input_TS.yml",
    Path.cwd().parent.parent / "elion" / "src" / "elion" / "input_TS.yml",
]


def locate() -> Path | None:
    for cand in CANDIDATES:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def insert(text: str) -> tuple[str, str]:
    """Return (new_text, status). status is 'inserted' | 'present' | an error."""
    lines = text.splitlines(keepends=True)

    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"^visualizer:\s*(#.*)?$", ln)), None)
    if start is None:
        return text, "no `visualizer:` section found"

    # End of the block: the next line at column 0 that is neither blank nor a
    # comment. Trailing blank/comment lines stay outside, so the block is
    # appended tight against the last real key rather than after a comment
    # that introduces the NEXT section.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln[0].isspace():
            end = i
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    tail = end
    while tail > start + 1 and lines[tail - 1].lstrip().startswith("#"):
        tail -= 1
    end = tail

    if re.search(r"^\s{2,}devmap:\s*$", "".join(lines[start:end]), re.M):
        return text, "present"

    body = BLOCK if lines[end - 1].endswith("\n") else "\n" + BLOCK
    return "".join(lines[:end]) + body + "".join(lines[end:]), "inserted"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yml", nargs="?", help="path to input_TS.yml")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the block and exit (paste it yourself)")
    ap.add_argument("--dry-run", action="store_true", help="show the result, write nothing")
    args = ap.parse_args()

    if args.show:
        print(BLOCK.rstrip())
        return 0

    path = Path(args.yml).expanduser() if args.yml else locate()
    if path is None:
        print("could not find input_TS.yml. Tried:", file=sys.stderr)
        for c in CANDIDATES:
            print("   ", c, file=sys.stderr)
        print("\nPass the path explicitly, or run with --print and paste the "
              "block under `visualizer:` yourself.", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2

    original = path.read_text()
    updated, status = insert(original)
    if status == "present":
        print(f"{path}: visualizer.devmap already present — nothing to do")
        return 0
    if status != "inserted":
        print(f"{path}: {status}", file=sys.stderr)
        print("\nRun with --print and paste the block under `visualizer:` yourself.",
              file=sys.stderr)
        return 1

    try:
        import yaml
        parsed = yaml.safe_load(updated) or {}
        dm = (parsed.get("visualizer") or {}).get("devmap")
        if not isinstance(dm, dict) or "hotkey" not in dm:
            print("refusing to write: the result did not parse as expected",
                  file=sys.stderr)
            return 1
    except ImportError:
        print("note: pyyaml not importable here, skipping the parse check")

    if args.dry_run:
        print(updated)
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(original)
    path.write_text(updated)
    print(f"{path}: added visualizer.devmap   (backup: {backup.name})")
    print("Reload the browser — no Flask restart needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
