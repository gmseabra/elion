#!/usr/bin/env python
"""diagnose_models.py -- print exactly what the server sees when it builds the
Model dropdown. Run from the UI root (the directory holding uiapp/ and web/):

    python diagnose_models.py

Reads the same config file the app reads and runs the app's own
_pose_score_models() -- but lifts it out of pose_routes.py with ast instead of
importing it, so it needs only PyYAML and works in any environment, with the
Flask app stopped.
"""
import ast
import os
import re
import sys
import types
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "uiapp").is_dir() or not (ROOT / "web").is_dir():
    sys.exit("run this from the UI root (the directory holding uiapp/ and web/)")


def tick(b):
    return "OK  " if b else "FAIL"


# ---- which config file the app reads --------------------------------------
# uiapp/config.py:  INPUT_ROUTES_YML = _env_path("UI_INPUT_ROUTES_YML",
#                                                CONFIG_DIR / "input_routes.yml")
# with CONFIG_DIR = REPO_ROOT / "config".  input_TS.yml is NOT consulted for
# the pose: section.
env = os.environ.get("UI_INPUT_ROUTES_YML")
yml = Path(env).expanduser() if env else (ROOT / "config" / "input_routes.yml")

print("config file   (UI_INPUT_ROUTES_YML %s)" % ("set" if env else "unset"))
print("  %s %s" % (tick(yml.is_file()), yml))
if not yml.is_file():
    sys.exit("\nThat file does not exist. Nothing below can work until it does.")

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is not installed in this environment; run it in the UI's env.")
cfg = (yaml.safe_load(open(yml, encoding="utf-8")) or {}).get("pose") or {}

# ---- the four keys the dropdown depends on --------------------------------
print("\npose: keys")
for key in ("gign_script", "gign_model", "deepatom_ckpt_dir", "default_script"):
    val = (cfg.get(key) or "").strip()
    if not val:
        print("  ---  %-18s <blank>" % key)
        continue
    p = Path(val).expanduser()
    ok = p.is_dir() if key.endswith("_dir") else p.is_file()
    print("  %s %-18s %s" % (tick(ok), key, p))
    if ok:
        continue
    parent = p if key.endswith("_dir") else p.parent
    if parent.is_dir():
        kids = sorted(x.name for x in parent.iterdir())[:14]
        print("       %s exists, holding: %s"
              % (parent, ", ".join(kids) if kids else "(empty)"))
        # the mistake this script exists to catch: right file, wrong depth
        if not key.endswith("_dir"):
            found = [x for x in parent.rglob(p.name) if x.is_file()][:3]
            if found:
                print("       >>> the file IS here: %s" % found[0])
                print("       >>> fix: mv it to %s" % p)
                print("       >>>  or: set %s to the path above" % key)
    else:
        print("       %s does not exist either" % parent)

# ---- what the dropdown will actually show ---------------------------------
print("\nModel dropdown  (the app's own _pose_score_models)")
src_path = ROOT / "uiapp" / "routes" / "pose_routes.py"
src = src_path.read_text(encoding="utf-8")
if "_pose_score_models" not in src:
    sys.exit("  %s has no _pose_score_models -- the patched pose_routes.py is "
             "NOT deployed. That alone would leave the menu empty." % src_path)

mod = types.ModuleType("m")
mod.__dict__.update({"re": re, "logger": types.SimpleNamespace(
    info=lambda *a, **k: None, warning=lambda *a, **k: None)})
mod.__dict__["_pose_cfg"] = lambda: cfg
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in (
            "_pose_score_models", "_pose_resolve_model"):
        exec(compile(ast.Module([node], []), str(src_path), "exec"), mod.__dict__)

models = mod._pose_score_models()
if not models:
    print("  (empty -- neither gign_model nor deepatom_ckpt_dir yielded anything)")
for m in models:
    print("  %s %-34s engine=%-9s %s"
          % (tick(m["ok"]), m["label"], m["engine"], m["why"] or m["note"]))
live = [m for m in models if m["ok"]]
print("\n  %d of %d selectable" % (len(live), len(models)))
if models and not live:
    print("  Every entry is greyed. The FAIL lines above say which path is wrong.")
