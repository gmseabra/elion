"""
Route registration.

Every module below decorates the single global ``app`` created in
``uiapp/__init__.py`` — there are no Flask blueprints, so importing a module
*is* registering its routes.  The loader is explicit rather than a directory
walk so the order is visible and reviewable: it is load-bearing, because these
modules import each other at import time.

Order rules:
  * ``shared``                 first — defines every path constant and job store
  * ``cot_routes``             before its consumer ``mini_chat_routes``
  * ``session_routes``         before its consumers; opens/creates the SQLite DB
  * ``chembert_model``         before ``attn_routes`` / ``vina_chembert_routes``
  * ``mini_chat_routes``       owns ALL chat endpoints (it replaced the three
                               near-duplicate per-tool implementations, which is
                               why there is no longer a ``vina_chat_routes``)
  * ``ts_mol_render``          is a library, pulled in by ``ts_routes``

Every module here decorates ``app`` directly; there is no blueprint
registration step, and no module below is allowed to need one.

NOTE: tests/test_structure.py::test_route_loader_matches_disk parses the
_load(...) calls out of this file with a regex and asserts the matching
routes/<stem>.py exists on disk -- so keep the literal double-quoted form,
and do not write an example call in prose (the regex cannot tell them apart).
"""

import importlib.util, pathlib as _pl, sys


def _load(name, stem):
    if name in sys.modules:
        return sys.modules[name]
    path = _pl.Path(__file__).parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── foundations (no routes of their own, or libraries) ───────────────────────
_load("uiapp.routes.shared",                 "shared")
_load("uiapp.routes.cot_routes",             "cot_routes")
_load("uiapp.routes.session_routes",         "session_routes")
_load("uiapp.routes.chembert_model",         "chembert_model")
_load("uiapp.routes.ts_mol_render",          "ts_mol_render")

# ── page + tool routes ───────────────────────────────────────────────────────
_load("uiapp.routes.hub_routes",             "hub_routes")
_load("uiapp.routes.attn_routes",            "attn_routes")
_load("uiapp.routes.mini_chat_routes",       "mini_chat_routes")
_load("uiapp.routes.vina_chembert_routes",   "vina_chembert_routes")
_load("uiapp.routes.vina_dock_routes",       "vina_dock_routes")
_load("uiapp.routes.pose_routes",            "pose_routes")
_load("uiapp.routes.ts_routes",              "ts_routes")
_load("uiapp.routes.tools_routes",           "tools_routes")
_load("uiapp.routes.deepatom_routes",        "deepatom_routes")
