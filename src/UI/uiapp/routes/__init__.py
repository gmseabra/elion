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

_load("uiapp.routes.shared",               "shared")
_load("uiapp.routes.chembert_model",       "chembert_model")
_load("uiapp.routes.hub_routes",           "hub_routes")
_load("uiapp.routes.attn_routes",          "attn_routes")
_load("uiapp.routes.vina_chembert_routes", "vina_chembert_routes")
_load("uiapp.routes.vina_dock_routes",     "vina_dock_routes")
_load("uiapp.routes.vina_chat_routes",     "vina_chat_routes")
_load("uiapp.routes.ts_routes",            "ts_routes")
_load("uiapp.routes.tools_routes",         "tools_routes")