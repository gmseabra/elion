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

_load("nas_storage_app.routes.shared",               "shared")
_load("nas_storage_app.routes.chembert_model",       "chembert_model")
_load("nas_storage_app.routes.hub_routes",           "hub_routes")
_load("nas_storage_app.routes.attn_routes",          "attn_routes")
_load("nas_storage_app.routes.vina_chembert_routes", "vina_chembert_routes")
_load("nas_storage_app.routes.vina_dock_routes",     "vina_dock_routes")
_load("nas_storage_app.routes.vina_chat_routes",     "vina_chat_routes")
_load("nas_storage_app.routes.ts_routes",            "ts_routes")
_load("nas_storage_app.routes.tools_routes",         "tools_routes")