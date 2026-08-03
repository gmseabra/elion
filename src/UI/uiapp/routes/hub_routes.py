# =============================================================================
# routes/hub_routes.py
# Hub landing page and debug_static endpoint.
# =============================================================================

from flask import jsonify, render_template, current_app
from uiapp import app
from uiapp import config as _hubcfg
from uiapp.routes.shared import logger, _VIZ
import os as _os2

@app.route('/debug_static')
def debug_static():
    """
    Temporary debug endpoint — remove after confirming static file paths.
    GET /debug_static  → JSON showing Flask's static_folder and whether the JS files exist.
    """
    import os as _os2
    sf = current_app.static_folder or ''
    return jsonify({
        "static_folder":           sf,
        "static_url_path":         current_app.static_url_path,
        "static_folder_exists":    _os2.path.isdir(sf),
        "js_dir_exists":           _os2.path.isdir(_os2.path.join(sf, 'js')),
        "deepatom_js_exists":      _os2.path.isfile(_os2.path.join(sf, 'js', 'deepatom.js')),
        "vina_welcome_js_exists":  _os2.path.isfile(_os2.path.join(sf, 'js', 'vina_mini_chat_welcome.js')),
        "js_dir_contents":         _os2.listdir(_os2.path.join(sf, 'js')) if _os2.path.isdir(_os2.path.join(sf, 'js')) else [],
        "cwd":                     _os2.getcwd(),
        "routes_py_location":      __file__,
        "_VIZ":                    str(_VIZ),
    })

@app.route('/')
def hub():
    """Hub landing page linking to both visualizer tools.

    `reasoning_url` is the only template variable this page takes. It is the
    destination of the Reasoning sidebar button, which is an EXTERNAL link —
    there is no /reasoning/* endpoint on this server. Override it with the
    `UI_REASONING_URL` environment variable; see uiapp/config.py.
    """
    return render_template('hub.html', reasoning_url=_hubcfg.REASONING_URL)

# NOTE: the local `_cot_route_attn` copy that used to live here has been removed.
# It referenced four names this module never imported (ATTN_ACTION_KB_PATH,
# qwen_compat, COT_ROUTING_PARAMS, _cot_logger), was never called, and was one of
# four copy-pasted CoT routers. The single canonical implementation is now
# uiapp/routes/cot_routes.py::cot_route_attn.
