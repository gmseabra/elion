# =============================================================================
# uiapp/__init__.py
# Creates the Flask app. Route registration happens at the bottom to avoid
# circular imports (route modules need `app` to be defined first).
# =============================================================================

from flask import Flask

from uiapp import config

# Templates and static assets live under the repository's web/ directory
# (paths are resolved centrally in uiapp.config).
app = Flask(
    __name__,
    static_folder=config.STATIC_DIR,
    static_url_path="/static",
    template_folder=config.TEMPLATES_DIR,
)

# ── Register routes ───────────────────────────────────────────────────────────
# Imported at the bottom so `app` exists before any route module runs.
# routes/__init__.py imports every sub-module which decorates `app` with routes.
import uiapp.routes   # noqa: F401  (side-effect: registers all routes)
