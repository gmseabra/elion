# =============================================================================
# nas_storage_app/__init__.py
# Creates the Flask app. Route registration happens at the bottom to avoid
# circular imports (route modules need `app` to be defined first).
# =============================================================================

from flask import Flask

app = Flask(__name__)

# ── Register routes ───────────────────────────────────────────────────────────
# Imported at the bottom so `app` exists before any route module runs.
# routes/__init__.py imports every sub-module which decorates `app` with routes.
import nas_storage_app.routes   # noqa: F401  (side-effect: registers all routes)