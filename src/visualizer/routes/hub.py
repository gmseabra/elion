"""
hub.py — Hub landing page + static JS file serving.

The /static/js/<filename> route is what the browser requests when hub.html
loads <script src="/static/js/hub-core.js"> etc.
Flask's built-in static file handler serves these automatically when
static_folder is configured correctly in __init__.py.

This blueprint handles the root landing page only.
"""

import os
from flask import Blueprint, render_template, send_from_directory

bp = Blueprint('hub', __name__)

_JS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'js')


@bp.route('/')
def hub():
    """Hub landing page."""
    return render_template('hub.html')


# Explicit static JS route — belt-and-suspenders fallback in case Flask's
# built-in static handler doesn't pick up the correct folder.
@bp.route('/static/js/<path:filename>')
def serve_js(filename):
    """Serve JS files from visualizer/static/js/."""
    return send_from_directory(_JS_DIR, filename)