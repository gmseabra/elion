"""
Elion Platform — nas_storage_app package
Imports the merged Flask app with all routes registered.
"""
from flask import Flask

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'elion-agi-platform-key'

# Import routes — this registers all @app.route decorators
from nas_storage_app import routes  # noqa: F401, E402