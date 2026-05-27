"""
Elion Visualizer — Flask application factory.

Directory layout expected at runtime:
    visualizer/
      __init__.py          ← this file
      static/js/           ← hub-core.js, attn.js, vina.js, converter.js
      templates/           ← hub.html, vina_ajax.html, attn_ajax.html
      routes/
        __init__.py
        hub.py
        attn.py
        vina.py
        tools.py
"""

import os
import sys

# ── sys.path: make legacy packages importable ────────────────────────────────
_ATTN_BASE = "/blue/lic/huangzihang/repos/Elion-AGI-Ecosystem/attention_visualization"
if _ATTN_BASE not in sys.path:
    sys.path.insert(0, _ATTN_BASE)

# ── Hardware / engine env vars MUST be set before any vLLM / torch import ───
os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
os.environ.setdefault("VLLM_USE_RAY_COMPILED_DAG",      "0")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE",     "1")
os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM",    "0")
os.environ.setdefault("NCCL_P2P_DISABLE",    "0")
os.environ.setdefault("NCCL_IB_DISABLE",     "1")
os.environ.setdefault("NCCL_SOCKET_IFNAME",  "ibs8f0")
os.environ.setdefault("NCCL_CROSS_NIC",      "1")
os.environ.setdefault("NCCL_IB_GID_INDEX",   "3")
os.environ.setdefault("NCCL_NET_GDR_LEVEL",  "0")
os.environ.setdefault("NCCL_NET_MERGE_LEVEL","LOC")
os.environ.setdefault("NCCL_IB_HCA",         "^mlx5_6,mlx5_12")
os.environ.setdefault("NCCL_IB_RETRY_CNT",   "13")

import getpass
try:
    _persist = f"/blue/lic/{getpass.getuser()}/.cache/triton"
    os.makedirs(_persist, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = _persist
except Exception:
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_fallback"

# ── Flask app ────────────────────────────────────────────────────────────────
from flask import Flask

_HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder  = os.path.join(_HERE, "static"),
    static_url_path= "/static",
    template_folder= os.path.join(_HERE, "templates"),
)

# ── Register blueprints ──────────────────────────────────────────────────────
from visualizer.routes.hub   import bp as hub_bp
from visualizer.routes.attn  import bp as attn_bp
from visualizer.routes.vina  import bp as vina_bp
from visualizer.routes.tools import bp as tools_bp

app.register_blueprint(hub_bp)
app.register_blueprint(attn_bp,  url_prefix="/attention_visualization")
app.register_blueprint(vina_bp,  url_prefix="/vina_visualization")
app.register_blueprint(tools_bp, url_prefix="/tools")