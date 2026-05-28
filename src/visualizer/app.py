import os
import subprocess
import yaml
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Force InfiniBand Networking for Ray & vLLM
# ─────────────────────────────────────────────────────────────────────────────
def setup_distributed_env():
    try:
        cmd = "ip addr show bridge-1145 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1"
        ib_ip = subprocess.check_output(cmd, shell=True).decode().strip()
        if ib_ip:
            print(f"📡 Detected InfiniBand IP: {ib_ip}")
            os.environ["RAY_NODE_IP_ADDRESS"] = ib_ip
            os.environ["VLLM_HOST_IP"]        = ib_ip
            os.environ["HOST_IP"]             = ib_ip
            os.environ["RAY_ADDRESS"]         = f"{ib_ip}:6379"
        else:
            print("⚠️  Could not detect IP on bridge-1145.")
    except Exception as e:
        print(f"⚠️  InfiniBand setup error: {e}")

    ray_tmp = "../../../../raytmp"
    os.makedirs(ray_tmp, exist_ok=True)
    for k in ("RAY_TMPDIR", "TMPDIR", "TEMP", "TMP"):
        os.environ[k] = ray_tmp

setup_distributed_env()

# ── Determine the repo root automatically ─────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load input_routes.yml into Flask app.config BEFORE importing routes ───────
# routes.py reads vina config from current_app.config["VINA"] at request time.
# All box coords, paths, and hyperparameters live in input_routes.yml only.
_routes_yml = Path(SCRIPT_DIR) / "input_routes.yml"
try:
    with open(_routes_yml) as _f:
        _routes_cfg = yaml.safe_load(_f)
    print(f"✅ Loaded vina config from {_routes_yml}")
except FileNotFoundError:
    _routes_cfg = {}
    print(f"⚠️  input_routes.yml not found at {_routes_yml} — vina will use fallback defaults")

from nas_storage_app import app   # registers all merged routes

# Store vina config on Flask app so routes.py can access via current_app.config
app.config["VINA"] = _routes_cfg.get("vina", {})

# ── Point Jinja2 at the templates folder next to this app.py ─────────────────
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, 'templates')
app.template_folder = TEMPLATES_DIR

print("=== Elion Platform ===")
print(f"  Script dir  : {SCRIPT_DIR}")
print(f"  Templates   : {TEMPLATES_DIR}")
print(f"  Vina bin    : {app.config['VINA'].get('bin', '(not set)')}")
print(f"  Vina box    : center=({app.config['VINA'].get('center_x')}, "
      f"{app.config['VINA'].get('center_y')}, {app.config['VINA'].get('center_z')}) "
      f"size=({app.config['VINA'].get('size_x')}, {app.config['VINA'].get('size_y')}, "
      f"{app.config['VINA'].get('size_z')})")
print(f"  http://0.0.0.0:5000/")
print(f"  http://0.0.0.0:5000/attention_visualization/")
print(f"  http://0.0.0.0:5000/vina_visualization/")

# ── AutoLearn file log ───────────────────────────────────────────────────────
import logging as _logging
_qwen_dir = os.path.join(SCRIPT_DIR, "nas_storage_app", ".qwen")
os.makedirs(_qwen_dir, exist_ok=True)

class _AutoLearnFilter(_logging.Filter):
    """Pass only records that contain [AutoLearn] in the message."""
    def filter(self, record: _logging.LogRecord) -> bool:
        return "[AutoLearn]" in record.getMessage()

_al_log = os.path.join(_qwen_dir, "autolearn.log")
_al_fh  = _logging.FileHandler(_al_log)
_al_fh.setLevel(_logging.DEBUG)
_al_fh.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_al_fh.addFilter(_AutoLearnFilter())
_logging.getLogger().addHandler(_al_fh)
_logging.getLogger().setLevel(_logging.INFO)

if __name__ == "__main__":
    print("🚀 Starting Elion Platform on http://0.0.0.0:5000")
    print(f"📝 AutoLearn log → {_al_log}")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)