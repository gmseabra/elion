import os
import subprocess

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

    ray_tmp = "/blue/lic/huangzihang/raytmp"
    os.makedirs(ray_tmp, exist_ok=True)
    for k in ("RAY_TMPDIR", "TMPDIR", "TEMP", "TMP"):
        os.environ[k] = ray_tmp

setup_distributed_env()

# ── Determine the repo root automatically ─────────────────────────────────────
# Works whether running from elion_platform/ or the original repo path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from nas_storage_app import app   # registers all merged routes

# ── Point Jinja2 at the templates folder next to this app.py ─────────────────
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, 'templates')
app.template_folder = TEMPLATES_DIR
# No static/ folder — Flask serves no static assets directly

print("=== Elion Platform ===")
print(f"  Script dir  : {SCRIPT_DIR}")
print(f"  Templates   : {TEMPLATES_DIR}")
print(f"  http://0.0.0.0:5000/")
print(f"  http://0.0.0.0:5000/attention_visualization/")
print(f"  http://0.0.0.0:5000/vina_visualization/")

# ── AutoLearn file log ───────────────────────────────────────────────────────
import logging as _logging
_qwen_dir = os.path.join(SCRIPT_DIR, "nas_storage_app", ".qwen")
os.makedirs(_qwen_dir, exist_ok=True)
# ── Two-tier CoT logging ─────────────────────────────────────────────────────
# cot_main.log   → ALL CoT reasoning (written by routes._cot_logger directly)
# autolearn.log  → SUBSET: only feature-gap entries (written by kb_auto_learner)
#                  + [AutoLearn] tagged lines from the root logger (below)

class _AutoLearnFilter(_logging.Filter):
    """Pass only records that contain [AutoLearn] in the message — these
    are the human-reviewable lines about new actions being learned or missed."""
    def filter(self, record: _logging.LogRecord) -> bool:
        return "[AutoLearn]" in record.getMessage()

_al_log = os.path.join(_qwen_dir, "autolearn.log")
_al_fh  = _logging.FileHandler(_al_log)
_al_fh.setLevel(_logging.DEBUG)
_al_fh.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_al_fh.addFilter(_AutoLearnFilter())          # ← only [AutoLearn] lines
_logging.getLogger().addHandler(_al_fh)
_logging.getLogger().setLevel(_logging.INFO)

if __name__ == "__main__":
    print("🚀 Starting Elion Platform on http://0.0.0.0:5000")
    print(f"📝 AutoLearn log → {_al_log}")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)