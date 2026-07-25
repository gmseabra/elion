#!/usr/bin/env python
# =============================================================================
# run.py — entry point for the Elion UI platform.
#
#   python run.py            # start the Flask server on http://0.0.0.0:5000
#
# Hardware/engine environment variables are set BEFORE importing the app (and
# therefore before torch / vLLM / triton are imported). All filesystem paths
# come from uiapp/config.py and are relative to this repository — nothing is
# hard-coded to a user's home directory.
# =============================================================================

import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Engine environment (must be set before heavy imports)
# ─────────────────────────────────────────────────────────────────────────────
def setup_engine_env() -> None:
    # vLLM / NCCL tuning — opt-in defaults, never override an explicit setting.
    for key, val in {
        "VLLM_DISABLE_CUSTOM_ALL_REDUCE": "1",
        "VLLM_USE_RAY_COMPILED_DAG":      "0",
        "VLLM_DISABLE_COMPILE_CACHE":     "1",
        "VLLM_ALLREDUCE_USE_SYMM_MEM":    "0",
        "NCCL_P2P_DISABLE":   "0",
        "NCCL_IB_DISABLE":    "1",
        "NCCL_SOCKET_IFNAME": "ibs8f0",
        "NCCL_CROSS_NIC":     "1",
        "NCCL_IB_GID_INDEX":  "3",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_NET_MERGE_LEVEL": "LOC",
        "NCCL_IB_HCA":        "^mlx5_6,mlx5_12",
        "NCCL_IB_RETRY_CNT":  "13",
    }.items():
        os.environ.setdefault(key, val)

    # Triton kernel cache (repo-agnostic; override with TRITON_CACHE_DIR).
    triton_cache = os.environ.get("TRITON_CACHE_DIR") or str(Path.home() / ".cache" / "triton")
    try:
        os.makedirs(triton_cache, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache
    except OSError:
        os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_fallback"

    # Optional InfiniBand IP detection for Ray / vLLM (no-op off-cluster).
    try:
        cmd = "ip addr show bridge-1145 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1"
        ib_ip = subprocess.check_output(cmd, shell=True).decode().strip()
        if ib_ip:
            print(f"📡 Detected InfiniBand IP: {ib_ip}")
            os.environ["RAY_NODE_IP_ADDRESS"] = ib_ip
            os.environ["VLLM_HOST_IP"] = ib_ip
            os.environ["HOST_IP"] = ib_ip
            os.environ["RAY_ADDRESS"] = f"{ib_ip}:6379"
    except Exception as exc:  # noqa: BLE001 — best-effort, must never crash startup
        print(f"⚠️  InfiniBand setup skipped: {exc}")

    # Ray scratch dir — repo-relative by default (override with RAY_TMPDIR).
    ray_tmp = os.environ.get("RAY_TMPDIR") or str(REPO_ROOT / ".raytmp")
    os.makedirs(ray_tmp, exist_ok=True)
    for key in ("RAY_TMPDIR", "TMPDIR", "TEMP", "TMP"):
        os.environ[key] = ray_tmp


setup_engine_env()

# ── Import the app (env is now configured) ───────────────────────────────────
from uiapp import app, config  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Load vina docking config and resolve its relative paths to absolute ones
# ─────────────────────────────────────────────────────────────────────────────
def load_vina_config() -> dict:
    try:
        with open(config.INPUT_ROUTES_YML) as fh:
            cfg = yaml.safe_load(fh) or {}
        print(f"✅ Loaded config from {config.INPUT_ROUTES_YML}")
    except FileNotFoundError:
        print(f"⚠️  {config.INPUT_ROUTES_YML} not found — vina will use fallback defaults")
        return {}

    vina = cfg.get("vina", {})
    # Turn the repo-relative paths in the yml into absolute paths.
    for key in ("bin", "log", "ligand_pdbqt_dir"):
        if vina.get(key):
            vina[key] = config.resolve(vina[key])
    for protein in vina.get("proteins", []):
        for key in ("default_receptor", "default_ligand"):
            if protein.get(key):
                protein[key] = config.resolve(protein[key])
    return vina


app.config["VINA"] = load_vina_config()

# ── Banner ───────────────────────────────────────────────────────────────────
print("=== Elion UI Platform ===")
print(f"  Repo root : {REPO_ROOT}")
print(f"  Templates : {config.TEMPLATES_DIR}")
print(f"  Vina bin  : {app.config['VINA'].get('bin', '(not set)')}")
print("  http://0.0.0.0:5000/")
print("  http://0.0.0.0:5000/attention_visualization/")
print("  http://0.0.0.0:5000/vina_visualization/")

# ── AutoLearn file log ───────────────────────────────────────────────────────
import logging  # noqa: E402

os.makedirs(config.KB_SHADOW_DIR, exist_ok=True)


class _AutoLearnFilter(logging.Filter):
    """Pass only records that contain [AutoLearn] in the message."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "[AutoLearn]" in record.getMessage()


_al_fh = logging.FileHandler(config.AUTOLEARN_LOG_PATH)
_al_fh.setLevel(logging.DEBUG)
_al_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_al_fh.addFilter(_AutoLearnFilter())
logging.getLogger().addHandler(_al_fh)
logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    print("🚀 Starting Elion UI Platform on http://0.0.0.0:5000")
    print(f"📝 AutoLearn log → {config.AUTOLEARN_LOG_PATH}")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
