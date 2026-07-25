#!/bin/bash
# serve_qwen.sh — Qwen2.5-Instruct via llama-cpp-python (library mode).
# Bypasses the llama_cpp.server DNS bug. Binds to 127.0.0.1:8001.
#
# Usage:
#   QWEN_MODEL=/path/to/model.gguf nohup bash serve_qwen.sh > qwen_server.log 2>&1 &
# Check:
#   curl http://127.0.0.1:8001/health
#
# The GGUF weights are an external resource (not shipped in this repo). Point at
# them with the QWEN_MODEL environment variable; otherwise the repo-relative
# default below (models/…) is used.

set -e

HERE="$(dirname "$(realpath "$0")")"
REPO_ROOT="$(realpath "$HERE/../..")"

MODEL="${QWEN_MODEL:-$REPO_ROOT/models/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}"
SERVER_SCRIPT="$HERE/qwen_server.py"

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    echo "   Set QWEN_MODEL to the path of your Qwen .gguf weights and retry."
    exit 1
fi

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

echo "🚀 Starting Qwen2.5-Instruct (llama.cpp CUDA) on 127.0.0.1:8001..."
echo "   Model: $MODEL"
echo "   Flask restarts will NOT reload weights."
echo ""

python "$SERVER_SCRIPT" \
    --model         "$MODEL" \
    --port          8001     \
    --n_gpu_layers  35       \
    --n_ctx         8192     \
    --model_alias   qwen2.5-14b
