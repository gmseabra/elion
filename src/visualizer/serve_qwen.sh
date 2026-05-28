#!/bin/bash
# serve_qwen.sh — Qwen2.5-7B-Instruct via llama-cpp-python (library mode).
# Bypasses llama_cpp.server DNS bug. Binds to 127.0.0.1:8001.
#
# Usage:
#   nohup bash serve_qwen.sh > qwen_server.log 2>&1 &
# Check: curl http://127.0.0.1:8001/health

set -e

MODEL="/home/huangzihang/repos/LLM/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
SERVER_SCRIPT="$(dirname "$(realpath "$0")")/qwen_server.py"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

echo "🚀 Starting Qwen2.5-7B-Instruct (llama.cpp CUDA) on 127.0.0.1:8001..."
echo "   Flask restarts will NOT reload weights."
echo ""

python "$SERVER_SCRIPT" \
    --model         "$MODEL" \
    --port          8001     \
    --n_gpu_layers  35       \
    --n_ctx         8192     \
    --model_alias   qwen2.5-14b