#!/bin/bash
# serve_qwen.sh — Persistent Qwen2.5-14B-Instruct vLLM server for vina_visualization.
#
# Run ONCE. Keep running. Flask restarts never reload the model.
# Exposes OpenAI-compatible API on port 8001.
#
# Usage:
#   nohup bash serve_qwen.sh > qwen_server.log 2>&1 &
#
# Check alive:
#   curl http://localhost:8001/health

set -e

MODEL="/blue/lic/huangzihang/repos/Elion-AGI-Ecosystem/LLM/Qwen2.5-14B-Instruct"
RAY_TMP="/blue/lic/huangzihang/raytmp"
TRITON_CACHE="/blue/lic/huangzihang/.cache/triton"

export RAY_TMPDIR="$RAY_TMP"
export TMPDIR="$RAY_TMP"
export TRITON_CACHE_DIR="$TRITON_CACHE"

export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bridge-1145
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=0

mkdir -p "$RAY_TMP" "$TRITON_CACHE"

echo "🚀 Starting Qwen2.5-14B-Instruct on port 8001..."
echo "   Flask restarts will NOT reload weights."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model              "$MODEL"     \
    --host               0.0.0.0      \
    --port               8001         \
    --tensor-parallel-size 1          \
    --quantization       fp8          \
    --max-model-len      16384        \
    --max-num-batched-tokens 16384    \
    --max-num-seqs       32           \
    --gpu-memory-utilization 0.95     \
    --trust-remote-code               \
    --enforce-eager                   \
    --disable-custom-all-reduce       \
    --enable-prefix-caching           \
    --served-model-name  qwen2.5-14b  \
    --allowed-origins    '["*"]'