#!/usr/bin/env bash
# Start the Elion UI platform (http://0.0.0.0:5000 by default).
#
# The Qwen LLM server is a separate process — start it first if you need the
# chat/routing features:
#     QWEN_MODEL=/path/to/model.gguf bash uiapp/llm/serve_qwen.sh &
set -e
cd "$(dirname "$0")"
exec python run.py
