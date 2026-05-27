"""
qwen_client.py — Drop-in HTTP client for the persistent vLLM OpenAI server.

Replaces all llm.generate() calls in routes.py.
Flask restarts do NOT reload the model — only serve_qwen.sh touches it.

Architecture (DeepSeek-V4 paper §3.5 inspiration):
  serve_qwen.sh  →  vLLM OpenAI server (port 8001)  ←  qwen_client.py  ←  routes.py
                     [weights stay in GPU memory]

Usage in routes.py:
    from qwen_client import qwen, QwenParams
    response_text = qwen(prompt, QwenParams(temperature=0.7, max_tokens=600))
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Server config ─────────────────────────────────────────────────────────────
VLLM_BASE_URL   = "http://localhost:8001"
VLLM_MODEL_NAME = "qwen2.5-14b"           # matches --served-model-name in serve_qwen.sh
_COMPLETIONS_URL = f"{VLLM_BASE_URL}/v1/completions"
_CHAT_URL        = f"{VLLM_BASE_URL}/v1/chat/completions"
_HEALTH_URL      = f"{VLLM_BASE_URL}/health"

DEFAULT_TIMEOUT = 120   # seconds per request


# ── Params dataclass (mirrors SamplingParams) ─────────────────────────────────
@dataclass
class QwenParams:
    temperature: float  = 0.7
    max_tokens:  int    = 600
    top_p:       float  = 0.95
    stop:        list   = field(default_factory=list)


# ── Pre-defined param sets (replace SamplingParams constants in routes.py) ────
PRO_PARAMS          = QwenParams(temperature=0.7,  max_tokens=8192, top_p=0.95)
FREE_PARAMS         = QwenParams(temperature=0.8,  max_tokens=2048, top_p=0.9)
COACH_PARAMS        = QwenParams(temperature=0.75, max_tokens=600,  top_p=0.95)
COT_ROUTING_PARAMS  = QwenParams(temperature=0.3,  max_tokens=1024, top_p=0.9)
PLANNER_PARAMS      = QwenParams(temperature=0.3,  max_tokens=512,  top_p=0.9)
PROFILER_PARAMS     = QwenParams(temperature=0.0,  max_tokens=15,   top_p=1.0)
TOOL_PARAMS         = QwenParams(temperature=0.2,  max_tokens=1024, top_p=0.9)
EXTRACTOR_PARAMS    = QwenParams(temperature=0.2,  max_tokens=400,  top_p=0.9)
COMPACT_PARAMS      = QwenParams(temperature=0.3,  max_tokens=800,  top_p=0.9)
REFLECT_PARAMS      = QwenParams(temperature=0.4,  max_tokens=400,  top_p=0.9)
CORRECTION_PARAMS   = QwenParams(temperature=0.5,  max_tokens=600,  top_p=0.9)


# ── Core function: send a raw ChatML prompt, return text ──────────────────────
def qwen(prompt: str, params: QwenParams | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Send a raw ChatML-formatted prompt to the vLLM completions endpoint.
    Returns the generated text string (equivalent to llm.generate(prompt, sp)[0].outputs[0].text).

    Drop-in for:
        outputs = llm.generate(prompt, sampling_params)
        text    = outputs[0].outputs[0].text.strip()
    →
        text = qwen(prompt, params)
    """
    p = params or QwenParams()
    payload = {
        "model":       VLLM_MODEL_NAME,
        "prompt":      prompt,
        "temperature": p.temperature,
        "max_tokens":  p.max_tokens,
        "top_p":       p.top_p,
        "stop":        p.stop or ["<|im_end|>", "<|endoftext|>"],
    }
    try:
        resp = requests.post(_COMPLETIONS_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Qwen server unreachable at port 8001. "
            "Run: nohup bash serve_qwen.sh > qwen_server.log 2>&1 &"
        )
    except Exception as e:
        logger.error("[qwen_client] request failed: %s", e)
        raise


def qwen_chat(messages: list[dict], params: QwenParams | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Send OpenAI-style messages list to the chat completions endpoint.
    Useful when you already have a messages array instead of a raw ChatML string.
    """
    p = params or QwenParams()
    payload = {
        "model":       VLLM_MODEL_NAME,
        "messages":    messages,
        "temperature": p.temperature,
        "max_tokens":  p.max_tokens,
        "top_p":       p.top_p,
    }
    try:
        resp = requests.post(_CHAT_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Qwen server unreachable at port 8001.")
    except Exception as e:
        logger.error("[qwen_client] chat request failed: %s", e)
        raise


def is_server_alive() -> bool:
    """Check if the vLLM server is running."""
    try:
        return requests.get(_HEALTH_URL, timeout=3).status_code == 200
    except Exception:
        return False


# ── Compatibility shim: fake vLLM output object ───────────────────────────────
# Lets old code that does outputs[0].outputs[0].text keep working unchanged.
class _FakeOutput:
    def __init__(self, text: str):
        self.text = text

class _FakeResult:
    def __init__(self, text: str):
        self.outputs = [_FakeOutput(text)]

def qwen_compat(prompt: str, params: QwenParams | None = None) -> list:
    """
    Exact drop-in for llm.generate(prompt, sampling_params).
    Returns a list with one fake result object so existing code needs zero changes:
        outputs = qwen_compat(prompt, COACH_PARAMS)
        text    = outputs[0].outputs[0].text.strip()   # works as before
    """
    text = qwen(prompt, params)
    return [_FakeResult(text)]


def qwen_stream(prompt: str, params: QwenParams | None = None, timeout: int = DEFAULT_TIMEOUT):
    """
    Stream tokens from the vLLM completions endpoint.
    Yields decoded text chunks as they arrive (Server-Sent Events compatible).
    Each chunk is a raw string fragment — caller assembles the full response.
    """
    import json as _json
    p = params or QwenParams()
    payload = {
        "model":       VLLM_MODEL_NAME,
        "prompt":      prompt,
        "temperature": p.temperature,
        "max_tokens":  p.max_tokens,
        "top_p":       p.top_p,
        "stop":        p.stop or ["<|im_end|>", "<|endoftext|>"],
        "stream":      True,
    }
    try:
        with requests.post(
            _COMPLETIONS_URL, json=payload,
            stream=True, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        return
                    try:
                        chunk = _json.loads(data)
                        text  = chunk["choices"][0].get("text", "")
                        if text:
                            yield text
                    except Exception as _parse_err:
                        import logging as _l
                        _l.getLogger(__name__).warning(
                            "[qwen_stream] parse error: %s | raw: %s", _parse_err, data[:200]
                        )
                        continue
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Qwen server unreachable at port 8001. "
            "Run: nohup bash serve_qwen.sh > qwen_server.log 2>&1 &"
        )