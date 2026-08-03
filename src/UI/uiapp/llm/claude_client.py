"""
claude_client.py — Claude (via UF gateway) client for the Elion reasoning tool.
================================================================================

Pairs with qwen_client.py. Division of labor:

    Qwen2.5-14B  (vLLM :8001)   → in-app chat, UI action routing, fast turns
    Claude       (UF gateway)   → OFFLINE memory pipeline:
                                     extraction → consolidation → reconciliation
                                     + read-path answer composition

These models are GENERATIVE. The retrieval index needs embeddings, but this key
has no embedding model available — see EMBEDDING_NOTE.

--------------------------------------------------------------------------------
CONFIG  (defaults target UF NaviGator; override via env if needed)
--------------------------------------------------------------------------------
    CLAUDE_BASE_URL    default https://api.ai.it.ufl.edu/v1   (NaviGator gateway)
    CLAUDE_API_KEY     key from https://api.ai.it.ufl.edu/ui  (GatorLink login)
    CLAUDE_API_STYLE   default "openai"  (NaviGator is OpenAI-API-compatible)

NOTE: This specific key is scoped to TWO models — gpt-oss-120b and
nemotron-3-super-120b-a12b — plus no embedding model. Other team models and
Claude (cloud) return 401 until added to the key / onboarded (UFIT Help Portal).
The pipeline runs on gpt-oss-120b. If your gateway were AWS Bedrock, tell me.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# ── Gateway config ────────────────────────────────────────────────────────────
# Defaults target UF's NaviGator gateway (OpenAI-compatible). Override via env.
CLAUDE_BASE_URL   = os.environ.get("CLAUDE_BASE_URL", "https://api.ai.it.ufl.edu/v1").rstrip("/")
CLAUDE_API_KEY    = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_API_STYLE  = os.environ.get("CLAUDE_API_STYLE", "openai").lower()
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
EMBED_MODEL       = os.environ.get("CLAUDE_EMBED_MODEL", "gte-large-en-v1.5")
DEFAULT_TIMEOUT   = 120  # seconds


# ── Model catalog: the models your NaviGator KEY actually allows → roles ──────
# The key is scoped to just these two (the probe's /models + 401s confirm it).
# The other team models (gpt-oss-20b, gemma, granite, codestral) 401 until you
# add them to the key at https://api.ai.it.ufl.edu/ui. Both below are US-origin.
MODELS = {
    "gpt-oss-120b":               {"tier": "strong", "label": "GPT-OSS 120B",
                                   "note": "OpenAI · confirmed on your key (200 OK) · all stages + chat."},
    "nemotron-3-super-120b-a12b": {"tier": "strong", "label": "Nemotron 3 Super 120B",
                                   "note": "NVIDIA · on your key · strong-reasoning alternate (verify chat)."},
}

# Pipeline stages (the Claude side of the memory architecture)
STAGES = {
    "extraction":     "Per-session extraction → typed records (high volume)",
    "consolidation":  "Merge records into entity dossiers (fidelity-critical)",
    "reconciliation": "Resolve contradictions / supersession (hard reasoning)",
    "readpath":       "Compose the assistant answer from retrieved memory",
}

# Recommended default stage → model assignment (only gpt-oss-120b is confirmed)
DEFAULT_ASSIGNMENT = {
    "extraction":     "gpt-oss-120b",
    "consolidation":  "gpt-oss-120b",
    "reconciliation": "gpt-oss-120b",
    "readpath":       "gpt-oss-120b",
}

EMBEDDING_NOTE = (
    "Your key has NO embedding model — gte-large 401s, and gte is Alibaba/Chinese "
    "anyway. Add a non-Chinese embedder to the key if your team has one, or self-host "
    "one (e.g. nomic-embed-text, US) on Lysine. embed() targets the gateway; set "
    "CLAUDE_EMBED_MODEL once you have access."
)


# ── Params (mirrors QwenParams) ───────────────────────────────────────────────
@dataclass
class ClaudeParams:
    temperature: float = 0.3
    max_tokens:  int   = 1024
    top_p:       float = 0.95
    stop:        list  = field(default_factory=list)


# Per-stage param sets (parallel to your COACH_PARAMS / COT_ROUTING_PARAMS etc.)
EXTRACT_PARAMS     = ClaudeParams(temperature=0.2, max_tokens=1500)
CONSOLIDATE_PARAMS = ClaudeParams(temperature=0.3, max_tokens=2000)
RECONCILE_PARAMS   = ClaudeParams(temperature=0.4, max_tokens=2000)
READPATH_PARAMS    = ClaudeParams(temperature=0.6, max_tokens=1200)


# ── Transport ─────────────────────────────────────────────────────────────────
def is_configured() -> bool:
    return bool(CLAUDE_BASE_URL and CLAUDE_API_KEY)


def _anthropic_call(messages, model, params, system=None) -> str:
    url = f"{CLAUDE_BASE_URL}/messages"
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if params.stop:
        payload["stop_sequences"] = params.stop
    r = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # content is a list of blocks; concatenate the text blocks
    return "".join(
        b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
    ).strip()


def _openai_call(messages, model, params, system=None) -> str:
    url = f"{CLAUDE_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CLAUDE_API_KEY}",
        "Content-Type": "application/json",
    }
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    payload = {
        "model": model,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "messages": msgs,
    }
    if params.stop:
        payload["stop"] = params.stop
    r = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def claude_chat(messages, model, params: ClaudeParams | None = None, system: str | None = None) -> str:
    """
    Send an OpenAI-style messages list to the configured gateway, return text.
    `model` is one of MODELS. NOTE: the *-thinking alias is assumed to enable
    extended thinking server-side; if your gateway needs an explicit thinking
    param, add it in _anthropic_call.
    """
    if not is_configured():
        raise RuntimeError("Claude gateway not configured — set CLAUDE_BASE_URL and CLAUDE_API_KEY.")
    if model not in MODELS:
        logger.warning("[claude_client] unknown model id: %s", model)
    p = params or ClaudeParams()
    try:
        if CLAUDE_API_STYLE == "openai":
            return _openai_call(messages, model, p, system)
        return _anthropic_call(messages, model, p, system)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Claude gateway unreachable at {CLAUDE_BASE_URL}. On the UF VPN?")
    except Exception as e:
        logger.error("[claude_client] request failed: %s", e)
        raise


def health() -> dict:
    """Cheap connectivity probe (≤5 tokens). Used by /reasoning/health."""
    if not is_configured():
        return {"ok": False, "reason": "not_configured"}
    model = DEFAULT_ASSIGNMENT["extraction"]
    try:
        out = claude_chat([{"role": "user", "content": "ping"}],
                          model=model, params=ClaudeParams(max_tokens=5))
        return {"ok": True, "model": model, "sample": out[:40], "api_style": CLAUDE_API_STYLE}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:200]}


# ── Embeddings (retrieval index) ──────────────────────────────────────────────
def embed(texts, model: str | None = None) -> list:
    """
    Vector embeddings via NaviGator /v1/embeddings (OpenAI-shaped).
    Pass a string or a list of strings; returns a list of float vectors.
    """
    if not is_configured():
        raise RuntimeError("Gateway not configured — set CLAUDE_API_KEY.")
    if isinstance(texts, str):
        texts = [texts]
    url = f"{CLAUDE_BASE_URL}/embeddings"
    headers = {"Authorization": f"Bearer {CLAUDE_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers,
                          json={"model": model or EMBED_MODEL, "input": texts},
                          timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Gateway unreachable at {CLAUDE_BASE_URL}. On the UF VPN?")


# ── JSON helper ───────────────────────────────────────────────────────────────
def _safe_json(text: str):
    t = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                pass
        return {"_parse_error": True, "raw": t[:500]}


# ── Pipeline stage helpers (the Claude side of the 6D memory architecture) ────
_EXTRACT_SYS = (
    "You are a memory extractor for the Elion drug-discovery platform. Read one "
    "session transcript and emit ONLY a JSON object with this exact schema:\n"
    "{\n"
    '  "decisions":    [{"claim": str, "rationale": str, "supersedes": str|null}],\n'
    '  "constraints":  [str],\n'
    '  "artifacts":    [str],   # files, endpoints, model ids, hostnames, IPs\n'
    '  "open_threads": [str],\n'
    '  "deltas":       [str],   # what changed vs the prior known state\n'
    '  "preferences":  [str]\n'
    "}\n"
    "Preserve the LOGIC BACKBONE: every decision must carry WHY it was made and "
    "what it replaces. Be terse. No prose, no markdown fences — JSON only."
)


def extract_session(transcript: str, model: str | None = None) -> dict:
    """Turn a raw session transcript into typed, schema-bound records."""
    model = model or DEFAULT_ASSIGNMENT["extraction"]
    raw = claude_chat(
        [{"role": "user", "content": transcript[:20000]}],
        model=model, params=EXTRACT_PARAMS, system=_EXTRACT_SYS,
    )
    return _safe_json(raw)


_CONSOLIDATE_SYS = (
    "You maintain a compact, append-only DOSSIER for one entity (a project, a "
    "tool, or a recurring problem) in the Elion memory system. You are given the "
    "current dossier JSON and a batch of NEW extracted records. Merge them:\n"
    "  - de-duplicate against existing entries\n"
    "  - when a new decision contradicts an old one, KEEP BOTH but set the old "
    "one's \"status\":\"superseded\" and link \"superseded_by\"\n"
    "  - keep the dossier small: re-summarize verbose entries, drop dead threads\n"
    "Output ONLY the merged dossier JSON, same schema as the input dossier."
)


def consolidate(dossier: dict, new_records: dict, model: str | None = None) -> dict:
    """Merge new records into an entity dossier, preserving supersession history."""
    model = model or DEFAULT_ASSIGNMENT["consolidation"]
    payload = json.dumps({"dossier": dossier, "new_records": new_records})[:20000]
    raw = claude_chat(
        [{"role": "user", "content": payload}],
        model=model, params=CONSOLIDATE_PARAMS, system=_CONSOLIDATE_SYS,
    )
    return _safe_json(raw)


_RECONCILE_SYS = (
    "You resolve contradictions in the Elion memory system. You are given a list "
    "of conflicting claim pairs (old vs new). For EACH pair decide which holds "
    "now and why, reasoning carefully about recency, specificity, and dependency. "
    "Output ONLY JSON: "
    '{"resolutions":[{"winner":"old"|"new","claim":str,"reason":str}]}'
)


def reconcile(conflicts: list, model: str | None = None) -> dict:
    """Hard reasoning stage — decide what supersedes what."""
    model = model or DEFAULT_ASSIGNMENT["reconciliation"]
    raw = claude_chat(
        [{"role": "user", "content": json.dumps({"conflicts": conflicts})[:20000]}],
        model=model, params=RECONCILE_PARAMS, system=_RECONCILE_SYS,
    )
    return _safe_json(raw)