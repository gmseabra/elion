#!/usr/bin/env python3
"""
gateway_probe.py — confirm your NaviGator key reaches the approved local models.
================================================================================

Your key has access to NaviGator LOCAL models only (no Claude, no Qwen). This
checks the ones the Elion pipeline now runs on:

    [1] GET /models       → everything this key can reach (flags the LLMs)
    [2] chat ping each LLM → gpt-oss-120b / gpt-oss-20b / gemma / granite / codestral
    [3] /embeddings ping   → retrieval-index path (gte-large-en-v1.5)

Run on the UF VPN:

    export CLAUDE_API_KEY="<your key from https://api.ai.it.ufl.edu/ui>"
    python gateway_probe.py

    # NAVIGATOR_TOOLKIT_API_KEY also accepted; override base/key via --base/--key.

Throwaway diagnostic — run from anywhere, delete after. Costs a few tokens.
"""

import os
import sys
import json
import time
import argparse

try:
    import requests
except ImportError:
    sys.exit("Needs `requests` — run inside your elion-app env, or: pip install requests")


DEFAULT_BASE = "https://api.ai.it.ufl.edu/v1"
EMBED_MODEL  = "gte-large-en-v1.5"

# The LLMs the Elion pipeline + chat run on (approved, non-Chinese, local on NaviGator)
LLM_MODELS = [
    "gpt-oss-120b",            # OpenAI · workhorse: consolidation / reconciliation / read-path
    "gpt-oss-20b",             # OpenAI · fast: high-volume extraction
    "gemma-3-27b-it",          # Google · mid alternate
    "granite-3.3-8b-instruct", # IBM · fast structured alternate
    "codestral-22b",           # Mistral · code tasks
]

# Checked against /models only (informational) — should be ABSENT until cloud onboarding
CLAUDE_MODELS = [
    "claude-3-haiku", "claude-3.5-haiku", "claude-3.5-sonnet", "claude-3.5-sonnet-v2",
    "claude-3.7-sonnet", "claude-4-sonnet", "claude-4-sonnet-thinking",
]

TIMEOUT = 20


def _headers(key):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _err_detail(r):
    try:
        body = r.json()
        return json.dumps(body.get("error", body))[:160]
    except Exception:
        return (r.text or "")[:160]


def list_models(base, key):
    try:
        r = requests.get(f"{base}/models", headers=_headers(key), timeout=TIMEOUT)
    except requests.exceptions.ConnectionError as e:
        return None, f"CONN_ERR: {str(e)[:110]}"
    except requests.exceptions.Timeout:
        return None, "TIMEOUT"
    if r.status_code != 200:
        return None, f"{r.status_code}: {_err_detail(r)}"
    try:
        data = r.json()
        items = data.get("data", data.get("models", []))
        ids = sorted({(it.get("id") or it.get("name")) for it in items if isinstance(it, dict)})
        return [i for i in ids if i], None
    except Exception as e:
        return None, f"parse error: {e}"


def chat_ping(base, key, model):
    body = {"model": model, "max_tokens": 5, "messages": [{"role": "user", "content": "ping"}]}
    t0 = time.time()
    try:
        r = requests.post(f"{base}/chat/completions", headers=_headers(key), json=body, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "code": "CONN_ERR", "ms": 0, "detail": str(e)[:60]}
    except requests.exceptions.Timeout:
        return {"ok": False, "code": "TIMEOUT", "ms": TIMEOUT * 1000, "detail": "timed out"}
    ms = int((time.time() - t0) * 1000)
    if r.status_code == 200:
        try:
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            txt = ""
        return {"ok": True, "code": 200, "ms": ms, "detail": txt[:30]}
    return {"ok": False, "code": r.status_code, "ms": ms, "detail": _err_detail(r)}


def embed_ping(base, key, model):
    try:
        r = requests.post(f"{base}/embeddings", headers=_headers(key),
                          json={"model": model, "input": "ping"}, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "code": "CONN_ERR", "detail": str(e)[:80]}
    except requests.exceptions.Timeout:
        return {"ok": False, "code": "TIMEOUT", "detail": "timed out"}
    if r.status_code == 200:
        try:
            return {"ok": True, "dim": len(r.json()["data"][0]["embedding"])}
        except Exception as e:
            return {"ok": False, "code": 200, "detail": f"parse: {e}"}
    return {"ok": False, "code": r.status_code, "detail": _err_detail(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("CLAUDE_BASE_URL", DEFAULT_BASE))
    ap.add_argument("--key",  default=os.environ.get("CLAUDE_API_KEY")
                              or os.environ.get("NAVIGATOR_TOOLKIT_API_KEY", ""))
    ap.add_argument("--embed-model", default=os.environ.get("CLAUDE_EMBED_MODEL", EMBED_MODEL))
    args = ap.parse_args()

    base = args.base.rstrip("/")
    key = args.key.strip()
    if not key:
        sys.exit("Set CLAUDE_API_KEY (or NAVIGATOR_TOOLKIT_API_KEY), or pass --key.")

    print(f"\nNaviGator gateway : {base}")
    print(f"Key               : ...{key[-4:]} (len {len(key)})")
    print("=" * 72)

    # ── [1] models the key can reach ──────────────────────────────────────────
    print("\n[1] GET /models — everything this key can reach")
    print("-" * 72)
    models, err = list_models(base, key)
    if err:
        print(f"  ✗ {err}")
        if err.startswith(("401", "403")):
            print("    → key invalid or not authorized. Check it at https://api.ai.it.ufl.edu/ui")
        elif err.startswith("CONN_ERR"):
            print("    → can't reach the gateway. On the UF VPN? URL correct?")
    else:
        print(f"  {len(models)} model(s) visible to this key:")
        for m in models:
            tag = "   ← pipeline LLM" if m in LLM_MODELS else ("   ← claude" if m in CLAUDE_MODELS else "")
            print(f"    • {m}{tag}")
        missing_llm = [m for m in LLM_MODELS if m not in models]
        if missing_llm:
            print(f"\n  Expected LLMs NOT visible: {', '.join(missing_llm)}")
            print("    → check the key's Models list on its Key ID page, or pick a different one.")
        has_claude = [m for m in CLAUDE_MODELS if m in models]
        print("\n  Claude access: " + (", ".join(has_claude) if has_claude
              else "none (expected — needs team cloud-budget onboarding via UFIT Help Portal)."))

    # ── [2] chat ping each pipeline LLM ───────────────────────────────────────
    print("\n[2] Chat ping each pipeline LLM (max_tokens=5)")
    print("-" * 72)
    print(f"  {'model':<28}{'code':<10}{'ms':<7}result")
    reachable = []
    for m in LLM_MODELS:
        res = chat_ping(base, key, m)
        if res["ok"]:
            reachable.append(m)
            shown = f'"{res["detail"]}"' if res["detail"] else "ok"
        else:
            shown = str(res["detail"])[:36]
        print(f"  {m:<28}{str(res['code']):<10}{res['ms']:<7}{shown}")

    # ── [3] embeddings ────────────────────────────────────────────────────────
    print(f"\n[3] Embeddings ping — /embeddings ({args.embed_model})")
    print("-" * 72)
    emb = embed_ping(base, key, args.embed_model)
    if emb.get("ok"):
        print(f"  ✓ vector dim {emb['dim']}")
    else:
        print(f"  ✗ {emb.get('code', '')} {emb.get('detail', '')}")

    # ── verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICT")
    print("-" * 72)
    if reachable:
        print(f"  ✓ Usable LLMs: {', '.join(reachable)}")
        rec = "gpt-oss-120b" if "gpt-oss-120b" in reachable else reachable[0]
        print(f"  → Recommended default for chat + heavy pipeline stages: {rec}")
    else:
        print("  ✗ No pipeline LLM responded — key may lack these models, or key/VPN issue.")
    print(f"  {'✓' if emb.get('ok') else '✗'} Embeddings via {args.embed_model}")
    print("\n  claude_client.py is already configured for these defaults:")
    print(f'    export CLAUDE_BASE_URL="{base}"')
    print(f'    export CLAUDE_API_STYLE="openai"')
    print(f'    export CLAUDE_API_KEY="<your key>"')
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()