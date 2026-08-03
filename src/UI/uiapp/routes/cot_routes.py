# =============================================================================
# routes/cot_routes.py
# Central CoT (Chain-of-Thought) module for the Elion platform.
#
# Previously duplicated across:
#   vina_chat_routes.py  — had logger setup + _cot_route_action
#   attn_routes.py       — had _cot_route_attn (no FileHandler — silent)
#   hub_routes.py        — duplicate of _cot_route_attn (undefined _cot_logger)
#
# Registration: add to routes/__init__.py SECOND, after shared, before hub_routes:
#   _load("uiapp.routes.shared",     "shared")
#   _load("uiapp.routes.cot_routes", "cot_routes")   ← ADD HERE
#   _load("uiapp.routes.chembert_model", "chembert_model")
#   _load("uiapp.routes.hub_routes", "hub_routes")
#   ...
#
# Exports:
#   cot_logger          — shared logger → routes/.qwen/cot_main.log
#   cot_parse_json()    — robust JSON extractor for CoT output
#   build_cot_prompt()  — ChatML prompt with <|im_start|>think\n token
#   cot_route_vina()    — Vina UI action CoT router
#   cot_route_attn()    — Attn UI action CoT router (with emotional state)
# =============================================================================

import json
import logging
import os
import re
from pathlib import Path

from uiapp.llm.qwen_client import qwen_compat, QwenParams, COT_ROUTING_PARAMS
from uiapp.routes.shared import logger, VINA_ACTION_KB_PATH, ATTN_ACTION_KB_PATH

# =============================================================================
# ── Log directory ─────────────────────────────────────────────────────────────
# _HERE = .../routes/   (same as shared.py convention)
# All CoT logs live in routes/.qwen/
# =============================================================================
_HERE    = Path(__file__).resolve().parent
_COT_DIR = _HERE / ".qwen"
_COT_DIR.mkdir(parents=True, exist_ok=True)

_COT_MAIN_LOG    = str(_COT_DIR / "cot_main.log")    # vina + attn routing

# =============================================================================
# ── Formatters ────────────────────────────────────────────────────────────────
# =============================================================================
_MAIN_FMT = logging.Formatter(
    "\n" + "=" * 70 + "\n"
    "[%(asctime)s]  source=%(name)s\n"
    "%(message)s\n"
    + "─" * 70
)

# =============================================================================
# ── cot_logger → cot_main.log ─────────────────────────────────────────────────
# Receives entries from: vina_chat_routes, attn_routes, hub_routes
# source= field in formatter distinguishes them.
# =============================================================================
cot_logger = logging.getLogger("elion.cot")
_main_handler = logging.FileHandler(_COT_MAIN_LOG)
_main_handler.setFormatter(_MAIN_FMT)
cot_logger.addHandler(_main_handler)
cot_logger.setLevel(logging.DEBUG)
cot_logger.propagate = False

# =============================================================================


# =============================================================================
# ── Shared helpers ────────────────────────────────────────────────────────────
# =============================================================================

def build_cot_prompt(system: str, user: str) -> str:
    """
    Assemble a ChatML prompt ending with <|im_start|>think\\n.
    The think token elicits step-by-step reasoning before the JSON answer.
    Used by both CoT routers.
    """
    return (
        "<|im_start|>system\n" + system + "\n<|im_end|>\n"
        "<|im_start|>user\n"   + user   + "\n<|im_end|>\n"
        "<|im_start|>think\n"
    )


def cot_parse_json(text: str) -> dict | None:
    """
    Robust JSON extractor for CoT output.

    Qwen2.5-7B may output any of:
      - Raw JSON object:          {"action": "none", ...}
      - Fenced JSON:              ```json\n{...}\n```
      - Prose then --- then JSON: <think>...</think>\n---\n{...}
      - Prose with JSON embedded: lots of text ... {"action":"none"} ... more text

    Strategy (in order):
      1. Strip ```json / ``` fences
      2. Find the last {...} block via regex (handles prose prefix)
      3. Try to parse the full cleaned text as JSON
    Returns None if no valid JSON found.
    """
    if not text:
        return None

    # Strip markdown fences
    cleaned = re.sub(r"```json\s*|```", "", text).strip()

    # Extract last {...} block — handles prose before/after JSON
    matches = re.findall(r'\{[^{}]+\}', cleaned, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass

    # Fallback: try the whole cleaned text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# =============================================================================
# ── KB caches ─────────────────────────────────────────────────────────────────
# =============================================================================
_vina_kb_cache: str | None = None
_attn_kb_cache: str | None = None


def _load_kb(path: str, cache_attr: str) -> str:
    """Generic KB loader with module-level cache."""
    global _vina_kb_cache, _attn_kb_cache
    cache = _vina_kb_cache if cache_attr == "vina" else _attn_kb_cache
    if cache is not None:
        return cache
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info("[CoT] %s KB loaded (%d chars)", cache_attr, len(content))
        if cache_attr == "vina":
            _vina_kb_cache = content
        else:
            _attn_kb_cache = content
        return content
    except Exception as e:
        logger.warning("[CoT] Could not load %s KB: %s", cache_attr, e)
        if cache_attr == "vina":
            _vina_kb_cache = ""
        else:
            _attn_kb_cache = ""
        return ""


# =============================================================================
# ── cot_route_vina ────────────────────────────────────────────────────────────
# Moved from vina_chat_routes.py
# =============================================================================

def cot_route_vina(user_message: str, coach_response: str, history: list) -> dict | None:
    """
    DeepSeek-V4-style CoT router for Vina UI actions.
    Replaces _cot_route_action in vina_chat_routes.py.
    Logs to routes/.qwen/cot_main.log with source=elion.cot.
    """
    kb = _load_kb(VINA_ACTION_KB_PATH, "vina")
    if not kb:
        return None

    history_summary = "".join(
        f"  {role.upper()}: {text[:200].replace(chr(10), ' ')}\n"
        for role, text in history[-3:]
    )

    system = (
        "You are Elion's UI routing agent for a molecular docking visualizer. "
        "Decide which UI action (if any) to trigger based on the conversation. "
        "Think step-by-step, then output one raw JSON line."
    )
    user = (
        f"## UI Action Knowledge Base\n{kb}\n\n"
        f"## Recent history\n{history_summary}\n"
        f"## User message\n{user_message}\n\n"
        f"## Elion response\n{coach_response[:400]}\n\n"
        "Think step-by-step about user intent. "
        "After your reasoning write exactly --- on its own line then raw JSON:\n"
        '{"action": "none", "confidence": "high", "reason": "..."}'
    )

    cot_prompt = build_cot_prompt(system, user)

    try:
        out  = qwen_compat(cot_prompt, COT_ROUTING_PARAMS)
        text = out[0].outputs[0].text.strip()
        cot_logger.info(
            "VINA CoT\nUSER: %s\nPROMPT (tail 600):\n...%s\n\nRAW OUTPUT:\n%s",
            user_message, cot_prompt[-600:], text
        )
        logger.info("[CoT] vina output (first 400): %s", text[:400])

        decision   = cot_parse_json(text)
        if decision is None:
            return None
        action     = decision.get("action", "none")
        confidence = decision.get("confidence", "low")
        reason     = decision.get("reason", "")
        logger.info("[CoT] vina → action=%s confidence=%s", action, confidence)

        _VALID_VINA = {
            "open_visualizer", "open_voxel_inspector", "toggle_grid_box",
            "run_visualization", "run_docking", "switch_ligand_view",
            "switch_protein_view", "load_ligand", "load_receptor",
            "vina_dock_guided", "guide_pdb_conversion", "none",
        }
        if action in _VALID_VINA and confidence in ("high", "medium") and action != "none":
            return {"type": "ui_action", "action": action,
                    "confidence": confidence, "reason": reason}
        return None
    except Exception as e:
        logger.warning("[CoT] vina routing failed: %s", e)
        return None


# =============================================================================
# ── cot_route_attn ────────────────────────────────────────────────────────────
# Moved from attn_routes.py and hub_routes.py (they were identical duplicates)
# =============================================================================

def cot_route_attn(user_message: str, coach_response: str, history: list) -> dict | None:
    """
    CoT router for ChemBERT Attention UI actions + emotional state detection.
    Replaces _cot_route_attn in attn_routes.py and hub_routes.py.
    Logs to routes/.qwen/cot_main.log with source=elion.cot.
    """
    kb = _load_kb(ATTN_ACTION_KB_PATH, "attn")
    if not kb:
        return None

    history_summary = "".join(
        f"  {role.upper()}: {text[:200].replace(chr(10), ' ')}\n"
        for role, text in history[-3:]
    )

    system = (
        "You are Elion's UI routing agent for the CHEM-BERT attention visualizer. "
        "Decide which UI action (if any) to trigger based on the conversation. "
        "Also detect the user's emotional state from their message tone.\n"
        "Think step-by-step, then output one raw JSON line."
    )
    user = (
        f"## UI Action Knowledge Base\n{kb}\n\n"
        f"## Recent history\n{history_summary}\n"
        f"## User message\n{user_message}\n\n"
        f"## Elion response\n{coach_response[:400]}\n\n"
        "Think step-by-step about user intent AND emotional state. "
        "Emotional state options: calm | frustrated | confused | excited | overwhelmed\n"
        "After your reasoning write exactly --- on its own line then raw JSON:\n"
        '{"action": "none", "confidence": "high", "reason": "...", "emotional_state": "calm"}'
    )

    cot_prompt = build_cot_prompt(system, user)

    try:
        out  = qwen_compat(cot_prompt, COT_ROUTING_PARAMS)
        text = out[0].outputs[0].text.strip()
        cot_logger.info(
            "ATTN CoT\nUSER: %s\nPROMPT (tail 600):\n...%s\n\nRAW OUTPUT:\n%s",
            user_message, cot_prompt[-600:], text
        )
        logger.info("[CoT] attn output (first 400): %s", text[:400])

        decision        = cot_parse_json(text)
        if decision is None:
            return None
        action          = decision.get("action", "none")
        confidence      = decision.get("confidence", "low")
        reason          = decision.get("reason", "")
        emotional_state = decision.get("emotional_state", "calm")
        logger.info("[CoT] attn → action=%s confidence=%s emotional_state=%s",
                    action, confidence, emotional_state)

        _VALID_ATTN = {
            "open_visualizer", "show_3d", "compare_molecules",
            "fine_tune_model", "load_model", "open_vina",
            "explain_ask_elion", "none",
        }
        if action in _VALID_ATTN and confidence in ("high", "medium") and action != "none":
            return {"type": "ui_action", "action": action,
                    "confidence": confidence, "reason": reason,
                    "emotional_state": emotional_state}
        return None
    except Exception as _e:
        logger.warning("[CoT] attn routing failed: %s", _e)
        return None