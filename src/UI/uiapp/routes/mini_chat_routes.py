# =============================================================================
# routes/mini_chat_routes.py
#
# Single source of truth for all Elion mini-chat endpoints:
#   - Vina Docking chat      → /vina_visualization/chat(/stream|/clear)
#   - ChemBERT Attn chat     → /attention_visualization/chat(/stream|/clear)
#   - DeepAtom CNN chat      → /attention_visualization/chat(/stream|/clear)
#                              (shares attn endpoint, distinguished by chat_type)
#
# Previously split across vina_chat_routes.py and attn_routes.py.
# Now unified here so all three chats share:
#   - Markdown component introductions (used on first message)
#   - Conversational detection (suppress UI context for greetings)
#   - Emotion detection + relational rules
#   - Session DB persistence (SQLite keyed by IP + chat_type)
#   - CoT UI action routing (from cot_routes.py)
#
# Registration: add to __init__.py after session_routes, before deepatom_routes:
#   _load("uiapp.routes.mini_chat_routes", "mini_chat_routes")
#
# vina_chat_routes.py and the chat section of attn_routes.py should import
# their endpoints FROM this file rather than defining them independently.
# =============================================================================

import json as _json
import logging
import os

from flask import Response, jsonify, request, stream_with_context

from uiapp import app
from uiapp.llm.qwen_client import (
    COACH_PARAMS, COT_ROUTING_PARAMS, QwenParams, qwen_compat, qwen_stream,
)
from uiapp.routes.shared import (
    ATTN_ACTION_KB_PATH, VINA_ACTION_KB_PATH, logger,
)
from uiapp.routes.chembert_model import _detect_emotion
from uiapp.routes.cot_routes import (
    cot_logger,
    cot_route_attn   as _cot_route_attn,
    cot_route_vina   as _cot_route_action,
)
from uiapp.routes.session_routes import (
    _client_ip as _get_ip,
    load_history,
    save_turn,
)

# =============================================================================
# ── Markdown component introductions ─────────────────────────────────────────
#
# Each chat type shows a brief plain-English description on first contact.
# This is injected into the system prompt so Qwen can refer to platform
# capabilities when a user asks "what can you do?" or "what is this?".
# =============================================================================

_VINA_INTRO_MD = """
## Vina Docking Assistant

You are embedded in **Elion's AutoDock Vina Visualizer** — a molecular docking tool.

**What this tool does:**
- Predicts how tightly a small-molecule ligand binds to a protein receptor
- Shows per-atom binding energy decomposition in 3D
- Reads `.pdbqt` files (convert `.pdb` first using the PDB→PDBQT tool)

**Workflow:** Convert → Load receptor → Load ligand → Vina Dock → Visualize

**Key numbers:**
- Binding score in `kcal/mol` — more negative = stronger binding
- Below −7 = good; below −9 = very strong
""".strip()

_ATTN_INTRO_MD = """
## ChemBERT Attention Visualizer Assistant

You are embedded in **Elion's ChemBERT Attention Visualizer**.

**What this tool does:**
- Runs a fine-tuned BERT model on SMILES strings
- Highlights per-atom `weight_a` attention scores in 3D
- Shows which atoms the model considers most important for binding affinity
- Supports single-compound and side-by-side Compare mode

**Workflow:** Enter SMILES → select model → Visualize → optionally Compare

**Key numbers:**
- `weight_a` = attention score per atom (higher = more important to the CNN prediction)
- Predicted binding affinity in `kcal/mol`
""".strip()

_DEEPATOM_INTRO_MD = """
## DeepAtom CNN Saliency Assistant

You are embedded in **Elion's DeepAtom CNN Saliency Visualizer**.

**What this tool does:**
- Runs a ShuffleNetV3 × 2.0 3D-CNN on a 32³ voxel grid built from `.atomtypes` files
- Computes per-atom importance = ‖∂(predicted pK)/∂(input voxel)‖₂ over 24 channels
- Atom colour & size encode importance — larger/brighter = more critical to pK prediction
- Reports predicted binding pK (−log₁₀ Kd)

**Workflow:** Enter compound ID (e.g. BM-1-57) or path → set data dir → Make Atomtypes → Visualize

**Key numbers:**
- Predicted pK — higher = tighter predicted binding (pK = −log₁₀ Kd)
- Per-atom importance score — guides structural modification priorities
""".strip()

# =============================================================================
# ── System prompts ────────────────────────────────────────────────────────────
# =============================================================================

VINA_SYSTEM_PROMPT = (
    "You are Elion, an expert AI assistant for computational drug discovery.\n"
    "You are embedded in a Vina molecular docking visualizer. You help researchers understand:\n"
    "- AutoDock Vina docking results and binding energies (kcal/mol)\n"
    "- Molecular interactions between ligands and protein receptors\n"
    "- SMILES notation, molecular properties, and drug-likeness\n"
    "- ChemBERT attention scores and what they reveal about binding sites\n"
    "- How to interpret 3D molecular visualizations and interaction energies\n"
    "- Next steps in the drug discovery pipeline\n\n"
    "Be concise, scientific, and practical. When given docking scores or SMILES, "
    "analyze them directly. Negative binding energies (e.g. -8.5 kcal/mol) indicate "
    "stronger binding. Below -7 is considered good, below -9 is very strong.\n\n"
    + _VINA_INTRO_MD
)

VINA_GUIDED_PROMPT = (
    "You are Elion, a guided assistant inside the Elion Vina Visualizer.\n\n"
    "STRICT RULES — follow every one:\n"
    "1. Reply in MAXIMUM 1-2 short sentences. No markdown headers. No bullet lists. No step numbers.\n"
    "2. Give exactly ONE concrete UI action. Name the specific button or field.\n"
    "3. If the receptor and ligand paths are already filled in (visible in context), "
    "tell the user to verify the paths and click the Vina Dock button.\n"
    "4. Detect where the user is in the workflow from the conversation history, "
    "then give ONLY the single immediate next micro-step.\n"
    "5. NEVER explain concepts, list alternatives, or describe future steps.\n"
    "6. FIRST-TIME USER DETECTION: If the user mentions having a .pdb file (not .pdbqt), "
    "or asks how to get started, or seems unfamiliar with the workflow, "
    "tell them to use the PDB → PDBQT converter tool first via the Ask Elion menu.\n"
    "7. POST-CONVERSION: If [Just converted] appears in context, the file was already saved "
    "and the path auto-filled. Tell the user which field still needs filling, then guide "
    "them to click Vina Dock. Do NOT ask them to convert again.\n\n"
    "Workflow: 1→Convert .pdb to .pdbqt (if needed) 2→Load receptor 3→Load ligand "
    "4→Click Vina Dock 5→Click Visualize\n\n"
    + _VINA_INTRO_MD
)

ATTN_SYSTEM_PROMPT = (
    "You are Elion, an expert AI assistant for medicinal chemistry and machine learning.\n"
    "You are embedded in the Elion CHEM-BERT Attention Visualizer. You help researchers:\n"
    "- Interpret ChemBERT attention weight_a scores for individual atoms\n"
    "- Understand predicted binding affinity scores\n"
    "- Compare two compounds side-by-side using the Compare mode\n"
    "- Guide fine-tuning of CHEM-BERT on custom SMILES datasets\n"
    "- Explain SMILES notation and molecular structure\n"
    "- Suggest structural modifications to improve binding affinity\n"
    "- Interpret 3D visualization of per-atom attention weights\n\n"
    "Be concise, scientific, and practical. Higher weight_a scores indicate atoms more "
    "important to the model's binding prediction. Negative binding scores (kcal/mol) are "
    "favorable; more negative = stronger predicted binding.\n\n"
    + _ATTN_INTRO_MD
)

ATTN_GUIDED_PROMPT = (
    "You are Elion guiding a researcher through the CHEM-BERT Attention Visualizer.\n\n"
    "STRICT RULES:\n"
    "1. MAXIMUM 2 sentences. No markdown headers. No bullet lists.\n"
    "2. ONE concrete next UI action. Name the specific button or field.\n"
    "3. Do NOT explain concepts. One step only.\n\n"
    + _ATTN_INTRO_MD
)

DEEPATOM_SYSTEM_PROMPT = (
    "You are Elion, an expert AI assistant for 3D CNN-based binding affinity prediction.\n"
    "You are embedded in the Elion DeepAtom CNN Saliency Visualizer. You help researchers:\n"
    "- Understand per-atom importance scores from the ShuffleNetV3 CNN\n"
    "- Interpret predicted binding pK values\n"
    "- Use Make Atomtypes to generate .atomtypes input files\n"
    "- Navigate compound IDs and data directory paths\n"
    "- Suggest which atoms to modify based on saliency\n\n"
    "Be concise, scientific, and practical.\n\n"
    + _DEEPATOM_INTRO_MD
)

DEEPATOM_GUIDED_PROMPT = (
    "You are Elion guiding a researcher through the DeepAtom CNN Saliency Visualizer.\n\n"
    "STRICT RULES:\n"
    "1. MAXIMUM 2 sentences. No markdown headers. No bullet lists.\n"
    "2. ONE concrete next UI action. Name the specific button or field.\n"
    "3. Do NOT explain concepts. One step only.\n\n"
    "Workflow: Enter compound ID → set data dir → Make Atomtypes → Visualize\n\n"
    + _DEEPATOM_INTRO_MD
)

# =============================================================================
# ── Shared helpers ────────────────────────────────────────────────────────────
# =============================================================================

_CONVERSATIONAL = frozenset({
    "hi", "hello", "hey", "sup", "yo", "howdy", "hiya",
    "good morning", "good afternoon", "good evening",
    "how are you", "what's up", "whats up", "who are you",
    "what can you do", "help", "thanks", "thank you", "bye",
    "what is this", "what's this", "what is elion", "what are you",
})

# In-memory fallback histories (used if DB unavailable)
_vina_chat_history:     list[tuple[str, str]] = []
_attn_chat_history:     list[tuple[str, str]] = []
_deepatom_chat_history: list[tuple[str, str]] = []

_HISTORY_MAP = {
    "vina":     _vina_chat_history,
    "attn":     _attn_chat_history,
    "deepatom": _deepatom_chat_history,
}


def _pick_prompts(chat_type: str, guided: bool) -> tuple[str, str]:
    """Return (system_prompt, guided_prompt) for the given chat_type."""
    if chat_type == "vina":
        return VINA_SYSTEM_PROMPT, VINA_GUIDED_PROMPT
    if chat_type == "deepatom":
        return DEEPATOM_SYSTEM_PROMPT, DEEPATOM_GUIDED_PROMPT
    return ATTN_SYSTEM_PROMPT, ATTN_GUIDED_PROMPT


def _build_ctx_lines(chat_type: str, ctx: dict, is_conv: bool) -> list[str]:
    """Build context lines injected before the user message."""
    if is_conv:
        return []
    lines = []
    if chat_type == "vina":
        if ctx.get("smiles"):
            lines.append(f"Current ligand SMILES: {ctx['smiles']}")
        if ctx.get("score") is not None:
            lines.append(f"Best docking score: {ctx['score']} kcal/mol")
        if ctx.get("mode_energies"):
            lines.append(f"Top binding modes: {', '.join(str(e) for e in ctx['mode_energies'][:5])} kcal/mol")
        if ctx.get("ligand_path"):
            lines.append(f"Ligand: {ctx['ligand_path']}")
        if ctx.get("receptor_path"):
            lines.append(f"Receptor: {ctx['receptor_path']}")
        if ctx.get("n_atoms"):
            lines.append(f"Ligand heavy atoms: {ctx['n_atoms']}")
        # NOT SET only after workflow has been attempted
        if ctx.get("last_conversion") or ctx.get("ligand_path") or ctx.get("receptor_path"):
            if not ctx.get("ligand_path"):   lines.append("Ligand path: NOT SET")
            if not ctx.get("receptor_path"): lines.append("Receptor path: NOT SET")
        if ctx.get("last_conversion"):
            lc = ctx["last_conversion"]
            lines.append(
                f"[Just converted] {lc.get('filename','?')} → {lc.get('mol_type','?')} PDBQT "
                f"saved at: {lc.get('output_path','?')} — path was auto-filled in the UI."
            )
    elif chat_type == "deepatom":
        if ctx.get("ligand"):    lines.append(f"Compound: {ctx['ligand']}")
        if ctx.get("data_dir"):  lines.append(f"Data dir: {ctx['data_dir']}")
        if ctx.get("pred_pk") is not None: lines.append(f"Predicted pK: {ctx['pred_pk']}")
    else:  # attn / chembert
        if ctx.get("smiles"):  lines.append(f"SMILES: {ctx['smiles']}")
        if ctx.get("score") is not None: lines.append(f"Affinity: {ctx['score']} kcal/mol")
        if ctx.get("model"):   lines.append(f"Model: {ctx['model']}")
        if ctx.get("mode"):    lines.append(f"Mode: {ctx['mode']}")
    return lines


def _keyword_route_vina(msg: str) -> dict | None:
    """Keyword/RAG-based UI action router for Vina."""
    try:
        from uiapp.router.ui_action_router import route_ui_action as _rag
        return _rag(msg, tool_hint="vina")
    except Exception:
        pass
    u = msg.lower()
    if any(w in u for w in ["run docking", "start docking", "dock this", "now what",
                              "what now", "next step"]):
        return {"action": "vina_dock_guided", "confidence": "high",
                "reason": "keyword: dock/progression"}
    if any(w in u for w in ["load ligand", "ligand path"]):
        return {"action": "load_ligand", "confidence": "medium", "reason": "keyword: ligand"}
    if any(w in u for w in ["load receptor", "receptor path", "protein file"]):
        return {"action": "load_receptor", "confidence": "medium", "reason": "keyword: receptor"}
    return None


def _keyword_route_attn(msg: str) -> dict | None:
    """Keyword/RAG-based UI action router for ChemBERT/Attn."""
    try:
        from uiapp.router.ui_action_router import route_ui_action as _rag
        return _rag(msg, tool_hint="attn")
    except Exception:
        pass
    u = msg.lower()
    if any(w in u for w in ["visualize", "show 3d", "render"]):
        return {"action": "show_3d", "confidence": "medium", "reason": "keyword: visualize"}
    if any(w in u for w in ["compare", "side by side", "two compound"]):
        return {"action": "compare_mode", "confidence": "medium", "reason": "keyword: compare"}
    if any(w in u for w in ["fine-tune", "finetune", "fine tune", "train"]):
        return {"action": "run_finetune", "confidence": "medium", "reason": "keyword: finetune"}
    return None


def _keyword_route_deepatom(msg: str) -> dict | None:
    """Keyword/RAG-based UI action router for DeepAtom."""
    try:
        from uiapp.router.ui_action_router import route_ui_action as _rag
        return _rag(msg, tool_hint="deepatom")
    except Exception:
        pass
    u = msg.lower()
    if any(w in u for w in ["visualize", "run", "show", "compute"]):
        return {"action": "run_deepatom", "confidence": "medium", "reason": "keyword: visualize"}
    if any(w in u for w in ["make atomtypes", "atomtypes", "convert"]):
        return {"action": "make_atomtypes", "confidence": "medium",
                "reason": "keyword: atomtypes"}
    return None


def _route_ui(chat_type: str, msg: str, response: str,
              history: list, ctx: dict) -> dict | None:
    """Dispatch UI action routing to the right router for each chat type."""
    resp_lower = response.lower()
    guided     = ctx.get("guided_mode", False)

    if chat_type == "vina":
        if guided:
            u = msg.lower()
            # Post-conversion
            if ctx.get("last_conversion") and any(
                w in u for w in ["now what", "next", "proceed", "ready", "done"]
            ):
                rec, lig = ctx.get("receptor_path","").strip(), ctx.get("ligand_path","").strip()
                if rec and lig:
                    return {"type":"ui_action","action":"vina_dock_guided",
                            "confidence":"high","reason":"post-conversion: both paths set"}
                return {"type":"ui_action",
                        "action":"load_receptor" if not rec else "load_ligand",
                        "confidence":"high","reason":"post-conversion: fill remaining path"}
            # First-time / PDB detection
            if ("pdb" in u and "pdbqt" not in u) or any(
                w in u for w in ["how do i start","i'm new","im new","first time",
                                   "don't know","dont know","no idea"]
            ) or any(w in resp_lower for w in ["ask elion","convert it","pdb to pdbqt",
                                                "converter","tool menu"]):
                return {"type":"ui_action","action":"guide_pdb_conversion",
                        "confidence":"high","reason":"guided: user has .pdb"}
            # Response contains dock/load/visualize cues
            if any(w in resp_lower for w in ["vina dock","click dock","paths look correct",
                                               "click the vina","click vina dock"]):
                return {"type":"ui_action","action":"vina_dock_guided",
                        "confidence":"high","reason":"guided: verify paths + dock"}
            if any(w in resp_lower for w in ["click load","ligand field","load the ligand"]):
                return {"type":"ui_action","action":"load_ligand",
                        "confidence":"high","reason":"guided: load ligand"}
            if any(w in resp_lower for w in ["click visualize","visualize button"]):
                return {"type":"ui_action","action":"run_visualization",
                        "confidence":"high","reason":"guided: visualize"}
            if any(w in resp_lower for w in ["receptor field","receptor path","load receptor"]):
                return {"type":"ui_action","action":"load_receptor",
                        "confidence":"high","reason":"guided: load receptor"}
            return _keyword_route_vina(msg)
        else:
            ua = _keyword_route_vina(msg)
            if ua is None:
                ua = _cot_route_action(msg, response, history[-6:])
            if ua is None:
                if any(w in msg.lower() for w in ["now what","what now","next step",
                                                    "what do i","proceed","ready"]):
                    if ctx.get("ligand_path") or ctx.get("receptor_path"):
                        ua = {"type":"ui_action","action":"vina_dock_guided",
                              "confidence":"high","reason":"paths set, next step is Vina Dock"}
            return ua

    elif chat_type == "deepatom":
        if guided:
            if any(w in resp_lower for w in ["click visualize","click the visualize",
                                               "run deepatom","compute"]):
                return {"type":"ui_action","action":"run_deepatom",
                        "confidence":"high","reason":"guided: visualize"}
            if any(w in resp_lower for w in ["make atomtypes","atomtypes","convert"]):
                return {"type":"ui_action","action":"make_atomtypes",
                        "confidence":"high","reason":"guided: make atomtypes"}
            return _keyword_route_deepatom(msg)
        else:
            ua = _keyword_route_deepatom(msg)
            if ua is None:
                ua = _cot_route_attn(msg, response, history[-6:])
            return ua

    else:  # attn / chembert
        if guided:
            if any(w in resp_lower for w in ["click visualize","visualize button",
                                               "click the ⚡","the visualize"]):
                return {"type":"ui_action","action":"show_3d",
                        "confidence":"high","reason":"guided: click Visualize"}
            if any(w in resp_lower for w in ["compare","compare button"]):
                return {"type":"ui_action","action":"compare_mode",
                        "confidence":"high","reason":"guided: compare mode"}
            if any(w in resp_lower for w in ["fine-tune","finetune"]):
                return {"type":"ui_action","action":"run_finetune",
                        "confidence":"high","reason":"guided: fine-tune"}
            if any(w in resp_lower for w in ["select a model","finetuned","pretrained","load model"]):
                return {"type":"ui_action","action":"load_model",
                        "confidence":"high","reason":"guided: select model"}
            return _keyword_route_attn(msg)
        else:
            ua = _keyword_route_attn(msg)
            if ua is None:
                ua = _cot_route_attn(msg, response, history[-6:])
            return ua


# =============================================================================
# ── Core streaming generator ──────────────────────────────────────────────────
# Single implementation used by all three chat types.
# =============================================================================

def _mini_chat_stream_generate(msg: str, ctx: dict):
    """
    Core SSE generator for all mini-chat types.
    Reads chat_type from ctx (default "vina").
    Yields SSE events: token | ui_action | done | error
    """
    try:
        if not msg:
            yield "event: error\ndata: Empty message\n\n"
            return

        chat_type = ctx.get("chat_type", "vina")   # "vina" | "attn" | "deepatom"
        ip        = _get_ip()

        # Load persistent history (DB first, in-memory fallback)
        db_hist   = load_history(ip, chat_type, last_n=20)
        mem_hist  = _HISTORY_MAP.get(chat_type, _vina_chat_history)
        history   = db_hist if db_hist else mem_hist

        # Emotion detection
        emotion = _detect_emotion(msg)
        if emotion != "calm":
            ctx["detected_emotion"] = emotion
        logger.info("[MiniChat] chat_type=%s emotion=%s msg=%r", chat_type, emotion, msg[:80])

        # Conversational detection
        msg_lower  = msg.lower().strip("!?.，。 ")
        is_conv    = msg_lower in _CONVERSATIONAL

        # Context block
        ctx_lines  = _build_ctx_lines(chat_type, ctx, is_conv)
        if emotion != "calm":
            ctx_lines.append(
                f"[User emotional state: {emotion} — "
                + ("validate first, then one short step." if emotion == "frustrated" else
                   "one concept only, no jargon." if emotion == "overwhelmed" else
                   "clarify gently before proceeding." if emotion == "confused" else
                   "match their energy briefly, then continue." if emotion == "excited" else "")
                + "]"
            )

        # Token budget
        if ctx.get("guided_mode"):      tok = 120
        elif emotion in ("frustrated", "overwhelmed"): tok = 80
        else:                           tok = 200

        # Select prompts
        sys_full, sys_guided = _pick_prompts(chat_type, ctx.get("guided_mode", False))
        sys_prompt   = sys_guided if ctx.get("guided_mode") else sys_full
        ctx_block    = ("\n[Session]\n" + "\n".join(ctx_lines)) if ctx_lines else ""

        # Build ChatML prompt
        parts = [f"<|im_start|>system\n{sys_prompt}<|im_end|>"]
        for role, content in history[-6:]:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{ctx_block}\n\n{msg}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)

        # Stream
        stream_params = QwenParams(temperature=0.5, max_tokens=tok, top_p=0.9)
        full_response = ""
        for chunk in qwen_stream(prompt, stream_params):
            full_response += chunk
            yield f"event: token\ndata: {_json.dumps(chunk)}\n\n"

        reply = full_response.strip()

        # Persist to in-memory + DB
        mem_hist.append(("user", msg))
        mem_hist.append(("assistant", reply))
        if len(mem_hist) > 20:
            del mem_hist[:2]
        save_turn(ip, chat_type, "user",      msg)
        save_turn(ip, chat_type, "assistant", reply)

        # CoT / keyword UI action routing
        ui_action = _route_ui(chat_type, msg, reply, history, ctx)

        # Auto-learn fallback
        if ui_action is None:
            try:
                from uiapp.router.kb_auto_learner import maybe_learn_and_route
                hint = "vina" if chat_type == "vina" else \
                       "deepatom" if chat_type == "deepatom" else "attn"
                ui_action = maybe_learn_and_route(msg, tool_hint=hint)
                if ui_action:
                    logger.info("[AutoLearn] %s learned: %s", chat_type, ui_action.get("action"))
            except Exception as _ale:
                logger.warning("[AutoLearn] %s error: %s", chat_type, _ale)

        logger.info("[MiniChat] chat_type=%s ui_action=%s", chat_type, ui_action)
        if ui_action:
            yield f"event: ui_action\ndata: {_json.dumps(ui_action)}\n\n"

        yield "event: done\ndata: {}\n\n"

    except Exception as exc:
        logger.error("[MiniChat] stream error: %s", exc)
        yield f"event: error\ndata: {_json.dumps(str(exc))}\n\n"


# =============================================================================
# ── Routes ────────────────────────────────────────────────────────────────────
# =============================================================================

# ── Vina ──────────────────────────────────────────────────────────────────────

@app.route('/vina_visualization/chat', methods=['POST'])
def vina_chat_ep():
    """Non-streaming Vina chat fallback."""
    try:
        data    = request.get_json(force=True) or {}
        message = (data.get('message') or '').strip()
        context = data.get('context', {})
        context.setdefault('chat_type', 'vina')
        if not message:
            return jsonify({"status": "error", "message": "Empty message"}), 400

        is_conv    = message.lower().strip("!?.，。 ") in _CONVERSATIONAL
        ctx_lines  = _build_ctx_lines("vina", context, is_conv)
        ctx_block  = ("\n[Current docking session]\n" + "\n".join(ctx_lines)) if ctx_lines else ""
        sys_prompt = VINA_GUIDED_PROMPT if context.get('guided_mode') else VINA_SYSTEM_PROMPT

        ip      = _get_ip()
        history = load_history(ip, "vina", last_n=20) or _vina_chat_history

        parts = [f"<|im_start|>system\n{sys_prompt}<|im_end|>"]
        for role, content in history[-6:]:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{ctx_block}\n\n{message}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        out      = qwen_compat("\n".join(parts), COACH_PARAMS)
        response = out[0].outputs[0].text.strip()

        _vina_chat_history.append(("user", message))
        _vina_chat_history.append(("assistant", response))
        if len(_vina_chat_history) > 20: del _vina_chat_history[:2]
        save_turn(ip, "vina", "user",      message)
        save_turn(ip, "vina", "assistant", response)

        ui_action = _route_ui("vina", message, response, history, context)
        return jsonify({"status": "success", "response": response, "ui_action": ui_action})

    except Exception as e:
        logger.error("Vina chat error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/vina_visualization/chat/stream', methods=['POST'])
def vina_chat_stream_ep():
    """Streaming SSE Vina chat."""
    _req = request.get_json(force=True) or {}
    _msg = (_req.get('message') or '').strip()
    _ctx = _req.get('context', {})
    _ctx.setdefault('chat_type', 'vina')
    return Response(
        stream_with_context(_mini_chat_stream_generate(_msg, _ctx)),
        mimetype="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"},
    )


@app.route('/vina_visualization/chat/clear', methods=['POST'])
def vina_chat_clear_ep():
    global _vina_chat_history
    _vina_chat_history = []
    try:
        ip = _get_ip()
        from uiapp.routes.session_routes import clear_session
        clear_session(ip, "vina")
    except Exception: pass
    return jsonify({"status": "success"})


# ── ChemBERT / Attn ───────────────────────────────────────────────────────────

@app.route('/attention_visualization/chat', methods=['POST'])
def attn_chat_ep():
    """Non-streaming ChemBERT/DeepAtom chat fallback."""
    try:
        data    = request.get_json(force=True) or {}
        message = (data.get('message') or '').strip()
        context = data.get('context', {})
        if not message:
            return jsonify({"status": "error", "message": "Empty message"}), 400

        chat_type = context.get('chat_type', 'attn')
        context.setdefault('chat_type', chat_type)

        is_conv    = message.lower().strip("!?.，。 ") in _CONVERSATIONAL
        ctx_lines  = _build_ctx_lines(chat_type, context, is_conv)
        ctx_block  = ("\n[Current session]\n" + "\n".join(ctx_lines)) if ctx_lines else ""
        sys_full, sys_guided = _pick_prompts(chat_type, context.get('guided_mode', False))
        sys_prompt = sys_guided if context.get('guided_mode') else sys_full

        ip      = _get_ip()
        mem     = _HISTORY_MAP.get(chat_type, _attn_chat_history)
        history = load_history(ip, chat_type, last_n=20) or mem

        parts = [f"<|im_start|>system\n{sys_prompt}<|im_end|>"]
        for role, content in history[-6:]:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{ctx_block}\n\n{message}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        out      = qwen_compat("\n".join(parts), COACH_PARAMS)
        response = out[0].outputs[0].text.strip()

        mem.append(("user", message));  mem.append(("assistant", response))
        if len(mem) > 20: del mem[:2]
        save_turn(ip, chat_type, "user",      message)
        save_turn(ip, chat_type, "assistant", response)

        ui_action = _route_ui(chat_type, message, response, history, context)
        return jsonify({"status": "success", "response": response, "ui_action": ui_action})

    except Exception as e:
        logger.error("Attn chat error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/attention_visualization/chat/stream', methods=['POST'])
def attn_chat_stream_ep():
    """Streaming SSE ChemBERT / DeepAtom chat."""
    _req = request.get_json(force=True) or {}
    _msg = (_req.get('message') or '').strip()
    _ctx = _req.get('context', {})
    _ctx.setdefault('chat_type', 'attn')
    return Response(
        stream_with_context(_mini_chat_stream_generate(_msg, _ctx)),
        mimetype="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"},
    )


@app.route('/attention_visualization/chat/clear', methods=['POST'])
def attn_chat_clear_ep():
    global _attn_chat_history, _deepatom_chat_history
    _attn_chat_history = []
    _deepatom_chat_history = []
    try:
        ip = _get_ip()
        from uiapp.routes.session_routes import clear_session
        clear_session(ip, "attn")
        clear_session(ip, "deepatom")
    except Exception: pass
    return jsonify({"status": "success"})