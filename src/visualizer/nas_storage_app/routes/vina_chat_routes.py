# =============================================================================
# routes/vina_chat_routes.py
# /vina_visualization/chat, chat/stream, chat/clear
# Elion AI assistant for Vina docking — Qwen-based SSE chat.
# =============================================================================

import os
import json as _json
import logging as _logging
from flask import jsonify, request, Response, stream_with_context
from nas_storage_app import app
from nas_storage_app.qwen_client import qwen_stream, qwen_compat, QwenParams, COACH_PARAMS, COT_ROUTING_PARAMS

from nas_storage_app.routes.shared import (
    logger, VINA_ACTION_KB_PATH, VINA_BASE, VINA_LOG, _INPUT_ROUTES_YML, _ROOT,
)

# ==============================================================================
# ── QWEN2.5-14B CHAT ROUTES ───────────────────────────────────────────────────
# Model runs in serve_qwen.sh on port 8001 — no weights loaded here.
# ==============================================================================


# ==============================================================================
# ── CoT UI Action Routing ─────────────────────────────────────────────────────
# ==============================================================================

# Dedicated CoT log
import logging as _logging
_cot_logger = _logging.getLogger("elion.vina.cot")

# cot_main.log — receives ALL CoT calls from routes.py (vina + attn routing)
# autolearn.log is written by kb_auto_learner._write_cot_log (feature-gap subset only)
_COT_MAIN_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".qwen", "cot_main.log"
)
os.makedirs(os.path.dirname(_COT_MAIN_LOG), exist_ok=True)
_cot_handler = _logging.FileHandler(_COT_MAIN_LOG)
_cot_handler.setFormatter(_logging.Formatter(
    "\n" + "=" * 60 + "\n"
    "[%(asctime)s]  source=routes\n"
    "%(message)s\n"
    + "─" * 60
))
_cot_logger.addHandler(_cot_handler)
_cot_logger.setLevel(_logging.DEBUG)
_cot_logger.propagate = False  # don't bubble up to root/autolearn.log

_vina_kb_cache: str | None = None

# Live stdout queue for vina_dock_progress SSE stream

def _load_vina_kb() -> str:
    global _vina_kb_cache
    if _vina_kb_cache is not None:
        return _vina_kb_cache
    try:
        with open(VINA_ACTION_KB_PATH, "r", encoding="utf-8") as f:
            _vina_kb_cache = f.read()
        logger.info("[CoT] vina_action_kb.md loaded (%d chars)", len(_vina_kb_cache))
    except Exception as e:
        logger.warning("[CoT] Could not load vina KB: %s", e)
        _vina_kb_cache = ""
    return _vina_kb_cache


def _keyword_route_action(user_message: str) -> dict | None:
    """
    RAG-powered UI action router for the Vina Docking visualizer.
    Delegates to ElionUIRouter; falls back gracefully if not available.
    """
    try:
        from elion_ui_router import route_ui_action as _rag_route
        return _rag_route(user_message, tool_hint="vina")
    except Exception as _e:
        logger.warning("[UIRouter] RAG unavailable for vina, using legacy keywords: %s", _e)

    # ── Legacy keyword fallback (kept for safety) ─────────────────────────────
    u = user_message.lower()
    if any(w in u for w in ["run docking", "start docking", "dock this", "perform docking",
                              "now what", "what now", "next step", "then", "and then"]):
        return {"action": "vina_dock_guided", "confidence": "high",
                "reason": "legacy: dock/progression keyword"}
    if any(w in u for w in ["load ligand", "ligand path", "ligand file"]):
        return {"action": "load_ligand", "confidence": "medium",
                "reason": "legacy: ligand keyword"}
    if any(w in u for w in ["load receptor", "receptor path", "protein file"]):
        return {"action": "load_receptor", "confidence": "medium",
                "reason": "legacy: receptor keyword"}
    if any(w in u for w in ["open chembert", "chembert", "attention"]):
        return {"action": "open_visualizer", "confidence": "medium",
                "reason": "legacy: chembert keyword"}
    return None


def _cot_route_action(user_message: str, coach_response: str, history: list) -> dict | None:
    """
    DeepSeek-V4-style two-stage CoT routing for Vina UI actions.
    Stage 1: Qwen thinks through intent using the KB.
    Stage 2: Parses JSON conclusion → ui_action.
    """
    import json, re as _re

    kb = _load_vina_kb()
    if not kb:
        return None

    history_summary = ""
    for role, text in history[-3:]:
        history_summary += f"  {role.upper()}: {text[:200].replace(chr(10), ' ')}\n"

    cot_prompt = (
        "<|im_start|>system\n"
        "You are Elion's UI routing agent for a molecular docking visualizer. "
        "Decide which UI action (if any) to trigger based on the conversation. "
        "Think step-by-step, then output one raw JSON line.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"## UI Action Knowledge Base\n{kb}\n\n"
        f"## Recent history\n{history_summary}\n"
        f"## User message\n{user_message}\n\n"
        f"## Elion response\n{coach_response[:400]}\n\n"
        "Think step-by-step about user intent. "
        "After your reasoning write exactly --- on its own line then raw JSON:\n"
        '{"action": "none", "confidence": "high", "reason": "..."}\n'
        "<|im_end|>\n"
        "<|im_start|>think\n"
    )

    try:
        out  = qwen_compat(cot_prompt, COT_ROUTING_PARAMS)
        text = out[0].outputs[0].text.strip()
        _cot_logger.info(
            "USER: %s\n\nPROMPT (tail 600):\n...%s\n\nRAW OUTPUT:\n%s",
            user_message, cot_prompt[-600:], text
        )
        logger.info("[CoT] vina routing output (first 400): %s", text[:400])

        matches = _re.findall(r'\{[^{}]+\}', text)
        if not matches:
            return None
        decision   = json.loads(matches[-1])
        action     = decision.get("action", "none")
        confidence = decision.get("confidence", "low")
        reason     = decision.get("reason", "")
        logger.info("[CoT] → action=%s confidence=%s reason=%s", action, confidence, reason)

        valid = {"open_visualizer","open_voxel_inspector","toggle_grid_box",
                 "run_visualization","run_docking","switch_ligand_view",
                 "switch_protein_view","load_ligand","load_receptor",
                 "vina_dock_guided","guide_pdb_conversion","none"}
        if action in valid and confidence in ("high", "medium") and action != "none":
            return {"type": "ui_action", "action": action,
                    "confidence": confidence, "reason": reason}
        return None
    except Exception as e:
        logger.warning("[CoT] vina routing failed: %s", e)
        return None

VINA_SYSTEM_PROMPT = """You are Elion, an expert AI assistant for computational drug discovery.
You are embedded in a Vina molecular docking visualizer. You help researchers understand:
- AutoDock Vina docking results and binding energies (kcal/mol)
- Molecular interactions between ligands and protein receptors
- SMILES notation, molecular properties, and drug-likeness
- ChemBERT attention scores and what they reveal about binding sites
- How to interpret 3D molecular visualizations and interaction energies
- Next steps in the drug discovery pipeline

Be concise, scientific, and practical. When given docking scores or SMILES, analyze them directly.
Negative binding energies (e.g. -8.5 kcal/mol) indicate stronger binding. Below -7 is considered
good, below -9 is very strong."""

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
    "tell them to use the PDB → PDBQT converter tool first via the Ask Elion menu. "
    "Say something like: \"Since you have a .pdb file, let me guide you to convert it — "
    "click the glowing Ask Elion button above to open the tool menu.\"\n"
    "7. POST-CONVERSION: If [Just converted] appears in context, the file was already saved "
    "and the path auto-filled. Tell the user which field still needs filling, then guide them "
    "to click Vina Dock. Do NOT ask them to convert again.\n\n"
    "Workflow order: 1→Convert .pdb to .pdbqt (if needed) 2→Load receptor 3→Load ligand "
    "4→Click Vina Dock 5→Click Visualize\n\n"
    "Good responses (copy this style):\n"
    '"If those two paths look correct, click the Vina Dock button to start docking."\n'
    '"Enter your ligand .pdbqt path in the LIGAND field, then click Load."\n'
    '"Click the Visualize button to render the per-atom energy decomposition."\n'
    '"Since you have a .pdb file, let me guide you to convert it — '
    'click the glowing Ask Elion button above to open the tool menu."\n'
)

_vina_chat_history = []   # in-memory session history


@app.route('/vina_visualization/chat', methods=['POST'])
def vina_chat_ep():
    """
    Non-streaming chat endpoint (fallback).
    POST { "message": str, "context": { smiles, score, mode_energies, ... } }
    """
    try:
        data    = request.get_json(force=True) or {}
        message = (data.get('message') or '').strip()
        context = data.get('context', {})
        if not message:
            return jsonify({"status": "error", "message": "Empty message"}), 400

        # Build context block from current visualization state
        ctx_lines = []
        if context.get('smiles'):
            ctx_lines.append(f"Current ligand SMILES: {context['smiles']}")
        if context.get('score') is not None:
            ctx_lines.append(f"Best docking score: {context['score']} kcal/mol")
        if context.get('mode_energies'):
            energies = context['mode_energies'][:5]
            ctx_lines.append(f"Top binding modes (kcal/mol): {', '.join(str(e) for e in energies)}")
        if context.get('ligand_path'):
            ctx_lines.append(f"Ligand file: {context['ligand_path']}")
        if context.get('receptor_path'):
            ctx_lines.append(f"Receptor file: {context['receptor_path']}")
        if context.get('n_atoms'):
            ctx_lines.append(f"Ligand heavy atoms: {context['n_atoms']}")

        ctx_block = ("\n[Current docking session context]\n" + "\n".join(ctx_lines)) if ctx_lines else ""

        # Build ChatML prompt with history
        parts = [f"<|im_start|>system\n{VINA_SYSTEM_PROMPT}<|im_end|>"]
        for role, content in _vina_chat_history[-6:]:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{ctx_block}\n\n{message}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)

        out      = qwen_compat(prompt, COACH_PARAMS)
        response = out[0].outputs[0].text.strip()

        _vina_chat_history.append(("user",      message))
        _vina_chat_history.append(("assistant", response))
        # Keep last 20 turns
        if len(_vina_chat_history) > 20:
            del _vina_chat_history[:2]

        # CoT UI action routing
        ui_action = _keyword_route_action(message)
        if ui_action is None:
            ui_action = _cot_route_action(message, response, _vina_chat_history[-6:])
        # Auto-learn: if still no match, use Qwen CoT to generate and persist new KB entry
        if ui_action is None:
            try:
                from kb_auto_learner import maybe_learn_and_route
                ui_action = maybe_learn_and_route(message, tool_hint="vina")
                if ui_action:
                    logger.info("[AutoLearn] vina learned: %s", ui_action.get("action"))
            except Exception as _ale:
                logger.warning("[AutoLearn] vina learner error: %s", _ale)
        logger.info("[UI] vina ui_action=%s", ui_action)

        return jsonify({"status": "success", "response": response, "ui_action": ui_action})

    except Exception as e:
        logger.error(f"Vina chat error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/vina_visualization/chat/stream', methods=['POST'])
def vina_chat_stream_ep():
    """
    Streaming SSE chat endpoint.
    POST { "message": str, "context": { smiles, score, mode_energies, ... } }
    Streams: event: token  data: "chunk"
             event: done   data: {}
             event: error  data: "msg"
    """
    import json as _json

    # Read request data BEFORE entering generator (request context safety)
    _req = request.get_json(force=True) or {}
    _msg = (_req.get('message') or '').strip()
    _ctx = _req.get('context', {})

    def generate():
        try:
            if not _msg:
                yield "event: error\ndata: Empty message\n\n"
                return

            # ── Pre-detect emotion BEFORE prompt — relational rules fire THIS reply ──
            emotion = _detect_emotion(_msg)
            if emotion != "calm":
                _ctx["detected_emotion"] = emotion
            logger.info("[Relational] vina detected_emotion=%s msg=%r", emotion, _msg[:80])

            # Build context block
            ctx_lines = []
            if _ctx.get('smiles'):
                ctx_lines.append(f"Current ligand SMILES: {_ctx['smiles']}")
            if _ctx.get('score') is not None:
                ctx_lines.append(f"Best docking score: {_ctx['score']} kcal/mol")
            if _ctx.get('mode_energies'):
                energies = _ctx['mode_energies'][:5]
                ctx_lines.append(f"Top binding modes (kcal/mol): {', '.join(str(e) for e in energies)}")
            if _ctx.get('ligand_path'):
                ctx_lines.append(f"Ligand: {_ctx['ligand_path']}")
            else:
                ctx_lines.append("Ligand path: NOT SET")
            if _ctx.get('receptor_path'):
                ctx_lines.append(f"Receptor: {_ctx['receptor_path']}")
            else:
                ctx_lines.append("Receptor path: NOT SET")
            if _ctx.get('last_conversion'):
                lc = _ctx['last_conversion']
                ctx_lines.append(
                    f"[Just converted] {lc.get('filename','?')} → {lc.get('mol_type','?')} PDBQT "
                    f"saved at: {lc.get('output_path','?')} — path was auto-filled in the UI."
                )
            if _ctx.get('receptor_missing') and _ctx.get('last_conversion', {}).get('mol_type') == 'ligand':
                ctx_lines.append(
                    "[ACTION NEEDED] Ligand was just converted but Receptor path is still missing. "
                    "Ask the user if they also need to convert a receptor .pdb file, "
                    "and guide them to open the converter again via Ask Elion → PDB → PDBQT."
                )
            elif _ctx.get('ligand_missing') and _ctx.get('last_conversion', {}).get('mol_type') == 'receptor':
                ctx_lines.append(
                    "[ACTION NEEDED] Receptor was just converted but Ligand path is still missing. "
                    "Ask the user if they also need to convert a ligand .pdb file, "
                    "and guide them to open the converter again via Ask Elion → PDB → PDBQT."
                )
            if emotion != "calm":
                ctx_lines.append(
                    f"[User emotional state: {emotion} — "
                    + ("validate first, keep to ONE sentence." if emotion == "frustrated" else
                       "one idea only, plain language." if emotion == "overwhelmed" else
                       "clarify gently before next step." if emotion == "confused" else
                       "match energy briefly, then continue." if emotion == "excited" else "")
                    + "]"
                )

            ctx_block = ("\n[Current docking session]\n" + "\n".join(ctx_lines)) if ctx_lines else ""

            # Build ChatML prompt — use guided prompt in mini-chat mode
            sys_prompt = VINA_GUIDED_PROMPT if _ctx.get('guided_mode') else VINA_SYSTEM_PROMPT
            parts = [f"<|im_start|>system\n{sys_prompt}<|im_end|>"]
            for role, content in _vina_chat_history[-6:]:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append(f"<|im_start|>user\n{ctx_block}\n\n{_msg}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(parts)

            # Token budget scales with emotion and mode
            if _ctx.get('guided_mode'):
                _tok = 120
            elif emotion in ("frustrated", "overwhelmed"):
                _tok = 80
            else:
                _tok = 200
            stream_params = QwenParams(temperature=0.5, max_tokens=_tok, top_p=0.9)
            full_response = ""
            for chunk in qwen_stream(prompt, stream_params):
                full_response += chunk
                yield f"event: token\ndata: {_json.dumps(chunk)}\n\n"

            # Save to history
            _vina_chat_history.append(("user",      _msg))
            _vina_chat_history.append(("assistant", full_response.strip()))
            if len(_vina_chat_history) > 20:
                del _vina_chat_history[:2]

            # CoT UI action routing
            resp_lower = full_response.lower()
            if _ctx.get('guided_mode'):
                # First-time / PDB detection: user mentions .pdb (not .pdbqt) or is clearly new
                _u_lower = _msg.lower()
                _has_pdb_only = (
                    ('pdb' in _u_lower and 'pdbqt' not in _u_lower) or
                    any(w in _u_lower for w in ['i have pdb', 'i have a pdb', 'my pdb',
                                                 'only have pdb', 'got pdb', 'got a pdb',
                                                 'how do i start', 'where do i start',
                                                 'i\'m new', 'im new', 'first time',
                                                 'don\'t know', 'dont know', 'no idea'])
                )
                _resp_suggests_convert = any(w in resp_lower for w in [
                    'ask elion', 'convert it', 'pdb to pdbqt', 'pdb → pdbqt',
                    'convert your', 'converter', 'tool menu', 'glowing ask elion',
                    'convert a receptor', 'convert a ligand', 'convert the receptor',
                    'convert the ligand', 'open the converter', 'open converter'
                ])
                # Post-conversion: file was just saved, guide to fill remaining field + dock
                _is_post_conversion = bool(_ctx.get('last_conversion'))
                _resp_post_conv = any(w in resp_lower for w in [
                    'auto-filled', 'already saved', 'path was', 'now fill', 'fill in the',
                    'other field', 'fill the', 'fill your'
                ])
                if (_is_post_conversion and any(w in _u_lower for w in [
                    'now what', 'what now', 'next', 'what do i', 'proceed',
                    'ready', 'done', 'converted', 'what should'
                ])) or _resp_post_conv:
                    # Check if both paths are set → go straight to dock
                    _rec = _ctx.get('receptor_path', '').strip()
                    _lig = _ctx.get('ligand_path', '').strip()
                    if _rec and _lig:
                        ui_action = {"type": "ui_action", "action": "vina_dock_guided",
                                     "confidence": "high", "reason": "post-conversion: both paths set, dock now"}
                    else:
                        # Highlight whichever field is still empty
                        _missing = []
                        if not _rec: _missing += ['recPath', 'recLoadBtn']
                        if not _lig: _missing += ['ligPath', 'ligLoadBtn']
                        ui_action = {"type": "ui_action", "action": "load_receptor" if not _rec else "load_ligand",
                                     "confidence": "high", "reason": "post-conversion: fill remaining path"}
                elif _has_pdb_only or _resp_suggests_convert:
                    ui_action = {"type": "ui_action", "action": "guide_pdb_conversion",
                                 "confidence": "high", "reason": "guided: user has .pdb, needs conversion flow"}
                elif any(w in resp_lower for w in ['vina dock', 'click dock', 'click the vina', 'paths look correct',
                                                  'paths are correct', 'if those', 'if the paths', 'click vina dock']):
                    ui_action = {"type": "ui_action", "action": "vina_dock_guided", "confidence": "high", "reason": "guided: verify paths + dock"}
                elif any(w in resp_lower for w in ['click load', 'click the load', 'load button', 'ligand field', 'load the ligand']):
                    ui_action = {"type": "ui_action", "action": "load_ligand", "confidence": "high", "reason": "guided: load ligand"}
                elif any(w in resp_lower for w in ['click visualize', 'visualize button', 'click the visualize', 'click ⚡']):
                    ui_action = {"type": "ui_action", "action": "run_visualization", "confidence": "high", "reason": "guided: visualize"}
                elif any(w in resp_lower for w in ['receptor field', 'receptor path', 'load receptor']):
                    ui_action = {"type": "ui_action", "action": "load_receptor", "confidence": "high", "reason": "guided: load receptor"}
                else:
                    ui_action = _keyword_route_action(_msg)
            else:
                ui_action = _keyword_route_action(_msg)
                if ui_action is None:
                    ui_action = _cot_route_action(_msg, full_response, _vina_chat_history[-6:])
                # If CoT still returns none for "now what?" style questions,
                # and context has both paths → default to vina_dock_guided
                if ui_action is None:
                    _u = _msg.lower()
                    if any(w in _u for w in ["now what", "what now", "next step", "what next",
                                              "what do i", "what should", "proceed", "ready"]):
                        if _ctx.get("ligand_path") or _ctx.get("receptor_path"):
                            ui_action = {"type": "ui_action", "action": "vina_dock_guided",
                                         "confidence": "high", "reason": "paths set, next step is Vina Dock"}
            # Auto-learn: still no match → Qwen CoT generates + persists new KB entry
            if ui_action is None:
                try:
                    from kb_auto_learner import maybe_learn_and_route
                    ui_action = maybe_learn_and_route(_msg, tool_hint="vina")
                    if ui_action:
                        logger.info("[AutoLearn] vina stream learned: %s", ui_action.get("action"))
                except Exception as _ale:
                    logger.warning("[AutoLearn] vina stream error: %s", _ale)

            logger.info("[UI] vina stream ui_action=%s", ui_action)
            if ui_action:
                yield f"event: ui_action\ndata: {_json.dumps(ui_action)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            logger.error(f"Vina stream error: {e}")
            yield f"event: error\ndata: {_json.dumps(str(e))}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        }
    )


@app.route('/vina_visualization/chat/clear', methods=['POST'])
def vina_chat_clear_ep():
    """Clear in-memory chat history."""
    global _vina_chat_history
    _vina_chat_history = []
    return jsonify({"status": "success"})