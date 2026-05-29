# =============================================================================
# routes/hub_routes.py
# Hub landing page and debug_static endpoint.
# =============================================================================

from flask import jsonify, render_template, current_app
from nas_storage_app import app
from nas_storage_app.routes.shared import logger, _VIZ
import os as _os2

@app.route('/debug_static')
def debug_static():
    """
    Temporary debug endpoint — remove after confirming static file paths.
    GET /debug_static  → JSON showing Flask's static_folder and whether the JS files exist.
    """
    import os as _os2
    sf = current_app.static_folder or ''
    return jsonify({
        "static_folder":           sf,
        "static_url_path":         current_app.static_url_path,
        "static_folder_exists":    _os2.path.isdir(sf),
        "js_dir_exists":           _os2.path.isdir(_os2.path.join(sf, 'js')),
        "deepatom_js_exists":      _os2.path.isfile(_os2.path.join(sf, 'js', 'deepatom.js')),
        "vina_welcome_js_exists":  _os2.path.isfile(_os2.path.join(sf, 'js', 'vina_mini_chat_welcome.js')),
        "js_dir_contents":         _os2.listdir(_os2.path.join(sf, 'js')) if _os2.path.isdir(_os2.path.join(sf, 'js')) else [],
        "cwd":                     _os2.getcwd(),
        "routes_py_location":      __file__,
        "_VIZ":                    str(_VIZ),
    })

@app.route('/')
def hub():
    """Hub landing page linking to both visualizer tools."""
    return render_template('hub.html')

def _cot_route_attn(user_message: str, coach_response: str, history: list) -> dict | None:
    """
    CoT routing for the attn stream — mirrors _cot_route_action for vina.
    Logs to cot_main.log via _cot_logger.
    """
    import json, re as _re
    try:
        kb_path = ATTN_ACTION_KB_PATH
        with open(kb_path, "r", encoding="utf-8") as _f:
            kb = _f.read()
    except Exception:
        return None

    history_summary = ""
    for role, text in history[-3:]:
        history_summary += f"  {role.upper()}: {text[:200].replace(chr(10), ' ')}\n"

    cot_prompt = (
        "<|im_start|>system\n"
        "You are Elion's UI routing agent for the CHEM-BERT attention visualizer. "
        "Decide which UI action (if any) to trigger based on the conversation. "
        "Also detect the user's emotional state from their message tone.\n"
        "Think step-by-step, then output one raw JSON line.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"## UI Action Knowledge Base\n{kb}\n\n"
        f"## Recent history\n{history_summary}\n"
        f"## User message\n{user_message}\n\n"
        f"## Elion response\n{coach_response[:400]}\n\n"
        "Think step-by-step about user intent AND emotional state. "
        "Emotional state options: calm | frustrated | confused | excited | overwhelmed\n"
        "After your reasoning write exactly --- on its own line then raw JSON:\n"
        '{"action": "none", "confidence": "high", "reason": "...", "emotional_state": "calm"}\n'
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
        logger.info("[CoT] attn routing output (first 400): %s", text[:400])
        matches = _re.findall(r'\{[^{}]+\}', text)
        if not matches:
            return None
        decision        = json.loads(matches[-1])
        action          = decision.get("action", "none")
        confidence      = decision.get("confidence", "low")
        reason          = decision.get("reason", "")
        emotional_state = decision.get("emotional_state", "calm")
        logger.info("[CoT] attn → action=%s confidence=%s emotional_state=%s",
                    action, confidence, emotional_state)
        valid = {"open_visualizer","show_3d","compare_molecules","fine_tune_model",
                 "load_model","open_vina","explain_ask_elion","none"}
        if action in valid and confidence in ("high","medium") and action != "none":
            return {"type": "ui_action", "action": action,
                    "confidence": confidence, "reason": reason,
                    "emotional_state": emotional_state}
    except Exception as _e:
        logger.warning("[CoT] attn routing error: %s", _e)
    return None


# ==============================================================================
# ══ ATTENTION VISUALIZER  /attention_visualization/* ═════════════════════════