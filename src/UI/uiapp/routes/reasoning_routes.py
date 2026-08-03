"""
reasoning_routes.py — Flask blueprint for the Reasoning tool.
==============================================================

Register it in app.py (after the app is created):

    from uiapp.routes.reasoning_routes import reasoning_bp
    app.register_blueprint(reasoning_bp)

Endpoints (all under /reasoning):
    GET  /reasoning/models   → model catalog, stages, current assignment, config status
    GET  /reasoning/health   → cheap connectivity probe to the UF Claude gateway
    POST /reasoning/assign   → persist a {stage: model} mapping
    POST /reasoning/run      → demo: run extraction on a transcript with the assigned model
"""

import os
import json
import logging

from flask import Blueprint, jsonify, request

try:
    from uiapp.llm import claude_client as cc   # when imported as a package
except Exception:                                     # noqa: BLE001
    from uiapp.llm import claude_client as cc                         # when uiapp is on sys.path

logger = logging.getLogger(__name__)

reasoning_bp = Blueprint("reasoning", __name__, url_prefix="/reasoning")

# Small JSON state file — lives next to your other .qwen / .reasoning state
_STATE_DIR   = os.environ.get("REASONING_STATE_DIR", ".reasoning")
_ASSIGN_PATH = os.path.join(_STATE_DIR, "assignments.json")


def _load_assignment() -> dict:
    try:
        with open(_ASSIGN_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        saved = {}
    # start from defaults, override with anything valid that was saved
    out = dict(cc.DEFAULT_ASSIGNMENT)
    for stage, model in saved.items():
        if stage in cc.STAGES and model in cc.MODELS:
            out[stage] = model
    return out


def _save_assignment(assignment: dict) -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    with open(_ASSIGN_PATH, "w", encoding="utf-8") as f:
        json.dump(assignment, f, indent=2)


@reasoning_bp.route("/models")
def models():
    return jsonify({
        "models":         cc.MODELS,
        "stages":         cc.STAGES,
        "assignment":     _load_assignment(),
        "configured":     cc.is_configured(),
        "api_style":      cc.CLAUDE_API_STYLE,
        "embedding_note": cc.EMBEDDING_NOTE,
        "qwen_role":      "In-app chat + UI action routing (vLLM :8001)",
    })


@reasoning_bp.route("/health")
def health():
    return jsonify(cc.health())


@reasoning_bp.route("/assign", methods=["POST"])
def assign():
    body = request.get_json(force=True, silent=True) or {}
    assignment = _load_assignment()
    changed = []
    for stage in cc.STAGES:
        chosen = body.get(stage)
        if chosen in cc.MODELS:
            assignment[stage] = chosen
            changed.append(stage)
    _save_assignment(assignment)
    return jsonify({"ok": True, "assignment": assignment, "changed": changed})


@reasoning_bp.route("/run", methods=["POST"])
def run():
    """Demo path: extract typed records from a transcript using the assigned model."""
    body = request.get_json(force=True, silent=True) or {}
    transcript = (body.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"ok": False, "error": "no transcript provided"}), 400
    if not cc.is_configured():
        return jsonify({"ok": False, "error": "gateway not configured"}), 503

    assignment = _load_assignment()
    model = assignment["extraction"]
    try:
        records = cc.extract_session(transcript, model=model)
        return jsonify({"ok": True, "stage": "extraction", "model": model, "records": records})
    except Exception as e:
        logger.error("[reasoning] run failed: %s", e)
        return jsonify({"ok": False, "error": str(e)[:300]}), 500