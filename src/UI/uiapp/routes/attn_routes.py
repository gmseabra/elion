# =============================================================================
# routes/attn_routes.py
# All /attention_visualization/* endpoints:
#   chembert_models, adj_3d_viz, chembert_compare,
#   finetune_chembert, finetune_status, prepare_smiles,
#   chat, chat/stream, chat/clear
# =============================================================================

import os, threading, uuid, queue, json as _json
from pathlib import Path
from rdkit import Chem
from flask import jsonify, render_template, request, Response, stream_with_context
from uiapp import app
from uiapp.llm.qwen_client import qwen_stream, qwen_compat, QwenParams, COACH_PARAMS, COT_ROUTING_PARAMS

from uiapp.routes.shared import (
    logger, DEFAULT_FINETUNED, DEFAULT_PRETRAINED, CHEMBERT_BASE,
    ATTN_ACTION_KB_PATH, _ROOT, _VIZ,
    _attn_finetune_jobs, _attn_finetune_lock,
)
from uiapp.routes.chembert_model import (
    _load_chembert, AdjacencyWeightVisualizer, _get_viz,
    _load_attn_kb, _detect_emotion, _keyword_route_attn,
    _attn_chat_history,
)

import logging as _logging
_cot_logger = _logging.getLogger("elion.attn.cot")

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

# ══ ATTENTION VISUALIZER  /attention_visualization/* ═════════════════════════
# ==============================================================================
@app.route('/attention_visualization')
def attn_home():
    return render_template('hub.html')


@app.route('/attention_visualization/chembert_models', methods=['GET'])
def attn_chembert_models():
    """
    GET /chembert_models
    Returns the two preset paths so the front-end can populate a selector.
    """
    return jsonify({
        "status": "success",
        "presets": [
            {"label": "Finetuned (Finetuned_model_5.pt)", "path": DEFAULT_FINETUNED,  "tag": "finetuned"},
            {"label": "Pretrained (pretrained_model.pt)",  "path": DEFAULT_PRETRAINED, "tag": "pretrained"},
        ]
    })


@app.route('/attention_visualization/adj_3d_viz', methods=['POST'])
def attn_adj_3d_viz():
    """
    POST /adj_3d_viz
    Body: { "smiles": "<SMILES>", "model_path": "<optional path>" }

    Returns:
    {
      "status":          "success",
      "smiles":          str,
      "predicted_score": float | null,
      "model_tag":       "finetuned" | "pretrained",
      "model_path":      str,
      "atoms":           [...],
      "bonds":           [...],
      "weight_vector":   {...}
    }
    """
    try:
        data       = request.get_json(force=True) or {}
        smiles     = data.get("smiles", "").strip()
        model_path = data.get("model_path", DEFAULT_FINETUNED).strip() or DEFAULT_FINETUNED

        if not smiles:
            return jsonify({"status": "error", "message": "No SMILES provided."}), 400
        if Chem.MolFromSmiles(smiles) is None:
            return jsonify({"status": "error",
                            "message": f"Invalid SMILES: {smiles}"}), 400

        viz               = _get_viz(model_path)
        payload           = viz.build_3d_payload(smiles)
        payload["status"] = "success"
        return jsonify(payload)

    except FileNotFoundError as exc:
        logger.error(f"adj_3d_viz model not found: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:
        logger.error(f"adj_3d_viz error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/attention_visualization/chembert_compare', methods=['POST'])
def attn_chembert_compare():
    """
    POST /chembert_compare
    Body: { "smiles_a": str, "smiles_b": str, "model_path": str (optional) }

    Returns both payloads in one round-trip for side-by-side rendering.
    """
    try:
        data       = request.get_json(force=True) or {}
        smiles_a   = data.get("smiles_a", "").strip()
        smiles_b   = data.get("smiles_b", "").strip()
        model_path = data.get("model_path", DEFAULT_FINETUNED).strip() or DEFAULT_FINETUNED

        errors = {}
        if not smiles_a:
            errors["smiles_a"] = "No SMILES provided."
        elif Chem.MolFromSmiles(smiles_a) is None:
            errors["smiles_a"] = f"Invalid SMILES: {smiles_a}"
        if not smiles_b:
            errors["smiles_b"] = "No SMILES provided."
        elif Chem.MolFromSmiles(smiles_b) is None:
            errors["smiles_b"] = f"Invalid SMILES: {smiles_b}"
        if errors:
            return jsonify({"status": "error", "errors": errors}), 400

        viz       = _get_viz(model_path)
        payload_a = viz.build_3d_payload(smiles_a)
        payload_b = viz.build_3d_payload(smiles_b)

        return jsonify({"status": "success", "a": payload_a, "b": payload_b})

    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:
        logger.error(f"chembert_compare error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500

# ==============================================================================
# Fine-tuning routes
# ==============================================================================

# Job registry: job_id -> {"status": "running"|"done"|"error", "queue": Queue}
_finetune_jobs: dict[str, dict] = {}
_finetune_lock = threading.Lock()


def _run_finetune_job(job_id: str, smiles_file: str, pretrained_model: str,
                      max_time: int, max_epochs: int, task: str):
    """
    Runs uiapp/finetune_CHEMBERT.py as a subprocess.
    Interface: finetune_CHEMBERT.py [-m MODEL] [-t MAX_TIME] smiles_file
    """
    import subprocess, sys as _sys
    q = _finetune_jobs[job_id]["queue"]
    def _push(line: str): q.put(line)

    _script   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "finetune_CHEMBERT.py")
    _viz_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        if not os.path.isfile(_script):
            raise FileNotFoundError(f"finetune_CHEMBERT.py not found at {_script}")

        cmd = [_sys.executable, _script]
        if pretrained_model:
            cmd += ["-m", pretrained_model]
        if max_time:
            cmd += ["-t", str(max_time)]
        cmd += [smiles_file]   # positional — must be last

        _env = os.environ.copy()
        _env["PYTHONPATH"] = _viz_root + os.pathsep + _env.get("PYTHONPATH", "")

        logger.info("[Finetune] cmd: %s", " ".join(cmd))
        _push(f"$ {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=_viz_root, env=_env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line: _push(line)
        proc.wait()

        if proc.returncode == 0:
            with _finetune_lock: _finetune_jobs[job_id]["status"] = "done"
        else:
            with _finetune_lock: _finetune_jobs[job_id]["status"] = "error"
            _push(f"ERROR: process exited with code {proc.returncode}")
        _push("__DONE__")

    except Exception as exc:
        logger.error("[Finetune] job %s failed: %s", job_id, exc)
        with _finetune_lock: _finetune_jobs[job_id]["status"] = "error"
        _push(f"ERROR: {exc}")
        _push("__DONE__")


@app.route('/attention_visualization/finetune_chembert', methods=['POST'])
def attn_finetune_chembert():
    """
    POST /finetune_chembert
    Body: {
        "smiles_file":       str,   # path to CSV with SMILES,LABELS
        "pretrained_model":  str,   # optional, defaults to DEFAULT_PRETRAINED
        "max_time":          int,   # minutes, default 720
        "max_epochs":        int,   # default 15
        "task":              str    # "regression" | "classification", default "regression"
    }
    Returns: { "status": "started", "job_id": str }
    """
    try:
        data            = request.get_json(force=True) or {}
        smiles_file     = data.get("smiles_file", "").strip()
        pretrained_model = (data.get("pretrained_model") or "").strip()  # use exactly what the UI sends; no hidden default
        max_time        = int(data.get("max_time", 720))
        max_epochs      = int(data.get("max_epochs", 15))
        task            = data.get("task", "regression").strip()

        if not smiles_file:
            return jsonify({"status": "error", "message": "No smiles_file provided."}), 400
        if not Path(smiles_file).is_file():
            return jsonify({"status": "error", "message": f"File not found: {smiles_file}"}), 404
        if task not in ("regression", "classification"):
            return jsonify({"status": "error", "message": "task must be 'regression' or 'classification'"}), 400

        job_id = str(uuid.uuid4())
        with _finetune_lock:
            _finetune_jobs[job_id] = {
                "status": "running",
                "queue":  queue.Queue(),
                "params": {
                    "smiles_file":      smiles_file,
                    "pretrained_model": pretrained_model,
                    "max_time":         max_time,
                    "max_epochs":       max_epochs,
                    "task":             task,
                },
            }

        t = threading.Thread(
            target=_run_finetune_job,
            args=(job_id, smiles_file, pretrained_model, max_time, max_epochs, task),
            daemon=True,
        )
        t.start()

        return jsonify({"status": "started", "job_id": job_id})

    except Exception as exc:
        logger.error(f"finetune_chembert error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/attention_visualization/finetune_status/<job_id>', methods=['GET'])
def attn_finetune_status(job_id: str):
    """
    GET /finetune_status/<job_id>
    Server-Sent Events stream: each event is a log line.
    Sends "data: __DONE__\\n\\n" when the job finishes.
    """
    if job_id not in _finetune_jobs:
        return jsonify({"status": "error", "message": "Unknown job_id"}), 404

    job = _finetune_jobs[job_id]
    q   = job["queue"]

    def generate():
        while True:
            try:
                line = q.get(timeout=30)
                yield f"data: {line}\n\n"
                if line == "__DONE__":
                    break
            except queue.Empty:
                # Send a keep-alive comment
                yield ": keep-alive\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ==============================================================================
# Prepare SMILES training data  (AGI_get_smile_fine_tune_ChemBERT logic)
# ==============================================================================

@app.route('/attention_visualization/prepare_smiles', methods=['POST'])
def attn_prepare_smiles():
    """
    POST /prepare_smiles
    Body: {
        "subset_path":    str,   # CSV with Ligand_ID + Affinity columns
        "reference_path": str,   # CSV with Name + SMILES columns
        "output_path":    str    # destination .smi / .csv path
    }

    Merges subset_path (docking scores) with reference_path (SMILES) on
    Ligand_ID == Name, writes SMILES,LABELS to output_path, and returns a
    preview of the first 10 rows plus row count.
    """
    try:
        import pandas as pd

        data           = request.get_json(force=True) or {}
        subset_path    = data.get("subset_path",    "").strip()
        reference_path = data.get("reference_path", "").strip()
        output_path    = data.get("output_path",    "").strip()

        # ── Validate inputs ────────────────────────────────────────────────────
        missing = [k for k, v in [
            ("subset_path",    subset_path),
            ("reference_path", reference_path),
            ("output_path",    output_path),
        ] if not v]
        if missing:
            return jsonify({"status": "error",
                            "message": f"Missing fields: {', '.join(missing)}"}), 400

        for label, p in [("subset_path",    subset_path),
                         ("reference_path", reference_path)]:
            if not Path(p).is_file():
                return jsonify({"status": "error",
                                "message": f"File not found ({label}): {p}"}), 404

        # ── Load & merge ───────────────────────────────────────────────────────
        subset_df    = pd.read_csv(subset_path)
        reference_df = pd.read_csv(reference_path)

        required_subset = {"Ligand_ID", "Affinity"}
        required_ref    = {"Name", "SMILES"}
        missing_s = required_subset - set(subset_df.columns)
        missing_r = required_ref    - set(reference_df.columns)
        if missing_s:
            return jsonify({"status": "error",
                            "message": f"subset_path missing columns: {missing_s}"}), 400
        if missing_r:
            return jsonify({"status": "error",
                            "message": f"reference_path missing columns: {missing_r}"}), 400

        merged_df = subset_df.merge(
            reference_df[["Name", "SMILES"]],
            left_on="Ligand_ID",
            right_on="Name",
            how="inner",
        )

        final_df = (merged_df[["SMILES", "Affinity"]]
                    .rename(columns={"Affinity": "LABELS"}))

        # ── Write output ───────────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)

        logger.info(f"prepare_smiles: wrote {len(final_df)} rows to {output_path}")

        preview = final_df.head(10).to_dict(orient="records")

        return jsonify({
            "status":       "success",
            "total_rows":   len(final_df),
            "output_path":  output_path,
            "preview":      preview,
        })

    except Exception as exc:
        logger.error(f"prepare_smiles error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ==============================================================================
# ── QWEN2.5-14B CHAT ROUTES ───────────────────────────────────────────────────
# Chat endpoints for /attention_visualization/chat(*) live in mini_chat_routes.py.
# That module is loaded independently by routes/__init__.py — no import needed here.
# ==============================================================================

_attn_chat_history: list = []  # kept for backward compat; mini_chat_routes owns it


# ==============================================================================
# ══ VINA VISUALIZER  /vina_visualization/* ════════════════════════════════════