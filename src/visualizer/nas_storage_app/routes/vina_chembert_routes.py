# =============================================================================
# routes/vina_chembert_routes.py
# /vina_visualization/* endpoints for ChemBERT, finetune, prepare_smiles,
# vina_defaults, vina_proteins, vina_select_protein.
# =============================================================================

import os, sys, threading, uuid, queue
from flask import jsonify, render_template, request, Response, stream_with_context, current_app
from nas_storage_app import app
from nas_storage_app.qwen_client import qwen_stream, qwen_compat, QwenParams

from nas_storage_app.routes.shared import (
    logger, DEFAULT_FINETUNED, DEFAULT_PRETRAINED, CHEMBERT_BASE,
    VINA_BASE, VINA_BIN, VINA_LOG, _INPUT_ROUTES_YML, _ROOT, _VIZ,
    _vina_finetune_jobs, _vina_finetune_lock,
)
from nas_storage_app.routes.chembert_model import _load_chembert, AdjacencyWeightVisualizer, _get_viz

# ══ VINA VISUALIZER  /vina_visualization/* ════════════════════════════════════
# ==============================================================================
@app.route('/vina_visualization/vina_defaults', methods=['GET'])
def vina_defaults():
    """
    GET /vina_visualization/vina_defaults
    Returns default receptor/ligand paths and box parameters from input_routes.yml
    (loaded into app.config['VINA'] by app.py at startup).
    Used by the frontend to pre-fill the receptor/ligand path fields.
    """
    cfg = current_app.config.get("VINA", {})
    return jsonify({
        "status":           "success",
        "default_receptor": cfg.get("default_receptor", ""),
        "default_ligand":   cfg.get("default_ligand",   ""),
        "center_x":         cfg.get("center_x", -25.7),
        "center_y":         cfg.get("center_y",   0.22),
        "center_z":         cfg.get("center_z",  28.39),
        "size_x":           cfg.get("size_x", 20),
        "size_y":           cfg.get("size_y", 20),
        "size_z":           cfg.get("size_z", 20),
        "exhaustiveness":   cfg.get("exhaustiveness", 8),
    })


# ==============================================================================
# ── Vina Protein Library  (reads input_routes.yml) ────────────────────────────
# ==============================================================================
import yaml as _yaml  # noqa: E402  (placed here to keep top-of-file clean)

# Absolute path to the YAML config — same file app.py reads at startup.
_INPUT_ROUTES_YML = str(_VIZ / "input_routes.yml")


def _load_proteins_from_yml() -> list:
    """
    Parse input_routes.yml and return the vina.proteins list.
    Falls back to a single synthetic entry built from app.config['VINA']
    if the file is missing or has no proteins key (backwards-compat).
    """
    try:
        with open(_INPUT_ROUTES_YML, "r", encoding="utf-8") as _f:
            raw = _yaml.safe_load(_f)
        proteins = raw.get("vina", {}).get("proteins")
        if proteins and isinstance(proteins, list):
            return proteins
    except Exception as _e:
        logger.warning("[vina_proteins] Could not read YAML: %s", _e)

    # Fallback: synthesise one entry from whatever is already in app.config
    cfg = current_app.config.get("VINA", {})
    return [{
        "id":               cfg.get("id", "default"),
        "label":            cfg.get("id", "Default protein"),
        "description":      "",
        "default_receptor": cfg.get("default_receptor", ""),
        "default_ligand":   cfg.get("default_ligand",   ""),
        "center_x":         cfg.get("center_x",  0),
        "center_y":         cfg.get("center_y",  0),
        "center_z":         cfg.get("center_z",  0),
        "size_x":           cfg.get("size_x",   20),
        "size_y":           cfg.get("size_y",   20),
        "size_z":           cfg.get("size_z",   20),
        "exhaustiveness":   cfg.get("exhaustiveness", 8),
        "num_modes":        cfg.get("num_modes",       9),
        "energy_range":     cfg.get("energy_range",    3),
    }]


@app.route('/vina_visualization/vina_proteins', methods=['GET'])
def vina_proteins():
    """
    GET /vina_visualization/vina_proteins
    Returns all protein targets defined in input_routes.yml so the
    mini-chat welcome screen can render one button per protein.

    Response:
    {
      "status":         "success",
      "active_protein": "8P0M",
      "proteins": [
        {
          "id": "8P0M", "label": "TEAD3 / 8P0M",
          "description": "...",
          "default_receptor": "...", "default_ligand": "...",
          "center_x": -25.7, "center_y": 0.22, "center_z": 28.39,
          "size_x": 20, "size_y": 20, "size_z": 20
        }, ...
      ]
    }
    """
    try:
        proteins = _load_proteins_from_yml()
        cfg      = current_app.config.get("VINA", {})
        active   = cfg.get("id", proteins[0]["id"] if proteins else "")
        return jsonify({"status": "success", "active_protein": active, "proteins": proteins})
    except Exception as exc:
        logger.error("[vina_proteins] error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/vina_visualization/vina_select_protein', methods=['POST'])
def vina_select_protein():
    """
    POST /vina_visualization/vina_select_protein
    Body: { "protein_id": "8P0M" }

    Finds the matching entry in input_routes.yml and merges its fields
    into app.config['VINA'] so all subsequent vina_dock calls use the
    chosen target without any restart.

    Response:
    {
      "status": "success", "protein_id": "8P0M",
      "default_receptor": "...", "default_ligand": "...",
      "center_x": ..., "center_y": ..., "center_z": ...,
      "size_x": ..., "size_y": ..., "size_z": ...,
      "exhaustiveness": 8
    }
    """
    try:
        body       = request.get_json(force=True) or {}
        protein_id = (body.get("protein_id") or "").strip()
        if not protein_id:
            return jsonify({"status": "error", "message": "protein_id is required"}), 400

        proteins = _load_proteins_from_yml()
        match    = next((p for p in proteins if p.get("id") == protein_id), None)
        if match is None:
            ids = [p.get("id") for p in proteins]
            return jsonify({
                "status":  "error",
                "message": f"protein_id '{protein_id}' not found. Available: {ids}"
            }), 404

        # Merge chosen protein fields into the live VINA config so vina_dock
        # picks them up immediately without a server restart.
        cfg = current_app.config.setdefault("VINA", {})
        cfg.update({
            "id":               match["id"],
            "default_receptor": match.get("default_receptor", cfg.get("default_receptor", "")),
            "default_ligand":   match.get("default_ligand",   cfg.get("default_ligand",   "")),
            "center_x":         match.get("center_x",         cfg.get("center_x",    0)),
            "center_y":         match.get("center_y",         cfg.get("center_y",    0)),
            "center_z":         match.get("center_z",         cfg.get("center_z",    0)),
            "size_x":           match.get("size_x",           cfg.get("size_x",     20)),
            "size_y":           match.get("size_y",           cfg.get("size_y",     20)),
            "size_z":           match.get("size_z",           cfg.get("size_z",     20)),
            "exhaustiveness":   match.get("exhaustiveness",   cfg.get("exhaustiveness", 8)),
            "num_modes":        match.get("num_modes",        cfg.get("num_modes",       9)),
            "energy_range":     match.get("energy_range",     cfg.get("energy_range",    3)),
            "cpu":              match.get("cpu",              cfg.get("cpu",          None)),
        })
        logger.info("[vina_select_protein] switched to protein_id=%s", protein_id)

        return jsonify({
            "status":           "success",
            "protein_id":       cfg["id"],
            "default_receptor": cfg["default_receptor"],
            "default_ligand":   cfg["default_ligand"],
            "center_x":         cfg["center_x"],
            "center_y":         cfg["center_y"],
            "center_z":         cfg["center_z"],
            "size_x":           cfg["size_x"],
            "size_y":           cfg["size_y"],
            "size_z":           cfg["size_z"],
            "exhaustiveness":   cfg["exhaustiveness"],
        })

    except Exception as exc:
        logger.error("[vina_select_protein] error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500



@app.route('/vina_visualization')
def vina_home():
    return render_template('hub.html')


@app.route('/vina_visualization/chembert_models', methods=['GET'])
def vina_chembert_models():
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


@app.route('/vina_visualization/adj_3d_viz', methods=['POST'])
def vina_adj_3d_viz():
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


@app.route('/vina_visualization/chembert_compare', methods=['POST'])
def vina_chembert_compare():
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
    Runs nas_storage_app/finetune_CHEMBERT.py as a subprocess.
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


@app.route('/vina_visualization/finetune_chembert', methods=['POST'])
def vina_finetune_chembert():
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


@app.route('/vina_visualization/finetune_status/<job_id>', methods=['GET'])
def vina_finetune_status(job_id: str):
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

@app.route('/vina_visualization/prepare_smiles', methods=['POST'])
def vina_prepare_smiles():
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
# Vina Score Decomposition Visualizer
# ==============================================================================

import re
import math
from collections import defaultdict

_XS_META = {
    0:("C_H",True,False,False), 1:("C_P",True,False,False),
    2:("N_P",False,False,False), 3:("N_D",False,True,False),
    4:("N_A",False,False,True), 5:("N_DA",False,True,True),
    6:("O_P",False,False,False), 7:("O_D",False,True,False),
    8:("O_A",False,False,True), 9:("O_DA",False,True,True),
    10:("S_P",False,False,False), 11:("P_P",False,False,False),
    12:("F_H",False,False,False), 13:("Cl_H",False,False,False),
    14:("Br_H",False,False,False), 15:("I_H",False,False,False),
    16:("Met",False,False,False),
}
_XS_RADIUS = {
    0:1.9,1:1.9,2:1.8,3:1.8,4:1.8,5:1.8,
    6:1.7,7:1.7,8:1.7,9:1.7,
    10:2.0,11:2.1,12:1.5,13:1.8,14:2.0,15:2.2,16:1.2,
}
_VINA_W = {
    "gauss1":-0.035579,"gauss2":-0.005156,"repulsion":0.840245,
    "hydrophobic":-0.035069,"hbond":-0.587439,
}