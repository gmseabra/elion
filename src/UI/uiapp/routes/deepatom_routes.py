# =============================================================================
# routes/deepatom_routes.py
# DeepAtom Virtual Screening endpoints:
#   deepatom_datasets        — GET  /vina_visualization/deepatom_datasets
#   deepatom_estimate_stream — GET  /vina_visualization/deepatom_estimate_stream
#   deepatom_estimate        — POST /vina_visualization/deepatom_estimate
# =============================================================================

from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

# yaml is imported lazily inside _load_deepatom_cfg() only — not at module top.

from flask import Response, jsonify, request, stream_with_context
from uiapp import app
from uiapp import config as _dacfg
from uiapp.routes.shared import logger, _INPUT_ROUTES_YML

# ── DeepAtom defaults ────────────────────────────────────────────────────────
# DeepAtom is an external project. These are the fallbacks used when the caller
# omits `data_dir` / `test_type`, or when input_routes.yml has no `deepatom:`
# block at all. Config-driven (DEEPATOM_SCRIPT / DEEPATOM_DATA_DIR /
# DEEPATOM_TEST_TYPE) so a host that has DeepAtom installed can point at it
# without editing this file — and so that a host that does NOT have it gets an
# empty string rather than a NameError on every request.
DEEPATOM_SCRIPT            = _dacfg.DEEPATOM_SCRIPT
DEEPATOM_DEFAULT_DATA_DIR  = _dacfg.DEEPATOM_DATA_DIR
DEEPATOM_DEFAULT_TEST_TYPE = _dacfg.DEEPATOM_TEST_TYPE

# Per-run log queue — populated by deepatom_estimate, drained by deepatom_estimate_stream
_da_log_q: queue.Queue = queue.Queue()
_da_run_lock = threading.Lock()


# ==============================================================================
# YAML helpers
# ==============================================================================

def _load_deepatom_cfg() -> dict:
    """Return the full `deepatom:` section from input_routes.yml, or {}."""
    if not _INPUT_ROUTES_YML:
        return {}
    try:
        import yaml as _yaml
        with open(_INPUT_ROUTES_YML, "r", encoding="utf-8") as fh:
            raw = _yaml.safe_load(fh)
        return raw.get("deepatom", {}) if raw else {}
    except Exception as exc:
        logger.warning("[deepatom_routes] Could not read YAML: %s", exc)
        return {}



def _read_deepatom_csv(data_dir: str) -> list:
    """Read vs_ZccE.csv directly; pK = -deltaG/1.36"""
    import csv as _csv
    from pathlib import Path as _P
    _STAT = {'count','mean','std','min','25%','50%','75%','max','number','avg','sum'}
    for csv_path in sorted(_P(data_dir).rglob("*.csv"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            compounds = []
            with open(csv_path, newline='') as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    pdb = (row.get('PDB') or row.get('pdb') or '').strip()
                    dg  = (row.get('deltaG_kcal_mol') or row.get('dG') or '').strip()
                    if not pdb or pdb.lower() in _STAT:
                        continue
                    try:
                        compounds.append({'id': pdb,
                                          'pred_pk': round(-float(dg)/1.36, 4)})
                    except (ValueError, ZeroDivisionError):
                        pass
            if compounds:
                logger.info("[deepatom] %d compounds from %s", len(compounds), csv_path)
                return compounds
        except Exception as _e:
            logger.warning("[deepatom] csv read error %s: %s", csv_path, _e)
    return []

def _load_deepatom_datasets_from_yml() -> list:
    """
    Return the `deepatom.datasets` list from input_routes.yml.
    Falls back to a single synthetic entry if the file is missing or
    has no deepatom key (backwards-compatible).
    """
    datasets = _load_deepatom_cfg().get("datasets")
    if datasets and isinstance(datasets, list):
        return datasets
    return [{
        "id":          "ZccE_VS",
        "label":       "ZccE · Virtual Screening",
        "description": "ZccE compound library — virtual screening mode",
        "test_type":   DEEPATOM_DEFAULT_TEST_TYPE,
        "data_dir":    DEEPATOM_DEFAULT_DATA_DIR,
        "active":      True,
    }]


# ==============================================================================
# Output parser
# ==============================================================================

def _parse_deepatom_output(stdout_text: str) -> dict:
    """
    Parse DeepAtom virtual-screening stdout into structured results.

    The script emits progress lines like:
        batch_name:  00000
        batch_dir:   /home/.../DEEP_MODEL_temp.xxx/vs/ZccE/00000
        LIG: BM-1-57.pdb
        dataset_name:  ZccE
        ================================================================
        BM-1-57:   complex 1 (out of 15)

    Then final scores in one of these formats:
        A) "BM-1-57  7.23"          (ID  score, whitespace-sep)
        B) "BM-1-57: 7.23"          (ID: score)
        C) "BM-1-57, 7.23, 6.80"    (CSV with optional exp pK)
        D) A summary table with a header row

    Progress lines like "BM-1-57:   complex 1 (out of 15)" are excluded
    because they contain "complex" as a non-numeric token.
    """
    lines     = stdout_text.splitlines()
    compounds = []
    raw_lines = []

    # ── Find the separator line (===...) to skip preamble ────────────────────
    sep_idx = 0
    for i, line in enumerate(lines):
        if re.match(r'^={10,}', line.strip()):
            sep_idx = i + 1
            break

    result_lines = lines[sep_idx:]

    # ── Pattern A/B: "<ID>:?  <float>"  (excluding progress "complex N" lines)
    SCORE_RE    = re.compile(
        r'^([A-Za-z0-9_\-\.]+)\s*:?\s+([-]?\d+\.\d+)'
        r'(?:\s+([-]?\d+\.\d+))?'
        r'\s*$'
    )
    PROGRESS_RE = re.compile(r'complex\s+\d+', re.IGNORECASE)

    seen_ids: set[str] = set()
    for line in result_lines:
        stripped = line.strip()
        if not stripped or PROGRESS_RE.search(stripped):
            raw_lines.append(line)
            continue

        m = SCORE_RE.match(stripped)
        if m:
            cid  = m.group(1)
            pred = float(m.group(2))
            exp  = float(m.group(3)) if m.group(3) else None
            # Skip pandas describe() stat rows — not real compound IDs
            _STAT = {'count','mean','std','min','25%','50%','75%','max','number','avg','sum'}
            if cid.lower() in _STAT:
                raw_lines.append(line)
                continue
            if cid not in seen_ids:
                seen_ids.add(cid)
                entry: dict = {'id': cid, 'pred_pk': round(pred, 4)}
                if exp is not None:
                    entry['exp_pk'] = round(exp, 4)
                compounds.append(entry)
        else:
            raw_lines.append(line)

    # ── Fallback: whitespace/CSV table with a header row ─────────────────────
    if not compounds:
        HEADER_HINTS = {
            'id', 'name', 'complex', 'complex_id', 'pred', 'predicted',
            'pred_pk', 'exp', 'experimental', 'exp_pk', 'score', 'affinity',
        }
        header_idx = None
        for i, line in enumerate(result_lines):
            parts = re.split(r'[,\t\s]+', line.strip().lower())
            if len(parts) >= 2 and HEADER_HINTS & set(parts):
                header_idx = i
                break

        if header_idx is not None:
            header = re.split(r'[,\t\s]+', result_lines[header_idx].strip().lower())

            def col_of(*cands):
                for c in cands:
                    if c in header:
                        return header.index(c)
                return None

            id_c = col_of('complex_id', 'id', 'name', 'complex')
            pr_c = col_of('pred_pk', 'predicted', 'pred', 'score', 'affinity')
            ex_c = col_of('exp_pk', 'experimental', 'exp', 'target', 'label')

            if pr_c is not None:
                for line in result_lines[header_idx + 1:]:
                    parts = re.split(r'[,\t\s]+', line.strip())
                    if len(parts) <= pr_c:
                        continue
                    try:
                        cid  = parts[id_c] if id_c is not None and id_c < len(parts) \
                               else str(len(compounds) + 1)
                        pred = float(parts[pr_c])
                        entry = {'id': cid, 'pred_pk': round(pred, 4)}
                        if ex_c is not None and ex_c < len(parts):
                            try:
                                entry['exp_pk'] = round(float(parts[ex_c]), 4)
                            except ValueError:
                                pass
                        compounds.append(entry)
                    except (ValueError, IndexError):
                        raw_lines.append(line)

    out: dict = {}
    if compounds:
        out['compounds'] = compounds
    out['_raw'] = stdout_text.strip()
    return out


# ==============================================================================
# Flask routes
# ==============================================================================

@app.route('/vina_visualization/deepatom_datasets', methods=['GET'])
def deepatom_datasets():
    """
    GET /vina_visualization/deepatom_datasets

    Returns all DeepAtom dataset entries from input_routes.yml so the
    mini-chat welcome screen can render one button per dataset.

    Response:
    {
      "status":         "success",
      "active_dataset": "ZccE_VS",
      "datasets": [
        { "id": "ZccE_VS", "label": "ZccE · Virtual Screening",
          "description": "...", "test_type": "vs",
          "data_dir": "/path/to/ZccE" },
        ...
      ]
    }
    """
    try:
        cfg       = _load_deepatom_cfg()
        datasets  = cfg.get("datasets") or _load_deepatom_datasets_from_yml()
        active_id = cfg.get("active_dataset", "")
        return jsonify({
            "status":         "success",
            "active_dataset": active_id,
            "datasets":       datasets,
        })
    except Exception as exc:
        logger.exception("[deepatom_datasets] error")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/vina_visualization/deepatom_estimate_stream', methods=['GET'])
def deepatom_estimate_stream():
    """
    GET /vina_visualization/deepatom_estimate_stream
    Server-Sent Events — streams raw stdout lines from the running
    deepatom_estimate job in real time.
    Sends "data: __DONE__\n\n" when the job finishes.
    """
    def generate():
        while True:
            try:
                line = _da_log_q.get(timeout=30)
                yield f"data: {line}\n\n"
                if line == "__DONE__" or line.startswith("__ERROR__"):
                    break
            except queue.Empty:
                yield ": keep-alive\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/vina_visualization/deepatom_estimate', methods=['POST'])
def deepatom_estimate():
    """
    POST /vina_visualization/deepatom_estimate
    Body: {
        "data_dir":  "<absolute path to dataset directory>",
        "test_type": "vs"   // maps to script's -t flag (default: "vs")
    }

    Runs:
        predict_binding_affinity_v4_2_data_split_ZccE_elion.sh
            -t <test_type>
            -d <data_dir>

    Returns JSON:
        { "status": "success"|"partial"|"error",
          "compounds": [{"id": str, "pred_pk": float, "exp_pk": float}, ...],
          "_raw": str,
          "stderr_snippet": str }
    """
    try:
        body      = request.get_json(force=True) or {}
        data_dir  = (body.get('data_dir')  or DEEPATOM_DEFAULT_DATA_DIR).strip()
        test_type = (body.get('test_type') or DEEPATOM_DEFAULT_TEST_TYPE).strip()

        if not data_dir:
            return jsonify({'status': 'error', 'message': 'data_dir is required'}), 400

        # Resolve script path — prefer YAML, fall back to constant
        script = _load_deepatom_cfg().get("script") or DEEPATOM_SCRIPT

        if not Path(script).is_file():
            return jsonify({
                'status':  'error',
                'message': f'DeepAtom script not found: {script}',
            }), 500

        if not Path(data_dir).is_dir():
            return jsonify({
                'status':  'error',
                'message': f'Data directory not found: {data_dir}',
            }), 400

        shell_cmd = (
            'source "$(conda info --base)/etc/profile.d/conda.sh" && '
            'conda activate elion_backend && '
            f'"{script}" -t {test_type} -d "{data_dir}" && '
            'conda deactivate'
        )
        logger.info('[deepatom_estimate] cmd: %s', shell_cmd)

        # Clear the log queue from any previous run
        while not _da_log_q.empty():
            try: _da_log_q.get_nowait()
            except Exception: pass

        import datetime as _dt
        _da_log_q.put(f"[deepatom] {'=' * 60}")
        _da_log_q.put(f"[deepatom] Run started  {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        _da_log_q.put(f"[deepatom] test_type    -t {test_type}")
        _da_log_q.put(f"[deepatom] data_dir     -d {data_dir}")
        _da_log_q.put(f"[deepatom] script       {script}")
        _da_log_q.put(f"[deepatom] {'=' * 60}")
        _da_log_q.put("")

        proc = subprocess.Popen(
            shell_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → stdout for unified stream
            text=True,
            bufsize=1,                  # line-buffered
        )

        stdout_lines: list[str] = []

        def _stream_to_queue():
            """Push each stdout line into _da_log_q for SSE delivery."""
            for raw in proc.stdout:
                line = raw.rstrip('\n')
                stdout_lines.append(line)
                _da_log_q.put(line)
            proc.wait()

        # Read timeout from input_routes.yml (deepatom.timeout_seconds)
        # Falls back to 1200 s (20 min) if not configured. 0 = wait forever.
        _timeout = _load_deepatom_cfg().get("timeout_seconds", 1200)
        _timeout = None if _timeout == 0 else int(_timeout)

        _t = threading.Thread(target=_stream_to_queue, daemon=True)
        _t.start()
        _t.join(timeout=_timeout)

        if _t.is_alive():
            proc.kill()
            _msg = f"script timed out (>{_timeout} s)"
            _da_log_q.put(f"__ERROR__: {_msg}")
            return jsonify({'status': 'error', 'message': f'DeepAtom {_msg}.'}), 504

        stdout = '\n'.join(stdout_lines)
        stderr = ''
        logger.info('[deepatom_estimate] exit=%d stdout_len=%d',
                    proc.returncode, len(stdout))

        _da_log_q.put("")
        _da_log_q.put(f"[deepatom] {'─' * 60}")
        _da_log_q.put(f"[deepatom] Script finished — exit code {proc.returncode}")
        _da_log_q.put(f"[deepatom] Stdout lines captured: {len(stdout_lines)}")
        _da_log_q.put(f"[deepatom] {'─' * 60}")

        if proc.returncode != 0:
            _da_log_q.put(f"__ERROR__: script exited with code {proc.returncode}")
            return jsonify({
                'status':  'error',
                'message': f'Script exited with code {proc.returncode}.',
                'stdout':  stdout[:1000],
            }), 500

        parsed = _parse_deepatom_output(stdout)
        # If only stat rows were found (or none), read the CSV directly
        if not parsed.get('compounds'):
            csv_compounds = _read_deepatom_csv(data_dir)
            if csv_compounds:
                parsed['compounds'] = csv_compounds
                logger.info('[deepatom] using %d compounds from CSV', len(csv_compounds))
        parsed['status'] = 'success' if parsed.get('compounds') else 'partial'
        if not parsed.get('compounds'):
            parsed['message'] = 'Script ran but no scored compounds found in output yet.'
        if stderr:
            parsed['_raw'] = (parsed.get('_raw') or '') + (
                '\n\n── STDERR ──────────────────────────────\n' + stderr
            )
        parsed['stderr_snippet'] = stderr[:1000] if stderr else ''

        # Emit parsed results summary to the log stream
        compounds = parsed.get('compounds') or []
        if compounds:
            _da_log_q.put("")
            _da_log_q.put(f"[deepatom] Parsed {len(compounds)} compounds:")
            preds = [c['pred_pk'] for c in compounds]
            _da_log_q.put(f"[deepatom]   best  pK = {max(preds):.4f}")
            _da_log_q.put(f"[deepatom]   mean  pK = {sum(preds)/len(preds):.4f}")
            _da_log_q.put(f"[deepatom]   worst pK = {min(preds):.4f}")
            _da_log_q.put("")
            # Top-5 hits
            top5 = sorted(compounds, key=lambda c: c['pred_pk'], reverse=True)[:5]
            _da_log_q.put(f"[deepatom] Top {min(5,len(top5))} hits:")
            for rank, c in enumerate(top5, 1):
                exp_str = f"  exp={c['exp_pk']:.2f}" if c.get('exp_pk') is not None else ""
                _da_log_q.put(f"[deepatom]   #{rank:2d}  {c['id']:<30s}  pred={c['pred_pk']:.4f}{exp_str}")
        else:
            _da_log_q.put("[deepatom] No compounds parsed from output.")
        _da_log_q.put('__DONE__')
        return jsonify(parsed)

    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error',
                        'message': 'DeepAtom timed out (>300 s).'}), 504
    except Exception as exc:
        logger.exception('[deepatom_estimate] error')
        return jsonify({'status': 'error', 'message': str(exc)}), 500



# ==============================================================================
# Model Weights Visualization
# GET /vina_visualization/deepatom_weights
# Loads model.pth.tar and returns per-layer L2 norm + shape + param count.
# ==============================================================================

_DEEPATOM_MODEL_PATH = os.path.join(
    _dacfg.DEEPATOM_ROOT, "model_split_data", "model", "model.pth.tar"
)

# Stage → colour mapping for the frontend chart
_STAGE_COLORS = {
    "input_block": "#67e8f9",   # cyan
    "stage2":      "#818cf8",   # indigo
    "stage3":      "#c084fc",   # purple
    "stage4":      "#f472b6",   # pink
    "out_block":   "#fbbf24",   # amber
}


@app.route("/vina_visualization/deepatom_weights", methods=["GET"])
def deepatom_weights():
    """
    GET /vina_visualization/deepatom_weights

    Loads the ShuffleNetV3 checkpoint and returns per-layer weight statistics:
      layer_name, stage, shape, n_params, l2_norm, mean, std

    Used by the DeepAtom modal to visualise the model architecture + weights.
    """
    import torch
    import math

    model_path = request.args.get("model_path", _DEEPATOM_MODEL_PATH).strip()

    try:
        from pathlib import Path as _PPath
        if not _PPath(model_path).is_file():
            return jsonify({"status": "error",
                            "message": f"Model not found: {model_path}"}), 404

        ckpt = torch.load(model_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        # Remove DataParallel "module." prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}

        layers = []
        total_params = 0

        for name, tensor in state.items():
            if tensor.dim() == 0:
                continue                        # skip scalars (BN running stats)
            if "running_mean" in name or "running_var" in name or "num_batches" in name:
                continue                        # skip BN tracking buffers

            n_params = tensor.numel()
            total_params += n_params
            l2_norm    = float(tensor.float().norm(p=2).item())
            t_mean     = float(tensor.float().mean().item())
            t_std      = float(tensor.float().std().item()) if n_params > 1 else 0.0

            # Map layer name → stage
            stage = "other"
            for s in ("input_block", "stage2", "stage3", "stage4", "out_block"):
                if s in name:
                    stage = s
                    break

            layers.append({
                "name":     name,
                "stage":    stage,
                "shape":    list(tensor.shape),
                "n_params": n_params,
                "l2_norm":  round(l2_norm, 4),
                "mean":     round(t_mean, 6),
                "std":      round(t_std, 6),
                "color":    _STAGE_COLORS.get(stage, "#475569"),
            })

        # Summary per stage
        stage_summary = {}
        for ly in layers:
            s = ly["stage"]
            if s not in stage_summary:
                stage_summary[s] = {"n_params": 0, "l2_norm_sq": 0.0,
                                    "color": ly["color"]}
            stage_summary[s]["n_params"]   += ly["n_params"]
            stage_summary[s]["l2_norm_sq"] += ly["l2_norm"] ** 2

        for s, v in stage_summary.items():
            v["l2_norm"] = round(math.sqrt(v["l2_norm_sq"]), 4)
            del v["l2_norm_sq"]

        return jsonify({
            "status":       "success",
            "model_path":   model_path,
            "total_params": total_params,
            "layers":       layers,
            "stage_summary": stage_summary,
        })

    except Exception as exc:
        logger.exception("[deepatom_weights] error")
        return jsonify({"status": "error", "message": str(exc)}), 500







# ==============================================================================
# Make Atomtypes — generate .atomtypes for a single compound
# POST /vina_visualization/deepatom_make_atomtypes
# GET  /vina_visualization/deepatom_make_atomtypes_stream  (SSE progress)
#
# Steps (tried in order):
#   0. Return immediately if data_dir/atomtypes/<id>.atomtypes already exists
#   1. Copy from newest DEEP_MODEL_temp dir if one exists there
#   2. Run arpeggio_mod2/arpeggio.py on the complex PDB
#   3. Run pipeline_VS.py --stages 1,3 as last resort
# ==============================================================================

_DA_SCRIPTS_DIR = Path(
    _dacfg.DEEPATOM_SCRIPTS_DIR
)
_DA_PRE_DIR  = _DA_SCRIPTS_DIR / "00_preprocess"
_DA_ARPEGGIO = _DA_SCRIPTS_DIR / "arpeggio_mod2" / "arpeggio.py"
_DA_PIPELINE = _DA_PRE_DIR / "pipeline_VS.py"

_da_at_log_q: queue.Queue = queue.Queue()


def _at_log(msg: str) -> None:
    logger.info("[make_atomtypes] %s", msg)
    _da_at_log_q.put(msg)


@app.route('/vina_visualization/deepatom_make_atomtypes_stream', methods=['GET'])
def deepatom_make_atomtypes_stream():
    """GET — SSE stream for make_atomtypes progress."""
    def generate():
        while True:
            try:
                line = _da_at_log_q.get(timeout=60)
                yield f"data: {line}\n\n"
                if line == "__DONE__" or line.startswith("__ERROR__"):
                    break
            except queue.Empty:
                yield ": keep-alive\n\n"
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/vina_visualization/deepatom_make_atomtypes', methods=['POST'])
def deepatom_make_atomtypes():
    """
    POST /vina_visualization/deepatom_make_atomtypes
    Body: { "compound_id": "BM-1-57", "data_dir": "/path/to/ZccE" }

    Returns:
      { "status": "success"|"error",
        "atomtypes_path": str,
        "method": "cached"|"temp_copy"|"arpeggio"|"pipeline",
        "message": str }
    """
    import shutil as _sh
    import subprocess as _sp
    import tempfile as _tf

    # Drain old log queue
    while not _da_at_log_q.empty():
        try: _da_at_log_q.get_nowait()
        except Exception: pass

    try:
        body        = request.get_json(force=True) or {}
        compound_id = (body.get('compound_id') or '').strip()
        data_dir    = (body.get('data_dir')    or '').strip()

        if not compound_id or not data_dir:
            return jsonify({'status': 'error',
                            'message': 'compound_id and data_dir are required'}), 400

        base    = Path(data_dir)
        out_dir = base / 'atomtypes'
        out_dir.mkdir(parents=True, exist_ok=True)
        dest    = out_dir / f"{compound_id}.atomtypes"

        _at_log(f"[deepatom] make_atomtypes: {compound_id}")
        _at_log(f"[deepatom] data_dir:       {data_dir}")
        _at_log(f"[deepatom] output:         {dest}")
        _at_log("")

        # ── Step 0: already cached ────────────────────────────────────────────
        if dest.is_file():
            _at_log(f"[deepatom] Step 0: cached → {dest.name}")
            _da_at_log_q.put("__DONE__")
            return jsonify({'status': 'success', 'atomtypes_path': str(dest),
                            'method': 'cached', 'message': 'Already exists — ready to visualize.'})

        # ── Step 1: copy from DEEP_MODEL_temp ────────────────────────────────
        _at_log("[deepatom] Step 1: searching DEEP_MODEL_temp dirs...")
        temp_dirs = sorted(base.glob("DEEP_MODEL_temp.*"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        for td in temp_dirs:
            hits = [h for h in td.rglob(f"{compound_id}.atomtypes")
                    if 'augmented' not in h.name]
            if hits:
                _sh.copy2(str(hits[0]), str(dest))
                _at_log(f"[deepatom] Step 1: copied from {td.name}")
                _da_at_log_q.put("__DONE__")
                return jsonify({'status': 'success', 'atomtypes_path': str(dest),
                                'method': 'temp_copy',
                                'message': f'Copied from {td.name}.'})
        _at_log("[deepatom] Step 1: not found in temp dirs")

        # Find the complex PDB (needed for steps 2 and 3)
        complex_pdb = None
        for td in temp_dirs:
            hits = list(td.rglob(f"{compound_id}_complex.pdb"))
            if hits:
                complex_pdb = hits[0]
                break
        if not complex_pdb:
            hits = list(base.rglob(f"{compound_id}_complex.pdb"))
            if hits:
                complex_pdb = hits[0]
        _at_log(f"[deepatom] complex PDB: {complex_pdb or 'NOT FOUND'}")

        # ── Step 2: run arpeggio directly ─────────────────────────────────────
        if complex_pdb and _DA_ARPEGGIO.is_file():
            _at_log(f"[deepatom] Step 2: arpeggio on {complex_pdb.name}")
            with _tf.TemporaryDirectory(prefix='elion_at_') as tmp:
                result = _sp.run(
                    [sys.executable, str(_DA_ARPEGGIO), str(complex_pdb), tmp],
                    capture_output=True, text=True, timeout=180
                )
                produced = (list(Path(tmp).glob("*.atomtypes")) +
                            list(complex_pdb.parent.glob(complex_pdb.stem + "*.atomtypes")))
                produced = [p for p in produced if 'augmented' not in p.name]
                if produced:
                    _sh.copy2(str(produced[0]), str(dest))
                    _at_log(f"[deepatom] Step 2: arpeggio OK → {dest.name}")
                    _da_at_log_q.put("__DONE__")
                    return jsonify({'status': 'success', 'atomtypes_path': str(dest),
                                    'method': 'arpeggio',
                                    'message': 'Generated with arpeggio.'})
                _at_log(f"[deepatom] Step 2: arpeggio produced nothing — stderr: {result.stderr[:200]}")
        else:
            _at_log("[deepatom] Step 2: skipped (no complex PDB or arpeggio.py not found)")

        # ── Step 3: run pipeline_VS.py --stages 1,3 ───────────────────────────
        if not _DA_PIPELINE.is_file():
            msg = f"pipeline_VS.py not found at {_DA_PIPELINE}"
            _at_log(f"[deepatom] Step 3: FAILED — {msg}")
            _da_at_log_q.put(f"__ERROR__: {msg}")
            return jsonify({'status': 'error', 'message': msg}), 404

        if not complex_pdb:
            msg = (f"No complex PDB found for '{compound_id}'. "
                   "Run Virtual Screening first to build the complex.")
            _at_log(f"[deepatom] Step 3: FAILED — {msg}")
            _da_at_log_q.put(f"__ERROR__: {msg}")
            return jsonify({'status': 'error', 'message': msg}), 404

        _at_log(f"[deepatom] Step 3: pipeline_VS.py --stages 1,3 for {compound_id}")
        with _tf.TemporaryDirectory(prefix='elion_pipe_') as tmp:
            tmp_batch = Path(tmp) / 'batch'
            ds_vs = tmp_batch / 'Dataset_VS' / compound_id
            ds_vs.mkdir(parents=True)
            _sh.copy2(str(complex_pdb), str(ds_vs / f"{compound_id}_complex.pdb"))
            lig_hits = list(complex_pdb.parent.glob(f"{compound_id}_ligand.pdb"))
            lig_src  = lig_hits[0] if lig_hits else complex_pdb
            _sh.copy2(str(lig_src), str(ds_vs / f"{compound_id}_ligand.pdb"))

            proc = _sp.Popen(
                [sys.executable, str(_DA_PIPELINE),
                 '--batch-dir',   str(tmp_batch),
                 '--pre-dir',     str(_DA_PRE_DIR),
                 '--scripts-dir', str(_DA_SCRIPTS_DIR),
                 '--stages', '1,3'],
                stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1
            )
            for raw in proc.stdout:
                _at_log(raw.rstrip('\n'))
            proc.wait()

            produced = [p for p in Path(tmp_batch).rglob(f"{compound_id}.atomtypes")
                        if 'augmented' not in p.name]
            if produced:
                _sh.copy2(str(produced[0]), str(dest))
                _at_log(f"[deepatom] Step 3: pipeline OK → {dest.name}")
                _da_at_log_q.put("__DONE__")
                return jsonify({'status': 'success', 'atomtypes_path': str(dest),
                                'method': 'pipeline',
                                'message': 'Generated via pipeline_VS.py stages 1+3.'})

        msg = f"All steps failed — could not generate .atomtypes for '{compound_id}'"
        _at_log(f"[deepatom] {msg}")
        _da_at_log_q.put(f"__ERROR__: {msg}")
        return jsonify({'status': 'error', 'message': msg}), 500

    except Exception as exc:
        logger.exception('[make_atomtypes] error')
        _da_at_log_q.put(f"__ERROR__: {exc}")
        return jsonify({'status': 'error', 'message': str(exc)}), 500


# ==============================================================================
# Register deepatom_saliency route
# deepatom_saliency.py lives at uiapp/ root and owns:
#   POST /vina_visualization/deepatom_saliency
# Importing it here causes its @app.route to register with Flask.
# ==============================================================================
try:
    import uiapp.core.deepatom_saliency  # noqa: F401
except Exception as _e:
    logger.warning("[deepatom_routes] deepatom_saliency import failed: %s", _e)