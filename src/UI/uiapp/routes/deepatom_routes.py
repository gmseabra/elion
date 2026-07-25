# ==============================================================================
# deepatom_routes.py — DeepAtom Virtual Screening routes
#
# Extracted from uiapp/routes.py.
# Registered by importing this module from routes.py:
#
#     from uiapp.routes import deepatom_routes as _deepatom_routes_mod
#     _deepatom_routes_mod.configure(input_routes_yml=_INPUT_ROUTES_YML)
#
# Flask routes registered here:
#   GET  /vina_visualization/deepatom_datasets
#   POST /vina_visualization/deepatom_estimate
# ==============================================================================

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import yaml
from flask import jsonify, request
from uiapp import app

logger = logging.getLogger(__name__)

# ── Defaults (overridden by configure() or by input_routes.yml) ───────────────
# DeepAtom is an external project; point at it with the DEEPATOM_SCRIPT and
# DEEPATOM_DATA_DIR environment variables (see uiapp/config.py) or via
# config/input_routes.yml. Empty defaults mean "not configured".
from uiapp import config as _cfg

_INPUT_ROUTES_YML: str = ""

DEEPATOM_SCRIPT = _cfg.DEEPATOM_SCRIPT
DEEPATOM_DEFAULT_DATA_DIR = _cfg.DEEPATOM_DATA_DIR
DEEPATOM_DEFAULT_TEST_TYPE = _cfg.DEEPATOM_TEST_TYPE


def configure(input_routes_yml: str) -> None:
    """Call once from routes.py after _INPUT_ROUTES_YML is resolved."""
    global _INPUT_ROUTES_YML
    _INPUT_ROUTES_YML = input_routes_yml


# ==============================================================================
# YAML helpers
# ==============================================================================

def _load_deepatom_cfg() -> dict:
    """Return the full `deepatom:` section from input_routes.yml, or {}."""
    if not _INPUT_ROUTES_YML:
        return {}
    try:
        with open(_INPUT_ROUTES_YML, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return raw.get("deepatom", {}) if raw else {}
    except Exception as exc:
        logger.warning("[deepatom_routes] Could not read YAML: %s", exc)
        return {}


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
            'conda activate binding_affinity_27 && '
            f'"{script}" -t {test_type} -d "{data_dir}" && '
            'conda deactivate'
        )
        logger.info('[deepatom_estimate] cmd: %s', shell_cmd)

        proc = subprocess.run(
            shell_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )

        stdout = proc.stdout or ''
        stderr = proc.stderr or ''
        logger.info('[deepatom_estimate] exit=%d stdout_len=%d',
                    proc.returncode, len(stdout))
        if stderr:
            logger.warning('[deepatom_estimate] stderr: %s', stderr[:500])

        if proc.returncode != 0:
            return jsonify({
                'status':  'error',
                'message': f'Script exited with code {proc.returncode}.',
                'stderr':  stderr[:1000],
                'stdout':  stdout[:1000],
            }), 500

        parsed = _parse_deepatom_output(stdout)
        parsed['status'] = 'success' if parsed.get('compounds') else 'partial'
        if not parsed.get('compounds'):
            parsed['message'] = 'Script ran but no scored compounds found in output yet.'
        if stderr:
            parsed['_raw'] = (parsed.get('_raw') or '') + (
                '\n\n── STDERR ──────────────────────────────\n' + stderr
            )
        parsed['stderr_snippet'] = stderr[:1000] if stderr else ''
        return jsonify(parsed)

    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error',
                        'message': 'DeepAtom timed out (>300 s).'}), 504
    except Exception as exc:
        logger.exception('[deepatom_estimate] error')
        return jsonify({'status': 'error', 'message': str(exc)}), 500