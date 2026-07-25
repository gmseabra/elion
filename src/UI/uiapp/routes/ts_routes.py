# =============================================================================
# routes/ts_routes.py
# Thompson Sampling generator endpoints:
#   ts_run    — POST: launch python elion.py -i input_TS.yml in a thread
#   ts_status — GET SSE: stream stdout line-by-line until __DONE__
#   ts_kill   — POST: SIGTERM the elion.py subprocess
# =============================================================================

import os, re, subprocess, threading, uuid, queue, signal, shutil, tempfile
import sys as _sys
from flask import jsonify, request, Response, stream_with_context
from uiapp import app

from uiapp.routes.shared import (
    logger, _ts_jobs, _ts_lock, _ELION_CWD, _ELION_YML, _ELION_VENV,
)

# ══ Thompson Sampling Generator  /vina_visualization/ts_* ══════════════════

# Module-level job store — same pattern as _finetune_jobs
_ts_jobs: dict[str, dict] = {}
_ts_lock = threading.Lock()

# Elion project root — elion.py lives here.
# Centralised in uiapp.config (override with the ELION_CWD / ELION_VENV env vars).
from uiapp import config as _cfg
_ELION_CWD  = _cfg.ELION_CWD
_ELION_YML  = _cfg.ELION_YML          # relative to _ELION_CWD
_ELION_VENV = _cfg.ELION_VENV


def _run_ts_job(job_id: str, yml_path: str, extra_env: dict) -> None:
    """
    Worker thread: runs  python elion.py -i <yml>  in _ELION_CWD.

    tqdm writes progress bars with \r (carriage return), NOT \n.
    Reading proc.stdout line-by-line (iterator or readline) buffers until \n,
    so tqdm bars arrive late and get merged with the next real log line.

    Fix: read raw bytes, split on BOTH \r and \n, push each non-empty fragment.
    This gives the frontend both the tqdm progress AND the [add_score/warmup]
    lines as separate SSE events the moment they're written.
    """
    import subprocess, re as _re_ts
    q = _ts_jobs[job_id]["queue"]

    def push(line: str):
        q.put(line)

    import sys as _sys
    python = _ELION_VENV if os.path.isfile(_ELION_VENV) else _sys.executable

    cmd = [python, "elion.py", "-i", yml_path]
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONPATH"] = _ELION_CWD + os.pathsep + env.get("PYTHONPATH", "")
    # Force line-buffered output from Python subprocess
    env["PYTHONUNBUFFERED"] = "1"

    logger.info("[TS] job %s starting: %s (cwd=%s)", job_id, " ".join(cmd), _ELION_CWD)
    push(f"$ cd {_ELION_CWD} && {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Binary mode so we control splitting — text=False
            bufsize=0,
            cwd=_ELION_CWD,
            env=env,
        )
        with _ts_lock:
            _ts_jobs[job_id]["pid"] = proc.pid

        # Read line-by-line using text mode (handles \n-terminated lines correctly).
        # tqdm uses \r to overwrite progress bars — we convert \r to \n via universal newlines.
        # Use a large buffer (65536) to handle long logging lines without mid-line splits.
        buf = b""
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            # Split on \r and \n
            parts = _re_ts.split(rb'[\r\n]+', buf)
            buf = parts[-1]
            for part in parts[:-1]:
                line = part.decode("utf-8", errors="replace").strip()
                if line:
                    push(line)
        if buf:
            line = buf.decode("utf-8", errors="replace").strip()
            if line:
                push(line)

        proc.wait()
        rc = proc.returncode
        with _ts_lock:
            _ts_jobs[job_id]["returncode"] = rc
            _ts_jobs[job_id]["status"] = "done" if rc == 0 else "error"

        if rc != 0:
            push(f"ERROR: elion.py exited with code {rc}")
        push("__DONE__")

    except Exception as exc:
        logger.exception("[TS] job %s failed", job_id)
        with _ts_lock:
            _ts_jobs[job_id]["status"] = "error"
        push(f"ERROR: {exc}")
        push("__DONE__")


@app.route('/vina_visualization/ts_run', methods=['POST'])
def vina_ts_run():
    """
    POST /vina_visualization/ts_run
    Body (all optional):
      {
        "yml_path":          str,   # path to input yml (default: input_TS.yml)
        "reaction_smarts":   str,   # if set, patches yml before running
        "num_ts_iterations": int    # if set, patches yml before running
      }

    Launches  python elion.py -i <yml>  in a daemon thread.
    Returns immediately with { "status": "started", "job_id": "<uuid>" }.
    Poll GET /vina_visualization/ts_status/<job_id>  (SSE) for live output.
    """
    import shutil, tempfile, re as _re

    try:
        data      = request.get_json(force=True) or {}
        yml_src   = data.get("yml_path", "").strip() or os.path.join(_ELION_CWD, _ELION_YML)
        smarts    = data.get("reaction_smarts", "").strip()
        iters     = data.get("num_ts_iterations")

        if not os.path.isfile(yml_src):
            return jsonify({"status": "error",
                            "message": f"yml not found: {yml_src}"}), 404

        # Always copy yml to a temp file so we can force log_level: DEBUG
        # (DEBUG is required for [add_score/warmup] and [evaluate] lines which
        #  feed the live warmup visualizer — at INFO they are suppressed)
        tmp_dir    = tempfile.mkdtemp(prefix="elion_ts_")
        yml_to_run = os.path.join(tmp_dir, "input_TS_run.yml")
        shutil.copy2(yml_src, yml_to_run)

        content = open(yml_to_run).read()
        if smarts:
            content = _re.sub(
                r'(reaction_smarts\s*:\s*).*',
                lambda m: m.group(1) + f'"{smarts}"',
                content,
            )
        if iters:
            content = _re.sub(
                r'(num_ts_iterations\s*:\s*)\d+',
                lambda m: m.group(1) + str(int(iters)),
                content,
            )
        # Force DEBUG so [evaluate] and [add_score/warmup] lines stream to the UI
        if _re.search(r'log_level\s*:', content):
            content = _re.sub(r'(log_level\s*:\s*)\S+', r'\g<1>DEBUG', content)
        else:
            # Append under the TS block if key absent
            content = content.rstrip() + '\n    log_level: DEBUG\n'
        open(yml_to_run, 'w').write(content)
        logger.info("[TS] patched yml written to %s (log_level forced to DEBUG)", yml_to_run)

        job_id = str(uuid.uuid4())
        with _ts_lock:
            _ts_jobs[job_id] = {
                "status":  "running",
                "queue":   queue.Queue(),
                "yml":     yml_to_run,
                "pid":     None,
                "returncode": None,
            }

        t = threading.Thread(
            target=_run_ts_job,
            args=(job_id, yml_to_run, {}),
            daemon=True,
        )
        t.start()

        logger.info("[TS] job %s started", job_id)
        return jsonify({"status": "started", "job_id": job_id})

    except Exception as exc:
        logger.exception("[TS] ts_run error")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/vina_visualization/ts_status/<job_id>', methods=['GET'])
def vina_ts_status(job_id: str):
    """
    GET /vina_visualization/ts_status/<job_id>
    Server-Sent Events stream.
    Each event:  data: <log line>\\n\\n
    Terminal:    data: __DONE__\\n\\n
    Keep-alive:  : keep-alive\\n\\n   (every 15 s while idle)
    """
    if job_id not in _ts_jobs:
        return jsonify({"status": "error", "message": "Unknown job_id"}), 404

    job = _ts_jobs[job_id]
    q   = job["queue"]

    def generate():
        while True:
            try:
                line = q.get(timeout=15)
                # Escape SSE special chars — newlines inside data would break the frame
                safe = line.replace("\n", " ").replace("\r", "")
                yield f"data: {safe}\n\n"
                if line == "__DONE__":
                    break
            except queue.Empty:
                yield ": keep-alive\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route('/vina_visualization/ts_kill/<job_id>', methods=['POST'])
def vina_ts_kill(job_id: str):
    """
    POST /vina_visualization/ts_kill/<job_id>
    Sends SIGTERM to the elion.py subprocess so the user can cancel a run.
    """
    import signal
    if job_id not in _ts_jobs:
        return jsonify({"status": "error", "message": "Unknown job_id"}), 404

    pid = _ts_jobs[job_id].get("pid")
    if not pid:
        return jsonify({"status": "error", "message": "Process not started yet"}), 400

    try:
        os.kill(pid, signal.SIGTERM)
        with _ts_lock:
            _ts_jobs[job_id]["status"] = "killed"
        logger.info("[TS] job %s killed (pid %s)", job_id, pid)
        return jsonify({"status": "killed", "pid": pid})
    except ProcessLookupError:
        return jsonify({"status": "already_done"})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route('/vina_visualization/ts_config', methods=['GET'])
def vina_ts_config():
    """
    GET /vina_visualization/ts_config
    Reads input_TS.yml and returns the TS generator config as JSON
    for display in the modal sidebar.
    """
    import yaml as _yaml
    yml_path = os.path.join(_ELION_CWD, _ELION_YML)
    try:
        with open(yml_path) as f:
            cfg = _yaml.safe_load(f)
        ts = cfg.get('generator', {}).get('TS', {})
        return jsonify({
            'status':      'ok',
            'log_level':   ts.get('log_level', 'INFO'),
            'ts_mode':     ts.get('ts_mode', '—'),
            'iterations':  ts.get('num_ts_iterations', '—'),
            'warmup':      ts.get('num_warmup_trials', '—'),
            'batch':       ts.get('eval_batch_size', '—'),
            'smarts':      ts.get('reaction_smarts', '—'),
            'reagents':    ts.get('reagent_file_list', []),
            'results':     ts.get('results_filename', '—'),
        })
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': f'Not found: {yml_path}'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500