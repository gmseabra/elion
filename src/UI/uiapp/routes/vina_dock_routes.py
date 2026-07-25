# =============================================================================
# routes/vina_dock_routes.py
# Core Vina docking endpoints:
#   vina_viz, vina_check_file, vina_dock (subprocess + SSE queue),
#   vina_dock_progress (SSE consumer), vina_tail_log (SSE log tail),
#   vina_parse_log
# =============================================================================

import os, re, json as _json, datetime, queue
from pathlib import Path
from flask import jsonify, request, Response, stream_with_context, current_app
from uiapp import app

from uiapp.routes.shared import (
    logger, VINA_BASE, VINA_BIN, VINA_LOG, _INPUT_ROUTES_YML,
    _vina_progress_q,
)

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

def _parse_vina_log(log_text):
    pair_re  = re.compile(
        r'\[non_cache::eval pair\]\s+lig_atom=(\d+)\(xs=(\d+)\)\s+'
        r'rec_atom=(\d+)\(xs=(\d+)\)\s+opt\(ri\+rj\)=([\d.]+)\s+'
        r'r=([\d.]+)\s+s=r-opt=([-\d.]+)\s+pair_e=([-\d.]+)'
    )
    atom_re  = re.compile(
        r'\[non_cache::eval lig_atom=(\d+)\(xs=(\d+)\)\]\s+'
        r'this_e\(after curl\)=([-\d.]+)\s+out_of_bounds_penalty=([-\d.]+)'
    )
    total_re = re.compile(r'\[non_cache::eval TOTAL\] e=([-\d.]+)')
    mode_re  = re.compile(r'\[mode (\d+) energy assembly\]')
    ev_re    = re.compile(r'total \(E_vina\)\s*=\s*([-\d.]+)')
    conf_re  = re.compile(r'conf_independent \(Nrot penalty\)\s*=\s*([-\d.]+)')

    lig_atoms, lig_order, total_e = {}, [], None
    mode_energies, cur_mode = {}, None

    for line in log_text.splitlines():
        m = mode_re.search(line)
        if m: cur_mode = int(m.group(1)); mode_energies[cur_mode] = {}; continue
        if cur_mode:
            m = ev_re.search(line)
            if m: mode_energies[cur_mode]['total_e'] = float(m.group(1)); continue
            m = conf_re.search(line)
            if m: mode_energies[cur_mode]['nrot_penalty'] = float(m.group(1)); continue

        m = pair_re.search(line)
        if m:
            li,lxs,ri,rxs = int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4))
            opt,r,s,pe    = float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8))
            if li not in lig_atoms:
                lig_atoms[li] = {"xs":lxs,"pairs":[],"this_e":0.0,"out_of_bounds":0.0}
                lig_order.append(li)
            lig_atoms[li]["pairs"].append({"rec_idx":ri,"rec_xs":rxs,"opt":opt,"r":r,"s":s,"pair_e":pe})
            continue

        m = atom_re.search(line)
        if m:
            li,lxs,this_e,oob = int(m.group(1)),int(m.group(2)),float(m.group(3)),float(m.group(4))
            if li not in lig_atoms:
                lig_atoms[li] = {"xs":lxs,"pairs":[],"this_e":0.0,"out_of_bounds":0.0}
                lig_order.append(li)
            lig_atoms[li]["this_e"]       = this_e
            lig_atoms[li]["out_of_bounds"] = oob
            continue

        m = total_re.search(line)
        if m: total_e = float(m.group(1))

    return lig_atoms, total_e, lig_order, mode_energies


def _term_breakdown(xs1, xs2, r):
    opt = _XS_RADIUS.get(xs1,1.8) + _XS_RADIUS.get(xs2,1.8)
    s   = r - opt
    g1  = math.exp(-((s/0.5)**2))
    g2  = math.exp(-(((s-3.0)/2.0)**2))
    rp  = s*s if s < 0 else 0.0
    m1  = _XS_META.get(xs1,("?",False,False,False))
    m2  = _XS_META.get(xs2,("?",False,False,False))
    hp  = (1.0 if s<=0.5 else 0.0 if s>=1.5 else 1.0-(s-0.5)) if (m1[1] and m2[1]) else 0.0
    hb_ok = (m1[2] and m2[3]) or (m1[3] and m2[2])
    hb  = (1.0 if s<=-0.7 else 0.0 if s>=0.0 else 1.0-s/(-0.7)) if hb_ok else 0.0
    w   = _VINA_W
    return {
        "s": round(s,5),
        "gauss1_raw": round(g1,5),    "gauss2_raw": round(g2,5),
        "repulsion_raw": round(rp,5), "hydrophobic_raw": round(hp,5), "hbond_raw": round(hb,5),
        "gauss1_w": round(w["gauss1"]*g1,6),       "gauss2_w": round(w["gauss2"]*g2,6),
        "repulsion_w": round(w["repulsion"]*rp,6),  "hydrophobic_w": round(w["hydrophobic"]*hp,6),
        "hbond_w": round(w["hbond"]*hb,6),
        "pair_e_formula": round(w["gauss1"]*g1+w["gauss2"]*g2+w["repulsion"]*rp
                                +w["hydrophobic"]*hp+w["hbond"]*hb, 6),
    }


def _load_pdbqt_all(path):
    """Load ALL atoms from PDBQT MODEL 1 (including H, for index alignment)."""
    atoms = []
    try:
        with open(path) as fh:
            in_model = has_model = False
            for line in fh:
                if line.startswith("MODEL"):  has_model = in_model = True; continue
                if line.startswith("ENDMDL"): break
                if not has_model: in_model = True
                if not in_model:  continue
                if line[:4] not in ("ATOM","HEAT"): continue
                try:
                    atoms.append({
                        "serial":  int(line[6:11]),
                        "name":    line[12:16].strip(),
                        "resname": line[17:20].strip(),
                        "resseq":  line[22:26].strip(),
                        "chain":   line[21].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "atype":   line[77:].strip(),
                    })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        logger.warning(f"_load_pdbqt_all({path}): {e}")
    return atoms


def _load_pdbqt_heavy(path):
    """Load heavy atoms only (H/HD excluded) — used for vina_check_file."""
    return [a for a in _load_pdbqt_all(path) if a["atype"] not in ("H","HD")]


def _load_rec_heavy(path):
    """Load receptor heavy atoms for protein view.

    CRITICAL: idx stored here must match the rec_atom index in the Vina log,
    which is the position in m.grid_atoms[] — i.e. the ALL-atom line index
    (H atoms included in the count, just skipped for scoring).
    """
    atoms = []
    all_atom_idx = 0   # counts every ATOM/HETATM line including H
    try:
        with open(path) as fh:
            for line in fh:
                if line[:4] not in ("ATOM","HEAT"): continue
                atype = line[77:].strip()
                is_h  = atype in ("HD", "H")
                try:
                    rec = {
                        "serial":      int(line[6:11]),
                        "name":        line[12:16].strip(),
                        "resname":     line[17:20].strip(),
                        "resseq":      line[22:26].strip(),
                        "chain":       line[21].strip(),
                        "x":           float(line[30:38]),
                        "y":           float(line[38:46]),
                        "z":           float(line[46:54]),
                        "atype":       atype,
                        "all_atom_idx": all_atom_idx,  # ← Vina's rec_atom index
                    }
                    if not is_h:
                        atoms.append(rec)
                    all_atom_idx += 1
                except (ValueError, IndexError):
                    all_atom_idx += 1
                    continue
    except Exception as e:
        logger.warning(f"_load_rec_heavy({path}): {e}")
    return atoms


def _build_response(lig_atoms_raw, total_e, lig_order, mode_energies,
                    lig_path, rec_path, log_text):
    """Shared JSON builder for vina_viz and vina_dock."""
    lig_atoms_all = _load_pdbqt_all(lig_path)   if lig_path  else []
    rec_atoms_all = _load_rec_heavy(rec_path)    if rec_path  else []

    # Build lookup: all_atom_idx → atom record (matches m.grid_atoms[] indexing)
    rec_by_idx = {a['all_atom_idx']: a for a in rec_atoms_all}

    result_atoms = []
    for li in lig_order:
        info  = lig_atoms_raw[li]
        xs    = info["xs"]
        pairs = info["pairs"]

        pairs_sorted = sorted(pairs, key=lambda p: abs(p["pair_e"]), reverse=True)
        top_pairs = []
        for p in pairs_sorted[:15]:
            terms    = _term_breakdown(xs, p["rec_xs"], p["r"])
            rec_atom = rec_by_idx.get(p["rec_idx"])
            top_pairs.append({
                "rec_idx":      p["rec_idx"],
                "rec_xs":       p["rec_xs"],
                "rec_xs_label": _XS_META.get(p["rec_xs"],("?",))[0],
                "rec_atom":     rec_atom,
                "opt":  p["opt"], "r": p["r"], "s": p["s"],
                "pair_e": p["pair_e"], "terms": terms,
            })

        lig_atom = lig_atoms_all[li] if li < len(lig_atoms_all) else None
        result_atoms.append({
            "lig_idx":       li,
            "xs":            xs,
            "xs_label":      _XS_META.get(xs,("?",))[0],
            "this_e":        info["this_e"],
            "out_of_bounds": info["out_of_bounds"],
            "n_pairs":       len(pairs),
            "pair_sum":      round(sum(p["pair_e"] for p in pairs), 6),
            "top_pairs":     top_pairs,
            "lig_atom":      lig_atom,
        })

    result_atoms.sort(key=lambda a: a["this_e"])

    return jsonify({
        "status":        "success",
        "total_e":       total_e,
        "n_lig_atoms":   len(result_atoms),
        "lig_atoms":     result_atoms,
        "mode_energies": mode_energies,
        "log_preview":   log_text[:2000],
    })


@app.route('/vina_visualization/vina_viz', methods=['POST'])
def vina_viz():
    """
    POST /vina_viz
    Body: { "log_text": str, "ligand_path": str (opt), "receptor_path": str (opt) }
    Parse a captured non_cache::eval log and return structured JSON.
    """
    try:
        data     = request.get_json(force=True) or {}
        log_text = data.get("log_text","").strip()
        lig_path = data.get("ligand_path","").strip()
        rec_path = data.get("receptor_path","").strip()

        if not log_text:
            return jsonify({"status":"error","message":"log_text is required"}), 400

        lig_atoms_raw, total_e, lig_order, mode_energies = _parse_vina_log(log_text)

        if not lig_atoms_raw:
            return jsonify({"status":"error",
                            "message":"No [non_cache::eval pair] lines found in log_text"}), 400

        return _build_response(lig_atoms_raw, total_e, lig_order, mode_energies,
                               lig_path, rec_path, log_text)
    except Exception as exc:
        logger.error(f"vina_viz error: {exc}", exc_info=True)
        return jsonify({"status":"error","message":str(exc)}), 500


@app.route('/vina_visualization/vina_check_file', methods=['POST'])
def vina_check_file():
    """Check if a PDBQT file exists and count its heavy atoms."""
    try:
        data  = request.get_json(force=True) or {}
        path  = data.get('path','').strip()
        if not path or not Path(path).is_file():
            return jsonify({'exists': False, 'n_atoms': 0})
        atoms = _load_pdbqt_heavy(path)
        return jsonify({'exists': True, 'n_atoms': len(atoms)})
    except Exception as exc:
        return jsonify({'exists': False, 'n_atoms': 0, 'error': str(exc)})


@app.route('/vina_visualization/vina_dock', methods=['POST'])
def vina_dock():
    """
    POST /vina_dock
    Runs AutoDock Vina, streams stdout progress via SSE, writes a log file,
    parses the non_cache::eval output, and returns structured score data.

    Body: { "receptor_path": str, "ligand_path": str }
    """
    import subprocess, re, datetime
    try:
        data     = request.get_json(force=True) or {}
        rec_path = data.get('receptor_path', '').strip()
        lig_path = data.get('ligand_path',   '').strip()

        for label, p in [('receptor_path', rec_path), ('ligand_path', lig_path)]:
            if not p:
                return jsonify({'status': 'error', 'message': f'{label} is required'}), 400
            if not Path(p).is_file():
                return jsonify({'status': 'error', 'message': f'File not found: {p}'}), 404

        LOG_FILE  = VINA_LOG

        # Output alongside the ligand file, named after it
        lig_stem  = Path(lig_path).stem
        OUT_LIG   = str(Path(lig_path).parent / f"{lig_stem}_out.pdbqt")

        # ── All hyperparameters from input_routes.yml via app.config["VINA"] ──────
        # Loaded by app.py at startup — zero hardcoding here.
        _vcfg = current_app.config.get("VINA", {})
        _cpu  = _vcfg.get("cpu") or os.cpu_count() or 4
        cmd = [
            VINA_BIN,
            '--receptor',       rec_path,
            '--ligand',         lig_path,
            '--center_x',       str(_vcfg.get('center_x', -25.7)),
            '--center_y',       str(_vcfg.get('center_y',   0.22)),
            '--center_z',       str(_vcfg.get('center_z',  28.39)),
            '--size_x',         str(_vcfg.get('size_x',      20)),
            '--size_y',         str(_vcfg.get('size_y',      20)),
            '--size_z',         str(_vcfg.get('size_z',      20)),
            '--exhaustiveness', str(_vcfg.get('exhaustiveness', 8)),
            '--num_modes',      str(_vcfg.get('num_modes',      9)),
            '--energy_range',   str(_vcfg.get('energy_range',   3)),
            '--cpu',            str(_cpu),
            '--out',            OUT_LIG,
        ]

        logger.info(f'vina_dock: running {" ".join(cmd)}')

        # ── Run with real-time line capture ──────────────────────────────────
        # Clear the progress queue from any previous run
        while not _vina_progress_q.empty():
            try: _vina_progress_q.get_nowait()
            except: pass

        # Ensure conda lib dir is in LD_LIBRARY_PATH so the instrumented
        # vina binary finds libboost_system.so.1.84.0 at runtime.
        _conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
        _proc_env  = os.environ.copy()
        if _conda_lib and _conda_lib not in _proc_env.get("LD_LIBRARY_PATH", ""):
            _proc_env["LD_LIBRARY_PATH"] = (
                _conda_lib + ":" + _proc_env.get("LD_LIBRARY_PATH", "")
            ).strip(":")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0,   # unbuffered — essential for live progress bar
            env=_proc_env,
        )

        lines = []
        import threading as _threading

        def _stream_stdout():
            """
            Read stdout byte-by-byte.
            Bar chars (* | -) are batched: a background flush thread emits
            the accumulated bar buffer every 150ms so the browser isn't
            flooded with thousands of individual SSE events.
            Regular text lines are pushed whole on newline.
            """
            import time as _time
            import threading as _t2

            buf      = b""
            bar_buf  = {"stars": "", "sep": ""}   # shared mutable dict
            bar_lock = _t2.Lock()
            done_ev  = _t2.Event()
            in_progress_bar = False   # True only after first * — guards |/- capture

            BAR_CHARS = {b'*', b'|', b'-'}

            def _flush_bar():
                """Push accumulated bar chars every 150ms."""
                while not done_ev.is_set():
                    _time.sleep(0.15)
                    with bar_lock:
                        if bar_buf["stars"]:
                            _vina_progress_q.put("__BARCH__stars:" + bar_buf["stars"])
                        if bar_buf["sep"]:
                            _vina_progress_q.put("__BARCH__sep:" + bar_buf["sep"])

            flush_t = _t2.Thread(target=_flush_bar, daemon=True)
            flush_t.start()

            with open(LOG_FILE, 'wb') as lf:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                header = f'# Vina dock run {ts}\n# CMD: {" ".join(cmd)}\n\n'.encode()
                lf.write(header)
                while True:
                    ch = proc.stdout.read(1)
                    if not ch:
                        break
                    lf.write(ch)
                    lf.flush()

                    if ch in (b'\n', b'\r'):
                        line = buf.decode('utf-8', errors='replace').strip()
                        if line:
                            lines.append(line + '\n')
                            _vina_progress_q.put(line)
                        buf = b""
                        in_progress_bar = False  # reset bar context on newline

                    elif ch == b'*':
                        in_progress_bar = True
                        with bar_lock:
                            bar_buf["stars"] += "*"
                        buf = b""   # reset text buf — bar started

                    elif ch in (b'|', b'-') and in_progress_bar:
                        # Only capture as bar chars when inside the progress bar
                        with bar_lock:
                            bar_buf["sep"] += ch.decode()
                        buf = b""

                    else:
                        buf += ch

                # Flush remaining text
                if buf:
                    line = buf.decode('utf-8', errors='replace').strip()
                    if line:
                        lines.append(line + '\n')
                        _vina_progress_q.put(line)

            # Stop flush thread and do final push
            done_ev.set()
            flush_t.join(timeout=0.5)
            with bar_lock:
                if bar_buf["stars"]:
                    _vina_progress_q.put("__BARCH__stars:" + bar_buf["stars"])
                if bar_buf["sep"]:
                    _vina_progress_q.put("__BARCH__sep:" + bar_buf["sep"])

        t = _threading.Thread(target=_stream_stdout, daemon=True)
        t.start()
        t.join()
        proc.wait()
        _vina_progress_q.put('__DONE__')

        log_text = ''.join(lines)
        logger.info(f'vina_dock: finished. Log written to {LOG_FILE} ({len(lines)} lines)')

        # ── Read the actual log file for clean extraction ─────────────────────
        # The streamed `lines` buffer strips bar chars (|/-/*) and may miss content.
        # Read the file directly to get the complete, unmodified output.
        try:
            with open(LOG_FILE, 'r', errors='replace') as _lf:
                full_log = _lf.read()
        except Exception:
            full_log = log_text   # fallback to streamed buffer

        # ── Extract the tail block: energy assembly + mode table ──────────────
        # The user-facing content starts from [num_tors_div...] or [mode 1 energy...]
        # and ends at the last line of the mode table. Read from file tail.
        tail_block = None
        tail_lines = full_log.splitlines()

        # Find the start of the energy assembly / tors block (last occurrence)
        start_idx = None
        for i in range(len(tail_lines) - 1, -1, -1):
            if '[num_tors_div' in tail_lines[i] or '[mode 1 energy' in tail_lines[i]:
                start_idx = i
                break

        if start_idx is not None:
            tail_block = '\n'.join(tail_lines[start_idx:]).strip()

        # ── Extract mode table ────────────────────────────────────────────────
        mode_table = None
        tbl_lines  = []
        in_table   = False
        for raw_line in tail_lines:
            if re.search(r'mode\s+\|\s+affinity', raw_line):
                in_table = True
                tbl_lines = [raw_line]
                continue
            if in_table:
                if raw_line.strip() == '' and tbl_lines:
                    break
                tbl_lines.append(raw_line)
        if tbl_lines:
            mode_table = '\n'.join(tbl_lines)

        # ── Extract best affinity from the mode table ─────────────────────────
        best_affinity = None
        for raw_line in tail_lines:
            m = re.search(r'^\s*1\s+([-\d.]+)', raw_line)
            if m:
                best_affinity = float(m.group(1))
                break

        # ── Count non_cache pair lines (for UI feedback) ──────────────────────
        n_pair_lines = sum(1 for l in lines if '[non_cache::eval pair]' in l)
        logger.info(f'vina_dock: {n_pair_lines} pair lines, best_affinity={best_affinity}')

        # ── Parse non_cache::eval log ─────────────────────────────────────────
        lig_atoms_raw, total_e, lig_order, mode_energies = _parse_vina_log(log_text)

        if not lig_atoms_raw:
            return jsonify({
                'status':        'success',
                'best_affinity': best_affinity,
                'mode_table':    mode_table,
                'tail_block':    tail_block,
                'log_file':      LOG_FILE,
                'n_pair_lines':  n_pair_lines,
                'total_e':       None,
                'lig_atoms':     None,
                'message':       (
                    f'Docking complete (best={best_affinity} kcal/mol). '
                    f'Log saved to {LOG_FILE}.'
                ),
            })

        # ── Build full response ───────────────────────────────────────────────
        resp      = _build_response(lig_atoms_raw, total_e, lig_order, mode_energies,
                                    OUT_LIG, rec_path, log_text)
        resp_data = resp.get_json()
        resp_data['best_affinity'] = best_affinity
        resp_data['log_file']      = LOG_FILE
        resp_data['n_pair_lines']  = n_pair_lines
        resp_data['mode_table']    = mode_table
        resp_data['tail_block']    = tail_block
        # Return only the compact summary lines — skip per-pair term breakdowns
        # (gauss/repulsion/hbond/hydrophobic) which flood the console.
        _SKIP = ('[gauss ', '[repulsion ', '[hydrophobic ', '[hbond ',
                 '[non_cache::eval pair]')
        nc_lines = [l for l in lines if not any(l.lstrip().startswith(p) for p in _SKIP)]
        resp_data['log_text'] = ''.join(nc_lines)
        return jsonify(resp_data)

    except Exception as exc:
        logger.error(f'vina_dock error: {exc}', exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/vina_visualization/vina_dock_progress', methods=['GET'])
def vina_dock_progress_ep():
    """
    GET /vina_dock_progress
    SSE stream reading from _vina_progress_q — populated byte-by-byte from
    Vina's live stdout. Captures the *** progress bar in real time.
    """
    import re as _re2

    _SKIP = (
        # vLLM verbose energy terms
        '[gauss ', '[repulsion ', '[hydrophobic ', '[hbond ',
        '[non_cache', '[eval_interacting', '[pair]', '[num_tors',
        'after_curl=', 'running_e=',
        # Energy assembly lines
        'lig_grids ', 'inter_pairs ', 'inter = ', 'flex_grids ',
        'intra_pairs ', 'lig_intra ', 'intra = ', 'intramolecular',
        'e_in = ', 'conf_independent', 'total (E_vina)',
        'x (interaction', 'num_tors ', 'raw weight ',
        'weight = ', 'denom = ', 'result = ',
        # Per-atom coordinate dump lines
        'rec_in_box=', 'clipped=', 'oob_penalty=',
        # Curl accumulator lines
        '(before curl', '(= lig_grids',
        # RMSD table separator
        'dist from best', 'rmsd u.b.', 'rmsd l.b.',
        # Optimizer iteration lines e.g. "15.575(Δ=0.008Å)]"
        '(Δ=',
        # Energy assembly noise
        '(raw+1)', 'num_tors/5', '[mode ',
    )

    _SEP_RE = re.compile(r'^[-|]{6,}')   # Vina grid separator line

    def generate():
        idle = 0
        while idle < 60:
            try:
                item = _vina_progress_q.get(timeout=0.5)
                idle = 0
                stripped = item.strip()
                if item == "__DONE__":
                    yield "data: __DONE__\n\n"
                    return
                # Batched bar update: __BARCH__stars:<all_stars> or __BARCH__sep:<sep_chars>
                if item.startswith("__BARCH__"):
                    yield "data: " + item + "\n\n"
                    continue

                # Percentage line: "0%  10  20 ... 100%"
                import re as _re3
                _PCT_RE = _re3.compile(r"^\d+%")
                if _PCT_RE.match(stripped):
                    yield "data: __BAR__pct" + stripped + "\n\n"
                    continue
                # Skip if any SKIP pattern appears anywhere in the line
                # (coordinate lines start with numbers, not the pattern)
                if any(p in stripped for p in _SKIP):
                    continue
                # Skip Vina grid separator lines (long ----||||--- sequences)
                if _SEP_RE.match(stripped):
                    continue
                # Skip bare float-only lines (coordinate accumulator values)
                import re as _re_skip
                if _re_skip.match(r'^-?\d+\.\d+$', stripped):
                    continue
                # Skip lines that are ONLY numbers/spaces (partial coordinate lines)
                # BUT keep result table rows like "1  -9.3082  0.0000  0.0000"
                # Result rows have mode number + negative affinity — recognise by pattern
                is_result_row = bool(_re_skip.match(r'^\d+\s+-\d+\.\d', stripped))
                if (not is_result_row and stripped
                        and all(c in '0123456789.-,() ' for c in stripped)
                        and len(stripped) < 40):
                    continue
                # Restore mode result table lines (keep "mode |" header and "N  -X.X  ..." rows)
                is_mode_table = (stripped.startswith('mode |') or
                                 bool(_re_skip.match(r'^\d+\s+-\d+\.', stripped)))
                if "[non_cache::eval pair]" in stripped:
                    m = _re2.search(
                        r"lig_atom=(\d+).*?pair_e=([-\d.]+).*?rec_in_box=(\w+)",
                        stripped)
                    if m:
                        item = ("[pair] atom=" + m.group(1) + "  "
                                "e=" + f"{float(m.group(2)):+.5f}" + "  "
                                "in_box=" + m.group(3))
                    else:
                        item = stripped[:120]
                safe = item.replace("\n", " ").replace("\r", "")
                if safe.strip():
                    yield "data: " + safe + "\n\n"
            except queue.Empty:
                idle += 0.5
                yield ": ping\n\n"
            except Exception as exc:
                yield "data: Error: " + str(exc) + "\n\n"
                break
        yield "data: __DONE__\n\n"
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/vina_visualization/vina_tail_log', methods=['GET'])
def vina_tail_log_ep():
    """
    SSE: tails VINA_LOG and streams [non_cache::clip lig_atom=] and
    [non_cache::eval lig_atom=] lines live as Vina writes them.

    Run detection: records the log file's mtime at connection time, then
    waits until the file is replaced/truncated (mtime changes AND size resets),
    confirming this is a new run. Works across multiple gunicorn workers because
    it uses the filesystem, not a process-local global.
    """
    import time as _time
    import os as _os

    def generate():
        NC_RE   = re.compile(r'\[non_cache::(?:clip|eval lig_atom)=\d+')
        DONE_RE = re.compile(r'mode\s+\|\s+affinity')

        connect_t = _time.time()
        ping_t    = _time.time()

        # ── Step 1: wait for the log file to be (re)created for this run ────────
        # We know vina_dock truncates and rewrites the file. Poll until the file's
        # mtime is AFTER our connection time, which means a new run started.
        logger.info(f"[tail] connected at {connect_t:.3f}, waiting for fresh log")
        fh = None
        for _ in range(150):   # up to 15s
            try:
                st = _os.stat(VINA_LOG)
                # File was modified after we connected → new run
                if st.st_mtime >= connect_t - 1.0:
                    fh = open(VINA_LOG, 'r', errors='replace')
                    logger.info(f"[tail] log fresh (mtime={st.st_mtime:.3f}), opened")
                    break
            except FileNotFoundError:
                pass
            _time.sleep(0.1)
            if _time.time() - ping_t > 4:
                yield ": ping\n\n"
                ping_t = _time.time()

        if fh is None:
            logger.warning("[tail] fresh log never appeared")
            yield "data: [tail] Log file not refreshed — is vina_dock running?\n\n"
            yield "data: __DONE__\n\n"
            return

        # ── Step 2: verify the header line to confirm this is the right run ─────
        # Read the first 3 lines (comment header). They appear quickly.
        header_ok  = False
        header_buf = []
        for _ in range(30):
            line = fh.readline()
            if line:
                header_buf.append(line)
                if len(header_buf) >= 3 or 'CMD:' in line:
                    header_ok = True
                    break
            else:
                _time.sleep(0.1)
        logger.info(f"[tail] header={'OK' if header_ok else 'TIMEOUT'}: {''.join(header_buf[:2])[:120]!r}")

        # ── Step 3: stream matching lines until mode table ───────────────────────
        nc_count = 0
        idle     = 0.0
        try:
            while idle < 300:
                line = fh.readline()
                if line:
                    idle = 0.0
                    stripped = line.rstrip()
                    if NC_RE.search(stripped):
                        nc_count += 1
                        if nc_count <= 5 or nc_count % 100 == 0:
                            logger.info(f"[tail] nc#{nc_count}: {stripped[:80]}")
                        yield f"data: {stripped}\n\n"
                    elif DONE_RE.search(stripped):
                        logger.info(f"[tail] done after {nc_count} nc lines")
                        yield "data: __DONE__\n\n"
                        return
                else:
                    _time.sleep(0.04)
                    idle += 0.04
                    if _time.time() - ping_t > 4:
                        logger.info(f"[tail] still alive, {nc_count} nc lines, idle={idle:.1f}s")
                        yield ": ping\n\n"
                        ping_t = _time.time()
        finally:
            fh.close()
        yield "data: __DONE__\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/vina_visualization/vina_parse_log', methods=['POST'])
def vina_parse_log():
    """
    POST /vina_parse_log
    Reads vina_non_cache.log, finds best-pose [non_cache::eval pair] lines,
    builds a payload in the SAME shape as adj_3d_viz (atoms, bonds, weight_vector)
    so the frontend can call _render3D, _renderBar, _renderTable unchanged.

    Body: { "receptor_path": str (opt), "ligand_path": str (opt) }

    weight_vector.values[i]  = this_e for atom idx i  (sum of pair_e)
    weight_vector.top_indices = top 50 by |this_e|
    atoms[i].weight_raw      = this_e (raw kcal/mol)
    atoms[i].weight_norm     = normalised 0..1 for colour/size
    """
    import re as _re
    LOG_FILE = VINA_LOG
    try:
        data     = request.get_json(force=True) or {}
        lig_path = data.get('ligand_path',   '').strip()
        rec_path = data.get('receptor_path', '').strip()

        if not Path(LOG_FILE).is_file():
            return jsonify({'status': 'error',
                            'message': f'Log file not found: {LOG_FILE}'}), 404

        with open(LOG_FILE) as fh:
            log_text = fh.read()

        # ── Parse: extract ALL non_cache pairs, split by mode ────────────────
        # Strategy: find each "[non_cache::eval lig_atom=N]" block for mode 1
        # (the best pose is the FIRST complete block, ending at [non_cache::eval TOTAL])

        pair_re = _re.compile(
            r'\[non_cache::eval pair\]\s+lig_atom=(\d+)\(xs=(\d+)\)\s+'
            r'rec_atom=(\d+)\(xs=(\d+)\)\s+opt\(ri\+rj\)=([\d.]+)\s+'
            r'r=([\d.]+)\s+'
            r'(?:r_true=[\d.]+\s+)?'
            r's=r-opt=([-\d.]+)\s+pair_e=([-\d.]+)'
        )
        atom_re = _re.compile(
            r'\[non_cache::eval lig_atom=(\d+)\(xs=(\d+)\)\]\s+'
            r'this_e\(after curl\)=([-\d.]+)\s+out_of_bounds_penalty=([-\d.]+)'
        )
        total_re = _re.compile(r'\[non_cache::eval TOTAL\] e=([-\d.]+)')
        mode_re  = _re.compile(r'\[mode (\d+) energy assembly\]')
        aff_re   = _re.compile(r'^\s*1\s+([-\d.]+)', _re.MULTILINE)

        # Split log into sections by [mode N] markers
        # Best pose = first complete non_cache block (before or within mode 1)
        lines = log_text.splitlines()

        # Find the FIRST occurrence of [non_cache::eval TOTAL]
        # and take all pair/atom lines before it
        best_pairs:  dict[int, dict] = {}
        best_order:  list[int]       = []
        best_total:  float | None    = None
        best_mode:   int             = 1

        collecting = False
        for line in lines:
            # Start collecting when we see the first pair line
            m = pair_re.search(line)
            if m:
                collecting = True
                li  = int(m.group(1))
                lxs = int(m.group(2))
                ri  = int(m.group(3))
                rxs = int(m.group(4))
                opt,r,s,pe = float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8))
                if li not in best_pairs:
                    best_pairs[li] = {'xs':lxs,'pairs':[],'this_e':0.0}
                    best_order.append(li)
                best_pairs[li]['pairs'].append({'rec_idx':ri,'rec_xs':rxs,'opt':opt,'r':r,'s':s,'pair_e':pe})
                continue

            if collecting:
                m = atom_re.search(line)
                if m:
                    li = int(m.group(1))
                    if li not in best_pairs:
                        best_pairs[li] = {'xs':int(m.group(2)),'pairs':[],'this_e':0.0}
                        best_order.append(li)
                    best_pairs[li]['this_e'] = float(m.group(3))
                    continue

                m = total_re.search(line)
                if m:
                    best_total = float(m.group(1))
                    break    # stop after first complete block

        if not best_pairs:
            return jsonify({'status': 'error',
                            'message': 'No [non_cache::eval pair] lines found in log. '
                                       'Make sure the instrumented Vina binary was used.'}), 400

        # Best affinity from mode table
        aff_m = aff_re.search(log_text)
        best_affinity = float(aff_m.group(1)) if aff_m else None

        # ── Load ligand PDBQT for 3D coords + bonds ───────────────────────────
        lig_atoms_all = _load_pdbqt_all(lig_path) if lig_path else []
        rec_atoms_all = _load_rec_heavy(rec_path)  if rec_path  else []

        # Build this_e per atom idx
        # best_pairs keys = lig_atom indices in non_cache (= PDBQT line order incl H)
        this_e_map: dict[int, float] = {li: info['this_e'] for li, info in best_pairs.items()}

        # Map to HEAVY atoms only (matching PDBQT ordering)
        # non_cache skips H (t1 >= n), so this_e only exists for heavy atoms
        # lig_atoms_all[i].atype gives the type; we need i → this_e
        heavy_atoms_indexed = [
            (i, a) for i, a in enumerate(lig_atoms_all)
            if a['atype'] not in ('H', 'HD')
        ]

        # ── Compute this_e for each heavy atom (from log) ─────────────────────
        # Non_cache iterates ALL movable atoms (incl H) but H are skipped.
        # The lig_atom index in the log == i in VINA_FOR(i, m.num_movable_atoms())
        # which iterates lig_atoms_all in PDBQT order.
        # So this_e_map[i] directly corresponds to lig_atoms_all[i].

        # Build this_e_by_pdbqt_idx: for each heavy atom's pdbqt idx → this_e
        values_raw = []
        for pdbqt_i, atom in heavy_atoms_indexed:
            te = this_e_map.get(pdbqt_i, 0.0)
            values_raw.append(te)

        if not values_raw:
            return jsonify({'status': 'error', 'message': 'No heavy atoms found in ligand PDBQT'}), 400

        # Normalise: most negative (most favourable) → 1 (red = most important),
        # least negative / zero / positive → 0 (blue = least important).
        # Vina scores: lower is better, so we INVERT the usual direction.
        min_e = min(values_raw)
        max_e = max(values_raw)
        rng   = max_e - min_e if max_e != min_e else 1.0

        def _coolwarm_hex(t):
            t = max(0.0, min(1.0, t))
            if t < 0.5:
                s = t * 2
                r = int(59  + s * (221 - 59))
                g = int(76  + s * (220 - 76))
                b = int(192 + s * (220 - 192))
            else:
                s = (t - 0.5) * 2
                r = int(221 + s * (180 - 221))
                g = int(220 + s * (4   - 220))
                b = int(220 + s * (38  - 220))
            return f'#{r:02x}{g:02x}{b:02x}'

        # ── Build atoms list (weight_a shape) ─────────────────────────────────
        atoms_out = []
        symbols   = []
        values_normalised = []

        for local_i, (pdbqt_i, atom) in enumerate(heavy_atoms_indexed):
            te   = this_e_map.get(pdbqt_i, 0.0)
            # Invert: most negative this_e → norm=1 (most important/large/red),
            # near-zero or positive → norm=0 (least important/small/blue)
            norm = (max_e - te) / rng
            atoms_out.append({
                'idx':         local_i,          # sequential heavy-atom index
                'symbol':      atom['name'],      # e.g. "O1", "C6"
                'x':           round(atom['x'], 4),
                'y':           round(atom['y'], 4),
                'z':           round(atom['z'], 4),
                'weight_raw':  round(te, 5),      # this_e in kcal/mol
                'weight_norm': round(norm, 5),    # 0..1 for colour
                'color':       _coolwarm_hex(norm),
            })
            symbols.append(atom['name'])
            values_normalised.append(round(norm, 5))

        n_atoms = len(atoms_out)

        # top 50 by |this_e|
        top_50 = sorted(range(n_atoms), key=lambda i: abs(values_raw[i]), reverse=True)[:50]

        weight_vector = {
            'values':       [round(v, 5) for v in values_raw],
            'top_indices':  top_50,
            'atom_symbols': symbols,
            'n_atoms':      n_atoms,
        }

        # ── Build bonds (PDBQT has no bond info → infer from BRANCH records) ──
        # Simple fallback: no bonds (3D still looks good with atoms only)
        # For real bonds, parse BRANCH/ENDBRANCH + ROOT connectivity
        bonds = _infer_bonds_from_pdbqt(lig_path) if lig_path else []

        # ── pairs_by_lig_atom: local_heavy_idx → [{rec_idx, pair_e}] ────────────
        pairs_by_lig = {}
        for local_i, (pdbqt_i, _) in enumerate(heavy_atoms_indexed):
            if pdbqt_i in best_pairs:
                pairs_by_lig[local_i] = [
                    {'rec_idx': p['rec_idx'], 'pair_e': round(p['pair_e'], 5)}
                    for p in best_pairs[pdbqt_i]['pairs']
                ]

        # ── rec_atoms: receptor atom coords + labels for protein view ─────────
        rec_atoms_out = [
            {
                'idx':     a['all_atom_idx'],  # ← matches m.grid_atoms[] index in Vina log
                'name':    a['name'],
                'resname': a['resname'],
                'resseq':  a['resseq'],
                'x':       round(a['x'], 4),
                'y':       round(a['y'], 4),
                'z':       round(a['z'], 4),
                'atype':   a['atype'],
            }
            for a in rec_atoms_all
        ]

        return jsonify({
            'status':            'success',
            'atoms':             atoms_out,
            'bonds':             bonds,
            'weight_vector':     weight_vector,
            'total_e':           best_total,
            'best_affinity':     best_affinity,
            'best_mode':         best_mode,
            'n_lig_atoms':       n_atoms,
            'log_file':          LOG_FILE,
            'pairs_by_lig_atom': pairs_by_lig,
            'rec_atoms':         rec_atoms_out,
        })

    except Exception as exc:
        logger.error(f'vina_parse_log error: {exc}', exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


def _infer_bonds_from_pdbqt(path: str) -> list:
    """
    Infer bonds from PDBQT BRANCH/ENDBRANCH + ROOT structure.
    Returns list of {begin: local_heavy_idx, end: local_heavy_idx, order: 1}.
    Uses simple distance-based bond detection as fallback.
    """
    import math as _math
    atoms = [a for a in _load_pdbqt_all(path) if a['atype'] not in ('H','HD')]
    bonds = []
    n = len(atoms)
    # Distance threshold per element pair (simplified)
    for i in range(n):
        for j in range(i+1, n):
            dx = atoms[i]['x']-atoms[j]['x']
            dy = atoms[i]['y']-atoms[j]['y']
            dz = atoms[i]['z']-atoms[j]['z']
            d  = _math.sqrt(dx*dx+dy*dy+dz*dz)
            # Typical covalent bond lengths: C-C~1.54, C-N~1.47, C-O~1.43, C-F~1.35
            # Use generous 1.8 Å cutoff for all heavy-heavy bonds
            if d < 1.85:
                bonds.append({'begin': i, 'end': j, 'order': 1})
    return bonds