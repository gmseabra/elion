# =============================================================================
# routes/vina_dock_routes.py
# Core Vina docking endpoints:
#   vina_viz, vina_check_file, vina_dock (subprocess + SSE queue),
#   vina_dock_progress (SSE consumer), vina_tail_log (SSE log tail),
#   vina_parse_log
# =============================================================================

import os, re, json as _json, datetime, queue, threading as _threading
import math          # _term_breakdown() below calls math.exp but math was never
                     # imported — _build_response() raised NameError on the dock path.
from pathlib import Path
from flask import jsonify, request, Response, stream_with_context, current_app
from uiapp import app

from uiapp.routes.shared import (
    logger, VINA_BASE, VINA_BIN, VINA_LOG, _INPUT_ROUTES_YML,
    _vina_progress_q, ensure_vina_safe_pdbqt,
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


# AutoDock atom type → element symbol (for the PDB element column when converting).
_ADT_ELEM = {
    'A':'C', 'C':'C', 'N':'N', 'NA':'N', 'NS':'N', 'O':'O', 'OA':'O', 'OS':'O',
    'S':'S', 'SA':'S', 'H':'H', 'HD':'H', 'HS':'H', 'P':'P', 'B':'B', 'SE':'Se',
    'F':'F', 'CL':'Cl', 'BR':'Br', 'I':'I', 'MG':'Mg', 'ZN':'Zn', 'MN':'Mn',
    'CA':'Ca', 'FE':'Fe', 'CU':'Cu', 'NI':'Ni', 'CO':'Co', 'K':'K', 'NA_':'Na',
}


def _adt_to_elem(adt):
    a = (adt or '').strip().upper()
    return _ADT_ELEM.get(a, ((adt or '').strip().capitalize() or 'C')[:2])


def _pdbqt_best_pose_to_pdb(pdbqt_text, src_name='', conect=None):
    """
    Convert the BEST (first) pose of a Vina .pdbqt to a plain .pdb string.
    Vina writes poses best-first as MODEL 1..N; we keep only MODEL 1. For each
    ATOM/HETATM we keep the standard PDB columns 1-66 (through tempFactor), drop the
    PDBQT-only trailing partial-charge + AutoDock-type columns, and set the PDB
    element symbol (cols 77-78) from the AutoDock type. ROOT/BRANCH/TORSDOF topology
    records are dropped. A REMARK header carries the source name and the pose affinity.
    """
    def _atom(line):
        toks = line.rstrip().split()
        elem = _adt_to_elem(toks[-1] if toks else '')
        return line[:66].ljust(76) + f"{elem:>2}"

    atoms, affinity, model_seen = [], None, 0
    for line in pdbqt_text.splitlines():
        rec = line[:6].strip()
        if rec == 'MODEL':
            model_seen += 1
            if model_seen > 1:
                break                     # keep only the best (first) pose
            continue
        if rec == 'ENDMDL':
            break
        if line.startswith('REMARK VINA RESULT') and affinity is None:
            try: affinity = float(line.split()[3])
            except (IndexError, ValueError): pass
            continue
        if rec in ('ATOM', 'HETATM'):
            atoms.append(_atom(line))

    if not atoms:                          # single-pose file with no MODEL wrapper
        for line in pdbqt_text.splitlines():
            if line[:6].strip() in ('ATOM', 'HETATM'):
                atoms.append(_atom(line))
            elif line.startswith('REMARK VINA RESULT') and affinity is None:
                try: affinity = float(line.split()[3])
                except (IndexError, ValueError): pass

    header = []
    if src_name:
        header.append(f"REMARK    Best Vina pose exported from {src_name}")
    if affinity is not None:
        header.append(f"REMARK    VINA RESULT affinity (kcal/mol): {affinity:.3f}")
    # CONECT goes after the coordinates and before END, per the PDB spec.
    return '\n'.join(header + atoms + list(conect or []) + ['END']) + '\n'


def _conect_from_pdbqt(path):
    """CONECT records for the exported PDB, carrying bond ORDER.

    Without any CONECT the exported ligand opens in PyMOL / Chimera / Discovery
    Studio as a cloud of unbonded atoms: the residue is UNK, so their own residue
    templates have nothing to match and they fall back to their own guesswork.

    A PDB has no bond-order column. The convention every major viewer follows —
    and what Open Babel and PyMOL write — is to repeat the partner's serial once
    per bond order, so a double bond lists its partner twice and a triple three
    times. Orders come from the same perception the 3-D viewer uses, so an
    exported file and the on-screen structure agree.
    """
    atoms = _load_pdbqt_all(path)
    if not atoms:
        return []
    for a in atoms:
        a['elem'] = _adt_to_elem(a['atype'])
    heavy_pos = [i for i, a in enumerate(atoms) if a['elem'] != 'H']

    partners = {}

    def _add(i, j, order):
        si, sj = atoms[i]['serial'], atoms[j]['serial']
        partners.setdefault(si, []).extend([sj] * order)
        partners.setdefault(sj, []).extend([si] * order)

    for b in _infer_bonds_from_pdbqt(path):
        try:
            order = max(1, min(3, int(b.get('order', 1))))
        except (TypeError, ValueError):
            order = 1
        _add(heavy_pos[b['begin']], heavy_pos[b['end']], order)

    # Polar hydrogens (the only H a PDBQT keeps): bond each to its nearest heavy
    # atom, so an -OH or -NH exports as a real hydroxyl / amine rather than a
    # floating H.
    for i, a in enumerate(atoms):
        if a['elem'] != 'H':
            continue
        best, best_d = None, 1e9
        for j in heavy_pos:
            d = _xyz_dist(a, atoms[j])
            if d < best_d:
                best, best_d = j, d
        if best is not None and \
           best_d < _RCOV['H'] + _RCOV.get(atoms[best]['elem'], 0.77) + 0.45:
            _add(i, best, 1)

    out = []
    for serial in sorted(partners):
        lst = partners[serial]
        for k in range(0, len(lst), 4):          # max 4 partner fields per record
            out.append('CONECT' + f'{serial:>5}' +
                       ''.join(f'{t:>5}' for t in lst[k:k + 4]))
    return out


def _count_pdbqt_models(path):
    """How many MODEL records a PDBQT holds (Vina _out.pdbqt = one per docked mode).

    The viewer renders MODEL 1 only (see _load_pdbqt_all, which breaks at the first
    ENDMDL). Returning the count lets the UI state 'pose 1 of N' so the displayed
    coordinates can't be mistaken for a different mode's coordinates in the same file.
    Returns 1 for a single-pose / MODEL-less file.
    """
    try:
        with open(path) as fh:
            n = sum(1 for line in fh if line.startswith("MODEL"))
        return n if n > 0 else 1
    except Exception as e:
        logger.warning(f"_count_pdbqt_models({path}): {e}")
        return 1


def _load_pdbqt_all(path):
    """Load ALL atoms from PDBQT MODEL 1 (including H, for index alignment).

    Accepts BOTH record names. MGLTools' prepare_ligand4.py emits HETATM for a
    ligand; Open Babel emits ATOM. Vina parses either (see parse_pdbqt.cpp) and
    echoes the original record name back into <stem>_out.pdbqt, so a file that
    docks fine can still be invisible here. The old test was
    `line[:4] not in ("ATOM","HEAT")` — "HEAT" is a typo for "HETA", so every
    HETATM line was dropped, leaving zero atoms and a bogus
    "No heavy atoms found in ligand PDBQT" for MGLTools-prepared ligands.
    """
    atoms = []
    try:
        with open(path) as fh:
            in_model = has_model = False
            for line in fh:
                if line.startswith("MODEL"):  has_model = in_model = True; continue
                if line.startswith("ENDMDL"): break
                if not has_model: in_model = True
                if not in_model:  continue
                if line[:6].rstrip() not in ("ATOM", "HETATM"): continue
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

    Vina's rigid-receptor parser accepts ATOM *and* HETATM, so both must be
    counted here. The old "HEAT" typo dropped HETATM lines without incrementing
    all_atom_idx, which silently shifted every rec_idx after the first
    cofactor/metal/water and mislabelled contacts in the pair table.
    """
    atoms = []
    all_atom_idx = 0   # counts every ATOM/HETATM line including H
    try:
        with open(path) as fh:
            for line in fh:
                if line[:6].rstrip() not in ("ATOM", "HETATM"): continue
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


# =============================================================================
# Which vina binary can actually be run
# -----------------------------------------------------------------------------
# Used by BOTH the dock route and the rescore route, so a checkout that can dock
# can always rescore and vice versa.
#
# This exists because VINA_BIN — engines/vina/vina — is frequently not a runnable
# file. A fresh `git clone` of this repo leaves that path either absent or as a
# DIRECTORY, while the CPU build sits one level down in engines/vina/vina_cpu/.
# The two failures that produces are both misleading:
#
#   directory  -> exec raises EACCES, surfaced as
#                 "[Errno 13] Permission denied: .../engines/vina/vina"
#                 which reads as a permissions problem and sends you chmod'ing a
#                 path that was never wrong.
#   absent     -> "[Errno 2] No such file or directory: .../engines/vina/vina"
#                 which does not say that a perfectly good binary is sitting in
#                 the sibling directory.
#
# is_file() + X_OK rejects both up front, and the candidate list travels back in
# the error so the answer is in the response rather than in this comment.
#
# Only CPU builds are candidates. engines/vina/ also ships AutoDock-Vina-GPU-2-1,
# but that binary takes a different command line (config file, no --center_x /
# --size_x), so falling back to it would turn a clear "no binary" error into a
# confusing argument-parsing one.
# =============================================================================
def _resolve_vina_bin(vcfg):
    """(path_or_None, [what was tried and why each failed]).

    Order: the configured vina.bin (already absolutised by load_vina_config),
    then the module constant, then the CPU build's conventional home.
    """
    tried, seen = [], set()
    for cand in (vcfg.get('bin'), VINA_BIN, os.path.join(VINA_BASE, 'vina_cpu', 'vina')):
        if not cand or cand in seen:
            continue
        seen.add(cand)
        p = Path(cand)
        if p.is_file() and os.access(str(p), os.X_OK):
            return str(p), tried
        tried.append('%s (%s)' % (cand, 'is a directory' if p.is_dir()
                                  else 'not executable' if p.exists() else 'missing'))
    return None, tried


def _vina_bin_error(tried):
    """One message for both callers, naming the fix rather than just the symptom."""
    msg = ('No runnable AutoDock Vina binary. Tried, in order: ' + '; '.join(tried) + '. ')
    if any('not executable' in x for x in tried):
        msg += ('One candidate exists but has no execute bit — a fresh clone can drop '
                'it; `chmod +x` that path. ')
    msg += ('The CPU build normally lives in engines/vina/vina_cpu/vina. To point '
            'somewhere else, set vina.bin in config/input_routes.yml.')
    return msg


@app.route('/vina_visualization/vina_dock', methods=['POST'])
def vina_dock():
    """
    POST /vina_dock
    Runs AutoDock Vina, streams stdout progress via SSE, writes a log file,
    parses the non_cache::eval output, and returns structured score data.

    Body: {
      "receptor_path": str, "ligand_path": str,
      # optional docking box (from the UI Box ctr / len inputs); each falls back
      # to input_routes.yml then a built-in default if omitted:
      "center_x": float, "center_y": float, "center_z": float,
      "size_x": float, "size_y": float, "size_z": float
    }
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

        # ── Self-healing pre-flight ───────────────────────────────────────────
        # Vina rejects TITLE/REMARK/etc. in either input and returns best=None.
        # These paths can arrive already-converted from anywhere (the modal
        # converter, an older route, Glide/MGLTools, a hand-placed file), so
        # clean both in place right before docking no matter their origin.
        # No-op if already clean.
        try:
            rec_clean = ensure_vina_safe_pdbqt(rec_path, 'receptor')
            lig_clean = ensure_vina_safe_pdbqt(lig_path, 'ligand')
            for tag, r in (('receptor', rec_clean), ('ligand', lig_clean)):
                if r['was_dirty']:
                    logger.info('vina_dock: auto-cleaned %s PDBQT (%d illegal '
                                'line(s) stripped) before docking: %s',
                                tag, r['stripped_lines'], r['path'])
        except Exception as _san_exc:
            logger.warning('vina_dock: PDBQT pre-flight sanitize failed '
                           '(continuing): %s', _san_exc)

        LOG_FILE  = VINA_LOG

        # Output alongside the ligand file, named after it
        lig_stem  = Path(lig_path).stem
        OUT_LIG   = str(Path(lig_path).parent / f"{lig_stem}_out.pdbqt")

        # ── All hyperparameters from input_routes.yml via app.config["VINA"] ──────
        # Loaded by app.py at startup — zero hardcoding here.
        _vcfg = current_app.config.get("VINA", {})
        _cpu  = _vcfg.get("cpu") or os.cpu_count() or 4

        # Docking box: prefer the values POSTed from the UI (the "Box ctr / len"
        # inputs in the Visualize row → center_x/y/z + size_x/y/z), then fall back
        # to the input_routes.yml config, then to the built-in default. This is what
        # lets the editable grid box actually drive the vina --center/--size flags;
        # previously the box was read only from _vcfg, so the UI value was ignored.
        def _box(key, default):
            v = data.get(key, None)
            if v is None or v == '':
                v = _vcfg.get(key, default)
            try:
                return str(float(v))
            except (TypeError, ValueError):
                return str(default)

        # Resolve before building the command: exec'ing VINA_BIN blind is what
        # produced "[Errno 13] Permission denied" / "[Errno 2] No such file or
        # directory" on a fresh clone, neither of which points at the fix.
        _vbin, _tried = _resolve_vina_bin(_vcfg)
        if not _vbin:
            logger.error('vina_dock: %s', _vina_bin_error(_tried))
            return jsonify({'status': 'error', 'message': _vina_bin_error(_tried)}), 500

        cmd = [
            _vbin,
            '--receptor',       rec_path,
            '--ligand',         lig_path,
            '--center_x',       _box('center_x', -25.7),
            '--center_y',       _box('center_y',   0.22),
            '--center_z',       _box('center_z',  28.39),
            '--size_x',         _box('size_x',      20),
            '--size_y',         _box('size_y',      20),
            '--size_z',         _box('size_z',      20),
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


@app.route('/vina_visualization/vina_export_pdb', methods=['POST'])
def vina_export_pdb():
    """
    POST /vina_export_pdb
    Convert the best-pose Vina output to PDB and return it for the browser to download.

    Body: { "ligand_path": str }   → reads <ligand_stem>_out.pdbqt next to the ligand
       or { "out_path": str }      → explicit path to the *_out.pdbqt to convert
    Returns: { ok: true, filename, pdb_text }  |  { ok: false, err }

    The *_out.pdbqt path mirrors what vina_dock writes (OUT_LIG). Best (first) pose only.
    """
    try:
        data     = request.get_json(force=True) or {}
        out_path = (data.get('out_path') or '').strip()
        if not out_path:
            lig = (data.get('ligand_path') or '').strip()
            if not lig:
                return jsonify({'ok': False, 'err': 'ligand_path or out_path is required'}), 400
            # If the ligand field already points AT a docked output, the naive
            # stem + '_out.pdbqt' derives <stem>_out_out.pdbqt — a path that never
            # exists — so Export PDB 404'd with "run a dock first" even though the
            # dock had just succeeded. vina_parse_log already guards this; the
            # export did not.
            _stem = Path(lig).stem
            out_path = (lig if _stem.endswith('_out')
                        else str(Path(lig).parent / f"{_stem}_out.pdbqt"))

        p = Path(out_path)
        if not p.is_file():
            return jsonify({'ok': False, 'err': f'output pose not found: {out_path} '
                                                 '(run a dock first)'}), 404

        conect   = _conect_from_pdbqt(str(p))
        pdb_text = _pdbqt_best_pose_to_pdb(p.read_text(errors='ignore'),
                                           src_name=p.name, conect=conect)
        filename = p.stem + '.pdb'          # e.g. HJL-1_out.pdb
        return jsonify({'ok': True, 'filename': filename, 'pdb_text': pdb_text,
                        'source': str(p), 'n_conect': len(conect)}), 200

    except Exception as exc:
        logger.warning(f'vina_export_pdb error: {exc}')
        return jsonify({'ok': False, 'err': str(exc)}), 200


# =============================================================================
# Docked results — read straight off the `*_out.pdbqt` files. No database.
#
# A Vina output file already carries everything a results view needs, and it
# carries it per pose:
#
#     MODEL 1
#     REMARK VINA RESULT:    -9.871      0.000      0.000    <- affinity, rmsd l.b./u.b.
#     REMARK INTER + INTRA:         -16.439
#     REMARK INTER:                 -14.198
#     REMARK INTRA:                  -2.240
#     REMARK UNBOUND:                -2.240
#     ...
#     TORSDOF 8
#     ENDMDL
#
# So the history is the directory listing. Nothing has to be recorded at dock
# time and nothing can drift out of sync with the files, which is the failure
# mode a side-car index would introduce.
#
# The one thing the file does NOT record is which receptor it was docked into.
# That is inferred, and the inference is reported rather than assumed:
#   * geometric — the pose has to sit inside the search box that produced it, so
#     the mode-1 centroid is tested against every configured protein's box. A
#     single containing box is strong evidence; several means the boxes overlap
#     and the answer is genuinely ambiguous.
#   * by name  — a configured protein whose `default_ligand` stem matches this
#     file's stem.
# Both are returned, along with `receptor_confidence`, so the UI can say "8P0M
# (box + name)" or "ambiguous" instead of quietly picking one.
# =============================================================================

_VINA_REMARK_KEYS = (
    ('INTER + INTRA', 'inter_intra'),
    ('INTER',         'inter'),
    ('INTRA',         'intra'),
    ('UNBOUND',       'unbound'),
)


def _parse_out_pdbqt(path):
    """Parse one Vina `*_out.pdbqt`. Returns a dict, or None if it has no poses.

    Streams the file: these can run to tens of thousands of lines for a big
    num_modes and nothing here needs the whole thing in memory.
    """
    modes, cur = [], None
    n_atoms_first, torsdof = 0, None
    sx = sy = sz = 0.0
    n_xyz = 0
    elements = {}
    try:
        with open(path, 'r', errors='replace') as fh:
            for line in fh:
                if line.startswith('MODEL'):
                    cur = {'mode': len(modes) + 1}
                    modes.append(cur)
                    continue
                if line.startswith('REMARK VINA RESULT:'):
                    parts = line.split(':', 1)[1].split()
                    try:
                        if cur is None:
                            cur = {'mode': 1}
                            modes.append(cur)
                        cur['affinity'] = float(parts[0])
                        cur['rmsd_lb'] = float(parts[1])
                        cur['rmsd_ub'] = float(parts[2])
                    except (IndexError, ValueError):
                        pass
                    continue
                if line.startswith('REMARK ') and cur is not None and ':' in line:
                    label, _, rest = line[7:].partition(':')
                    label = label.strip()
                    for key, field in _VINA_REMARK_KEYS:
                        if label == key:
                            try:
                                cur[field] = float(rest.split()[0])
                            except (IndexError, ValueError):
                                pass
                            break
                    continue
                if line.startswith('TORSDOF'):
                    try:
                        torsdof = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                    continue
                if line[:6] in ('ATOM  ', 'HETATM') and len(modes) <= 1:
                    # Geometry from pose 1 only: every mode has the same atoms, and
                    # the centroid is what the box test needs.
                    n_atoms_first += 1
                    try:
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        sx += x; sy += y; sz += z; n_xyz += 1
                    except (ValueError, IndexError):
                        pass
                    # Column 78-79 is the AutoDock TYPE, not the element: aromatic
                    # carbon is `A`, carbonyl oxygen is `OA`, polar hydrogen `HD`.
                    # Title-casing it gives you elements that do not exist ("Oa",
                    # "Hd") and an H filter that never fires. The module already
                    # has the mapping — use it.
                    el = _adt_to_elem(line[77:79].strip() or line[12:16].strip()[:2])
                    if el != 'H':
                        elements[el] = elements.get(el, 0) + 1
    except OSError as e:
        logger.warning('[vina] could not read %s: %s', path, e)
        return None

    modes = [m for m in modes if 'affinity' in m]
    if not modes:
        return None

    st = os.stat(path)
    centroid = [round(sx / n_xyz, 3), round(sy / n_xyz, 3), round(sz / n_xyz, 3)] if n_xyz else None
    p = Path(path)
    stem = p.name[:-len('_out.pdbqt')] if p.name.endswith('_out.pdbqt') else p.stem
    src = p.parent / (stem + '.pdbqt')
    return {
        'name': p.name,
        'stem': stem,
        'path': str(p),
        'source_ligand': str(src) if src.is_file() else '',
        'best': modes[0].get('affinity'),
        'n_modes': len(modes),
        'modes': modes,
        'torsdof': torsdof,
        'n_heavy': sum(elements.values()),
        'n_atoms': n_atoms_first,
        'elements': elements,
        'centroid': centroid,
        'size': st.st_size,
        'mtime': datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'),
        'mtime_epoch': st.st_mtime,
    }


def _attribute_receptor(centroid, stem, proteins):
    """Which configured protein does this pose belong to? Returns
    (receptor_id, receptor_path, confidence, [candidate ids], box).

    A docked pose cannot lie outside the box it was searched in, so box
    containment is real evidence rather than a guess — but overlapping boxes make
    it non-unique, and that is reported rather than resolved by picking the first.

    `box` is the matched protein's search box, carried back so the client can
    refill the box inputs without re-deriving it: this function already had to
    read it to run the containment test.
    """
    in_box, by_name = [], []
    for pr in proteins or []:
        pid = pr.get('id') or ''
        try:
            c = [float(pr.get('center_x')), float(pr.get('center_y')), float(pr.get('center_z'))]
            s = [float(pr.get('size_x', 20)), float(pr.get('size_y', 20)), float(pr.get('size_z', 20))]
        except (TypeError, ValueError):
            c = s = None
        if centroid and c:
            # half-size plus a little: the centroid of a pose that hugs one face
            # of the box still sits inside, but rounding should not exclude it.
            if all(abs(centroid[k] - c[k]) <= s[k] / 2.0 + 1e-6 for k in range(3)):
                in_box.append(pid)
        dl = pr.get('default_ligand') or ''
        if dl and Path(dl).stem == stem:
            by_name.append(pid)

    both = [p for p in in_box if p in by_name]
    if both:
        pick, conf = both[0], 'box + name'
    elif len(in_box) == 1:
        pick, conf = in_box[0], 'box'
    elif by_name:
        pick, conf = by_name[0], 'name'
    elif len(in_box) > 1:
        pick, conf = in_box[0], 'ambiguous (%d boxes contain this pose)' % len(in_box)
    else:
        return '', '', 'unknown', [], None

    path, box = '', None
    for pr in (proteins or []):
        if pr.get('id') == pick:
            path = pr.get('default_receptor') or ''
            try:
                box = {'center_x': float(pr.get('center_x')), 'center_y': float(pr.get('center_y')),
                       'center_z': float(pr.get('center_z')), 'size_x': float(pr.get('size_x', 20)),
                       'size_y': float(pr.get('size_y', 20)), 'size_z': float(pr.get('size_z', 20))}
            except (TypeError, ValueError):
                box = None
            break
    return pick, path, conf, sorted(set(in_box) | set(by_name)), box


# Records Vina's ligand parser accepts. A *_out.pdbqt additionally carries
# MODEL/ENDMDL and REMARK lines, and Vina rejects all of them on input
# ("Unknown or inappropriate tag" / "Unexpected multi-MODEL tag"), so a pose has
# to be split back out before it can be fed to the binary again.
_VINA_LIG_RECORDS = ('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF',
                     'ATOM', 'HETATM')

# One process, one log file. Serialise the writers so a rescore can never
# interleave with (or truncate) a dock that is still streaming into it.
_VINA_LOG_LOCK = _threading.Lock()


def _split_model1(src, dest):
    """Write MODEL 1 of `src` to `dest` as a Vina-loadable ligand.

    Never modifies `src`: the *_out.pdbqt is the record of the dock, and its
    REMARK VINA RESULT lines are what the results list reads. (ensure_vina_safe_pdbqt
    would have stripped them in place — hence this separate, copy-only path.)

    Returns the number of atom records written.
    """
    n = 0
    with open(src, 'r', errors='replace') as fh, open(dest, 'w') as out:
        in_model = seen_model = False
        for line in fh:
            if line.startswith('MODEL'):
                if seen_model:
                    break                      # MODEL 2 — stop, we only want the best
                seen_model = in_model = True
                continue
            if line.startswith('ENDMDL'):
                break
            if not seen_model:
                in_model = True                # single-pose file, no MODEL wrapper
            if not in_model:
                continue
            if not line.startswith(_VINA_LIG_RECORDS):
                continue                       # REMARK, TITLE, USER, blank …
            out.write(line)
            if line.startswith(('ATOM', 'HETATM')):
                n += 1
    return n


def _mode_table(modes):
    """Reproduce the mode table Vina prints, from the file's own REMARK lines.

    Not fabricated data — `REMARK VINA RESULT` *is* what Vina wrote for this
    file. Restating it at the top of a rescored log matters for two reasons:
    /vina_parse_log reads mode 1 from here for the score card, and
    _live_log_signature() reads it to decide the log already belongs to a given
    result — which is what lets a second Visualize click skip the rescore.
    """
    rows = ['mode |   affinity | dist from best mode',
            '     | (kcal/mol) | rmsd l.b.| rmsd u.b.',
            '-----+------------+----------+----------']
    for m in modes:
        rows.append('%4d%13.4f%11.4f%11.4f' % (
            m.get('mode', 0), m.get('affinity', 0.0) or 0.0,
            m.get('rmsd_lb', 0.0) or 0.0, m.get('rmsd_ub', 0.0) or 0.0))
    return '\n'.join(rows)


def _vina_env():
    """Vina's own runtime env: the instrumented binary links boost from the
    conda prefix, which is not on the default loader path."""
    lib = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib')
    env = os.environ.copy()
    if lib and lib not in env.get('LD_LIBRARY_PATH', ''):
        env['LD_LIBRARY_PATH'] = (lib + ':' + env.get('LD_LIBRARY_PATH', '')).strip(':')
    return env


# =============================================================================
# POST /vina_visualization/dock_result_rescore
# -----------------------------------------------------------------------------
# Make an OLD docked result visualizable.
#
# The per-atom energy view is rebuilt by /vina_parse_log from vina_non_cache.log,
# and there is exactly one of those — every dock truncates and rewrites it. So
# out of the box only the most recent dock could ever be visualized, and pointing
# the viewer at an older result would paint this molecule's coordinates with the
# previous molecule's energies while looking entirely plausible.
#
# Rather than disable the older ones, regenerate the log for the pose being
# asked for: run the same instrumented binary in --score_only mode over MODEL 1
# of its *_out.pdbqt. No search, no randomness — it re-evaluates coordinates
# Vina already chose, so it reproduces that dock's own numbers (measured on
# Structures_for_Vina_originalprotonated_out.pdbqt: score_only -5.445 / inter
# -8.310 / intra -1.535 against REMARK -5.446 / -8.311 / -1.535) and emits the
# identical [non_cache::eval pair] / [atom_map] instrumentation a dock does.
# =============================================================================
@app.route('/vina_visualization/dock_result_rescore', methods=['POST'])
def vina_dock_result_rescore():
    import subprocess, tempfile, shutil
    try:
        data = request.get_json(force=True) or {}
        vcfg = current_app.config.get('VINA', {}) or {}

        # ── resolve the result, inside the configured directory only ─────────
        name = (data.get('name') or '').strip()
        raw_dir = (data.get('dir') or '').strip() or (vcfg.get('ligand_pdbqt_dir') or '').strip()
        if not raw_dir:
            return jsonify({'ok': False, 'err': 'vina.ligand_pdbqt_dir is not configured'}), 200
        d = Path(raw_dir).expanduser().resolve()
        if not name:
            return jsonify({'ok': False, 'err': 'name is required'}), 200
        # Basename only, then re-check containment: a name is a value from the
        # results list, but it arrives over HTTP and "../../etc/passwd" is a
        # perfectly valid string.
        out_path = (d / Path(name).name).resolve()
        if out_path.parent != d or not out_path.is_file():
            return jsonify({'ok': False, 'err': 'no such result: %s' % name}), 200

        rec = _parse_out_pdbqt(str(out_path))
        if not rec:
            return jsonify({'ok': False, 'err': '%s holds no scored pose' % out_path.name}), 200

        # ── receptor ─────────────────────────────────────────────────────────
        # The client already has one from the scan of this same file; take it when
        # it holds up, and only re-derive when it does not. _attribute_receptor is
        # the single source either way, so the two paths cannot disagree.
        rpath = (data.get('receptor_path') or '').strip()
        if not rpath or not Path(rpath).is_file():
            _, rpath, _, _, _ = _attribute_receptor(
                rec['centroid'], rec['stem'], vcfg.get('proteins') or [])
        if not rpath or not Path(rpath).is_file():
            return jsonify({'ok': False, 'err':
                            'no receptor could be identified for %s — open the card and '
                            'check "inferred from"' % out_path.name}), 200

        vbin, tried = _resolve_vina_bin(vcfg)
        if not vbin:
            return jsonify({'ok': False, 'err': 'no runnable vina binary',
                            'detail': _vina_bin_error(tried)}), 200

        if not _VINA_LOG_LOCK.acquire(blocking=False):
            return jsonify({'ok': False, 'err': 'a dock or rescore is already writing '
                                                'the log — try again in a moment'}), 200
        tmp = None
        try:
            tmp = tempfile.mkdtemp(prefix='vina-rescore-')
            pose = os.path.join(tmp, 'pose.pdbqt')
            n_at = _split_model1(str(out_path), pose)
            if n_at == 0:
                return jsonify({'ok': False, 'err': 'MODEL 1 of %s has no atoms'
                                                    % out_path.name}), 200

            # --autobox, not the search box. score_only does not search, so the box
            # has no bearing on the pair energies — its only role is the
            # out-of-bounds penalty, and Vina hard-errors ("The ligand is outside
            # the grid box") if any atom falls outside. That is not an edge case:
            # docking constrains the ligand's CENTRE to the box, so a pose that
            # docked perfectly well can still poke out at the edges, and roughly
            # half the results here do. --autobox sizes the box to the pose, so the
            # penalty is zero and every result can be rescored. Verified to give
            # byte-identical energies on a pose that fits both ways
            # (Structures_for_Vina_originalprotonated: -5.445 / -8.310 / -1.535
            # either way), and to be the only way the ones that do not fit run at all
            # (G001a_IL-6_gp130: -6.641, exactly its REMARK).
            cmd = [vbin, '--receptor', str(rpath), '--ligand', pose,
                   '--score_only', '--autobox']

            logger.info('[vina] rescore: %s', ' '.join(cmd))
            started = datetime.datetime.now()
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, env=_vina_env(),
                                      timeout=float(vcfg.get('rescore_timeout', 300)))
            except subprocess.TimeoutExpired:
                return jsonify({'ok': False, 'err': 'vina --score_only timed out'}), 200
            body = proc.stdout.decode('utf-8', 'replace')

            if proc.returncode != 0 or '[non_cache::eval pair]' not in body:
                # Surface vina's own complaint rather than a generic failure —
                # it is usually a specific, fixable one (bad atom type, unreadable
                # receptor, a binary built without the instrumentation).
                tail = '\n'.join(l for l in body.splitlines()
                                 if l.strip() and not l.startswith('#'))[-600:]
                return jsonify({'ok': False, 'rc': proc.returncode, 'err':
                                ('vina --score_only produced no per-atom lines'
                                 if proc.returncode == 0 else
                                 'vina --score_only failed (rc=%d)' % proc.returncode),
                                'detail': tail}), 200

            header = ('# Vina rescore (--score_only) %s\n'
                      '# POSE: %s  (MODEL 1 of %d, %d atoms)\n'
                      '# CMD: %s\n\n%s\n\n'
                      % (started.strftime('%Y-%m-%d %H:%M:%S'), out_path,
                         len(rec['modes']), n_at, ' '.join(cmd),
                         _mode_table(rec['modes'])))
            with open(VINA_LOG, 'w') as lf:
                lf.write(header)
                lf.write(body)

            # score_only recomputes the affinity; the REMARK is what the dock
            # recorded. They describe the same pose under the same function, so a
            # disagreement means the file and the receptor no longer belong
            # together — worth saying, not worth refusing over.
            m = re.search(r'Estimated Free Energy of Binding\s*:\s*([-\d.]+)', body)
            rescored = float(m.group(1)) if m else None
            drift = (None if rescored is None or rec['best'] is None
                     else round(rescored - rec['best'], 4))
            if drift is not None and abs(drift) > 0.05:
                logger.warning('[vina] rescore of %s gave %.3f but its REMARK says %.3f '
                               '(%.3f apart) — receptor mismatch?', out_path.name,
                               rescored, rec['best'], drift)

            return jsonify({
                'ok': True, 'log': str(VINA_LOG), 'ligand': str(out_path),
                'receptor': str(rpath), 'best': rec['best'], 'rescored': rescored,
                'drift': drift, 'atoms': n_at,
                'pairs': body.count('[non_cache::eval pair]'),
                'seconds': round((datetime.datetime.now() - started).total_seconds(), 1),
            }), 200
        finally:
            _VINA_LOG_LOCK.release()
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        logger.warning('[vina] dock_result_rescore: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200


def _live_log_signature():
    """(best_affinity, has_pair_lines) for the ONE shared vina_non_cache.log.

    Why this exists: the per-atom energy view is rebuilt by /vina_parse_log from
    that log, and there is exactly one of it — every dock truncates and rewrites
    it. So the decomposition can only ever describe the MOST RECENT run. Pointing
    the viewer at an older result would render this molecule's coordinates against
    the previous molecule's energies, and nothing about the picture would look
    wrong.

    The log names neither its ligand nor a checksum, but it does carry the mode
    table, so mode 1's affinity identifies the run well enough to refuse a
    mismatch. Two results that scored identically are genuinely indistinguishable
    from it — both are offered, and the cost of picking the wrong one is bounded
    because they scored the same anyway.
    """
    try:
        p = Path(VINA_LOG)
        if not p.is_file():
            return None, False
        txt = p.read_text(errors='replace')
        m = re.search(r'^\s*1\s+([-\d.]+)', txt, re.MULTILINE)
        return (float(m.group(1)) if m else None), ('[non_cache::eval pair]' in txt)
    except Exception as e:
        logger.warning('[vina] could not read the live log: %s', e)
        return None, False


# =============================================================================
# GET /vina_visualization/dock_results?dir=<optional override>
#
# Every `*_out.pdbqt` in the ligand directory, newest first, fully parsed.
# One request serves the whole panel — these files are small and there are tens
# of them, not thousands.
#
# Returns: { ok, dir, results:[...], n } | { ok:false, err, results:[] }
# =============================================================================
@app.route('/vina_visualization/dock_results', methods=['GET'])
def vina_dock_results():
    try:
        vcfg = current_app.config.get('VINA', {}) or {}
        raw = (request.args.get('dir') or '').strip() or (vcfg.get('ligand_pdbqt_dir') or '').strip()
        if not raw:
            return jsonify({'ok': False, 'results': [],
                            'err': 'vina.ligand_pdbqt_dir is not configured'}), 200
        d = Path(raw).expanduser()
        if not d.is_dir():
            return jsonify({'ok': False, 'results': [], 'dir': str(d),
                            'err': 'not a directory: %s' % d}), 200

        proteins = vcfg.get('proteins') or []
        log_best, log_has_pairs = _live_log_signature()      # read once, not per file
        out, skipped = [], []
        for fp in sorted(d.iterdir()):
            if not fp.is_file() or not fp.name.endswith('_out.pdbqt'):
                continue
            rec = _parse_out_pdbqt(str(fp))
            if not rec:
                skipped.append(fp.name)          # exists but holds no scored pose
                continue
            rid, rpath, conf, cands, box = _attribute_receptor(rec['centroid'], rec['stem'], proteins)
            rec['receptor_id'] = rid
            rec['receptor'] = rpath
            rec['receptor_confidence'] = conf
            rec['receptor_candidates'] = cands
            rec['box'] = box
            # Can the per-atom decomposition be shown for this one? Only if the
            # single shared log is the one this file came from.
            rec['log_matches'] = bool(
                log_has_pairs and log_best is not None and rec['best'] is not None
                and abs(log_best - rec['best']) < 0.005)
            out.append(rec)

        out.sort(key=lambda r: r['mtime_epoch'], reverse=True)
        return jsonify({'ok': True, 'dir': str(d), 'results': out, 'n': len(out),
                        'skipped': skipped, 'log': str(VINA_LOG),
                        'log_best': log_best, 'log_usable': log_has_pairs}), 200
    except Exception as e:
        logger.warning('[vina] dock_results: %s', e)
        return jsonify({'ok': False, 'results': [], 'err': str(e)}), 200


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
            r'(?:\s+lig_xyz=\(([-\d.]+),([-\d.]+),([-\d.]+)\))?'   # groups 9,10,11: log's ligand coord
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

        # ── Vina index → PDBQT file index ────────────────────────────────────────
        # Vina's m.atoms order follows the PDBQT torsion tree (ROOT, then BRANCHes),
        # which is NOT the file's line order. Measured on this ligand, 31 of 41 atoms
        # differ — e.g. Vina's lig_atom=13 is C19 (file line 19) while C16 (file
        # line 14) is Vina's lig_atom=15. Keying this_e / pairs by the raw log index
        # therefore attributes one atom's contacts to a different atom.
        # The instrumented non_cache.cpp emits an [atom_map] block pairing each Vina
        # index with the original PDBQT line, so use it when present.
        # The instrumented non_cache.cpp echoes the ligand's ORIGINAL PDBQT record
        # name, so an MGLTools ligand prints pdbqt_line="HETATM   1 ...". Anchoring
        # on "ATOM" alone left vina_to_file empty for those files and silently fell
        # back to identity mapping — no error, just per-atom energies painted onto
        # the wrong atoms (31 of 41 differ on this ligand).
        atommap_re = _re.compile(
            r'\[atom_map\]\s+lig_atom=(\d+)\s+pdbqt_line="(?:ATOM|HETATM)\s*(\d+)')
        vina_to_file: dict[int, int] = {}
        for line in lines:
            m = atommap_re.search(line)
            if m:
                vina_to_file[int(m.group(1))] = int(m.group(2)) - 1   # serial is 1-based
        file_to_vina: dict[int, int] = {f: v for v, f in vina_to_file.items()}
        if vina_to_file:
            n_bad = sum(1 for v, f in vina_to_file.items() if v != f)
            logger.info('vina_parse_log: [atom_map] found — %d atoms, %d with '
                        'lig_atom != file index', len(vina_to_file), n_bad)
        else:
            logger.warning('vina_parse_log: no [atom_map] block in log — falling back to '
                           'identity mapping (lig_atom == file index). Rebuild Vina with '
                           'the instrumented non_cache.cpp so contacts are attributed to '
                           'the correct atoms.')

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
                # The log prints the ligand atom's coordinate (in Vina's scoring frame)
                # on every pair line for that atom; capture it once. Lets the client
                # optionally plot ligand atoms in the frame Vina actually scored.
                if m.group(9) is not None and 'log_xyz' not in best_pairs[li]:
                    best_pairs[li]['log_xyz'] = [float(m.group(9)), float(m.group(10)), float(m.group(11))]
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

        # ── Load the REAL docked pose Vina wrote to disk (authoritative) ───────
        # vina_dock() always writes <ligand_stem>_out.pdbqt next to the ligand (OUT_LIG).
        # Its MODEL 1 is the best pose in the coordinate frame Vina actually scored —
        # ground truth, vs. reconstructing an approximate frame from log text below.
        # Same atom count/order as the exported ligand (Vina repositions atoms, it
        # doesn't add/remove/reorder them), so we zip by index.
        docked_atoms_all = []
        docked_pose_path = None
        if lig_path:
            # vina_dock() writes <ligand_stem>_out.pdbqt next to the ligand. If the user
            # already pointed ligPath AT that _out file, naive stem+'_out.pdbqt' would
            # derive <stem>_out_out.pdbqt — a file that never exists — silently leaving
            # has_docked_pose false. Treat an existing *_out.pdbqt as its own docked pose.
            _stem = Path(lig_path).stem
            if _stem.endswith('_out'):
                out_path = str(lig_path)
            else:
                out_path = str(Path(lig_path).parent / (_stem + '_out.pdbqt'))
            if Path(out_path).is_file():
                try:
                    candidate = _load_pdbqt_all(out_path)
                    if len(candidate) == len(lig_atoms_all):
                        docked_atoms_all = candidate
                        docked_pose_path = out_path
                    else:
                        logger.warning('vina_parse_log: %s atom count (%d) != ligand (%d) — '
                                       'skipping docked-pose coordinates', out_path,
                                       len(candidate), len(lig_atoms_all))
                except Exception as _de:
                    logger.warning('vina_parse_log: could not read docked pose %s: %s', out_path, _de)

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
        # The log's lig_atom index is Vina's internal m.atoms index, which follows the
        # PDBQT torsion tree rather than file order. file_to_vina translates; it falls
        # back to identity for logs predating the [atom_map] instrumentation.
        def _vina_idx(pdbqt_i):
            return file_to_vina.get(pdbqt_i, pdbqt_i)

        # Build this_e_by_pdbqt_idx: for each heavy atom's pdbqt idx → this_e
        values_raw = []
        for pdbqt_i, atom in heavy_atoms_indexed:
            te = this_e_map.get(_vina_idx(pdbqt_i), 0.0)
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
            te   = this_e_map.get(_vina_idx(pdbqt_i), 0.0)
            # Invert: most negative this_e → norm=1 (most important/large/red),
            # near-zero or positive → norm=0 (least important/small/blue)
            norm = (max_e - te) / rng
            # Vina's own scoring-frame coordinate for this atom, if the log printed it.
            # Differs from x/y/z (exported-pose PDBQT) when the two frames disagree.
            log_xyz = best_pairs.get(_vina_idx(pdbqt_i), {}).get('log_xyz')
            # The REAL docked pose from <ligand_stem>_out.pdbqt (authoritative — see above).
            docked = docked_atoms_all[pdbqt_i] if pdbqt_i < len(docked_atoms_all) else None
            atoms_out.append({
                'idx':         local_i,          # sequential heavy-atom index
                'symbol':      atom['name'],      # e.g. "O1", "C6"
                'x':           round(atom['x'], 4),
                'y':           round(atom['y'], 4),
                'z':           round(atom['z'], 4),
                'log_x':       (round(log_xyz[0], 4) if log_xyz else None),
                'log_y':       (round(log_xyz[1], 4) if log_xyz else None),
                'log_z':       (round(log_xyz[2], 4) if log_xyz else None),
                'docked_x':    (round(docked['x'], 4) if docked else None),
                'docked_y':    (round(docked['y'], 4) if docked else None),
                'docked_z':    (round(docked['z'], 4) if docked else None),
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

        # ── pairs_by_lig_atom: local_heavy_idx → [{rec_idx, pair_e, rec_xs, lig_xs, r, s}]
        # rec_xs/lig_xs/r/s are carried so the client can draw the atom–atom distance
        # and reconstruct the Vina pair_e breakdown on demand (see _voxelPairClick).
        pairs_by_lig = {}
        for local_i, (pdbqt_i, _) in enumerate(heavy_atoms_indexed):
            vi = _vina_idx(pdbqt_i)
            if vi in best_pairs:
                lig_xs = best_pairs[vi].get('xs')
                pairs_by_lig[local_i] = [
                    {'rec_idx': p['rec_idx'], 'pair_e': round(p['pair_e'], 5),
                     'rec_xs':  p['rec_xs'],  'lig_xs':  lig_xs,
                     'r':       round(p['r'], 4), 's': round(p['s'], 4)}
                    for p in best_pairs[vi]['pairs']
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
            'has_docked_pose':   bool(docked_atoms_all),
            'docked_pose_path':  docked_pose_path,
            # Which pose is on screen. A Vina _out.pdbqt holds several MODELs (mode 1 =
            # best); _load_pdbqt_all() stops at the first ENDMDL, so the viewer always
            # shows MODEL 1. Reporting the count lets the UI say so explicitly instead
            # of silently showing one of N and inviting "these coords look wrong" when
            # someone reads a different MODEL out of the file.
            'lig_model_count':   _count_pdbqt_models(lig_path),
            'lig_model_shown':   1,
        })

    except Exception as exc:
        logger.error(f'vina_parse_log error: {exc}', exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


# ── Bond perception ───────────────────────────────────────────────────────────
# A PDBQT carries no bond block at all — only the ROOT/BRANCH torsion tree (which
# lists rotatable bonds, not all bonds) and AutoDock atom types. The old
# _infer_bonds_from_pdbqt() therefore emitted every bond as order 1, which is why a
# benzo ring drew as six identical single lines. Two things in the file let us do
# better:
#
#   1. AutoDock types aromatic carbons 'A' — MGLTools' prepare_ligand4.py and Open
#      Babel both do this — so aromatic rings are labelled, not guessed.
#   2. The coordinates are a real 3D structure, so bond LENGTH cleanly separates
#      single / double / triple for everything outside a ring.
#
# RDKit's rdDetermineBonds is the usual tool for this, but it needs a complete
# hydrogen count to balance valences and a PDBQT keeps only POLAR hydrogens.
# Measured on this ligand: heavy atoms alone raise "Final molecular charge (0) does
# not match input (-2); could not find valid bond ordering", and including the polar
# H raises AtomValenceException. Hence the geometry + atom-type route below, which
# adds no dependency.

# Covalent radii (Å), Cordero et al. 2008.
_RCOV = {'H':0.31,'B':0.84,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'S':1.05,
         'Cl':1.02,'Se':1.20,'Br':1.20,'I':1.39,'Mg':1.41,'Zn':1.22,'Mn':1.39,
         'Ca':1.76,'Fe':1.32,'Cu':1.32,'Ni':1.24,'Co':1.26,'K':2.03}

# (elem, elem) -> (longest triple bond, longest double bond), Å.
_BOND_MULT = {
    ('C','C'): (1.25, 1.365), ('C','N'): (1.21, 1.345), ('C','O'): (1.15, 1.29),
    ('C','S'): (1.55, 1.70),  ('N','N'): (1.20, 1.31),  ('N','O'): (1.15, 1.30),
    ('O','O'): (None, 1.30),  ('N','S'): (None, 1.60),  ('O','S'): (None, 1.55),
    ('P','O'): (None, 1.55),  ('P','N'): (None, 1.65),  ('P','S'): (None, 1.98),
}
_AROM_MAX_BOND = 1.45      # a ring bond longer than this is not aromatic


def _xyz_dist(a, b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)


def _small_rings(adj, max_size=8):
    """Simple cycles up to max_size, then a crude SSSR filter. Ligand-sized graphs."""
    seen, rings = set(), []
    for start in sorted(adj):
        stack = [(start, [start], {start})]
        while stack:
            cur, path, inpath = stack.pop()
            for nb in sorted(adj[cur]):
                if nb == start and len(path) >= 3:
                    key = frozenset(path)
                    if key not in seen:
                        seen.add(key); rings.append(list(path))
                elif nb not in inpath and nb > start and len(path) < max_size:
                    stack.append((nb, path + [nb], inpath | {nb}))
    rings.sort(key=len)
    kept, covered = [], set()
    for r in rings:
        edges = {(min(r[i], r[(i+1) % len(r)]), max(r[i], r[(i+1) % len(r)]))
                 for i in range(len(r))}
        if not edges <= covered:
            kept.append(r); covered |= edges
    return kept


def _max_matching(nodes, edges):
    """Maximum matching by augmenting paths — this is the Kekule assignment for one
    aromatic system. Matched pairs become the double bonds."""
    adj = {n: [] for n in nodes}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].append(v); adj[v].append(u)
    mate = {}

    def _aug(u, seen):
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in mate or _aug(mate[v], seen):
                mate[v] = u; mate[u] = v
                return True
        return False

    for u in nodes:
        if u not in mate:
            _aug(u, set())
    return {(min(u, v), max(u, v)) for u, v in mate.items()}


def _order_from_length(a, b, a_has_h, b_has_h, deg_a, deg_b):
    """Bond order for a NON-aromatic bond, from its measured length."""
    ea, eb = a['elem'], b['elem']
    key = (ea, eb) if (ea, eb) in _BOND_MULT else (eb, ea)
    lim = _BOND_MULT.get(key)
    if not lim:
        return 1
    # An O or N carrying a polar hydrogen is a hydroxyl / amine, never the
    # multiply-bonded partner. PDBQT keeps polar H, so this test is reliable — it is
    # what stops the C9-O1 hydroxyl in this ligand being drawn as a carbonyl.
    if a_has_h or b_has_h:
        return 1
    d = _xyz_dist(a, b)
    triple, double = lim
    if triple is not None and d <= triple and deg_a <= 2 and deg_b <= 2:
        return 3
    if d <= double:
        return 2
    return 1


def _infer_bonds_from_pdbqt(path: str) -> list:
    """Ligand bonds for the viewer, WITH bond orders.

    Returns [{begin, end, order, aromatic}] where begin/end index the heavy-atom
    list (the same ordering the atom payload uses) and order is the Kekule order
    1 / 2 / 3. Aromatic rings are kekulised so the viewer can draw alternating
    double bonds; `aromatic` is carried alongside for anything that would rather
    render the delocalised form.
    """
    all_atoms = _load_pdbqt_all(path)
    if not all_atoms:
        return []
    for a in all_atoms:
        a['elem'] = _adt_to_elem(a['atype'])

    heavy_pos = [i for i, a in enumerate(all_atoms) if a['elem'] != 'H']
    g2h   = {g: h for h, g in enumerate(heavy_pos)}
    heavy = [all_atoms[i] for i in heavy_pos]

    # ── connectivity: covalent radii + 0.45 Å slack. The old flat 1.85 Å cutoff
    #    over-bonds halogens and under-bonds S/P. ──────────────────────────────
    adj = {i: set() for i in range(len(all_atoms))}
    for i in range(len(all_atoms)):
        for j in range(i + 1, len(all_atoms)):
            ei, ej = all_atoms[i]['elem'], all_atoms[j]['elem']
            if ei == 'H' and ej == 'H':
                continue
            cut = _RCOV.get(ei, 0.77) + _RCOV.get(ej, 0.77) + 0.45
            if _xyz_dist(all_atoms[i], all_atoms[j]) < cut:
                adj[i].add(j); adj[j].add(i)

    has_h = {i: any(all_atoms[j]['elem'] == 'H' for j in adj[i])
             for i in range(len(all_atoms))}

    hbonds = sorted({(g2h[i], g2h[j])
                     for i in heavy_pos for j in adj[i]
                     if j in g2h and g2h[i] < g2h[j]})

    hadj = {i: set() for i in range(len(heavy))}
    for u, v in hbonds:
        hadj[u].add(v); hadj[v].add(u)

    # ── aromatic rings: flagged by AutoDock type 'A', confirmed by geometry ───
    arom_bonds, arom_atoms = set(), set()
    for ring in _small_rings(hadj, max_size=8):
        n = len(ring)
        if n not in (5, 6, 7):
            continue
        if any(_xyz_dist(heavy[ring[k]], heavy[ring[(k + 1) % n]]) > _AROM_MAX_BOND
               for k in range(n)):
            continue
        n_a = sum(1 for a in ring if heavy[a]['atype'].strip().upper() == 'A')
        if n_a * 2 < n:                    # majority must be aromatic carbons
            continue
        for k in range(n):
            a, b = ring[k], ring[(k + 1) % n]
            arom_bonds.add((min(a, b), max(a, b)))
            arom_atoms.update((a, b))

    # ── kekulise: pyrrole-type N/O/S donates a lone pair to the ring and so takes
    #    no double bond; everything else in the aromatic system is matched in
    #    pairs. For indole this gives three doubles in the benzo ring and one in
    #    the five-ring, which is the structure a chemist would draw. ───────────
    donors = set()
    for h in arom_atoms:
        g, e = heavy_pos[h], heavy[h]['elem']
        if e in ('O', 'S'):
            donors.add(h)
        elif e == 'N' and (has_h[g] or len(hadj[h]) >= 3):
            donors.add(h)
    matched = _max_matching(
        sorted(arom_atoms - donors),
        [(u, v) for (u, v) in arom_bonds if u not in donors and v not in donors])

    bonds = []
    for (u, v) in hbonds:
        aromatic = (u, v) in arom_bonds
        if aromatic:
            order = 2 if (u, v) in matched else 1
        else:
            order = _order_from_length(heavy[u], heavy[v],
                                       has_h[heavy_pos[u]], has_h[heavy_pos[v]],
                                       len(hadj[u]), len(hadj[v]))
        bonds.append({'begin': u, 'end': v, 'order': order, 'aromatic': aromatic})
    return bonds