#!/usr/bin/env python
"""add_model_picker.py -- add a "Model" dropdown to the 3D Builder toolbar.

Run from the UI checkout root (the directory holding uiapp/ and web/):

    python add_model_picker.py            # apply
    python add_model_picker.py --check    # report what would change, touch nothing
    python add_model_picker.py --revert   # restore the .bak files

What it adds
------------
web/static/js/pose.js
  * a Model button in builder row 4, styled like Move / Add Fragments, whose
    menu lists GIGN (GPLA.pt) and every DeepAtom checkpoint the server finds
  * the menu branch, the state, the loader, and the label
  * _bldrGign now sends the chosen model and posts to the engine's own endpoint

uiapp/routes/pose_routes.py
  * GET  /pose/score_models   -- enumerate the allowed checkpoints
  * POST /pose/gign_score     -- honours an optional "model" from the body,
                                 validated against that same enumeration

config/input_routes.yml
  * pose.deepatom_ckpt_dir    -- the directory the DeepAtom checkpoints live in

Every edit is idempotent: re-running is a no-op.  Originals are kept as
<file>.bak on first run.
"""
import argparse
import os
import re
import shutil
import sys

JS = "web/static/js/pose.js"
PY = "uiapp/routes/pose_routes.py"
YML = "config/input_routes.yml"

DEFAULT_CKPT_DIR = "/home/huangzihang/repos/user_data/deepatom/checkpoints"


# ---------------------------------------------------------------------------
# the edits: (file, marker-that-means-already-applied, anchor, replacement)
# ---------------------------------------------------------------------------

JS_STATE_ANCHOR = """  PG._bldr = {open:false, sel:[], elem:'C', undo:[], redo:[], base:null, menu:null,
              view:null, hover:-1, ligIdx:null, baseCol:null, baseSize:null, dragging:false, pickDone:false,
              lastSar:null};"""

JS_STATE_NEW = """  PG._bldr = {open:false, sel:[], elem:'C', undo:[], redo:[], base:null, menu:null,
              view:null, hover:-1, ligIdx:null, baseCol:null, baseSize:null, dragging:false, pickDone:false,
              lastSar:null,
              /* --- scoring model -------------------------------------------
                 `models` is the server's list (null until /pose/score_models
                 answers); `model` is the chosen entry, an object shaped
                 {id,label,engine,path,ok,why}.  Null means "the server's
                 configured default", which is what every call did before this
                 picker existed -- so a failed enumeration degrades to the old
                 behaviour instead of blocking the button. */
              models:null, model:null};"""

JS_ROW_ANCHOR = """            + T('', cell+'opacity:.5;', 'Draw new atoms and bonds by clicking — not wired', G.pencil+'<span style="margin-left:4px;">Draw</span>', 'PoseGen._bldrTodo(\\'Draw\\')')
          + '</div>'"""

JS_ROW_NEW = """            + T('', cell+'opacity:.5;', 'Draw new atoms and bonds by clicking — not wired', G.pencil+'<span style="margin-left:4px;">Draw</span>', 'PoseGen._bldrTodo(\\'Draw\\')')
          + '</div>'

          /* ---- row 4b: which trained model scores the ligand ------------------
             Full width, because the label is a checkpoint name and truncating it
             to a 34px cell would make two checkpoints from the same sweep
             indistinguishable. Not dimmed: unlike the row above it, this one is
             wired. */
          + '<div style="display:flex;gap:5px;align-items:center;">'
            + T('poseBldModel', wide,
                'Which trained model scores this ligand: GIGN (GPLA.pt) or a retrained DeepAtom checkpoint',
                '<span id="poseBldModelLbl" style="overflow:hidden;text-overflow:ellipsis;max-width:210px;">Model — loading…</span>'+chev,
                'PoseGen._bldrMenu(\\'model\\',event)')
          + '</div>'"""

JS_MENU_ANCHOR = """    if(kind==='elem'){"""

JS_MENU_NEW = """    if(kind==='model'){
      /* One row per checkpoint the server is willing to run. An entry the server
         marked not-ok is still LISTED, greyed, with its reason -- a checkpoint
         that silently vanishes from this menu is indistinguishable from one that
         was never dropped in the directory. */
      var ms = PG._bldr.models;
      if(ms === null){
        m.innerHTML = '<div style="padding:6px 8px;color:#7b8798;font-family:ui-monospace,monospace;'
          + 'font-size:10px;">loading checkpoints…</div>';
        PG._bldrLoadModels(true);
      } else if(!ms.length){
        m.innerHTML = '<div style="padding:6px 8px;color:#fbbf24;font-family:ui-monospace,monospace;'
          + 'font-size:10px;line-height:1.5;">no checkpoints found.<br>Set pose.gign_model and '
          + 'pose.deepatom_ckpt_dir in config/input_routes.yml.</div>';
      } else {
        var cur = PG._bldr.model, eng = null, out = [];
        ms.forEach(function(x){
          if(x.engine !== eng){
            eng = x.engine;
            out.push('<div style="padding:5px 8px 3px;color:#5c6a7d;font-family:ui-monospace,monospace;'
              + 'font-size:9px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;">'
              + (eng==='gign' ? 'GIGN' : 'DeepAtom') + '</div>');
          }
          var on = !!(cur && cur.id === x.id);
          out.push('<button onclick="PoseGen._bldrMenu(null);PoseGen._bldrSetModel(\\''
            + x.id.replace(/'/g, "\\\\'") + '\\')" title="' + (x.ok ? x.path : (x.why || 'unavailable'))
            + '" style="display:flex;width:100%;box-sizing:border-box;align-items:center;'
            + 'justify-content:space-between;gap:10px;text-align:left;padding:5px 8px;border:none;'
            + 'border-radius:5px;background:' + (on ? 'rgba(34,211,238,.12)' : 'none')
            + ';color:' + (x.ok ? (on ? '#67e8f9' : '#cbd5e1') : '#6b7280')
            + ';font-family:ui-monospace,monospace;font-size:10.5px;cursor:pointer;"'
            + ' onmouseover="this.style.background=\\'#232a35\\'"'
            + ' onmouseout="this.style.background=\\'' + (on ? 'rgba(34,211,238,.12)' : 'none') + '\\'">'
            + '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
            + (on ? '&#10003; ' : '') + x.label + '</span>'
            + '<span style="color:#5c6a7d;font-size:9px;white-space:nowrap;">'
            + (x.ok ? (x.note || '') : 'unavailable') + '</span></button>');
        });
        m.innerHTML = out.join('');
      }
    } else if(kind==='elem'){"""

JS_HELPERS_ANCHOR = """  /* ---- history ------------------------------------------------------------- */
  PG._bldrSnap=function(){ try{ return JSON.stringify(L); }catch(e){ return null; } };"""

JS_HELPERS_NEW = '''  /* ---- scoring model -------------------------------------------------------
     The picker chooses BOTH the engine and the checkpoint, because they are not
     independent: GPLA.pt is a GIGN graph network and a DeepAtom .pth.tar is a
     24-channel voxel CNN, so each only runs under its own scorer. The server
     enumerates what it is actually willing to run; this side never invents a
     path. */
  PG._bldrLoadModels = function(reopen){
    return fetch('/pose/score_models')
      .then(function(r){ return r.ok ? r.json() : null; })
      .then(function(res){
        var ms = (res && res.models) || [];
        PG._bldr.models = ms;
        /* Keep an explicit choice across reloads; otherwise take the server's
           default, which is the first runnable entry. */
        var keep = PG._bldr.model && ms.filter(function(x){ return x.id === PG._bldr.model.id; })[0];
        PG._bldr.model = keep || ms.filter(function(x){ return x.ok; })[0] || null;
        PG._bldrModelLabel();
        if(reopen && PG._bldr.menu === 'model'){ PG._bldr.menu = null; PG._bldrMenu('model', null); }
        return ms;
      })
      .catch(function(){
        PG._bldr.models = [];
        PG._bldrModelLabel();
        return [];
      });
  };

  PG._bldrSetModel = function(id){
    var x = (PG._bldr.models || []).filter(function(e){ return e.id === id; })[0];
    if(!x) return;
    if(!x.ok){ PG._bldrSay(x.why || (x.label + ' is not runnable on this server'), '#fbbf24'); return; }
    PG._bldr.model = x;
    PG._bldrModelLabel();
    PG._bldrSay('scoring model: ' + x.label, '#67e8f9');
    PG._miniLog('\\u269b scoring model \\u2192 ' + x.label + '  [' + x.engine + ']  ' + x.path, '#67e8f9');
  };

  PG._bldrModelName = function(){
    var x = PG._bldr.model;
    return x ? x.label : 'the server default';
  };

  PG._bldrModelLabel = function(){
    var el = $('poseBldModelLbl'); if(!el) return;
    var x = PG._bldr.model;
    el.textContent = x ? x.label : ((PG._bldr.models && !PG._bldr.models.length)
                                    ? 'Model — none found' : 'Model — server default');
    el.title = x ? x.path : '';
    var sar = $('poseBldSar');
    if(sar) sar.title = 'Structure-activity: score this ligand with ' + PG._bldrModelName()
      + ' twice \\u2014 as the panel found it and as it is now \\u2014 and report the change in '
      + 'predicted affinity. Needs a loaded receptor and at least one edit.';
  };

  /* ---- history ------------------------------------------------------------- */
  PG._bldrSnap=function(){ try{ return JSON.stringify(L); }catch(e){ return null; } };'''

JS_FETCH_ANCHOR = """  PG._bldrGign = function(els, world, smiles, tag, bonds){
    var pdb = PG.mc._ligandPdb(world, els, bonds);
    var od = $('poseGignOutDir') || $('poseDaOutDir');
    return fetch('/pose/gign_score', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ligand_pdb:pdb, receptor_pdb:protein.raw, name:PG.mc._ts()+'_'+tag,
                            smiles:(smiles||''), out_dir:(od?od.value.trim():'')})})"""

JS_FETCH_NEW = """  PG._bldrGign = function(els, world, smiles, tag, bonds){
    var pdb = PG.mc._ligandPdb(world, els, bonds);
    var od = $('poseGignOutDir') || $('poseDaOutDir');
    /* The endpoint follows the ENGINE, not the button: a DeepAtom checkpoint
       handed to /pose/gign_score would be loaded by yupu_GIGN_pose.py and fail
       on a state-dict mismatch, which reads as a corrupt file rather than as the
       wrong scorer. `model` is a server-issued id, never a path from here. */
    var mdl = PG._bldr.model;
    var url = (mdl && mdl.engine === 'deepatom') ? '/pose/deepatom_score' : '/pose/gign_score';
    return fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ligand_pdb:pdb, receptor_pdb:protein.raw, name:PG.mc._ts()+'_'+tag,
                            smiles:(smiles||''), out_dir:(od?od.value.trim():''),
                            model:(mdl?mdl.id:'')})})"""

JS_OPEN_ANCHOR = """    PG._builderInject();
    var p=$('poseBuilderPanel'); if(!p) return;
    PG._bldr.open = want;"""

JS_OPEN_NEW = """    PG._builderInject();
    var p=$('poseBuilderPanel'); if(!p) return;
    PG._bldr.open = want;
    /* Enumerate once per panel-open, not once per process: a checkpoint dropped
       into the directory while the tab is open should appear without a reload. */
    if(want) PG._bldrLoadModels(false);"""


PY_ROUTE = '''

# =============================================================================
# GET /pose/score_models
# Enumerate the trained models this server is willing to run on a pose.
#
# Returns: { ok, models: [ {id, label, engine, path, ok, why, note}, ... ] }
#
# Two sources, and they are not interchangeable:
#   pose.gign_model        one GIGN checkpoint (GPLA.pt) -- a graph network, run
#                          by pose.gign_script
#   pose.deepatom_ckpt_dir a DIRECTORY of DeepAtom checkpoints (*.pt, *.pth,
#                          *.pth.tar) -- 24-channel voxel CNNs, run by
#                          pose.default_script
#
# An entry is reported with ok=false and a `why` rather than being hidden, so a
# checkpoint that is present but unrunnable (its scorer script is not
# configured) is distinguishable from one that was never dropped in the
# directory. `id` is what the browser sends back; no path from the client is
# ever executed.
# =============================================================================
@app.route('/pose/score_models', methods=['GET'])
def pose_score_models():
    return jsonify({'ok': True, 'models': _pose_score_models()}), 200


def _pose_score_models():
    """The allow-list. Shared by /pose/score_models and the scoring endpoints."""
    from pathlib import Path
    pcfg = _pose_cfg()
    out = []

    gign_script = (pcfg.get('gign_script') or '').strip()
    gign_ok = bool(gign_script) and Path(gign_script).expanduser().is_file()
    gm = (pcfg.get('gign_model') or '').strip()
    if gm:
        p = Path(gm).expanduser()
        exists = p.is_file()
        out.append({
            'id': 'gign:' + p.name,
            'label': 'GIGN · ' + p.name,
            'engine': 'gign',
            'path': str(p),
            'ok': bool(exists and gign_ok),
            'why': ('' if exists else 'checkpoint not found: %s' % p) or
                   ('' if gign_ok else 'pose.gign_script is not set or not found'),
            'note': 'default' if (exists and gign_ok) else '',
        })

    da_script = (pcfg.get('default_script') or '').strip()
    da_ok = bool(da_script) and Path(da_script).expanduser().is_file()
    why_da = '' if da_ok else 'pose.default_script (the DeepAtom pose scorer) is not set'
    ckpt_dir = (pcfg.get('deepatom_ckpt_dir') or '').strip()
    if ckpt_dir:
        d = Path(ckpt_dir).expanduser()
        if d.is_dir():
            names = []
            for pat in ('*.pt', '*.pth', '*.pth.tar', '*.ckpt'):
                names.extend(d.glob(pat))
            for p in sorted(set(names), key=lambda q: q.name):
                try:
                    mb = p.stat().st_size / (1024.0 * 1024.0)
                except OSError:
                    mb = 0.0
                out.append({
                    'id': 'deepatom:' + p.name,
                    'label': 'DeepAtom · ' + p.name,
                    'engine': 'deepatom',
                    'path': str(p),
                    'ok': da_ok,
                    'why': why_da,
                    'note': '%.0f MB' % mb if mb else '',
                })
        else:
            logger.warning('[pose] deepatom_ckpt_dir does not exist: %s', d)
    return out


def _pose_resolve_model(model_id, engine):
    """A client-supplied id -> an absolute checkpoint path, or ('', reason).

    Never trusts a path from the browser: the id has to match an entry this
    server already published, and its engine has to be the one the endpoint
    runs. '' means "use whatever the config says", which is the pre-picker
    behaviour and the right fallback when the browser sends nothing.
    """
    model_id = (model_id or '').strip()
    if not model_id:
        return '', ''
    for m in _pose_score_models():
        if m['id'] == model_id:
            if m['engine'] != engine:
                return '', ('%s is a %s checkpoint; it cannot be run by the %s scorer'
                            % (m['label'], m['engine'], engine))
            if not m['ok']:
                return '', (m['why'] or ('%s is not runnable on this server' % m['label']))
            return m['path'], ''
    return '', 'unknown model id: %s' % model_id
'''

PY_GIGN_ANCHOR = """    model_ckpt = (pcfg.get('gign_model') or '').strip()        # blank -> script default"""

PY_GIGN_NEW = """    model_ckpt = (pcfg.get('gign_model') or '').strip()        # blank -> script default
    # An explicit pick from the 3D Builder's Model dropdown overrides the config
    # for this one call. Resolved through the published allow-list, so the body
    # carries an id and never a filesystem path.
    _picked, _why = _pose_resolve_model(data.get('model'), 'gign')
    if _why:
        return jsonify({'ok': False, 'err': _why}), 200
    if _picked:
        model_ckpt = _picked"""

PY_DA_ANCHOR = """        test_type = 'vs'
        pose_default_dir = (pcfg.get('default_dir') or '').strip()"""

PY_DA_NEW = """        test_type = 'vs'
        pose_default_dir = (pcfg.get('default_dir') or '').strip()

        # Which DeepAtom checkpoint. The 3D Builder's Model dropdown sends an id;
        # anything else means "whatever the script defaults to", which is what
        # this endpoint did before the picker existed.
        da_ckpt, da_why = _pose_resolve_model(data.get('model'), 'deepatom')
        if da_why:
            return jsonify({'ok': False, 'err': da_why}), 200"""

PY_CMD_ANCHOR = """        core_cmd = '%s -t %s -d %s' % (script, test_type, str(root))
        run_cwd = str(root)
        shell_cmd = (
            'source "$(conda info --base)/etc/profile.d/conda.sh" && '
            'conda activate elion_backend && '
            '"%s" -t %s -d "%s" && conda deactivate' % (script, test_type, str(root))
        )"""

PY_CMD_NEW = """        # --model is appended only when one was picked, so a scorer script that
        # predates the flag still runs unchanged. Same for the release paths:
        # pose.deepatom_arpeggio / pose.deepatom_repo_dir pin deepatom_pose.py
        # to one copy of the DeepAtom tree instead of letting it search.
        _m = (' --model "%s"' % da_ckpt) if da_ckpt else ''
        _arp = (pcfg.get('deepatom_arpeggio') or '').strip()
        _rep = (pcfg.get('deepatom_repo_dir') or '').strip()
        if _arp:
            _m += ' --arpeggio "%s"' % _arp
        if _rep:
            _m += ' --repo-dir "%s"' % _rep
        core_cmd = '%s -t %s -d %s%s' % (script, test_type, str(root), _m)
        run_cwd = str(root)
        shell_cmd = (
            'source "$(conda info --base)/etc/profile.d/conda.sh" && '
            'conda activate elion_backend && '
            '"%s" -t %s -d "%s"%s && conda deactivate'
            % (script, test_type, str(root), _m)
        )"""

# Applied on top of the v3 block, for a tree that already has it.
YML_MPY_ANCHOR = 're:^(  deepatom_repo_dir: "[^"\n]*"\n)'

YML_MPY_NEW = r"""\1
  # The network definition FILE, under any name. Set this when the copy of
  # shufflenet_v3.py matching your checkpoint is NOT the one the rest of the
  # pipeline uses -- it wins over deepatom_repo_dir and disturbs nothing else.
  # The elion tree builds repeat_block [3,4,4] with a 2048-wide out_block; a
  # checkpoint trained at [3,3,3] with 1024 needs its own file. Passed as
  # --model-py. Blank = fall back to deepatom_repo_dir.
  deepatom_model_py: "/home/huangzihang/repos/user_data/deepatom/checkpoints/shufflenet_v3_blue.py"
"""

PY_MPY_ANCHOR = """        if _rep:
            _m += ' --repo-dir "%s"' % _rep"""

PY_MPY_NEW = """        if _rep:
            _m += ' --repo-dir "%s"' % _rep
        # pose.deepatom_model_py names the network definition FILE directly and
        # wins over the directory above -- for when the copy of shufflenet_v3.py
        # matching the checkpoint is not the one the rest of the pipeline uses.
        _mpy = (pcfg.get('deepatom_model_py') or '').strip()
        if _mpy:
            _m += ' --model-py "%s"' % _mpy"""

PY_CMD_ANCHOR_V3 = """        _m = (' --model "%s"' % da_ckpt) if da_ckpt else ''
        core_cmd = '%s -t %s -d %s%s' % (script, test_type, str(root), _m)"""

PY_CMD_NEW_V3 = """        _m = (' --model "%s"' % da_ckpt) if da_ckpt else ''
        # pose.deepatom_arpeggio / pose.deepatom_repo_dir pin deepatom_pose.py to
        # ONE copy of the DeepAtom release instead of letting it search -- the
        # search can land on a different copy (~/repos/deepatom is Python 2).
        _arp = (pcfg.get('deepatom_arpeggio') or '').strip()
        _rep = (pcfg.get('deepatom_repo_dir') or '').strip()
        _mpy = (pcfg.get('deepatom_model_py') or '').strip()
        if _arp:
            _m += ' --arpeggio "%s"' % _arp
        if _rep:
            _m += ' --repo-dir "%s"' % _rep
        if _mpy:
            _m += ' --model-py "%s"' % _mpy
        core_cmd = '%s -t %s -d %s%s' % (script, test_type, str(root), _m)"""

YML_SCRIPT_ANCHOR = """  default_script: \"\""""

YML_SCRIPT_NEW = """  # deepatom_pose.py -- staged complex -> arpeggio -> 32^3 x 24 voxel grid ->
  # ShuffleNetV3 -> pK. Needs deepatom_featurize.py beside it, and must be
  # executable (chmod +x). Blank disables the DeepAtom options in the Model
  # dropdown; a path that does not exist leaves them listed but greyed.
  default_script: "/home/huangzihang/repos/user_data/deepatom/deepatom_pose.py\""""

# Inserted directly under default_script, whatever path it currently holds --
# the user edits that line, so it is matched by shape, not by value.
YML_TREE_ANCHOR = 're:^(  default_script: "[^"\n]*"\n)'

YML_TREE_NEW = r"""\1
  # Which DeepAtom release deepatom_pose.py runs. Set BOTH, to the same tree the
  # "⚛ Score best pose" pipeline uses (pipeline_VS.py:278 builds its arpeggio
  # path from this model_split_data directory), so the Model dropdown scores
  # with identical code. Passed to the script as --arpeggio / --repo-dir.
  # Blank = the script searches for them itself, which can land on a different
  # copy of the release (~/repos/deepatom is one such copy, and it is Python 2).
  deepatom_arpeggio: "/home/huangzihang/repos/elion/src/elion/properties/deepatom/model_split_data/arpeggio_mod2/arpeggio.py"
  deepatom_repo_dir: "/home/huangzihang/repos/elion/src/elion/properties/deepatom/model_split_data/02_pytorch"

  # The network definition FILE, under any name. Set this when the copy of
  # shufflenet_v3.py that matches your checkpoint is NOT the one the rest of the
  # pipeline uses -- it wins over deepatom_repo_dir and nothing else is touched.
  # The elion tree builds repeat_block [3,4,4] with a 2048-wide out_block; a
  # checkpoint trained at [3,3,3] with 1024 needs its own file. Passed as
  # --model-py. Blank = use deepatom_repo_dir.
  deepatom_model_py: "/home/huangzihang/repos/user_data/deepatom/checkpoints/shufflenet_v3_blue.py"
"""

YML_ANCHOR = """  gign_conda_env: "elion_backend\""""

YML_NEW = '''  gign_conda_env: "elion_backend"

  # DeepAtom checkpoints offered in the 3D Builder's Model dropdown.
  # A DIRECTORY, not a file: every *.pt / *.pth / *.pth.tar / *.ckpt inside it is
  # listed. They are run by pose.default_script (blank above = listed but greyed
  # out, because a DeepAtom checkpoint cannot be loaded by the GIGN scorer).
  deepatom_ckpt_dir: "''' + DEFAULT_CKPT_DIR + '"'


# The SAR flow names its scorer in six status strings. They were written when
# GIGN was the only option; with a picker in the panel they have to follow it,
# or the log will say "GIGN" while a DeepAtom checkpoint is running.
SAR_STRINGS = [
    ("""'load a target protein (.pdb) first — GIGN scores a complex'""",
     """'load a target protein (.pdb) first — ' + PG._bldrModelName() + ' scores a complex'"""),
    ("""'a GIGN score is already running'""",
     """'a score is already running'"""),
    ("""'GIGN 1 of 2 — scoring the ligand as the panel found it…'""",
     """PG._bldrModelName() + ' 1 of 2 — scoring the ligand as the panel found it…'"""),
    ("""'GIGN 2 of 2 — scoring the edited ligand…'""",
     """PG._bldrModelName() + ' 2 of 2 — scoring the edited ligand…'"""),
    ("""'⚛ SAR (GIGN) — ΔG in kcal/mol, lower = tighter\\n'""",
     """'⚛ SAR (' + PG._bldrModelName() + ') — ΔG in kcal/mol, lower = tighter\\n'"""),
    ("""'This is the SERVER-side GIGN scorer failing, not the SAR wiring — '""",
     """'This is the SERVER-side scorer failing, not the SAR wiring — '"""),
]

# v2 -- "is not set" was printed both when the key was blank AND when it pointed
# at a file that does not exist, which sends people to edit a yml that is already
# correct. Say which, and name the path.
PY_WHY_GIGN_ANCHOR = """            'why': ('' if exists else 'checkpoint not found: %s' % p) or
                   ('' if gign_ok else 'pose.gign_script is not set or not found'),"""

PY_WHY_GIGN_NEW = """            'why': ('' if exists else 'gign_model points at %s, which does not exist' % p)
                   or ('' if gign_ok else
                       ('pose.gign_script is blank' if not gign_script else
                        'pose.gign_script points at %s, which does not exist' % gign_script)),"""

PY_WHY_DA_ANCHOR = """    da_script = (pcfg.get('default_script') or '').strip()
    da_ok = bool(da_script) and Path(da_script).expanduser().is_file()
    why_da = '' if da_ok else 'pose.default_script (the DeepAtom pose scorer) is not set'"""

PY_WHY_DA_NEW = """    da_script = (pcfg.get('default_script') or '').strip()
    da_path = Path(da_script).expanduser() if da_script else None
    da_ok = bool(da_path) and da_path.is_file()
    if da_ok:
        why_da = ''
    elif not da_script:
        why_da = ('pose.default_script is blank -- set it to deepatom_pose.py in '
                  'config/input_routes.yml')
    else:
        # Name the path. A message that says "not set" for a key that IS set
        # sends people to edit a config file that is already correct; the usual
        # cause is the script sitting one directory away from where it points.
        why_da = 'pose.default_script points at %s, which does not exist' % da_path
        try:
            near = [q for q in da_path.parent.rglob(da_path.name) if q.is_file()]
            if near:
                why_da += ' (found at %s -- move it, or point the key there)' % near[0]
        except OSError:
            pass"""

# --- v4: use the shell pipeline when pose.default_script is blank -----------
DA_FALLBACK_ANCHOR = """    da_script = (pcfg.get('default_script') or '').strip()
    da_path = Path(da_script).expanduser() if da_script else None"""

DA_FALLBACK_NEW = """    da_script = (pcfg.get('default_script') or '').strip()
    # /pose/deepatom_score falls back to deepatom.script when pose.default_script
    # is blank -- that fallback is the shell pipeline the "Score best pose"
    # button has always used. Either one makes DeepAtom runnable, so both are
    # checked here; otherwise blanking default_script to get the working
    # pipeline would grey out the very option it enables.
    if not da_script:
        da_script = (_deepatom_cfg().get('script') or '').strip()
    da_path = Path(da_script).expanduser() if da_script else None"""

DA_BLANKMSG_ANCHOR = """        why_da = ('pose.default_script is blank -- set it to deepatom_pose.py in '
                  'config/input_routes.yml')"""

DA_BLANKMSG_NEW = """        why_da = ('neither pose.default_script nor deepatom.script is set in '
                  'config/input_routes.yml')"""

DA_CFG_ANCHOR = """def _pose_score_models():"""

DA_CFG_NEW = '\n'.join([
    'def _deepatom_cfg():',
    '    """The `deepatom:` section of input_routes.yml, for its `script` key --',
    '    what /pose/deepatom_score runs when pose.default_script is blank."""',
    '    from pathlib import Path',
    '    try:',
    '        import yaml',
    '        try:',
    '            from uiapp.routes.shared import _INPUT_ROUTES_YML as _YML',
    '        except Exception:',
    '            _YML = str(Path(__file__).resolve().parent.parent / "input_TS.yml")',
    '        return (yaml.safe_load(open(_YML, "r", encoding="utf-8")) or {}).get("deepatom") or {}',
    '    except Exception as e:',
    '        logger.warning("[pose] could not read deepatom config: %s", e)',
    '        return {}',
    '',
    '',
    'def _pose_score_models():',
])

DA_PYGATE_ANCHOR = """        _m = (' --model "%s"' % da_ckpt) if da_ckpt else ''"""

DA_PYGATE_NEW = """        # Only a PYTHON scorer understands these flags. The shell pipeline
        # (deepatom.script) takes -t and -d and nothing else -- its getopts would
        # choke on anything more -- so it keeps the invocation it always had.
        _py = str(script).lower().endswith('.py')
        _m = (' --model "%s"' % da_ckpt) if (da_ckpt and _py) else ''"""

DA_GATE2_ANCHOR = """        _arp = (pcfg.get('deepatom_arpeggio') or '').strip()
        _rep = (pcfg.get('deepatom_repo_dir') or '').strip()"""

DA_GATE2_NEW = """        _arp = (pcfg.get('deepatom_arpeggio') or '').strip() if _py else ''
        _rep = (pcfg.get('deepatom_repo_dir') or '').strip() if _py else ''"""

DA_GATE3_ANCHOR = """        _mpy = (pcfg.get('deepatom_model_py') or '').strip()"""

DA_GATE3_NEW = """        _mpy = (pcfg.get('deepatom_model_py') or '').strip() if _py else ''"""

# --- the SAR button names the backend it is about to call -------------------
JS_SARSPAN_ANCHOR = """              + '<span style="color:#8b5cf6;font-size:9px;font-weight:600;margin-left:7px;">before &#8594; after</span>'"""

JS_SARSPAN_NEW = """              + '<span id="poseBldSarWhat" style="color:#8b5cf6;font-size:9px;font-weight:600;margin-left:7px;">before &#8594; after</span>'"""

JS_SARLBL_ANCHOR = """    var sar = $('poseBldSar');
    if(sar) sar.title = 'Structure-activity: score this ligand with ' + PG._bldrModelName()"""

JS_SARLBL_NEW = """    var sar = $('poseBldSar');
    /* The sub-label names the machinery, because for DeepAtom the two SAR passes
       go through /pose/deepatom_score -- the same endpoint, script and pipeline
       the "Score best pose" button uses -- while GIGN goes through
       /pose/gign_score. Same before/after comparison either way. */
    var what = $('poseBldSarWhat');
    if(what) what.textContent = (x && x.engine === 'deepatom')
      ? 'Score best pose \\u00d7 2' : 'before \\u2192 after';
    if(sar) sar.title = 'Structure-activity: score this ligand with ' + PG._bldrModelName()"""

# --- v5: repo-relative paths, and a message that names the right key --------
# `deepatom.script` is stored as "../deepatom/bin/...", relative to the UI root.
# Path(...).is_file() on that tests it against the FLASK PROCESS's working
# directory, which is only the UI root if the app happened to be started there.
# uiapp/config.resolve() exists for exactly this ("turn the relative paths stored
# in config/input_routes.yml into absolute paths"), so use REPO_ROOT.
ABS_HELPER_ANCHOR = """def _deepatom_cfg():"""

ABS_HELPER_NEW = '\n'.join([
    'def _abs_repo(path_str):',
    '    """Absolute path for a value out of input_routes.yml.',
    '',
    '    Absolute input is returned unchanged; a relative one is resolved against',
    '    the UI root, not the process CWD -- the same rule uiapp.config.resolve()',
    '    applies to every other path in that file. Falling back to the CWD as a',
    '    second try keeps any setup that already relied on it working."""',
    '    from pathlib import Path',
    '    if not path_str:',
    '        return path_str',
    '    p = Path(str(path_str)).expanduser()',
    '    if p.is_absolute():',
    '        return str(p)',
    '    try:',
    '        root = Path(_pcfg.REPO_ROOT)',
    '    except Exception:',
    '        root = Path(__file__).resolve().parent.parent.parent',
    '    cand = (root / p)',
    '    if cand.exists():',
    '        return str(cand.resolve())',
    '    if p.exists():',
    '        return str(p.resolve())',
    '    return str(cand)          # report the REPO-relative guess, not the CWD one',
    '',
    '',
    'def _deepatom_cfg():',
])

# the enumeration: resolve, and remember which key the value came from
DA_SRC_ANCHOR = """    if not da_script:
        da_script = (_deepatom_cfg().get('script') or '').strip()
    da_path = Path(da_script).expanduser() if da_script else None"""

DA_SRC_NEW = """    da_key = 'pose.default_script'
    if not da_script:
        da_script = (_deepatom_cfg().get('script') or '').strip()
        da_key = 'deepatom.script'
    da_script = _abs_repo(da_script)
    da_path = Path(da_script) if da_script else None"""

DA_MSG_ANCHOR = """        why_da = 'pose.default_script points at %s, which does not exist' % da_path"""

DA_MSG_NEW = """        # Name the key the value actually came from. Saying "pose.default_script"
        # for a path that came from deepatom.script sends people to edit a line
        # that is already blank.
        why_da = '%s points at %s, which does not exist' % (da_key, da_path)"""

# the route: resolve the same way, so the dropdown and the run agree
ROUTE_ABS_ANCHOR = """        script = (pcfg.get('default_script') or '').strip() or cfg.get('script')
        if not script or not Path(script).is_file():
            return jsonify({'ok': False,
                            'err': 'pose scoring script not found (set pose.default_script or deepatom.script in input_TS.yml)'}), 200"""

ROUTE_ABS_NEW = """        script = (pcfg.get('default_script') or '').strip() or cfg.get('script')
        # Resolved against the UI root rather than the process CWD -- see
        # _abs_repo. Without this, deepatom.script's "../deepatom/bin/..." only
        # resolves when Flask happens to have been started from the UI root, and
        # the Model dropdown would advertise a scorer this route cannot find.
        script = _abs_repo(script)
        if not script or not Path(script).is_file():
            return jsonify({'ok': False,
                            'err': ('pose scoring script not found: %s  (from '
                                    'pose.default_script, else deepatom.script, in '
                                    'config/input_routes.yml)' % script)}), 200"""

EDITS = [
    (PY, "points at %s, which does not exist", PY_WHY_DA_ANCHOR, PY_WHY_DA_NEW, "v2 deepatom why"),
    (PY, "gign_model points at", PY_WHY_GIGN_ANCHOR, PY_WHY_GIGN_NEW, "v2 gign why"),
    (JS, "models:null, model:null", JS_STATE_ANCHOR, JS_STATE_NEW, "builder state"),
    (JS, "row 4b: which trained model", JS_ROW_ANCHOR, JS_ROW_NEW, "Model button"),
    (JS, "if(kind==='model')", JS_MENU_ANCHOR, JS_MENU_NEW, "dropdown menu"),
    (JS, "PG._bldrLoadModels", JS_HELPERS_ANCHOR, JS_HELPERS_NEW, "loader + label"),
    (JS, "var url = (mdl && mdl.engine", JS_FETCH_ANCHOR, JS_FETCH_NEW, "engine routing"),
    (JS, "if(want) PG._bldrLoadModels(false)", JS_OPEN_ANCHOR, JS_OPEN_NEW, "load on open"),
    (PY, "_pose_score_models", None, PY_ROUTE, "/pose/score_models"),
    (PY, "_picked, _why = _pose_resolve_model", PY_GIGN_ANCHOR, PY_GIGN_NEW, "gign_score model"),
    (PY, "da_ckpt, da_why = _pose_resolve_model", PY_DA_ANCHOR, PY_DA_NEW, "deepatom_score model"),
    (PY, "_m = (' --model", PY_CMD_ANCHOR, PY_CMD_NEW, "deepatom --model flag"),
    (YML, "deepatom_ckpt_dir", YML_ANCHOR, YML_NEW, "deepatom_ckpt_dir"),
    (YML, "deepatom_pose.py", YML_SCRIPT_ANCHOR, YML_SCRIPT_NEW, "default_script"),
    (YML, "deepatom_arpeggio:", YML_TREE_ANCHOR, YML_TREE_NEW, "deepatom_arpeggio/repo"),
    (PY, "def _abs_repo(", ABS_HELPER_ANCHOR, ABS_HELPER_NEW, "_abs_repo helper"),
    (PY, "da_key = 'pose.default_script'", DA_SRC_ANCHOR, DA_SRC_NEW, "track which key"),
    (PY, "'%s points at %s, which does not exist'", DA_MSG_ANCHOR, DA_MSG_NEW, "name the right key"),
    (PY, "script = _abs_repo(script)", ROUTE_ABS_ANCHOR, ROUTE_ABS_NEW, "route resolves repo-rel"),
    (PY, "_deepatom_cfg().get('script')", DA_FALLBACK_ANCHOR, DA_FALLBACK_NEW, "deepatom.script fallback"),
    (PY, "neither pose.default_script nor", DA_BLANKMSG_ANCHOR, DA_BLANKMSG_NEW, "blank-both message"),
    (PY, "def _deepatom_cfg()", DA_CFG_ANCHOR, DA_CFG_NEW, "_deepatom_cfg helper"),
    (PY, "_py = str(script).lower().endswith", DA_PYGATE_ANCHOR, DA_PYGATE_NEW, "flags only for .py"),
    (PY, "deepatom_arpeggio') or '').strip() if _py", DA_GATE2_ANCHOR, DA_GATE2_NEW, "arpeggio/repo gated"),
    (PY, "deepatom_model_py') or '').strip() if _py", DA_GATE3_ANCHOR, DA_GATE3_NEW, "model_py gated"),
    (JS, 'id="poseBldSarWhat"', JS_SARSPAN_ANCHOR, JS_SARSPAN_NEW, "SAR label span"),
    (JS, "var what = $('poseBldSarWhat')", JS_SARLBL_ANCHOR, JS_SARLBL_NEW, "SAR label follows engine"),
    (PY, "_arp = (pcfg.get('deepatom_arpeggio')", PY_CMD_ANCHOR_V3, PY_CMD_NEW_V3, "route passes release paths"),
    (PY, "deepatom_model_py", PY_MPY_ANCHOR, PY_MPY_NEW, "route passes --model-py"),
    (YML, "deepatom_model_py", YML_MPY_ANCHOR, YML_MPY_NEW, "deepatom_model_py"),
] + [(JS, new, old, new, "SAR label %d" % (i + 1))
     for i, (old, new) in enumerate(SAR_STRINGS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--revert", action="store_true", help="restore the .bak files")
    a = ap.parse_args()

    for f in (JS, PY, YML):
        if not os.path.isfile(f):
            sys.exit("not in the UI checkout root: %s is missing "
                     "(cd to the directory holding uiapp/ and web/)" % f)

    if a.revert:
        for f in (JS, PY, YML):
            if os.path.isfile(f + ".bak"):
                shutil.copyfile(f + ".bak", f)
                print("reverted %s" % f)
            else:
                print("no backup for %s" % f)
        return 0

    text = {f: open(f, encoding="utf-8").read() for f in (JS, PY, YML)}
    plan, missing = [], []
    for f, marker, anchor, new, what in EDITS:
        if marker in text[f]:
            print("  [have] %-22s %s" % (what, f))
            continue
        if anchor is None:
            plan.append((f, anchor, new, what))
        elif anchor.startswith("re:"):
            n = len(re.findall(anchor[3:], text[f], re.M))
            if n == 1:
                plan.append((f, anchor, new, what))
            else:
                missing.append((f, what, n))
        elif text[f].count(anchor) == 1:
            plan.append((f, anchor, new, what))
        else:
            missing.append((f, what, text[f].count(anchor)))
    for f, what, n in missing:
        print("  [MISS] %-22s %s  (anchor found %d times, expected 1)" % (what, f, n))
    for f, anchor, new, what in plan:
        print("  [todo] %-22s %s" % (what, f))

    if missing:
        print("\nRefusing to edit: %d anchor(s) did not match exactly once.\n"
              "The file has moved on from the version this patch was written "
              "against; send me the file and I'll re-anchor." % len(missing))
        return 1
    if not plan:
        print("\nNothing to do -- already applied.")
        return 0
    if a.check:
        print("\n--check: %d edit(s) would be applied." % len(plan))
        return 0

    for f in {p[0] for p in plan}:
        if not os.path.isfile(f + ".bak"):
            shutil.copyfile(f, f + ".bak")
            print("backup -> %s.bak" % f)

    for f, anchor, new, what in plan:
        if anchor is None:
            text[f] = text[f].rstrip("\n") + "\n" + new
        elif anchor.startswith("re:"):
            text[f] = re.sub(anchor[3:], lambda m: new.replace("\\1", m.group(1)),
                             text[f], count=1, flags=re.M)
        else:
            text[f] = text[f].replace(anchor, new, 1)
    for f in {p[0] for p in plan}:
        open(f, "w", encoding="utf-8").write(text[f])
        print("patched %s" % f)

    # cheap syntax guard on the python side; the JS is checked by the browser
    import py_compile
    try:
        py_compile.compile(PY, doraise=True)
        print("python syntax ok")
    except py_compile.PyCompileError as e:
        print("SYNTAX ERROR in %s -- run --revert\n%s" % (PY, e))
        return 1

    print("\nDone. Restart the Flask app, then:\n"
          "  curl -s localhost:5000/pose/score_models | python -m json.tool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
