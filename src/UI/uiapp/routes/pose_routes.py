# =============================================================================
# routes/pose_routes.py
# RDKit-backed SMILES → 3D initial-pose endpoint for the Pose Generation tool.
#
#   POST /pose/ligand   body: {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
#   → {ok, smiles, formula, N, atoms[], REF[[x,y,z]], BONDS[...], ROOT[],
#      TORS[...], TINFO[...], source:"rdkit"}
#
# Register by importing this module once at app startup (same pattern as
# hub_routes.py):  from uiapp.routes import pose_routes  # noqa
# Requires rdkit in the server venv:  pip install rdkit
# =============================================================================
from collections import deque
import re
from flask import jsonify, request
from uiapp import app
from uiapp import config as _pcfg
try:
    from uiapp.routes.shared import logger
except Exception:                       # shared.py may not expose `logger`
    import logging
    logger = logging.getLogger("pose_routes")


def _pose_cfg():
    """Read the `pose:` section from input_TS.yml (a defaults source)."""
    from pathlib import Path
    try:
        import yaml
        try:
            from uiapp.routes.shared import _INPUT_ROUTES_YML as _YML
        except Exception:
            _YML = str(Path(__file__).resolve().parent.parent / 'input_TS.yml')
        cfg = yaml.safe_load(open(_YML, 'r', encoding='utf-8')) or {}
        return cfg.get('pose') or {}
    except Exception as e:
        logger.warning('[pose] could not read pose config: %s', e)
        return {}


def _fmt_lig(line, base_serial, counter):
    """Reformat one ligand atom line exactly as stage-2 preprocessing
    (generate_atomtypes.py) does: drop hydrogens, renumber the serial after the
    protein, give a unique name (element + running index), and force resName LIG
    and chain 'y' -- the chain ID the npz/grid stage uses to locate ligand atoms.
    Returns (formatted_line_or_None, counter)."""
    line = line.ljust(80)
    if line[:6].rstrip() not in ('ATOM', 'HETATM'):
        return None, counter
    if line[77] == 'H':                          # stage 2 drops ligand hydrogens
        return None, counter
    try:
        old = int(line[6:11].strip())
    except ValueError:
        return None, counter
    serial_str = '%5s' % str(base_serial + old)
    last_col = line[76:78].strip()
    sym = last_col if last_col else line[12:16].strip()[0]
    counter += 1
    ann = sym + str(counter)
    ann_str = (' ' + '%-3s' % ann) if len(ann) < 4 else ('%4s' % ann)
    return 'HETATM' + serial_str + ' ' + ann_str + ' LIG y' + line[22:], counter


def _clean_ligand(lig):
    """Ligand-only PDB with chain 'y' / unique names (stage-2 formatting)."""
    out, counter = [], 0
    for ln in lig.splitlines():
        fl, counter = _fmt_lig(ln, 0, counter)
        if fl:
            out.append(fl)
    out.append('END')
    return '\n'.join(out) + '\n'


def _clean_complex(rec, lig):
    """Merge receptor + ligand into the complex PDB the DeepAtom pipeline expects.

    The pose path runs generate_atomtypes with --stages 3,5(,4,6), skipping
    stage-2 preprocessing -- so we reproduce here what stage 2 does to the
    ligand: serials continued past the protein's last, unique atom names,
    resName LIG, and *critically* chain ID 'y'. The grid stage finds ligand
    atoms only via atomtypes lines whose chain is 'y' (generate_npz
    create_ligand_file / make_grid); without it the run dies with 'No ligand
    atoms found'. Unique serials also avoid arpeggio's AtomSerialError.
    """
    rec_lines = [ln for ln in rec.splitlines() if ln[:6] in ('ATOM  ', 'HETATM', 'TER   ')]
    last = 0
    for ln in rec_lines:
        if ln[:6] in ('ATOM  ', 'HETATM'):
            try:
                last = max(last, int(ln[6:11]))
            except ValueError:
                pass
    out = list(rec_lines)
    out.append('TER')                                  # separate protein from het ligand
    counter = 0
    for ln in lig.splitlines():
        fl, counter = _fmt_lig(ln, last + 1, counter)  # stage 2: last_prot_serial = last + 1
        if fl:
            out.append(fl)
    out.append('END')
    return '\n'.join(out) + '\n'

_ROT_STRICT = ('[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)'
               '&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])'
               '&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])'
               '&!$([#7!D1]-!@[CD3]=[N+])]'
               '-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)'
               '&!$(C([CH3])([CH3])[CH3])]')


def _build_tree(n, bond_pairs, rotatable_ids):
    rigid = [[] for _ in range(n)]
    for bid, (a, b) in enumerate(bond_pairs):
        if bid not in rotatable_ids:
            rigid[a].append(b); rigid[b].append(a)
    frag = [-1] * n; fragments = []
    for s in range(n):
        if frag[s] >= 0:
            continue
        fid = len(fragments); stack = [s]; frag[s] = fid; comp = []
        while stack:
            u = stack.pop(); comp.append(u)
            for v in rigid[u]:
                if frag[v] < 0:
                    frag[v] = fid; stack.append(v)
        fragments.append(comp)
    root_frag = max(range(len(fragments)), key=lambda i: len(fragments[i]))
    fadj = [[] for _ in fragments]
    for bid, (a, b) in enumerate(bond_pairs):
        if bid in rotatable_ids:
            fadj[frag[a]].append((frag[b], a, b))
            fadj[frag[b]].append((frag[a], b, a))
    par = [-1] * len(fragments); par_bond = [None] * len(fragments)
    vis = [False] * len(fragments); orderF = []
    q = deque([root_frag]); vis[root_frag] = True
    while q:
        u = q.popleft(); orderF.append(u)
        for (to, af, bf) in fadj[u]:
            if not vis[to]:
                vis[to] = True; par[to] = u; par_bond[to] = (af, bf); q.append(to)
    children = [[] for _ in fragments]
    for f in range(len(fragments)):
        if par[f] >= 0:
            children[par[f]].append(f)

    def subtree(f):
        out = []; stack = [f]
        while stack:
            u = stack.pop(); out.extend(fragments[u])
            for c in children[u]:
                stack.append(c)
        return out

    TORS = []
    for f in orderF:
        if par[f] < 0:
            continue
        af, bf = par_bond[f]
        TORS.append({"from": af, "to": bf, "moves": subtree(f)})
    return fragments[root_frag][:], TORS


def smiles_to_pose(smiles, seed=0xC0FFEE):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    smiles = (smiles or "").strip()
    if not smiles:
        return {"ok": False, "err": "Empty SMILES"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "err": "RDKit could not parse SMILES"}
    if mol.GetNumAtoms() > 150:
        return {"ok": False, "err": "Too large (%d heavy atoms)" % mol.GetNumAtoms()}
    formula = rdMolDescriptors.CalcMolFormula(mol)

    rot_pairs = set()
    patt = Chem.MolFromSmarts(_ROT_STRICT)
    if patt is not None:
        for a, b in mol.GetSubstructMatches(patt):
            rot_pairs.add((min(a, b), max(a, b)))

    molH = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = seed
    if AllChem.EmbedMolecule(molH, params) != 0:
        if AllChem.EmbedMolecule(molH, useRandomCoords=True, randomSeed=seed) != 0:
            return {"ok": False, "err": "3D embedding failed"}
    try:
        AllChem.MMFFOptimizeMolecule(molH, maxIters=400)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(molH, maxIters=400)
        except Exception:
            pass
    conf = molH.GetConformer()

    n = mol.GetNumAtoms()
    REF, atoms = [], []
    for i in range(n):
        p = conf.GetAtomPosition(i)
        REF.append([round(p.x, 4), round(p.y, 4), round(p.z, 4)])
        atoms.append(mol.GetAtomWithIdx(i).GetSymbol())

    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        kek = mol
    bond_pairs, rotatable_ids, bond_meta = [], set(), []
    for bond in kek.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bid = len(bond_pairs); bond_pairs.append((a, b))
        bt = bond.GetBondTypeAsDouble()
        order = 2 if bt == 2.0 else (3 if bt == 3.0 else 1)
        is_rot = (min(a, b), max(a, b)) in rot_pairs
        if is_rot:
            rotatable_ids.add(bid)
        bond_meta.append({"a": a, "b": b, "order": order,
                          "type": "rot" if is_rot else ("ring" if bond.IsInRing() else "rigid")})

    ROOT, TORS = _build_tree(n, bond_pairs, rotatable_ids)

    def tors_index(a, b):
        for i, t in enumerate(TORS):
            if (t["from"] == a and t["to"] == b) or (t["from"] == b and t["to"] == a):
                return i
        return -1
    BONDS = [{"a": m["a"], "b": m["b"], "order": m["order"], "type": m["type"],
              "ti": tors_index(m["a"], m["b"]) if m["type"] == "rot" else -1} for m in bond_meta]
    TINFO = [{"n": atoms[t["from"]] + "\u2013" + atoms[t["to"]],
              "a": "A%d\u2013A%d" % (t["from"], t["to"])} for t in TORS]

    return {"ok": True, "smiles": smiles, "formula": formula, "N": len(TORS),
            "atoms": atoms, "REF": REF, "BONDS": BONDS, "ROOT": ROOT,
            "TORS": TORS, "TINFO": TINFO, "source": "rdkit"}


@app.route('/pose/ligand', methods=['POST'])
def pose_ligand():
    """SMILES → 3D initial pose + torsion tree (RDKit). Frontend falls back to
    its in-browser parser if this endpoint is unavailable."""
    data = request.get_json(silent=True) or {}
    smiles = data.get('smiles', '')
    try:
        result = smiles_to_pose(smiles)
    except ImportError:
        logger.warning("[pose] rdkit not installed; client will fall back to JS parser")
        return jsonify({"ok": False, "err": "rdkit not installed on server"}), 200
    except Exception as e:
        logger.warning("[pose] smiles_to_pose error: %s", e)
        return jsonify({"ok": False, "err": "server error: %s" % e}), 200
    return jsonify(result), 200


# =============================================================================
# POST /pose/deepatom_score
# Score a single generated pose with DeepAtom (server-side ShuffleNetV3 CNN).
#
# Body: { "ligand_pdb": "<HETATM records of the posed ligand>",
#         "receptor_pdb": "<full receptor .pdb text>",
#         "name": "<id>", "smiles": "<optional>" }
# Returns: { ok, pred_pk, deltaG, elapsed, source:"deepatom" } | { ok:false, err }
#
# It assembles a one-compound Dataset_VS/<name>/ tree (ligand + complex pdb) and
# runs the same scoring script used by /vina_visualization/deepatom_estimate, then
# parses the predicted pK (ΔG = -pK * 1.36).
#
# REQUIRES (same as deepatom_estimate): the `elion_backend` conda env, the DeepAtom
# weights, arpeggio, and `deepatom.script` set in input_TS.yml. If any piece is
# missing it returns {ok:false, err:...} so the frontend degrades gracefully and the
# fast surrogate keeps driving the live search. This endpoint could not be executed
# in the build sandbox — verify on the host (lysine) before relying on it.
# =============================================================================
@app.route('/pose/deepatom_score', methods=['POST'])
def pose_deepatom_score():
    import os, time, tempfile, subprocess
    from pathlib import Path
    data = request.get_json(silent=True) or {}
    lig = (data.get('ligand_pdb') or '').strip()
    rec = (data.get('receptor_pdb') or '').strip()
    name = (re.sub(r'[^A-Za-z0-9_\-]', '_', (data.get('name') or 'pose'))[:40]) or 'pose'
    out_dir_req = (data.get('out_dir') or '').strip()
    if not lig or not rec:
        return jsonify({'ok': False, 'err': 'ligand_pdb and receptor_pdb are required'}), 200

    t0 = time.time()
    try:
        from uiapp.routes import deepatom_routes as DA
    except Exception as e:
        return jsonify({'ok': False, 'err': 'deepatom_routes not importable: %s' % e}), 200

    try:
        cfg = DA._load_deepatom_cfg() if hasattr(DA, '_load_deepatom_cfg') else {}
        pcfg = _pose_cfg()
        # "⚛ Score best pose" runs:  <pose.default_script> -t vs -d <root>
        # pose.default_script overrides deepatom.script for this endpoint; root is the
        # card's output-dir field if set, else pose.default_dir, else a temp dir.
        script = (pcfg.get('default_script') or '').strip() or cfg.get('script')
        if not script or not Path(script).is_file():
            return jsonify({'ok': False,
                            'err': 'pose scoring script not found (set pose.default_script or deepatom.script in input_TS.yml)'}), 200
        test_type = 'vs'
        pose_default_dir = (pcfg.get('default_dir') or '').strip()

        # Working/output root. When the user supplies a directory, the pipeline's intermediates
        # (.atomtypes from arpeggio, the .npz voxel grid) and the results CSV all land there and
        # persist for inspection; otherwise fall back to a throwaway temp dir.
        if out_dir_req:
            try:
                root = Path(out_dir_req).expanduser()
                root.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return jsonify({'ok': False, 'err': 'cannot use output dir "%s": %s' % (out_dir_req, e)}), 200
            ephemeral = False
        elif pose_default_dir:
            try:
                root = Path(pose_default_dir).expanduser()
                root.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return jsonify({'ok': False, 'err': 'cannot use pose.default_dir "%s": %s' % (pose_default_dir, e)}), 200
            ephemeral = False
        else:
            root = Path(tempfile.mkdtemp(prefix='elion_pose_da_'))
            ephemeral = True
        # give the pipeline a place to drop the atomtypes / npz intermediates
        for sub in ('atomtypes', 'npz'):
            try:
                (root / sub).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # one-compound dataset:  <root>/Dataset_VS/<name>/{<name>_ligand.pdb, <name>_complex.pdb}
        ds = root / 'Dataset_VS' / name
        ds.mkdir(parents=True, exist_ok=True)
        (ds / (name + '_ligand.pdb')).write_text(_clean_ligand(lig))
        (ds / (name + '_complex.pdb')).write_text(_clean_complex(rec, lig))

        # How many compounds this call will actually process.
        #
        # The script is a virtual-screening driver: `-d <root>` means "atom-type,
        # voxelise and predict EVERYTHING under <root>/Dataset_VS", not "the one
        # I just wrote". Reusing one root therefore makes call N redo calls
        # 1..N-1 — 78 complexes over a 12-pose trajectory instead of 12, and the
        # only visible symptom is that each score takes longer than the last.
        # Report it rather than let it look like the model got slower.
        try:
            n_ds = sum(1 for p in (root / 'Dataset_VS').iterdir() if p.is_dir())
        except Exception:
            n_ds = 1
        if n_ds > 1:
            logger.warning('[pose] deepatom_score: %s holds %d compounds; this run '
                           're-processes all of them. Use one root per pose.',
                           root / 'Dataset_VS', n_ds)

        # the command shown in the UI, and the directory the subprocess runs in.
        # This used to be the BARE script; the conda wrapper was stripped before
        # display, so "run the command below" reproduced a different thing than
        # the server ran and a failure inside `conda activate` was invisible.
        # cmd_core is kept for anyone who wants the script on its own.
        core_cmd = '%s -t %s -d %s' % (script, test_type, str(root))
        run_cwd = str(root)
        # pose.deepatom_conda_env, falling back to the historical hard-coded
        # name so an existing deployment keeps working. Blank disables activation.
        da_env = pcfg.get('deepatom_conda_env')
        da_env = 'elion_backend' if da_env is None else str(da_env).strip()
        shell_cmd = _conda_wrap('"%s" -t %s -d "%s"' % (script, test_type, str(root)), da_env)
        logger.info('[pose] deepatom_score cwd=%s cmd: %s', run_cwd, shell_cmd)
        timeout = int(cfg.get('timeout_seconds', 600) or 600)
        # AUGMENT=1 builds + averages augmented grids (matches the batch
        # methodology / avg_test). It is the default rather than a constant so
        # pose.deepatom_env can override it, and can unset whatever else the
        # host needs removed — see _subproc_env.
        da_over = pcfg.get('deepatom_env')
        if not isinstance(da_over, dict):
            da_over = {}
        da_over = dict({'AUGMENT': '1'}, **da_over)
        env, env_note = _subproc_env(da_over)
        logger.info('[pose] deepatom_score env overrides: %s', env_note)
        proc = subprocess.run(shell_cmd, shell=True, executable='/bin/bash',
                              cwd=run_cwd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout)
        out = proc.stdout or ''

        # always save the full run output as a debug log in the output dir, so it
        # can be inspected even on success (the mini-chat stays quiet when it works)
        log_path = None
        try:
            log_path = root / (name + '_score.log')
            log_path.write_text(
                ('# pose DeepAtom score\n# when : %s\n# cmd  : %s\n# cwd  : %s\n# exit : %d\n'
                 % (time.strftime('%Y-%m-%d %H:%M:%S'), core_cmd, run_cwd, proc.returncode))
                + ('-' * 70) + '\n' + out)
        except Exception as _e:
            logger.warning('[pose] could not write debug log: %s', _e)
            log_path = None

        # locate the prediction: stdout first, then any results CSV written anywhere under the root
        parsed = DA._parse_deepatom_output(out) if hasattr(DA, '_parse_deepatom_output') else {}
        comps = parsed.get('compounds') or (
            DA._read_deepatom_csv(str(root)) if hasattr(DA, '_read_deepatom_csv') else [])
        pred = None
        for c in comps:
            if c.get('id') == name or len(comps) == 1:
                pred = c.get('pred_pk')
                break

        def _ls(pat):
            try:
                return sorted(str(p) for p in root.rglob(pat))
            except Exception:
                return []
        produced = {'atomtypes': _ls('*.atomtypes'), 'npz': _ls('*.npz'), 'csv': _ls('*.csv')}

        if pred is None:
            return jsonify({
                'ok': False,
                'err': ('script exited %d but no pK was parsed — looked in stdout and %d CSV under the output dir '
                        '(%d .atomtypes, %d .npz produced). If these counts are 0 the pipeline never wrote there; '
                        'run the command below in the cwd below to see the real error (commonly arpeggio not on PATH '
                        'inside the scoring env).'
                        % (proc.returncode, len(produced['csv']), len(produced['atomtypes']), len(produced['npz']))
                        + _conda_diagnose(out, da_env) + _link_diagnose(out)),
                'cmd': shell_cmd,
                'cmd_core': core_cmd,
                'conda_env': da_env or '(none — running the script directly)',
                'env_overrides': env_note,
                'cwd': run_cwd,
                'log': (str(log_path) if log_path else None),
                'returncode': proc.returncode,
                'out_dir': str(root),
                'produced': produced,
                'compounds': n_ds,
                'stdout': out[-200000:],
            }), 200

        return jsonify({'ok': True,
                        'pred_pk': round(float(pred), 4),
                        'deltaG': round(-float(pred) * 1.36, 4),
                        'elapsed': '%.1fs' % (time.time() - t0),
                        'cmd': shell_cmd,
                        'cmd_core': core_cmd,
                        'conda_env': da_env or '(none — running the script directly)',
                        'cwd': run_cwd,
                        'log': (str(log_path) if log_path else None),
                        'out_dir': str(root),
                        'produced': produced,
                        'compounds': n_ds,
                        'source': 'deepatom'}), 200
    except subprocess.TimeoutExpired:
        return jsonify({'ok': False, 'err': 'DeepAtom timed out'}), 200
    except Exception as e:
        logger.warning('[pose] deepatom_score error: %s', e)
        return jsonify({'ok': False, 'err': 'server error: %s' % e}), 200

# =============================================================================
# POST /pose/gign_score
# Score the live pose with the Yupu_GIGN model: the uploaded receptor + the
# final posed ligand (the same two inputs DeepAtom receives). Backs the
# "Yupu_GIGN" option in the (3) Monte-Carlo "Scorer" dropdown.
#
# Body: { "ligand_pdb": "<posed ligand HETATM>", "receptor_pdb": "<full .pdb>",
#         "name": "<id>", "smiles": "<optional>", "out_dir": "<optional>" }
# Returns: { ok, pred_pk, deltaG, elapsed, cmd, cwd, stage, log,
#            source:"yupu_gign" } | { ok:false, err, cmd, cwd, stage, stdout }
#
# It stages <stage>/<name>_{ligand,protein}.pdb + <name>_meta.json, then runs
#     python <gign_script> --stage <stage> --name <name>
# (cwd = the script's own dir, inside <gign_conda_env>). yupu_GIGN_pose.py builds
# the GIGN graph from the posed ligand + receptor and runs the model, printing
# GIGN_PRED_PK which we parse here (dG = -1.36 * pK).
#
# By DEFAULT a 5 A binding pocket is cut (gign_pocket_cutoff = 5, waters dropped),
# matching how the model was TRAINED (preprocessing.py: pymol byres <lig> around 5,
# HOH + hydrogens removed). Set gign_pocket_cutoff = 0 to feed the FULL receptor
# instead, which reproduces predict_ZccE.py (a train/inference mismatch -- the
# model was trained on pockets, so full-protein scores are not meaningful).
#
# Config (input_TS.yml -> pose:):
#     gign_script:          ".../Yupu_GIGN/yupu_GIGN_pose.py"  (required)
#     gign_model:           ".../epoch-165...pt"   # optional; "" -> script default
#     gign_conda_env:       "elion_backend"        # optional; "" -> plain `python`
#     gign_pocket_cutoff:   5      # A; 5 = training pocket. 0 = full receptor
#     gign_drop_water:      true   # drop HOH/WAT (applies when cutting)
#     gign_scratch:         ""     # optional persistent scratch root; "" -> temp
#     gign_timeout_seconds: 1200
#
# This endpoint could not be executed in the build sandbox -- verify on the host.
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# conda wrapping, shared by the two scorer endpoints
# -----------------------------------------------------------------------------
# Both scorers shell out to something that must run in a specific environment.
# The env NAME is configuration, not a constant: /pose/deepatom_score hard-coded
# "elion_backend" and produced a one-line log —
#
#     Could not find conda environment: elion_backend
#
# — with no way to tell from the UI whether the name was wrong, the env lived in
# a conda installation `conda info --base` does not point at, or the script was
# meant to activate its own env all along. All three are common and all three
# looked identical.
#
# A blank env means "run the command as-is", which is the right answer when the
# script activates its own environment or when the server already runs inside the
# right one — previously impossible to express.
# ─────────────────────────────────────────────────────────────────────────────
def _subproc_env(overrides, base=None) -> tuple:   # noqa: C901
    """(env, note) for a scorer subprocess.

    `overrides` is a mapping from config. A null/blank value UNSETS the
    variable, which is the point: the common way a working conda env still
    fails to import torch is a stray LD_LIBRARY_PATH pointing at a system NCCL
    or CUDA that shadows the ones bundled in torch/lib. You cannot express
    "remove this" by setting it to something, so a null has to mean delete.

    Everything here used to be a hard-coded `env['AUGMENT'] = '1'` — same class
    of problem the conda env name had: a value only the source could change.
    """
    import os                    # imported per-route in this module, not at top level
    env = dict(base if base is not None else os.environ)
    applied = []
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            k = str(k)
            if v is None or (isinstance(v, str) and not v.strip()):
                if env.pop(k, None) is not None:
                    applied.append('-' + k)
            else:
                env[k] = str(v)
                applied.append('%s=%s' % (k, v))
    return env, (', '.join(applied) if applied else '(none)')


def _link_diagnose(out: str) -> str:
    """Hint for the dynamic-linker failures that look like a broken env."""
    if not out or 'undefined symbol' not in out:
        return ''
    lib = 'a shared library'
    for token in ('libtorch_cuda', 'libtorch', 'libcudart', 'libnccl'):
        if token in out:
            lib = token
            break
    return ('\n\nThis is a LINKER failure inside %s, not a missing package: the '
            'interpreter found the module and the module found a library of the '
            'right NAME but the wrong build. Almost always LD_LIBRARY_PATH (or a '
            'conda-installed nccl/cudatoolkit) shadowing the copies bundled in '
            'torch/lib.\n'
            'Check whether it imports on its own first:\n'
            '    conda run -n <env> python -c "import torch; print(torch.__version__)"\n'
            'If that works and this does not, the SCRIPT is changing the '
            'environment. Unset the offending variable for the subprocess with, '
            'in input_routes.yml:\n'
            '    pose:\n'
            '      deepatom_env:\n'
            '        LD_LIBRARY_PATH: null      # null = unset it\n'
            % lib)


def _conda_wrap(cmd: str, env_name: str) -> str:
    """`cmd` wrapped in `conda activate <env_name>`, or `cmd` unchanged if blank."""
    env_name = (env_name or '').strip()
    if not env_name:
        return cmd
    return ('source "$(conda info --base)/etc/profile.d/conda.sh" && '
            'conda activate %s && %s && conda deactivate' % (env_name, cmd))


def _conda_diagnose(out: str, env_name: str) -> str:
    """When activation is what failed, say which envs actually exist.

    Returns '' unless the output carries conda's not-found message, so the happy
    path costs nothing and an unrelated failure is not buried under env noise.
    """
    if not env_name or not out:
        return ''
    if ('Could not find conda environment' not in out
            and 'EnvironmentNameNotFound' not in out
            and 'CondaEnvironmentError' not in out):
        return ''
    import subprocess          # imported per-route in this module, not at top level
    listing = ''
    try:
        pr = subprocess.run(
            'source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null; conda env list',
            shell=True, executable='/bin/bash', stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=30)
        listing = (pr.stdout or '').strip()
    except Exception as e:                                        # noqa: BLE001
        listing = 'could not run `conda env list`: %s' % e
    return ('\n\nconda could not activate %r. Environments visible to the server '
            '(note this is the conda that `conda info --base` resolves to — an env '
            'created under a different installation will not appear here):\n%s\n\n'
            'Fix by pointing the config at a name from that list, or set it blank '
            'to run the command without activating anything.'
            % (env_name, listing or '(none)'))


@app.route('/pose/gign_score', methods=['POST'])
def pose_gign_score():
    import os, time, json, tempfile, subprocess
    from pathlib import Path
    t0 = time.time()
    data = request.get_json(silent=True) or {}
    lig = (data.get('ligand_pdb') or '').strip()
    rec = (data.get('receptor_pdb') or '').strip()
    smiles = (data.get('smiles') or '').strip()
    name = (re.sub(r'[^A-Za-z0-9_\-]', '_', (data.get('name') or 'pose'))[:40]) or 'pose'
    out_dir_req = (data.get('out_dir') or '').strip()
    if not lig or not rec:
        return jsonify({'ok': False, 'err': 'ligand_pdb and receptor_pdb are required'}), 200

    pcfg = _pose_cfg()
    script = (pcfg.get('gign_script') or
              _pcfg.POSE_GIGN_SCRIPT).strip()
    if not script or not Path(script).expanduser().is_file():
        return jsonify({'ok': False,
                        'err': 'GIGN predict script not found (set pose.gign_script in input_TS.yml): %s' % script}), 200
    # Blank is now meaningful: run yupu_GIGN_pose.py without activating anything,
    # for a server already inside the right env. `or` would swallow that, so read
    # the key explicitly and only default when it is absent.
    env_name = pcfg.get('gign_conda_env')
    env_name = 'elion_backend' if env_name is None else str(env_name).strip()
    model_ckpt = (pcfg.get('gign_model') or '').strip()        # blank -> script default
    try:
        cutoff = float(pcfg.get('gign_pocket_cutoff', 5) or 5)
    except (TypeError, ValueError):
        cutoff = 10.0
    drop_water = bool(pcfg.get('gign_drop_water', True))
    run_cwd = str(Path(script).expanduser().resolve().parent)

    # staging dir: explicit out_dir -> pose.gign_scratch -> temp
    scratch_cfg = (pcfg.get('gign_scratch') or '').strip()
    try:
        if out_dir_req:
            stage = Path(out_dir_req).expanduser() / ('gign_' + name)
        elif scratch_cfg:
            stage = Path(scratch_cfg).expanduser() / ('gign_' + name)
        else:
            stage = Path(tempfile.mkdtemp(prefix='elion_gign_'))
        stage.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return jsonify({'ok': False, 'err': 'cannot create staging dir: %s' % e}), 200

    lig_path = stage / (name + '_ligand.pdb')
    prot_path = stage / (name + '_protein.pdb')
    meta_path = stage / (name + '_meta.json')
    try:
        lig_path.write_text(lig if lig.endswith('\n') else lig + '\n')
        prot_path.write_text(rec if rec.endswith('\n') else rec + '\n')
        meta_path.write_text(json.dumps({
            'name': name,
            'ligand_pdb': str(lig_path),
            'protein_pdb': str(prot_path),
            'smiles': smiles,
            'pocket_cutoff': cutoff,
            'drop_water': drop_water,
            'model': model_ckpt,           # '' -> yupu_GIGN_pose.py picks its default
        }))
    except Exception as e:
        return jsonify({'ok': False, 'err': 'cannot write staging files: %s' % e}), 200

    core_cmd = 'python %s --stage %s --name %s' % (script, str(stage), name)   # see the note in deepatom_score
    shell_cmd = _conda_wrap('python "%s" --stage "%s" --name "%s"'
                            % (script, str(stage), name), env_name)
    logger.info('[pose] gign_score cwd=%s cmd: %s', run_cwd, shell_cmd)

    try:
        timeout = int(pcfg.get('gign_timeout_seconds', 1200) or 1200)
        gi_over = pcfg.get('gign_env')
        genv, genv_note = _subproc_env(gi_over if isinstance(gi_over, dict) else {})
        logger.info('[pose] gign_score env overrides: %s', genv_note)
        proc = subprocess.run(shell_cmd, shell=True, executable='/bin/bash',
                              cwd=run_cwd, env=genv,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout)
        out = proc.stdout or ''

        log_path = None
        try:
            log_path = stage / (name + '_gign.log')
            log_path.write_text(
                ('# pose GIGN score\n# when : %s\n# cmd  : %s\n# cwd  : %s\n# exit : %d\n'
                 % (time.strftime('%Y-%m-%d %H:%M:%S'), core_cmd, run_cwd, proc.returncode))
                + ('-' * 70) + '\n' + out)
        except Exception as _e:
            logger.warning('[pose] could not write gign debug log: %s', _e)
            log_path = None

        # parse the explicit marker first, then fall back to a 'pred:' decimal
        pk = None
        for ln in out.splitlines():
            if 'GIGN_PRED_PK' in ln:
                m = re.findall(r'[-+]?\d+\.\d+(?:[eE][-+]?\d+)?', ln.split('GIGN_PRED_PK', 1)[1])
                if m:
                    pk = float(m[0]); break
        if pk is None:
            for ln in out.splitlines():
                i = ln.find('pred:')
                if i >= 0:
                    m = re.findall(r'[-+]?\d+\.\d+(?:[eE][-+]?\d+)?', ln[i + 5:])
                    if m:
                        pk = float(m[0]); break

        if pk is None:
            return jsonify({
                'ok': False,
                'err': ('script exited %d but no GIGN_PRED_PK was parsed from stdout. '
                        'Run the command below in the cwd below to see the real error '
                        '(common causes: wrong conda env, missing torch/CUDA, the model '
                        'checkpoint not found, or RDKit failing to parse the pose/pocket).'
                        % proc.returncode) + _conda_diagnose(out, env_name) + _link_diagnose(out),
                'cmd': shell_cmd, 'cmd_core': core_cmd,
                'conda_env': env_name or '(none — running python directly)',
                'cwd': run_cwd, 'stage': str(stage),
                'log': (str(log_path) if log_path else None),
                'returncode': proc.returncode,
                'stdout': out[-200000:],
            }), 200

        return jsonify({'ok': True,
                        'pred_pk': round(pk, 4),
                        'deltaG': round(-pk * 1.36, 4),
                        'elapsed': '%.1fs' % (time.time() - t0),
                        'cmd': shell_cmd, 'cmd_core': core_cmd,
                        'conda_env': env_name or '(none — running python directly)',
                        'cwd': run_cwd, 'stage': str(stage),
                        'log': (str(log_path) if log_path else None),
                        'source': 'yupu_gign'}), 200
    except subprocess.TimeoutExpired:
        return jsonify({'ok': False, 'err': 'GIGN timed out', 'cmd': shell_cmd, 'cwd': run_cwd}), 200
    except Exception as e:
        logger.warning('[pose] gign_score error: %s', e)
        return jsonify({'ok': False, 'err': 'server error: %s' % e, 'cmd': core_cmd, 'cwd': run_cwd}), 200


# =============================================================================
# POST /pose/save_pdb
# Save a generated pose .pdb to a server directory. Used by the DeepAtom card's
# "⤓ Generate pose .pdb" button when the output-dir field is filled in; when it
# is blank the frontend downloads the file in-browser instead (no server write).
# Also used by the Monte-Carlo "▶ Run search" auto-save, which posts with no
# out_dir so the server writes to pose.search_pose_dir (input_TS.yml).
#
# Body: { "pdb": "<HETATM records>", "out_dir": "<server path, optional>", "name": "<id>" }
#   out_dir omitted/blank -> falls back to pose.search_pose_dir.
# Returns: { ok, path } | { ok:false, err }
# =============================================================================
@app.route('/pose/save_pdb', methods=['POST'])
def pose_save_pdb():
    from pathlib import Path
    data = request.get_json(silent=True) or {}
    pdb = data.get('pdb') or ''
    out_dir = (data.get('out_dir') or '').strip()
    name = (re.sub(r'[^A-Za-z0-9_\-]', '_', (data.get('name') or 'pose'))[:60]) or 'pose'
    if not pdb.strip():
        return jsonify({'ok': False, 'err': 'empty pdb'}), 200
    if not out_dir:
        # no explicit dir (e.g. the Monte-Carlo auto-save) -> pose.search_pose_dir
        out_dir = (_pose_cfg().get('search_pose_dir') or '').strip()
    if not out_dir:
        return jsonify({'ok': False, 'err': 'no output dir provided (set pose.search_pose_dir in input_TS.yml or pass out_dir)'}), 200
    try:
        root = Path(out_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        fpath = root / (name + '.pdb')
        fpath.write_text(pdb)
        logger.info('[pose] saved pose pdb -> %s', fpath)
        return jsonify({'ok': True, 'path': str(fpath)}), 200
    except Exception as e:
        logger.warning('[pose] save_pdb error: %s', e)
        return jsonify({'ok': False, 'err': 'cannot write to "%s": %s' % (out_dir, e)}), 200


# =============================================================================
# GET /pose/default_receptor
# Serve the default receptor .pdb configured in input_TS.yml under:
#     pose:
#       default_receptor_pdb: "/abs/path/to/receptor.pdb"
# The Pose Generation tool auto-loads this on open (the "⤓ Upload .pdb" button
# overrides it). Blank/missing config → {ok:false} and the tool starts with no
# target (docking box defaults to the ligand frame).
# =============================================================================
@app.route('/pose/default_receptor', methods=['GET'])
def pose_default_receptor():
    """Serve the Pose Generation tool's on-open defaults: the receptor .pdb (if
    configured), the docking box (3D-legend center x/y/z + edge length), and the
    output-dir field pre-fill (default_dir). Box/out_dir are returned whether or
    not a receptor exists, so the box still loads without a default target."""
    from pathlib import Path
    pcfg = _pose_cfg()
    resp = {'ok': False}

    # docking box defaults (included only if configured)
    box = {}
    for k, key in (('cx', 'box_center_x'), ('cy', 'box_center_y'),
                   ('cz', 'box_center_z'), ('len', 'box_length')):
        v = pcfg.get(key)
        if v is not None:
            try:
                box[k] = float(v)
            except (TypeError, ValueError):
                pass
    if box:
        resp['box'] = box

    # output-dir field pre-fill
    od = (pcfg.get('default_dir') or '').strip()
    if od:
        resp['out_dir'] = od

    # Where the trajectory scorers stage every new-minimum pose.
    #
    # Separate from default_dir on purpose. default_dir is "the box the user can
    # type in", one directory reused by every hand-clicked score; the trajectory
    # writes a dozen poses per run plus, for DeepAtom, a per-record subtree of
    # atomtypes and voxel grids. Pointing both at one directory is what made
    # /pose/deepatom_score re-atom-type every earlier complex on each call —
    # its pipeline processes everything under <root>/Dataset_VS by design.
    #
    # Blank falls back client-side to the panel's output-dir box, so an existing
    # deployment that has not set this key behaves as it did before.
    td = (pcfg.get('trajectory_dir') or '').strip()
    if td:
        resp['traj_dir'] = td

    # default ligand SMILES (optional) -- pre-fills the #poseSmiles box on open
    sm = (pcfg.get('default_smile') or '').strip()
    if sm:
        resp['default_smile'] = sm

    # default receptor .pdb (optional)
    rp = (pcfg.get('default_receptor_pdb') or '').strip()
    try:
        if not rp:
            resp['err'] = 'no default receptor configured'
        else:
            fp = Path(rp).expanduser()
            if fp.is_file():
                resp['ok'] = True
                resp['name'] = fp.stem
                resp['pdb'] = fp.read_text()
            else:
                resp['err'] = 'default receptor not found: %s' % rp
    except Exception as e:
        logger.warning('[pose] default_receptor error: %s', e)
        resp['err'] = str(e)

    return jsonify(resp), 200


# =============================================================================
# POST /pose/save_upload
# Archive a receptor .pdb uploaded via the "⤓ Upload .pdb" button to the folder
# configured in input_TS.yml under pose.default_upload_path, so the user can
# find the uploaded file on the server later. No-op (ok:false) if unconfigured.
#
# Body: { "pdb": "<.pdb text>", "name": "<id>" }  ->  { ok, path } | { ok:false, err }
# =============================================================================
@app.route('/pose/save_upload', methods=['POST'])
def pose_save_upload():
    from pathlib import Path
    data = request.get_json(silent=True) or {}
    pdb = data.get('pdb') or ''
    name = (re.sub(r'[^A-Za-z0-9_\-]', '_', (data.get('name') or 'receptor'))[:80]) or 'receptor'
    if not pdb.strip():
        return jsonify({'ok': False, 'err': 'empty pdb'}), 200
    try:
        import yaml
        try:
            from uiapp.routes.shared import _INPUT_ROUTES_YML as _YML
        except Exception:
            _YML = str(Path(__file__).resolve().parent.parent / 'input_TS.yml')
        cfg = yaml.safe_load(open(_YML, 'r', encoding='utf-8')) or {}
        updir = (((cfg.get('pose') or {}).get('default_upload_path')) or '').strip()
        if not updir:
            return jsonify({'ok': False, 'err': 'no default_upload_path configured'}), 200
        root = Path(updir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        fpath = root / (name + '.pdb')
        fpath.write_text(pdb)
        logger.info('[pose] archived upload -> %s', fpath)
        return jsonify({'ok': True, 'path': str(fpath)}), 200
    except Exception as e:
        logger.warning('[pose] save_upload error: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200

# =============================================================================
# POST /pose/vina_breakdown
# Authoritative, server-side Vina score for the LIVE pose, with a full
# per-atom-pair breakdown written to an auditable log. Backs the "⚛ Compute &
# log breakdown" button on the Pose Generation tool's "Vina" tab.
#
# The tab's on-screen card is a fast client-side preview over a 5 A pocket; this
# endpoint recomputes the SAME five-term AutoDock Vina empirical function over
# the FULL receptor (every heavy atom within the 8 A pair cutoff of the ligand),
# so the logged score is complete and reproducible from the raw coordinates
# rather than echoed from the browser.
#
# Body: { "ligand_pdb":   "<HETATM records of the posed ligand>",
#         "receptor_pdb": "<full receptor .pdb text>",
#         "name": "<id>", "smiles": "<optional>",
#         "nrot": <int active rotatable bonds, optional>,
#         "weights": { gauss1, gauss2, repulsion, hydrophobic, hbond, rot },  # optional; each defaults to the Vina value
#         "out_dir": "<optional; overrides pose.vina_breakdown_dir>" }
# Returns: { ok, dg, inter, nrot, npairs, terms{...}, raw_sums{...}, weights{...},
#            pk, top_pairs[...], log, out_dir, elapsed, source:"vina" }
#          | { ok:false, err }
#
# Per ligand-atom <-> receptor-atom pair, with surface gap s = r - Ri - Rj:
#     e = w1*gauss1 + w2*gauss2 + w3*repulsion + w4*hydrophobic + w5*Hbond
# and the reported affinity is  dG = (sum e) / (1 + w_rot * Nrot).
#
# Config (input_TS.yml -> pose:):
#     vina_breakdown_dir: "/abs/dir"   # where <name>_vina_breakdown.log is written.
#                                      # Blank -> computed but NOT written to disk
#                                      # (the response still carries the numbers).
# No conda env / external binary needed (pure-Python reimplementation of the same
# five terms used by the tab and by vina_dock_routes._term_breakdown).
# =============================================================================

# ── Vina five-term empirical scoring (element-based XS radii + typing) ────────
# Defaults are the standard AutoDock Vina weights (match the GPU binary and the
# "Vina" tab in the Pose Generation tool).
_VINA_DEFAULT_W = {'gauss1': -0.035579, 'gauss2': -0.005156, 'repulsion': 0.840245,
                   'hydrophobic': -0.035069, 'hbond': -0.587439, 'rot': 0.05846}
# element -> Vina XS van-der-Waals radius (A); default 1.8 for anything unlisted
_VINA_XS_RADIUS = {'C': 1.9, 'N': 1.8, 'O': 1.7, 'S': 2.0, 'P': 2.1,
                   'F': 1.5, 'Cl': 1.8, 'Br': 2.0, 'I': 2.2, 'B': 1.8}
_VINA_HYDROPHOBIC = {'C', 'F', 'Cl', 'Br', 'I'}   # hydrophobic XS types
_VINA_HB = {'N', 'O'}                             # H-bond donor/acceptor (simplified)


def _vina_rad(el):
    return _VINA_XS_RADIUS.get(el, 1.8)


def _polar_carbon_flags(atoms):
    """Vina C_P detection: mark carbons covalently bonded to N/O (<1.85 A). Those
    are polar carbons and earn no hydrophobic reward (unlike element-based typing,
    which counts every carbon). `atoms` = [(el, x, y, z, label), ...]."""
    n = len(atoms)
    polar = [False] * n
    cov2 = 1.85 * 1.85
    het = [k for k in range(n) if atoms[k][0] in ('N', 'O')]
    for i in range(n):
        if atoms[i][0] != 'C':
            continue
        xi, yi, zi = atoms[i][1], atoms[i][2], atoms[i][3]
        for j in het:
            dx = xi - atoms[j][1]; dy = yi - atoms[j][2]; dz = zi - atoms[j][3]
            if dx * dx + dy * dy + dz * dz < cov2:
                polar[i] = True
                break
    return polar


def _vina_terms(s):
    """The five Vina terms at surface gap s (A). Directional terms (hydrophobic,
    hbond) are returned unconditionally; the caller zeroes them for non-matching
    atom-type pairs."""
    import math
    g1 = math.exp(-((s / 0.5) ** 2))
    g2 = math.exp(-(((s - 3.0) / 2.0) ** 2))
    rep = (s * s) if s < 0 else 0.0
    hyd = 1.0 if s <= 0.5 else (0.0 if s >= 1.5 else (1.5 - s))     # ramp 1->0 over 0.5..1.5 A
    hb = 1.0 if s <= -0.7 else (0.0 if s >= 0.0 else (-s / 0.7))    # ramp 1->0 over -0.7..0 A
    return g1, g2, rep, hyd, hb


def _vina_pdb_atoms(text, kind='rec'):
    """Parse heavy atoms from PDB ATOM/HETATM lines.
    Returns [(element, x, y, z, label), ...]; hydrogens skipped.
    label: ligand -> '<El><serial>' (e.g. 'C7'); receptor -> '<name>/<resn><resseq>'
    (e.g. 'OG1/SER123'), so every logged pair names its exact protein atom."""
    out = []
    for ln in text.splitlines():
        if ln[:6].strip() not in ('ATOM', 'HETATM'):
            continue
        line = ln.ljust(80)
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        el = line[76:78].strip()
        name = line[12:16].strip()
        if el:                                             # element column present (normal case)
            el = el[:1].upper() + (el[1:2].lower() if len(el) > 1 else '')   # 'CL'->'Cl', ' C'->'C'
        else:                                              # infer from atom name: first letter only
            el = (re.sub(r'[^A-Za-z]', '', name)[:1] or 'C').upper()          # avoids 'CA'->'Ca'
        if el == 'H':
            continue
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = len(out) + 1
        if kind == 'lig':
            label = '%s%d' % (el, serial)
        else:
            resn = line[17:20].strip(); resseq = line[22:26].strip()
            label = ('%s/%s%s' % (name or el, resn, resseq)) if (resn or resseq) else (name or '%s%d' % (el, serial))
        out.append((el, x, y, z, label))
    return out


def _render_vina_log(name, w, nrot, sums, contrib, inter, denom, dg, pk,
                     n_lig, n_rec_near, npair, rows, t0, typing='element'):
    """Format the human-readable, auditable per-atom-pair breakdown log."""
    import time
    out = []
    out.append('# pose Vina score-breakdown (five-term AutoDock Vina empirical score)')
    out.append('# name          : %s' % name)
    out.append('# when          : %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    out.append('# ligand atoms  : %d (heavy)' % n_lig)
    out.append('# receptor atoms: %d within the pair cutoff of the ligand' % n_rec_near)
    out.append('# atom pairs    : %d evaluated (surface gap < 8.0 A)' % npair)
    out.append('# typing        : %s hydrophobic — %s'
               % (typing, 'C_H only (carbons bonded to N/O skipped)' if typing == 'vina' else 'every carbon counted'))
    out.append('# weights       : gauss1=%.6f gauss2=%.6f repulsion=%.6f hydrophobic=%.6f hbond=%.6f rot=%.5f'
               % (w['gauss1'], w['gauss2'], w['repulsion'], w['hydrophobic'], w['hbond'], w['rot']))
    out.append('#')
    out.append('# --- weighted term contributions (w_k * sum_pairs term_k) --------------')
    out.append('#   gauss1      = %+.5f   (raw sum %.4f)' % (contrib['gauss1'], sums['gauss1']))
    out.append('#   gauss2      = %+.5f   (raw sum %.4f)' % (contrib['gauss2'], sums['gauss2']))
    out.append('#   repulsion   = %+.5f   (raw sum %.4f)' % (contrib['repulsion'], sums['repulsion']))
    out.append('#   hydrophobic = %+.5f   (raw sum %.4f)' % (contrib['hydrophobic'], sums['hydrophobic']))
    out.append('#   hbond       = %+.5f   (raw sum %.4f)' % (contrib['hbond'], sums['hbond']))
    out.append('#   ------------------------------------------')
    out.append('#   c (inter)   = %+.5f  kcal/mol' % inter)
    out.append('#   Nrot        = %d' % nrot)
    out.append('#   denom       = 1 + %.5f * %d = %.5f' % (w['rot'], nrot, denom))
    out.append('#   dG          = c / denom = %+.5f  kcal/mol' % dg)
    out.append('#   pK (-dG/1.36) = %s' % (('%.3f' % pk) if pk is not None else '-- (dG >= 0)'))
    out.append('#')
    out.append('# --- per-atom-pair breakdown (sorted: most favourable pair_e first) ----')
    out.append('# lig_atom <-> rec_atom   r(A)   s(A)   gauss1 gauss2 repulsion hydrophobic hbond   pair_e')
    for (pair_e, llab, plab, r, s, g1, g2, rep, hyd, hb) in rows:
        out.append('[non_cache::eval pair]  %-10s <-> %-18s  r=%5.2f  s=%+5.2f  '
                   'g1=%.3f g2=%.3f rep=%.3f hyd=%.3f hb=%.3f  pair_e=%+.5f'
                   % (llab, plab, r, s, g1, g2, rep, hyd, hb, pair_e))
    return '\n'.join(out) + '\n'


@app.route('/pose/vina_breakdown', methods=['POST'])
def pose_vina_breakdown():
    import time, math
    from pathlib import Path
    t0 = time.time()
    data = request.get_json(silent=True) or {}
    lig = (data.get('ligand_pdb') or '').strip()
    rec = (data.get('receptor_pdb') or '').strip()
    name = (re.sub(r'[^A-Za-z0-9_\-]', '_', (data.get('name') or 'pose'))[:40]) or 'pose'
    if not lig or not rec:
        return jsonify({'ok': False, 'err': 'ligand_pdb and receptor_pdb are required'}), 200

    # weights: request overrides the Vina defaults; rot (Nrot penalty) clamped >= 0
    w = dict(_VINA_DEFAULT_W)
    req_w = data.get('weights') or {}
    for k in list(w):
        try:
            if req_w.get(k) is not None:
                w[k] = float(req_w[k])
        except (TypeError, ValueError):
            pass
    w['rot'] = max(0.0, w['rot'])
    try:
        nrot = int(data.get('nrot')) if data.get('nrot') is not None else 0
    except (TypeError, ValueError):
        nrot = 0
    typing = (data.get('typing') or 'element')
    if typing not in ('vina', 'element'):
        typing = 'element'

    try:
        lig_atoms = _vina_pdb_atoms(lig, kind='lig')
        rec_atoms = _vina_pdb_atoms(rec, kind='rec')
        if not lig_atoms:
            return jsonify({'ok': False, 'err': 'no ligand heavy atoms parsed from ligand_pdb'}), 200
        if not rec_atoms:
            return jsonify({'ok': False, 'err': 'no receptor heavy atoms parsed from receptor_pdb'}), 200

        # pre-filter receptor atoms to a box around the ligand (8 A cutoff + margin)
        CUT = 8.0; CUT2 = CUT * CUT; MARGIN = 2.5
        lxs = [a[1] for a in lig_atoms]; lys = [a[2] for a in lig_atoms]; lzs = [a[3] for a in lig_atoms]
        lo0, lo1, lo2 = min(lxs) - CUT - MARGIN, min(lys) - CUT - MARGIN, min(lzs) - CUT - MARGIN
        hi0, hi1, hi2 = max(lxs) + CUT + MARGIN, max(lys) + CUT + MARGIN, max(lzs) + CUT + MARGIN
        near = [a for a in rec_atoms
                if lo0 <= a[1] <= hi0 and lo1 <= a[2] <= hi1 and lo2 <= a[3] <= hi2]

        rows = []
        sums = {'gauss1': 0.0, 'gauss2': 0.0, 'repulsion': 0.0, 'hydrophobic': 0.0, 'hbond': 0.0}
        npair = 0
        # Vina XS typing: carbons bonded to N/O are polar (no hydrophobic reward).
        lig_polar = _polar_carbon_flags(lig_atoms) if typing == 'vina' else None
        near_polar = _polar_carbon_flags(near) if typing == 'vina' else None
        for i, (le, lx, ly, lz, llab) in enumerate(lig_atoms):
            lr = _vina_rad(le)
            lHy = (le in _VINA_HYDROPHOBIC) and not (typing == 'vina' and le == 'C' and lig_polar[i])
            lHB = le in _VINA_HB
            for j, (pe, px, py, pz, plab) in enumerate(near):
                dx = lx - px; dy = ly - py; dz = lz - pz
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > CUT2:
                    continue
                r = math.sqrt(r2)
                s = r - lr - _vina_rad(pe)
                g1, g2, rep, hyd, hb = _vina_terms(s)
                pHy = (pe in _VINA_HYDROPHOBIC) and not (typing == 'vina' and pe == 'C' and near_polar[j])
                if not (lHy and pHy):
                    hyd = 0.0
                if not (lHB and pe in _VINA_HB):
                    hb = 0.0
                pair_e = (w['gauss1'] * g1 + w['gauss2'] * g2 + w['repulsion'] * rep
                          + w['hydrophobic'] * hyd + w['hbond'] * hb)
                sums['gauss1'] += g1; sums['gauss2'] += g2; sums['repulsion'] += rep
                sums['hydrophobic'] += hyd; sums['hbond'] += hb
                npair += 1
                rows.append((pair_e, llab, plab, r, s, g1, g2, rep, hyd, hb))

        contrib = {k: w[k] * sums[k] for k in sums}
        inter = sum(contrib.values())
        denom = 1.0 + w['rot'] * nrot
        dg = (inter / denom) if denom else inter
        pk = round(-dg / 1.36, 3) if dg < 0 else None

        rows.sort(key=lambda t: t[0])            # most favourable (most negative) first

        # write the auditable log to pose.vina_breakdown_dir (out_dir overrides it)
        pcfg = _pose_cfg()
        brk_dir = (data.get('out_dir') or '').strip() or (pcfg.get('vina_breakdown_dir') or '').strip()
        log_path = None
        if brk_dir:
            try:
                root = Path(brk_dir).expanduser()
                root.mkdir(parents=True, exist_ok=True)
                log_path = root / (name + '_vina_breakdown.log')
                log_path.write_text(_render_vina_log(name, w, nrot, sums, contrib, inter, denom,
                                                     dg, pk, len(lig_atoms), len(near), npair, rows, t0, typing=typing))
                logger.info('[pose] vina_breakdown -> %s (%d pairs, dG=%.4f)', log_path, npair, dg)
            except Exception as _e:
                logger.warning('[pose] could not write vina breakdown log: %s', _e)
                log_path = None
        # else: pose.vina_breakdown_dir blank -> numbers returned, nothing written (per input_TS.yml)

        top = [{'lig': t[1], 'rec': t[2], 'r': round(t[3], 3), 's': round(t[4], 3),
                'pair_e': round(t[0], 5)} for t in rows[:12]]
        return jsonify({
            'ok': True,
            'dg': round(dg, 4),
            'inter': round(inter, 4),
            'nrot': nrot,
            'typing': typing,
            'npairs': npair,
            'terms': {k: round(contrib[k], 4) for k in contrib},
            'raw_sums': {k: round(sums[k], 4) for k in sums},
            'weights': {k: w[k] for k in w},
            'pk': pk,
            'top_pairs': top,
            'log': (str(log_path) if log_path else None),
            'out_dir': (str(log_path.parent) if log_path else None),
            'elapsed': '%.2fs' % (time.time() - t0),
            'source': 'vina',
        }), 200
    except Exception as e:
        logger.warning('[pose] vina_breakdown error: %s', e)
        return jsonify({'ok': False, 'err': 'server error: %s' % e}), 200


# =============================================================================
# Uploaded-file tracking (SQLite) + the receptors/ligands gallery
#
# All files uploaded via "⤓ Upload .pdb" are archived on disk in
# pose.default_upload_path. A small SQLite database (pose.uploads_db, or
# <default_upload_path>/../pose_uploads.db by default) tracks each file's
# classification — 'receptor' or 'ligand' — and its order within its section,
# so the two galleries and the user's drag-and-drop arrangement survive reloads.
# New files are auto-classified by atom count (small => ligand) on first sight;
# the user can drag a card to the other section to override it (persisted).
# =============================================================================
_POSE_LIGAND_MAX_ATOMS = 300     # first-sight heuristic: fewer heavy atoms => ligand


def _pose_pdb_meta(p):
    """Return {size, mtime, atoms, center} for a PDB path (single pass).
    center = bound-ligand centroid if present, else atom bounding-box center
    (matches parsePDB / what loading the file sets as the box center)."""
    import time as _t
    st = p.stat()
    atoms = 0
    mn = [1e9, 1e9, 1e9]; mx = [-1e9, -1e9, -1e9]
    hsum = [0.0, 0.0, 0.0]; hn = 0
    try:
        with p.open('r', errors='ignore') as fh:
            for ln in fh:
                rec = ln[:6]
                if rec not in ('ATOM  ', 'HETATM'):
                    continue
                try:
                    x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                except (ValueError, IndexError):
                    continue
                atoms += 1
                if x < mn[0]: mn[0] = x
                if y < mn[1]: mn[1] = y
                if z < mn[2]: mn[2] = z
                if x > mx[0]: mx[0] = x
                if y > mx[1]: mx[1] = y
                if z > mx[2]: mx[2] = z
                if rec == 'HETATM' and ln[17:20].strip() not in ('HOH', 'WAT', 'SO4', 'PO4'):
                    hsum[0] += x; hsum[1] += y; hsum[2] += z; hn += 1
    except Exception:
        atoms = None
    center = None
    if atoms:
        if hn:
            center = [round(hsum[0] / hn, 2), round(hsum[1] / hn, 2), round(hsum[2] / hn, 2)]
        elif mn[0] < 1e9:
            center = [round((mn[0] + mx[0]) / 2, 2), round((mn[1] + mx[1]) / 2, 2), round((mn[2] + mx[2]) / 2, 2)]
    return {'size': st.st_size, 'mtime': _t.strftime('%Y-%m-%d %H:%M', _t.localtime(st.st_mtime)),
            'atoms': atoms, 'center': center}


def _pose_uploads_db_path():
    """SQLite path: pose.uploads_db, else <default_upload_path>/../pose_uploads.db."""
    from pathlib import Path
    pcfg = _pose_cfg()
    dbp = (pcfg.get('uploads_db') or '').strip()
    if dbp:
        return str(Path(dbp).expanduser())
    updir = (pcfg.get('default_upload_path') or '').strip()
    if updir:
        return str(Path(updir).expanduser().parent / 'pose_uploads.db')
    return ''


def _pose_uploads_conn():
    """Open the uploads DB (creating dir + table). Returns a sqlite3 connection or None."""
    import sqlite3
    from pathlib import Path
    dbp = _pose_uploads_db_path()
    if not dbp:
        return None
    try:
        Path(dbp).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(dbp)
        conn.execute("""CREATE TABLE IF NOT EXISTS pose_uploads(
            name  TEXT PRIMARY KEY,
            kind  TEXT NOT NULL DEFAULT 'receptor',
            ord   INTEGER NOT NULL DEFAULT 0,
            size  INTEGER, atoms INTEGER,
            cx REAL, cy REAL, cz REAL,
            mtime TEXT, added_at TEXT, smiles TEXT)""")
        # migrate DBs created before the smiles column existed
        cols = {r[1] for r in conn.execute("PRAGMA table_info(pose_uploads)")}
        if 'smiles' not in cols:
            conn.execute("ALTER TABLE pose_uploads ADD COLUMN smiles TEXT")
        return conn
    except Exception as e:
        logger.warning('[pose] uploads DB open failed (%s): %s', dbp, e)
        return None


# =============================================================================
# GET /pose/uploaded_pdbs
# Scan pose.default_upload_path, reconcile it with the uploads DB (insert new
# files auto-classified by atom count, refresh metadata, drop rows for deleted
# files), and return the files split into two ordered sections.
#
# Returns: { ok, dir, db,
#            receptors: [ {name, kind, ord, size, mtime, atoms, center:[x,y,z]}, ... ],
#            ligands:   [ ... same shape ... ] }        (each ordered by the saved arrangement)
#          | { ok:false, err, receptors:[], ligands:[] }
# Metadata only (no file bodies); a chosen file's body is fetched via POST /pose/uploaded_pdb.
# =============================================================================
@app.route('/pose/uploaded_pdbs', methods=['GET'])
def pose_uploaded_pdbs():
    from pathlib import Path
    import time as _t
    updir = (_pose_cfg().get('default_upload_path') or '').strip()
    if not updir:
        return jsonify({'ok': False, 'err': 'no default_upload_path configured', 'receptors': [], 'ligands': []}), 200
    root = Path(updir).expanduser()

    # 1) gather on-disk .pdb/.ent files + metadata
    disk = {}
    if root.is_dir():
        try:
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() in ('.pdb', '.ent'):
                    try:
                        disk[p.name] = _pose_pdb_meta(p)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning('[pose] uploaded_pdbs listdir error: %s', e)
            return jsonify({'ok': False, 'err': str(e), 'receptors': [], 'ligands': []}), 200

    conn = _pose_uploads_conn()
    if conn is None:
        # DB unavailable -> classify by atom count without persistence
        recs, ligs = [], []
        for nm, m in sorted(disk.items(), key=lambda kv: kv[1]['mtime'], reverse=True):
            kind = 'ligand' if (m['atoms'] is not None and m['atoms'] < _POSE_LIGAND_MAX_ATOMS) else 'receptor'
            (ligs if kind == 'ligand' else recs).append(dict(m, name=nm, kind=kind, ord=0))
        return jsonify({'ok': True, 'dir': str(root), 'db': None, 'receptors': recs, 'ligands': ligs}), 200

    try:
        cur = conn.cursor()
        known = {r[0] for r in cur.execute("SELECT name FROM pose_uploads")}
        now = _t.strftime('%Y-%m-%d %H:%M:%S')
        # 2a) insert new files (auto-classified), refresh metadata for existing ones
        for nm, m in disk.items():
            c = m['center'] or [None, None, None]
            if nm not in known:
                kind = 'ligand' if (m['atoms'] is not None and m['atoms'] < _POSE_LIGAND_MAX_ATOMS) else 'receptor'
                nxt = cur.execute("SELECT COALESCE(MAX(ord), -1) + 1 FROM pose_uploads WHERE kind=?", (kind,)).fetchone()[0]
                cur.execute("INSERT INTO pose_uploads(name,kind,ord,size,atoms,cx,cy,cz,mtime,added_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (nm, kind, nxt, m['size'], m['atoms'], c[0], c[1], c[2], m['mtime'], now))
            else:
                cur.execute("UPDATE pose_uploads SET size=?,atoms=?,cx=?,cy=?,cz=?,mtime=? WHERE name=?",
                            (m['size'], m['atoms'], c[0], c[1], c[2], m['mtime'], nm))
        # 2b) drop rows whose file no longer exists on disk
        for nm in known:
            if nm not in disk:
                cur.execute("DELETE FROM pose_uploads WHERE name=?", (nm,))
        conn.commit()

        def _load(kind):
            out = []
            for r in cur.execute("SELECT name,kind,ord,size,atoms,cx,cy,cz,mtime FROM pose_uploads WHERE kind=? ORDER BY ord, mtime DESC", (kind,)):
                ctr = [r[5], r[6], r[7]] if r[5] is not None else None
                out.append({'name': r[0], 'kind': r[1], 'ord': r[2], 'size': r[3],
                            'atoms': r[4], 'center': ctr, 'mtime': r[8]})
            return out
        return jsonify({'ok': True, 'dir': str(root), 'db': _pose_uploads_db_path(),
                        'receptors': _load('receptor'), 'ligands': _load('ligand')}), 200
    except Exception as e:
        logger.warning('[pose] uploaded_pdbs DB error: %s', e)
        return jsonify({'ok': False, 'err': str(e), 'receptors': [], 'ligands': []}), 200
    finally:
        conn.close()


# =============================================================================
# POST /pose/uploaded_arrange   body: { "receptors": [name,...], "ligands": [name,...] }
# Persist the two-section arrangement after a drag: every listed file gets the
# given kind and an order equal to its index in that list. Called whenever the
# user drags a card within/between the "Uploaded receptors" / "Uploaded ligands"
# sections. Returns { ok, receptors, ligands } (counts) | { ok:false, err }.
# =============================================================================
@app.route('/pose/uploaded_arrange', methods=['POST'])
def pose_uploaded_arrange():
    data = request.get_json(silent=True) or {}
    receptors = data.get('receptors') or []
    ligands = data.get('ligands') or []
    conn = _pose_uploads_conn()
    if conn is None:
        return jsonify({'ok': False, 'err': 'uploads DB unavailable'}), 200
    try:
        cur = conn.cursor()
        for i, nm in enumerate(receptors):
            if isinstance(nm, str):
                cur.execute("UPDATE pose_uploads SET kind='receptor', ord=? WHERE name=?", (i, nm))
        for i, nm in enumerate(ligands):
            if isinstance(nm, str):
                cur.execute("UPDATE pose_uploads SET kind='ligand', ord=? WHERE name=?", (i, nm))
        conn.commit()
        return jsonify({'ok': True, 'receptors': len(receptors), 'ligands': len(ligands)}), 200
    except Exception as e:
        logger.warning('[pose] uploaded_arrange error: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200
    finally:
        conn.close()


# =============================================================================
# Deriving a SMILES for an uploaded ligand .pdb
# Priority: a sidecar "<name>.smi" file -> an embedded "REMARK ... SMILES ..."
# line -> RDKit (Chem.MolFromPDBBlock -> MolToSmiles). Returns None if none
# succeed (RDKit missing, or bond perception fails) — the caller then keeps the
# current SMILES rather than substituting a chemically wrong guess.
# =============================================================================
def _smiles_from_remark(pdb_text):
    import re as _re
    for ln in pdb_text.splitlines():
        if ln[:6].strip() == 'REMARK' and 'SMILES' in ln.upper():
            m = _re.search(r'SMILES[:=\s]+(\S+)', ln, _re.IGNORECASE)
            if m:
                return m.group(1)
    return None


def _smiles_via_rdkit(pdb_text):
    try:
        from rdkit import Chem
        try:
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except Exception:
            pass
        # preferred: perceive connectivity from 3D coords, then bond orders
        # (rdDetermineBonds) — more robust than pure distance-based PDB parsing,
        # especially for charged/aromatic groups.
        try:
            from rdkit.Chem import rdDetermineBonds
            raw = Chem.MolFromPDBBlock(pdb_text, removeHs=False, sanitize=False, proximityBonding=True)
            if raw is not None and raw.GetNumAtoms() > 0:
                rw = Chem.RWMol(raw)
                rdDetermineBonds.DetermineBonds(rw, charge=0)
                mol = Chem.RemoveHs(rw)
                if mol is not None and mol.GetNumAtoms() > 0:
                    smi = Chem.MolToSmiles(mol)
                    if smi and smi.strip():
                        return smi.strip()
        except Exception:
            pass
        # fallback: PDB-aware distance-based bond perception
        mol = Chem.MolFromPDBBlock(pdb_text, removeHs=True, sanitize=True)
        if mol is None:
            mol = Chem.MolFromPDBBlock(pdb_text, removeHs=True, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
        if mol is not None and mol.GetNumAtoms() > 0:
            smi = Chem.MolToSmiles(mol)
            if smi and smi.strip():
                return smi.strip()
    except Exception:
        pass
    return None


def _derive_ligand_smiles(pdb_text, path):
    from pathlib import Path
    # 1) sidecar <basename>.smi (first whitespace-delimited token)
    try:
        smi_path = Path(str(path)).with_suffix('.smi')
        if smi_path.is_file():
            toks = smi_path.read_text(errors='ignore').strip().split()
            if toks:
                return toks[0]
    except Exception:
        pass
    # 2) embedded REMARK SMILES
    s = _smiles_from_remark(pdb_text)
    if s:
        return s
    # 3) RDKit perception
    return _smiles_via_rdkit(pdb_text)


# =============================================================================
# POST /pose/uploaded_pdb   body: { "name": "<basename>.pdb", "smiles": <bool> }
# Return the text of one archived .pdb from pose.default_upload_path so the
# frontend can load it (clicking a card in the receptors/ligands gallery).
# When "smiles" is true (ligand click), also derive/cache and return a SMILES so
# the frontend can switch the ligand SMILES input to the clicked molecule.
#
# Returns: { ok, name, pdb, smiles? } | { ok:false, err }
# The name is treated as a basename only (no path components); anything with a
# separator, a non-.pdb/.ent suffix, or that resolves outside the upload dir is
# rejected, so this cannot be used to read arbitrary files.
# =============================================================================
@app.route('/pose/uploaded_pdb', methods=['POST'])
def pose_uploaded_pdb():
    from pathlib import Path
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    updir = (_pose_cfg().get('default_upload_path') or '').strip()
    if not updir:
        return jsonify({'ok': False, 'err': 'no default_upload_path configured'}), 200
    base = Path(name).name                                   # strip any path components
    if not base or base != name or base.startswith('.') or Path(base).suffix.lower() not in ('.pdb', '.ent'):
        return jsonify({'ok': False, 'err': 'invalid filename'}), 200
    try:
        root = Path(updir).expanduser().resolve()
        fp = (root / base).resolve()
        if fp.parent != root:                               # reject symlink / traversal escapes
            return jsonify({'ok': False, 'err': 'path outside upload dir'}), 200
        if not fp.is_file():
            return jsonify({'ok': False, 'err': 'file not found: %s' % base}), 200
        txt = fp.read_text(errors='ignore')
        resp = {'ok': True, 'name': base, 'pdb': txt}
        if data.get('smiles'):
            smi = None
            conn = _pose_uploads_conn()
            if conn is not None:
                try:
                    row = conn.execute("SELECT smiles FROM pose_uploads WHERE name=?", (base,)).fetchone()
                    if row and row[0]:
                        smi = row[0]                         # cached
                except Exception:
                    pass
            if not smi:
                smi = _derive_ligand_smiles(txt, fp)         # sidecar / REMARK / RDKit
                if smi and conn is not None:
                    try:
                        conn.execute("UPDATE pose_uploads SET smiles=? WHERE name=?", (smi, base))
                        conn.commit()
                    except Exception:
                        pass
            if conn is not None:
                conn.close()
            resp['smiles'] = smi
        return jsonify(resp), 200
    except Exception as e:
        logger.warning('[pose] uploaded_pdb error: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200

# ── Reaction builder ─────────────────────────────────────────────────────────
# React a ligand SMILES with every building block in a CSV via a reaction SMARTS,
# returning the product SMILES (canonical, de-duplicated). Consumed by the
# "React" popover on the pose 3D pane (pose.js -> PoseGen.reactConfirm).
def _read_bb_smiles(path, Chem, scan=25):
    """Return the list of SMILES strings from a building-block CSV.

    Sniffs comma vs tab, detects the SMILES column by header name, and falls
    back to the column whose sample cells parse as molecules. Works with files
    that have a header row or none.
    """
    import os
    import csv as _csv
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    # Sniff delimiter from the first non-empty line (default comma).
    delim = ','
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if line.strip():
                    delim = '\t' if line.count('\t') > line.count(',') else ','
                    break
    except Exception:
        pass

    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        rows_all = list(_csv.reader(fh, delimiter=delim))
    if not rows_all:
        return []

    first = rows_all[0]
    looks_header = any(
        ('smile' in str(c).lower() or 'name' in str(c).lower()
         or 'id' in str(c).lower() or 'reag' in str(c).lower())
        for c in first
    )
    header = [str(c) for c in first] if looks_header else None
    rows = rows_all[1:] if looks_header else rows_all

    col = None
    if header:
        for i, h in enumerate(header):
            if h.strip().lower() in ('smiles', 'smi', 'canonical_smiles', 'structure', 'smiles_1'):
                col = i
                break
    if col is None:
        ncol = max((len(r) for r in rows[:scan]), default=0)
        best_i, best_ok = None, 0
        for i in range(ncol):
            ok = tot = 0
            for r in rows[:scan]:
                if i < len(r) and str(r[i]).strip():
                    tot += 1
                    if Chem.MolFromSmiles(str(r[i]).strip()) is not None:
                        ok += 1
            if tot and (ok / tot) >= 0.6 and ok > best_ok:
                best_i, best_ok = i, ok
        col = best_i
    if col is None:
        return []

    out = []
    for r in rows:
        if col < len(r):
            s = str(r[col]).strip()
            if s:
                out.append(s)
    return out


@app.route('/pose/react', methods=['POST'])
def pose_react():
    """React #poseSmiles with each building block in a CSV via a reaction SMARTS.

    Body: {smiles, smarts, csv_path, [limit]}
    Returns: {ok, products:[canonical SMILES], n_products, n_bb, n_bb_total,
              limit, truncated}
    Reactant order in the SMARTS is unknown, so each building block is tried in
    both slots (ligand, bb) and (bb, ligand); the first order that fires wins.
    """
    data = request.get_json(silent=True) or {}
    ligand = (data.get('smiles') or '').strip()
    smarts = (data.get('smarts') or '').strip()
    csv_path = (data.get('csv_path') or '').strip()
    try:
        limit = max(1, int(data.get('limit') or 500))
    except Exception:
        limit = 500

    if not ligand or not smarts or not csv_path:
        return jsonify({"ok": False, "err": "smiles, smarts and csv_path are required"}), 200

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        return jsonify({"ok": False, "err": "rdkit not installed on server"}), 200

    try:
        bbs = _read_bb_smiles(csv_path, Chem)
    except FileNotFoundError:
        return jsonify({"ok": False, "err": "CSV not found: %s" % csv_path}), 200
    except Exception as e:
        return jsonify({"ok": False, "err": "could not read CSV: %s" % e}), 200
    if not bbs:
        return jsonify({"ok": False, "err": "no SMILES column detected in CSV"}), 200

    n_bb_total = len(bbs)
    truncated = n_bb_total > limit
    if truncated:
        bbs = bbs[:limit]

    rxn = AllChem.ReactionFromSmarts(smarts)
    if rxn is None:
        return jsonify({"ok": False, "err": "invalid reaction SMARTS"}), 200
    try:
        rxn.Initialize()
    except Exception:
        pass
    n_react = rxn.GetNumReactantTemplates()

    lig = Chem.MolFromSmiles(ligand)
    if lig is None:
        return jsonify({"ok": False, "err": "invalid ligand SMILES"}), 200

    products, seen, reacted = [], set(), 0
    for bb_smi in bbs:
        bb = Chem.MolFromSmiles(bb_smi)
        if bb is None:
            continue
        orders = [(lig, bb), (bb, lig)] if n_react >= 2 else [(lig,), (bb,)]
        outs = None
        for combo in orders:
            if len(combo) != n_react:
                continue
            try:
                res = rxn.RunReactants(combo)
            except Exception:
                res = None
            if res:
                outs = res
                break
        if not outs:
            continue
        got = False
        for tup in outs:
            for p in tup:
                try:
                    Chem.SanitizeMol(p)
                    smi = Chem.MolToSmiles(p)
                except Exception:
                    continue
                if smi and smi not in seen:
                    seen.add(smi)
                    products.append(smi)
                    got = True
        if got:
            reacted += 1

    return jsonify({
        "ok": True,
        "products": products,
        "n_products": len(products),
        "n_bb": reacted,
        "n_bb_total": n_bb_total,
        "limit": limit,
        "truncated": truncated,
    }), 200

# ── Import a folder of .pdbqt files ──────────────────────────────────────────
# Recursively find .pdbqt files under a folder, then serve them one at a time.
# Consumed by the "Import → scan folder for .pdbqt" control on the pose 3D
# pane (pose.js -> PoseGen.importConfirm / PoseGen._impLoad), which steps
# through the results with the same ◀ ▶ ▶ Play stepper the reacted products
# use. Paths are trusted as given (same model as /pose/react's csv_path) --
# this tool already reads arbitrary server-side paths for that feature.
_POSE_SCAN_MAX_FILES = 5000   # safety ceiling on a single recursive scan
_POSE_SCAN_MAX_LINES = 8000   # per-file line cap while probing metadata


def _pdbqt_probe(p):
    """Cheap per-file metadata: best Vina score + pose-1 heavy/H atom count.

    Vina writes its best pose first and puts the affinity on the file's second
    line, so this stops at the first ENDMDL — probing a folder of thousands of
    docked results stays fast. Files with no MODEL wrapper (prepared receptors)
    are read up to the line cap, which is far more than the 300-atom ligand
    threshold needs. Returns {'score': float|None, 'atoms': int} or None.
    """
    score, atoms = None, 0
    try:
        with p.open('r', errors='ignore') as fh:
            for i, ln in enumerate(fh):
                if i > _POSE_SCAN_MAX_LINES:
                    break
                head = ln[:6].rstrip()
                if head == 'ENDMDL':
                    break                        # end of the best pose → done
                if head in ('ATOM', 'HETATM'):
                    atoms += 1
                elif score is None and ln.startswith('REMARK') and 'VINA RESULT' in ln:
                    try:
                        score = float(ln.split(':', 1)[1].split()[0])
                    except (IndexError, ValueError):
                        pass
    except Exception:
        return None
    return {'score': score, 'atoms': atoms}


# =============================================================================
# POST /pose/scan_pdbqt
# Body: {"path": "/abs/or/~/folder"}
# Returns: {ok, dir, count, truncated, n_ligands, n_receptors, n_scored,
#           files:[{path, score, atoms, kind}, ...]}
#        | {ok:false, err}
# Each entry is probed for its best Vina affinity and pose-1 atom count, so the
# frontend can tally ligands vs receptors and filter by score without re-reading
# anything. File bodies are still fetched one at a time via POST /pose/read_pdbqt
# as the stepper visits them.
# =============================================================================
# Suffixes the folder import understands.
#
# This used to be '.pdbqt' alone, which made the tool unable to re-open its own
# output: "⬇ Download filtered batch" converts every match to .pdb and zips it,
# so the moment a user unzipped that batch and pointed the importer at it, the
# scan reported "no .pdbqt files found" — the one folder guaranteed to be full of
# real, scored, best-pose structures was the one folder it could not read.
# _pdbqt_scores already parses the REMARK line those .pdb files carry, and
# _pdbqt_probe already tolerates a file with no MODEL wrapper, so the only thing
# standing in the way was the extension test.
_POSE_IMPORT_SUFFIXES = ('.pdbqt', '.pdb', '.ent')


@app.route('/pose/scan_pdbqt', methods=['POST'])
def pose_scan_pdbqt():
    from pathlib import Path
    data = request.get_json(silent=True) or {}
    raw = (data.get('path') or '').strip()
    if not raw:
        return jsonify({'ok': False, 'err': 'no folder path given'}), 200
    try:
        root = Path(raw).expanduser()
    except Exception as e:
        return jsonify({'ok': False, 'err': 'bad path: %s' % e}), 200
    if not root.is_dir():
        return jsonify({'ok': False, 'err': 'not a folder: %s' % str(root)}), 200

    paths, truncated = [], False
    try:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in _POSE_IMPORT_SUFFIXES:
                paths.append(p)
                if len(paths) >= _POSE_SCAN_MAX_FILES:
                    truncated = True
                    break
    except Exception as e:
        logger.warning('[pose] scan_pdbqt error: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200

    if not paths:
        return jsonify({'ok': False, 'err': 'no %s files found under %s'
                        % ('/'.join(_POSE_IMPORT_SUFFIXES), str(root))}), 200

    paths.sort()
    files, n_lig, n_rec, n_scored = [], 0, 0, 0
    for p in paths:
        meta = _pdbqt_probe(p) or {'score': None, 'atoms': 0}
        kind = 'ligand' if meta['atoms'] <= _POSE_LIGAND_MAX_ATOMS else 'receptor'
        if kind == 'ligand':
            n_lig += 1
        else:
            n_rec += 1
        if meta['score'] is not None:
            n_scored += 1
        files.append({'path': str(p), 'score': meta['score'],
                      'atoms': meta['atoms'], 'kind': kind})

    return jsonify({'ok': True, 'dir': str(root), 'files': files,
                    'count': len(files), 'truncated': truncated,
                    'n_ligands': n_lig, 'n_receptors': n_rec,
                    'n_scored': n_scored}), 200


# =============================================================================
# POST /pose/read_pdbqt
# Body: {"path": "/abs/file.pdbqt"}
# Returns: {ok, name, path, pdb, [n_models, note]} | {ok:false, err}
# Same response shape as /pose/uploaded_pdb ({ok, name, pdb}) so the frontend
# reuses the exact same PE.parsePDB(res.pdb) parsing path. PDBQT differs from
# PDB in two ways parsePDB doesn't understand — see _pdbqt_to_pdb below.
# =============================================================================
# ── PDBQT → PDB conversion ───────────────────────────────────────────────────
# The client-side parsePDB reads standard PDB (element symbol in cols 76-77,
# one contiguous atom block) and knows nothing about PDBQT extras:
#   · AutoDock atom types (A=aromatic C, OA/NA/SA=polar O/N/S, HD/HS=polar H)
#     sit where the element symbol should be, so raw parsing colors aromatic
#     carbons as unknown "A".
#   · Vina *_out.pdbqt* files pack every docked pose into its own MODEL block;
#     without MODEL awareness parsePDB reads all N × pose_atoms as one blob,
#     so a 20-pose file with a 50-atom ligand looks like 1000 chaotic
#     superimposed atoms and the client-side classifier (>300 atoms ⇒
#     receptor) fires on what is really just one ligand pose.
# The map covers the AD 4.2 atom-type set; unknown types fall back to the
# first character (a safe approximation for C/N/O/S/P/H/F/I).
_AD_TO_ELEM = {
    'A': 'C', 'C': 'C',
    'N': 'N', 'NA': 'N', 'NS': 'N', 'NB': 'N',
    'O': 'O', 'OA': 'O', 'OS': 'O',
    'S': 'S', 'SA': 'S',
    'H': 'H', 'HD': 'H', 'HS': 'H', 'HB': 'H',
    'P': 'P', 'F': 'F', 'I': 'I',
    'CL': 'Cl', 'BR': 'Br',
    'MG': 'Mg', 'CA': 'Ca', 'MN': 'Mn', 'FE': 'Fe', 'ZN': 'Zn', 'CU': 'Cu',
}


def _pdbqt_atom_line(ln):
    """One PDBQT ATOM/HETATM line → standard PDB.

    The trailing AutoDock atom type is mapped back to a plain element symbol and
    written to the standard PDB element column (cols 76-77), which is what the
    client's ln.slice(76,78) reads. Columns 0-65 (record → temp factor) survive
    unchanged; the pdbqt partial charge past col 66 is dropped.
    """
    # AutoDock atom type = last whitespace-separated token past the Z column
    tail = ln[54:].split()
    adt = tail[-1].strip().upper() if tail else ''
    elem = _AD_TO_ELEM.get(adt, adt[:1] if adt else '')
    if not elem:                                       # last-ditch: atom name (cols 12-15)
        nm = ln[12:16].strip().upper()
        elem = nm[0] if nm else 'C'
    if len(elem) == 2:                                 # 'Cl'/'Br'/'Mg'… PDB convention: 2nd char lowercase
        elem = elem[0] + elem[1].lower()
    return ln[:66].rstrip().ljust(66) + '          ' + elem.rjust(2)


def _perceive_pdb_orders(atoms, bonds):
    """Bond orders from geometry — the Python port of the frontend's
    perceivePdbBondOrders. RDKit's coordinate-based perception is unreliable on
    docked poses (it returned all-single for real Vina output), so this uses the
    same planarity + bond-length heuristic that works client-side:
      · planar 5-/6-rings of C/N/O/S → alternating Kekulé doubles (benzene → 3)
      · non-ring bonds shorter than the single-bond reference (C=O, C=N, C=C) → 2
    `atoms` is a list of (el, x, y, z); `bonds` is a list of (i, j). Returns a
    parallel list of orders (1/2). Deterministic, no external dependency.
    """
    import math
    n = len(atoms)
    order = [1] * len(bonds)
    bidx = {}
    for i, (a, b) in enumerate(bonds):
        bidx[(a, b) if a < b else (b, a)] = i
    adj = [[] for _ in range(n)]
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)

    def P(i):
        return atoms[i]

    def D(i, j):
        ax, ay, az = atoms[i][1], atoms[i][2], atoms[i][3]
        bx, by, bz = atoms[j][1], atoms[j][2], atoms[j][3]
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    # small rings (size 5,6) via bounded DFS, dedup by sorted membership
    seen, rings = set(), []

    def dfs(start, cur, prev, path):
        if len(path) > 6:
            return
        for nx in adj[cur]:
            if nx == prev:
                continue
            if nx == start and 5 <= len(path) <= 6:
                key = tuple(sorted(path))
                if key not in seen:
                    seen.add(key)
                    rings.append(list(path))
                continue
            if nx in path:
                continue
            path.append(nx)
            dfs(start, nx, cur, path)
            path.pop()

    for i in range(n):
        if len(adj[i]) >= 2:
            dfs(i, i, -1, [i])

    def arom_el(el):
        return el in ('C', 'N', 'O', 'S')

    used_dbl = [False] * n

    for ring in rings:
        m = len(ring)
        if not all(arom_el(atoms[i][0]) for i in ring):
            continue
        a, b, c = P(ring[0]), P(ring[1]), P(ring[2])
        ux, uy, uz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
        vx, vy, vz = c[1] - a[1], c[2] - a[2], c[3] - a[3]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl < 1e-3:
            continue
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        maxdev = 0.0
        for i in ring:
            pi = P(i)
            dv = abs((pi[1] - a[1]) * nx + (pi[2] - a[2]) * ny + (pi[3] - a[3]) * nz)
            maxdev = max(maxdev, dv)
        if maxdev > 0.35:                        # non-planar → not aromatic
            continue
        # order the ring as a cycle
        cyc = [ring[0]]
        prev, cur = -1, ring[0]
        for _ in range(m - 1):
            nxt = None
            for w in adj[cur]:
                if w != prev and w in ring and w not in cyc:
                    nxt = w
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
            cyc.append(cur)
        if len(cyc) != m:
            continue
        for k in range(m):
            i, j = cyc[k], cyc[(k + 1) % m]
            if used_dbl[i] or used_dbl[j]:
                continue
            bi = bidx.get((i, j) if i < j else (j, i))
            if bi is None:
                continue
            if D(i, j) > 1.62:
                continue
            order[bi] = 2
            used_dbl[i] = used_dbl[j] = True

    REF = {'C-C': 1.54, 'C-N': 1.47, 'C-O': 1.43, 'N-O': 1.44,
           'C-S': 1.82, 'N-N': 1.45, 'O-P': 1.63, 'O-S': 1.57}
    # triple-bond upper bounds (Å): a C≡C ~1.20, C≡N ~1.16 — clearly shorter than a double
    TRIP = {'C-C': 1.27, 'C-N': 1.24, 'N-N': 1.20}
    for i, (a, b) in enumerate(bonds):
        if order[i] != 1:
            continue
        ea, eb = atoms[a][0], atoms[b][0]
        if ea == 'H' or eb == 'H':
            continue
        if used_dbl[a] or used_dbl[b]:
            continue
        pk = '-'.join(sorted((ea, eb)))
        d = D(a, b)
        tri = TRIP.get(pk)
        if tri is not None and d < tri:
            order[i] = 3
            used_dbl[a] = used_dbl[b] = True
            continue
        ref = REF.get(pk)
        if ref is None:
            continue
        if d < ref - 0.13:
            order[i] = 2
            used_dbl[a] = used_dbl[b] = True
    return order


def _infer_bonds_py(atoms):
    """Distance-based connectivity, mirroring the frontend's inferBonds."""
    import math
    COV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'P': 1.07,
           'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39, 'B': 0.84, 'Se': 1.20}
    n = len(atoms)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = atoms[i][1] - atoms[j][1]
            dy = atoms[i][2] - atoms[j][2]
            dz = atoms[i][3] - atoms[j][3]
            d2 = dx * dx + dy * dy + dz * dz
            mx = COV.get(atoms[i][0], 0.76) + COV.get(atoms[j][0], 0.76) + 0.45
            if 0.20 < d2 < mx * mx:
                bonds.append((i, j))
    return bonds


def _parse_pdb_atoms(atom_lines):
    """(element, x, y, z) from PDB ATOM/HETATM lines. Element from cols 76-77,
    falling back to the atom name."""
    out = []
    for ln in atom_lines:
        try:
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError:
            continue
        el = ln[76:78].strip()
        if not el:
            nm = ln[12:16].strip()
            el = ''.join(ch for ch in nm if ch.isalpha())[:2]
        if len(el) == 2:
            el = el[0] + el[1].lower()
        out.append((el, x, y, z))
    return out


def _pdb_block_with_conect(block):
    """Append Maestro-readable CONECT records (with bond orders) to a PDB block.

    A .pdbqt carries no bond orders, so Maestro/PyMOL show single sticks only.
    This perceives connectivity and orders from the 3D coordinates (geometry
    heuristic — see _perceive_pdb_orders) and writes CONECT in the SYMMETRIC form
    Maestro honours: for a bond a-b of order n, atom a lists b n times AND atom b
    lists a n times. (RDKit's MolToPDBBlock writes doubles one-directionally,
    which Maestro silently ignores — that was the bug.)

    `block` is a list of PDB lines. Returns a new list with CONECT inserted
    before END. On any failure the block is returned unchanged (single bonds
    only — never worse than before; geometry is always correct).
    """
    try:
        atom_lines = [l for l in block if l[:6].rstrip() in ('ATOM', 'HETATM')]
        if len(atom_lines) < 2:
            return block
        atoms = _parse_pdb_atoms(atom_lines)
        if len(atoms) != len(atom_lines):
            return block
        bonds = _infer_bonds_py(atoms)
        if not bonds:
            return block
        orders = _perceive_pdb_orders(atoms, bonds)

        from collections import defaultdict
        adj = defaultdict(list)                  # 1-based serials, neighbour repeated per order
        for (a, b), o in zip(bonds, orders):
            ia, ib = a + 1, b + 1
            for _ in range(o):
                adj[ia].append(ib)
                adj[ib].append(ia)               # SYMMETRIC — the part Maestro needs

        conect = []
        for a in sorted(adj):
            nb = sorted(adj[a])
            for k in range(0, len(nb), 4):       # ≤4 partners per CONECT line (PDB spec)
                grp = nb[k:k + 4]
                conect.append('CONECT' + ('%5d' % a) + ''.join('%5d' % nn for nn in grp))

        out = [l for l in block if l.strip() != 'END']
        out.extend(conect)
        return out
    except Exception:
        return block


def _pdbqt_models_to_pdb(text, already_pdb=False):
    """Convert every pose in a .pdbqt to PDB blocks.

    Converts all of them so callers can choose: the download route keeps only
    models[0] (Vina's best), and _best_poses.pdb reuses that same block. Returns
    (scores, models, had_models) where models[i] is the PDB lines for pose i+1,
    prefixed with a REMARK carrying that pose's Vina affinity. PDBQT-only records
    (ROOT/BRANCH/TORSDOF) are dropped — they have no meaning in PDB. had_models
    is False for files with no MODEL wrapper, i.e. prepared receptors.

    already_pdb=True is for the .pdb/.ent inputs the importer now accepts
    (including this tool's own downloaded batches). Those lines already carry a
    real element symbol in columns 76-77, so rewriting them through
    _pdbqt_atom_line — which infers the element from the last token past column
    54 — would be guessing at data that is already correct.
    """
    scores, models = [], []
    cur, cur_score, had_models = [], None, False
    for ln in text.splitlines():
        head = ln[:6].rstrip()
        if head == 'MODEL':
            if cur:
                models.append(cur), scores.append(cur_score)
            cur, cur_score, had_models = [], None, True
            continue
        if head == 'ENDMDL':
            if cur:
                models.append(cur), scores.append(cur_score)
            cur, cur_score = [], None
            continue
        if cur_score is None and ln.startswith('REMARK') and 'VINA RESULT' in ln:
            try:
                cur_score = float(ln.split(':', 1)[1].split()[0])
            except (IndexError, ValueError):
                pass
            continue
        if head in ('ATOM', 'HETATM'):
            cur.append(ln.rstrip('\n') if already_pdb else _pdbqt_atom_line(ln))
    if cur:
        models.append(cur), scores.append(cur_score)

    out = []
    for blk, sc in zip(models, scores):
        pre = ['REMARK 999 VINA RESULT: %.2f kcal/mol' % sc] if sc is not None else []
        out.append(pre + blk)
    return scores, out, had_models


def _pdb_text(models, had_models):
    """Assemble PDB blocks into one file, wrapping poses in MODEL/ENDMDL."""
    lines = []
    if had_models or len(models) > 1:
        for i, blk in enumerate(models, 1):
            lines.append('MODEL     %4d' % i)
            lines.extend(blk)
            lines.append('ENDMDL')
    else:
        for blk in models:
            lines.extend(blk)
    lines.append('END')
    return '\n'.join(lines) + '\n'


def _pdbqt_to_pdb(text):
    """Rewrite PDBQT text as a standard PDB that parsePDB can render correctly.

    Only the first MODEL block is kept (Vina packs docked poses into successive
    MODELs — the first is Vina's top-scoring pose). Files with no MODEL wrapper
    (prepared receptors and some ligand inputs) pass through as one block.
    Each ATOM/HETATM line's trailing AutoDock atom type is mapped back to a
    plain element symbol and written to the standard PDB element column
    (cols 76-77), which is what the client's ln.slice(76,78) reads.
    """
    lines = []
    seen_model = False
    for ln in text.splitlines():
        head = ln[:6].rstrip()
        if head == 'MODEL':
            if seen_model:                             # 2nd MODEL onwards → stop
                break
            seen_model = True
            continue
        if head == 'ENDMDL':
            if seen_model:                             # finished the 1st pose → done
                break
            continue
        if head not in ('ATOM', 'HETATM'):
            continue
        lines.append(_pdbqt_atom_line(ln))
    if not lines:                                      # nothing recognised — hand back the raw file
        return text
    return '\n'.join(lines) + '\n'


def _pdb_first_model(text):
    """Keep MODEL 1 of an already-standard .pdb, verbatim.

    Deliberately NOT _pdbqt_to_pdb. That function rebuilds every atom line and
    derives the element from the last whitespace token past column 54 — correct
    for PDBQT, where the AutoDock type sits there, but wrong in principle for a
    real PDB, which already carries the element in columns 76-77 where the
    client's ln.slice(76,78) reads it. Passing ATOM/HETATM through untouched
    keeps two-character elements (Br, Cl, Fe) and occupancy/B-factor columns
    exactly as written.

    _best_poses.pdb from the download button holds one MODEL per ligand, so the
    same first-MODEL rule the PDBQT path uses applies here: show pose 1, and let
    the caller report how many there were.
    """
    lines, seen_model = [], False
    for ln in text.splitlines():
        head = ln[:6].rstrip()
        if head == 'MODEL':
            if seen_model:
                break                                  # 2nd MODEL onwards → stop
            seen_model = True
            continue
        if head == 'ENDMDL':
            if seen_model:
                break
            continue
        if head in ('ATOM', 'HETATM'):
            lines.append(ln.rstrip('\n'))
    if not lines:
        return text
    return '\n'.join(lines) + '\n'



def _pdbqt_scores(text):
    """Vina affinities (kcal/mol) from a docked *_out.pdbqt*, best pose first.

    Vina writes one 'REMARK VINA RESULT:  <affinity>  <rmsd_lb>  <rmsd_ub>'
    per MODEL, already sorted best (most negative) first. Files that aren't
    Vina output (prepared receptors / input ligands) have none → [].
    """
    out = []
    for ln in text.splitlines():
        if not ln.startswith('REMARK') or 'VINA RESULT' not in ln:
            continue
        try:
            out.append(float(ln.split(':', 1)[1].split()[0]))
        except (IndexError, ValueError):
            continue
    return out


@app.route('/pose/read_pdbqt', methods=['POST'])
def pose_read_pdbqt():
    from pathlib import Path
    data = request.get_json(silent=True) or {}
    raw = (data.get('path') or '').strip()
    if not raw:
        return jsonify({'ok': False, 'err': 'no file path given'}), 200
    suffix = Path(raw).suffix.lower()
    if suffix not in _POSE_IMPORT_SUFFIXES:
        return jsonify({'ok': False, 'err': 'not a %s file: %s'
                        % ('/'.join(_POSE_IMPORT_SUFFIXES), raw)}), 200
    try:
        fp = Path(raw).expanduser()
        if not fp.is_file():
            return jsonify({'ok': False, 'err': 'file not found: %s' % raw}), 200
        raw_text = fp.read_text(errors='ignore')
        # Count MODELs before conversion (conversion keeps only the first)
        n_models = sum(1 for ln in raw_text.splitlines() if ln[:6].rstrip() == 'MODEL')
        # .pdbqt needs its AutoDock atom types mapped back to elements; .pdb/.ent
        # is already in the format parsePDB reads, so it passes through untouched.
        pdb = _pdbqt_to_pdb(raw_text) if suffix == '.pdbqt' else _pdb_first_model(raw_text)
        resp = {'ok': True, 'name': fp.name, 'path': str(fp), 'pdb': pdb}
        if n_models > 1:
            resp['n_models'] = n_models
            resp['note'] = 'showing pose 1 of %d' % n_models
        scores = _pdbqt_scores(raw_text)
        if scores:
            resp['score'] = scores[0]                  # pose 1 = Vina's best
            resp['scores'] = scores                    # every pose, best first
        return jsonify(resp), 200
    except Exception as e:
        logger.warning('[pose] read_pdbqt error: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200


# ── Download the filtered batch ──────────────────────────────────────────────
_POSE_ZIP_MAX_BYTES = 1 << 29     # 512 MB uncompressed budget for one download




# =============================================================================
# POST /pose/download_pdbqt
# Body: {"paths": ["/abs/a.pdbqt", ...], "name": "results_p100"}
# Returns: a .zip attachment of PDB files converted from the requested .pdbqt —
#          the best pose only (Vina's MODEL 1), AutoDock atom types mapped back
#          to real elements, plus _manifest.csv (score / atoms / source pose
#          count per file, best first) and _best_poses.pdb (all of them
#          concatenated as one multi-model PDB that loads as a single batch).
#          Counts ride back on X-Pose-Zip-* headers; errors return JSON.
# Paths come from the client's current filter, so the download always matches
# exactly what the stepper is showing.
# =============================================================================
@app.route('/pose/download_pdbqt', methods=['POST'])
def pose_download_pdbqt():
    from pathlib import Path
    import csv, io, os, tempfile, zipfile
    from flask import send_file, after_this_request

    data = request.get_json(silent=True) or {}
    raw_paths = data.get('paths') or []
    label = re.sub(r'[^A-Za-z0-9._-]+', '_', (data.get('name') or 'filtered').strip()) or 'filtered'
    if not isinstance(raw_paths, list) or not raw_paths:
        return jsonify({'ok': False, 'err': 'no files given'}), 200

    files = []
    for p in raw_paths[:_POSE_SCAN_MAX_FILES]:
        try:
            fp = Path(str(p)).expanduser()
        except Exception:
            continue
        if fp.is_file() and fp.suffix.lower() in _POSE_IMPORT_SUFFIXES:
            files.append(fp)
    if not files:
        return jsonify({'ok': False, 'err': 'none of those paths are readable %s files'
                        % '/'.join(_POSE_IMPORT_SUFFIXES)}), 200

    # Keep subfolder structure so same-named files in sibling batches don't collide
    try:
        base = Path(os.path.commonpath([str(f.parent) for f in files]))
    except Exception:
        base = None

    try:
        tmp = tempfile.NamedTemporaryFile(prefix='pose_batch_', suffix='.zip', delete=False)
        tmp.close()
    except Exception as e:
        logger.warning('[pose] download_pdbqt tempfile: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200

    @after_this_request
    def _cleanup(resp):                        # unlinking an open file is fine on POSIX
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return resp

    rows, best_poses = [], []
    used, added, skipped, total = set(), 0, 0, 0
    truncated = False
    try:
        with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for fp in files:
                if total >= _POSE_ZIP_MAX_BYTES:
                    truncated = True
                    break
                try:
                    raw = fp.read_text(errors='ignore')
                except Exception:
                    skipped += 1
                    continue

                scores, models, had_models = _pdbqt_models_to_pdb(
                    raw, already_pdb=(fp.suffix.lower() != '.pdbqt'))
                if not models:                 # nothing convertible in this file
                    skipped += 1
                    continue
                # Vina sorts poses best-first, so MODEL 1 is the top hit — keep
                # only that. One pose per file means no MODEL wrapper is needed.
                best_block = _pdb_block_with_conect(models[0])     # add Maestro-readable bond orders
                pdb = _pdb_text([best_block], False)
                total += len(pdb)

                arc = fp.name
                if base is not None:
                    try:
                        arc = str(fp.relative_to(base))
                    except ValueError:
                        arc = fp.name
                # Strip whichever accepted suffix the source had before appending
                # .pdb. This was '\.pdbqt$' only, which turned a re-downloaded
                # batch's "foo.pdb" into "foo.pdb.pdb".
                arc = re.sub(r'\.(pdbqt|pdb|ent)$', '', arc, flags=re.I) + '.pdb'
                if arc in used:                # last-ditch de-dupe
                    stem, ext = os.path.splitext(arc)
                    n = 2
                    while '%s_%d%s' % (stem, n, ext) in used:
                        n += 1
                    arc = '%s_%d%s' % (stem, n, ext)
                used.add(arc)

                zf.writestr('files/' + arc, pdb)
                added += 1

                score = scores[0] if scores else None
                atoms = sum(1 for l in models[0] if l[:6].rstrip() in ('ATOM', 'HETATM'))
                rows.append({'file': arc, 'score_kcal_mol': ('' if score is None else score),
                             'atoms': atoms, 'source_poses': len(models), 'source': str(fp)})
                if had_models:
                    best_poses.append((score, arc, models[0]))

            # manifest, best score first (unscored last)
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=['file', 'score_kcal_mol', 'atoms', 'source_poses', 'source'])
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r['score_kcal_mol'] == '', r['score_kcal_mol'])):
                w.writerow(r)
            zf.writestr('_manifest.csv', buf.getvalue())

            if best_poses:
                best_poses.sort(key=lambda t: (t[0] is None, t[0]))
                # each MODEL gets its own CONECT block (per-model bond orders)
                blocks = [_pdb_block_with_conect(['REMARK 999 SOURCE %s' % arc] + blk)
                          for _s, arc, blk in best_poses]
                zf.writestr('_best_poses.pdb', _pdb_text(blocks, True))
    except Exception as e:
        logger.warning('[pose] download_pdbqt zip error: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200

    zip_name = '%s_%d_pdb.zip' % (label, added)
    try:
        resp = send_file(tmp.name, as_attachment=True,
                         download_name=zip_name, mimetype='application/zip')
    except TypeError:                          # Flask < 2.0
        resp = send_file(tmp.name, as_attachment=True,
                         attachment_filename=zip_name, mimetype='application/zip')
    resp.headers['X-Pose-Zip-Count'] = str(added)
    resp.headers['X-Pose-Zip-Skipped'] = str(skipped)
    resp.headers['X-Pose-Zip-Poses'] = str(len(best_poses))
    resp.headers['X-Pose-Zip-Truncated'] = '1' if truncated else '0'
    return resp



# ── Full debug log to disk ───────────────────────────────────────────────────
# The browser console truncates long runs and is awkward to share, so pose.js
# buffers every [Pose3D] line (with a timestamp, and NOT truncated) and POSTs it
# here. Frontend: PG._log wrapper → PG.dumpLog() · auto-flushes every few seconds.
_POSE_DEBUG_DIR = _pcfg.POSE_DEBUG_DIR


# =============================================================================
# POST /pose/debug_log
# Body: {"text": "<log lines>", "session": "<id>", "reset": bool, "dir": "<opt>"}
# Returns: {ok, path, bytes} | {ok:false, err}
# One file per browser session: the first flush truncates it, later flushes
# append, so the file always holds the complete run rather than a fragment.
# =============================================================================
@app.route('/pose/debug_log', methods=['POST'])
def pose_debug_log():
    from pathlib import Path
    data = request.get_json(silent=True) or {}
    text = data.get('text') or ''
    if not text:
        return jsonify({'ok': False, 'err': 'empty log'}), 200
    # Separators are stripped, so the result is always a single filename component
    # and cannot escape the directory; trimming ._- keeps odd input from producing
    # names like "pose_.._.._etc.log" or hidden dotfiles.
    session = re.sub(r'[^A-Za-z0-9._-]+', '_', str(data.get('session') or 'session'))[:64]
    session = session.strip('._-') or 'session'
    try:
        d = Path(str(data.get('dir') or _POSE_DEBUG_DIR)).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        fp = d / ('pose_%s.log' % session)
        mode = 'w' if data.get('reset') else 'a'
        with fp.open(mode, encoding='utf-8') as fh:
            fh.write(text if text.endswith('\n') else text + '\n')
        return jsonify({'ok': True, 'path': str(fp), 'bytes': fp.stat().st_size}), 200
    except Exception as e:
        logger.warning('[pose] debug_log error: %s', e)
        return jsonify({'ok': False, 'err': str(e)}), 200