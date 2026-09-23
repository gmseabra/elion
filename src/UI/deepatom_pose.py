#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""deepatom_pose.py -- score one generated pose with a retrained DeepAtom checkpoint.

This is the script `pose.default_script` points at.  /pose/deepatom_score invokes
it exactly as the "⚛ Score best pose" button always has:

    <script> -t vs -d <root>          cwd=<root>, env AUGMENT=1, conda elion_backend

and it consumes the tree that route already staged:

    <root>/Dataset_VS/<name>/<name>_complex.pdb    receptor ATOM + ligand HETATM chain 'y'
    <root>/Dataset_VS/<name>/<name>_ligand.pdb     the ligand alone

WHAT IT DOES, AND WHY EACH STEP IS THERE
========================================
A DeepAtom checkpoint is a 3-D convolution over a 24-channel voxel grid, so the
pose has to become that grid before the model can see it.  Nothing here is new
work -- every step is the retraining pipeline's own code, imported rather than
reimplemented, so a pose scored in the UI goes through the identical transform
the training set did:

    complex.pdb
      -> arpeggio.py                    11 interaction types per atom
      -> read_atomtypes                 the typed atom list
      -> select_pocket_residues(5.0)    whole residues within 5 A of the ligand
      -> pcmax_grid                     32^3 x 24,  occ = 1 - exp(-(r_vdw/d)^12)
      -> ShuffleNetV3x2 forward         -> pK
      -> dG = -1.36 * pK

With AUGMENT=1 (which the route sets) it builds `--avg-num` rotated copies and
averages their predictions, matching --avg_num_test in the retrain protocol.
That is an ensemble of views, not a single forward pass; the CSV records which
was used.

OUTPUT
======
Two ways, because the route accepts either and one of them has to survive a
stdout the conda wrapper decorates:

    stdout   a "=====" separator line, then   <name>  <pk>
    CSV      <root>/vs_pose.csv  with PDB,deltaG_kcal_mol,pred_pK

CONFIGURATION
=============
Paths are resolved in this order, first hit wins:

    --model / $DEEPATOM_CKPT / DEFAULT_CKPT below
    --featurize-dir / $DEEPATOM_TOOLS / the directory holding this script
    --model-py / $DEEPATOM_MODEL_PY                        (a network FILE, wins)
    --repo-dir / $DEEPATOM_REPO / DEFAULT_REPO below       (shufflenet_v3.py)
    --arpeggio / $ARPEGGIO / DEFAULT_ARPEGGIO below
    --arpeggio-python / $ARPEGGIO_PYTHON / whichever interpreter can parse it

arpeggio_mod2/arpeggio.py is PYTHON 2.  The env this script runs in is Python 3,
so the interpreter for that one subprocess is chosen separately: every candidate
is asked to PARSE arpeggio.py, and the first that can is used and cached.

Every one is printed at startup, and a missing one is named in plain language
rather than surfacing as an import error three steps later.

    python deepatom_pose.py --selftest        # no arpeggio, no GPU, no checkpoint
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# --- edit these three, or set the environment variables ----------------------
DEFAULT_CKPT = "/home/huangzihang/repos/user_data/deepatom/checkpoints/best_val_rmse.pth.tar"
DEFAULT_REPO = "/home/huangzihang/repos/deepdrug_pt04-master/script/model/3dcnn/pytorch"
DEFAULT_ARPEGGIO = "/home/huangzihang/repos/deepatom/model_split_data/arpeggio_mod2/arpeggio.py"
# -----------------------------------------------------------------------------

POCKET_CUTOFF = 5.0
GRID_LENGTH = 32.0
RESOLUTION = 1.0
KCAL_PER_PK = 1.36


def log(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


def die(msg, code=2):
    log("ERROR: " + str(msg))
    sys.exit(code)


# =============================================================================
# locating the pieces
# =============================================================================

def find_tools(explicit=None):
    """The directory holding deepatom_featurize.py, and import it."""
    cands = [explicit, os.environ.get("DEEPATOM_TOOLS"),
             str(Path(__file__).resolve().parent)]
    for c in cands:
        if not c:
            continue
        p = Path(c).expanduser()
        if (p / "deepatom_featurize.py").is_file():
            sys.path.insert(0, str(p))
            import deepatom_featurize as F
            return p, F
    die("deepatom_featurize.py not found. Looked in: %s\n"
        "       Put this script beside it, or set --featurize-dir / $DEEPATOM_TOOLS."
        % ", ".join(str(c) for c in cands if c))


# --- discovery ---------------------------------------------------------------
# shufflenet_v3.py and arpeggio.py live wherever the DeepAtom release was
# unpacked, which differs per host -- and a wrong default here fails at the very
# end of a scoring run, after arpeggio and the grid build have already been paid
# for. So they are SEARCHED for, once, and the answer is cached beside this
# script. --rescan discards the cache.

CACHE_NAME = ".deepatom_pose_paths.json"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".conda", "site-packages",
             ".cache", "envs", ".venv", "venv", "data", "lib", "share"}


def _cache_path():
    return Path(__file__).resolve().parent / CACHE_NAME


def _cache_read():
    import json
    try:
        return json.loads(_cache_path().read_text())
    except Exception:
        return {}


def _cache_write(d):
    import json
    try:
        _cache_path().write_text(json.dumps(d, indent=2))
    except OSError:
        pass


def search_roots():
    """Where a DeepAtom release plausibly lives, nearest first."""
    here = Path(__file__).resolve().parent
    out = [here, here.parent, here.parent.parent]
    home = Path.home()
    out += [home / "repos", home / "repos" / "elion" / "src", home]
    seen, roots = set(), []
    for r in out:
        try:
            r = r.resolve()
        except OSError:
            continue
        if r.is_dir() and r not in seen:
            seen.add(r)
            roots.append(r)
    return roots


def discover(filename, roots, max_depth=7):
    """First `filename` under any root, breadth-first, pruning heavy dirs."""
    for root in roots:
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop(0)
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            for e in entries:
                if e.is_file() and e.name == filename:
                    return e
            if depth < max_depth:
                for e in entries:
                    if e.is_dir() and not e.name.startswith(".") \
                            and e.name not in SKIP_DIRS:
                        stack.append((e, depth + 1))
    return None


def _resolve(filename, explicit, env_var, default, want_dir, rescan, flag, describe):
    """explicit -> $env -> DEFAULT_* -> cache -> search. Then cache the hit."""
    for c in (explicit, os.environ.get(env_var), default):
        if not c:
            continue
        p = Path(c).expanduser()
        hit = (p / filename) if want_dir else p
        if hit.is_file():
            return hit.parent if want_dir else hit

    cache = _cache_read()
    cached = cache.get(filename)
    if cached and not rescan and Path(cached).is_file():
        p = Path(cached)
        return p.parent if want_dir else p

    roots = search_roots()
    log("search  : looking for %s ..." % filename)
    found = discover(filename, roots)
    if found:
        log("found   : %s" % found)
        cache[filename] = str(found)
        _cache_write(cache)
        return found.parent if want_dir else found

    die("%s not found.  It is %s\n"
        "       Searched to depth 7 under:\n         %s\n"
        "       Point at it with --%s <path>, or $%s, or by editing the "
        "DEFAULT_* line at the top of this script.\n"
        "       To locate it:  find ~ -name %s -not -path '*/__pycache__/*' 2>/dev/null"
        % (filename, describe,
           "\n         ".join(str(r) for r in roots),
           flag, env_var, filename))


def find_model_py(explicit=None):
    """--model-py / $DEEPATOM_MODEL_PY -- a network definition file, or None.

    Returned as a Path to the FILE (import_deepatom_model tells them apart), so
    an explicit choice never falls back to a search that could pick a different
    copy of the release.
    """
    for c in (explicit, os.environ.get("DEEPATOM_MODEL_PY")):
        if not c:
            continue
        p = Path(c).expanduser()
        if p.is_file():
            return p
        die("--model-py / $DEEPATOM_MODEL_PY points at %s, which is not a file." % p)
    return None


def find_repo(explicit=None, rescan=False):
    """The pytorch tree that defines ShuffleNetV3."""
    return _resolve("shufflenet_v3.py", explicit, "DEEPATOM_REPO", DEFAULT_REPO,
                    want_dir=True, rescan=rescan, flag="repo-dir",
                    describe="the release's script/model/3dcnn/pytorch/ directory.")


def find_arpeggio(explicit=None, rescan=False):
    return _resolve("arpeggio.py", explicit, "ARPEGGIO", DEFAULT_ARPEGGIO,
                    want_dir=False, rescan=rescan, flag="arpeggio",
                    describe="model_split_data/arpeggio_mod2/arpeggio.py in the deepatom tree.")


def find_ckpt(explicit=None):
    """The route always passes --model, so this only matters for a hand-run."""
    for c in (explicit, os.environ.get("DEEPATOM_CKPT"), DEFAULT_CKPT):
        if not c:
            continue
        p = Path(c).expanduser()
        if p.is_file():
            return p
    # last resort: the newest checkpoint beside this script, or in a
    # checkpoints/ directory next to it
    here = Path(__file__).resolve().parent
    cands = []
    for d in (here, here / "checkpoints", here.parent / "checkpoints"):
        if d.is_dir():
            for pat in ("*.pth.tar", "*.pt", "*.pth", "*.ckpt"):
                cands.extend(d.glob(pat))
    if cands:
        p = max(set(cands), key=lambda q: q.stat().st_mtime)
        log("model    : none given; using the newest checkpoint found: %s" % p)
        return p
    die("checkpoint not found. Pass --model, or set $DEEPATOM_CKPT, or leave one "
        "in %s (or its checkpoints/ subdirectory)." % here)


def find_complex(root):
    """The staged complex PDB, and the name the route used."""
    ds = Path(root) / "Dataset_VS"
    if not ds.is_dir():
        die("no Dataset_VS/ under %s -- this script expects the tree "
            "/pose/deepatom_score stages." % root)
    hits = sorted(ds.rglob("*_complex.pdb"))
    if not hits:
        die("no *_complex.pdb under %s" % ds)
    if len(hits) > 1:
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        log("note   : %d complexes staged; scoring the newest (%s)"
            % (len(hits), hits[0].name))
    pdb = hits[0]
    return pdb, pdb.name[:-len("_complex.pdb")]


# =============================================================================
# the pipeline
# =============================================================================

def _compiles_under(py, script, timeout=25):
    """Can interpreter `py` parse `script`? Works in both Python 2 and 3.

    arpeggio_mod2/arpeggio.py is PYTHON 2 -- it uses print statements -- so
    running it under the Python 3 that the elion_backend env provides dies with
    "Missing parentheses in call to 'print'" before it touches a single atom.
    Parsing is the cheap, side-effect-free way to tell which interpreters can
    run it at all.
    """
    code = ("import sys\n"
            "f = sys.argv[1]\n"
            "compile(open(f).read(), f, 'exec')\n")
    try:
        r = subprocess.run([str(py), "-c", code, str(script)],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return False, "cannot execute"
    if r.returncode == 0:
        return True, ""
    return False, (r.stderr or r.stdout or "").strip().splitlines()[-1:][0] \
        if (r.stderr or r.stdout).strip() else "exit %d" % r.returncode


def _has_biopython(py):
    try:
        r = subprocess.run([str(py), "-c", "import Bio"],
                           capture_output=True, text=True, timeout=25)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def python_candidates():
    """Interpreters that might be able to run a Python 2 arpeggio, best first."""
    import glob
    import shutil
    out = [sys.executable]
    for name in ("python2.7", "python2"):
        w = shutil.which(name)
        if w:
            out.append(w)
    pats = []
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        pats.append(str(Path(cp).parent / "*" / "bin" / "python*"))
    home = Path.home()
    for base in ("repos/envs", "envs", "miniconda3/envs", "repos/miniconda3/envs",
                 "anaconda3/envs", "miniconda2/envs"):
        pats.append(str(home / base / "*" / "bin" / "python*"))
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            # python, python2, python2.7, python3.10 -- but not python-config etc.
            if re.match(r"^python[0-9.]*$", Path(p).name) and os.access(p, os.X_OK):
                out.append(p)
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def arpeggio_python(arpeggio, explicit=None, rescan=False):
    """The interpreter that can actually parse arpeggio.py."""
    cache = _cache_read()
    ordered = []
    for c in (explicit, os.environ.get("ARPEGGIO_PYTHON"),
              None if rescan else cache.get("arpeggio_python")):
        if c:
            ordered.append(c)
    ordered += python_candidates()

    tried = []
    for py in ordered:
        ok, why = _compiles_under(py, arpeggio)
        tried.append((py, ok, why))
        if ok:
            if str(py) != str(sys.executable):
                log("python  : arpeggio needs a different interpreter -> %s" % py)
            if not _has_biopython(py):
                log("WARNING : %s has no BioPython; arpeggio imports Bio and will "
                    "fail. Install it there, or point --arpeggio-python elsewhere." % py)
            cache["arpeggio_python"] = str(py)
            _cache_write(cache)
            return py

    lines = ["no interpreter on this machine can parse %s." % arpeggio,
             "       arpeggio_mod2/arpeggio.py is PYTHON 2 (print statements), and "
             "the env running this script is Python 3.",
             "       Tried:"]
    for py, ok, why in tried[:12]:
        lines.append("         %-56s %s" % (py, why))
    lines.append("       Fix with --arpeggio-python <python2>, or $ARPEGGIO_PYTHON.")
    lines.append("       To find one:  ls -d ~/repos/envs/*/bin/python2* "
                 "~/*/envs/*/bin/python2* 2>/dev/null")
    lines.append("       Or create one: conda create -n arpeggio python=2.7 biopython -y")
    die("\n".join(lines))


def run_arpeggio(arpeggio, pdb, timeout=600, py=None):
    """complex.pdb -> complex.atomtypes, beside the input.

    The path MUST be absolute: arpeggio.py:936 does
    os.path.join(outdir, pdb.replace('.pdb', '.atomtypes')) without stripping the
    directory, so a relative input produces a doubled path and a
    FileNotFoundError, while an absolute one makes os.path.join discard outdir
    and land the file beside the PDB.
    """
    pdb = Path(pdb).resolve()
    out = pdb.with_suffix(".atomtypes")
    if out.is_file():
        out.unlink()                       # never score a previous pose's types
    cmd = [str(py or sys.executable), str(arpeggio), str(pdb), str(pdb.parent)]
    log("arpeggio: " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        die("arpeggio timed out after %ds" % timeout)
    final = out if out.is_file() else pdb.parent / out.name
    if not final.is_file():
        blob = (r.stderr or "") + (r.stdout or "")
        tail = blob.strip().splitlines()
        last = tail[-1][:300] if tail else "exit %d" % r.returncode
        hint = ""
        if "Missing parentheses in call to 'print'" in blob or "SyntaxError" in blob:
            hint = ("\n       That is a PYTHON 2 / PYTHON 3 mismatch: %s cannot parse "
                    "arpeggio.py.\n       Pass --arpeggio-python <a python2>, or set "
                    "$ARPEGGIO_PYTHON." % cmd[0])
        elif "ImportError" in blob or "ModuleNotFoundError" in blob:
            mod = re.search(r"No module named '?([A-Za-z0-9_.]+)", blob)
            hint = ("\n       %s is missing a module arpeggio needs%s. Install it in "
                    "that interpreter's environment." % (cmd[0], (": " + mod.group(1)) if mod else ""))
        die("arpeggio produced no .atomtypes (%s)%s" % (last, hint))
    return final


def check_ligand_present(F, atomtypes, complex_pdb):
    """A file that exists is not a file that worked.

    arpeggio streams atoms into an already-open handle, so an element it cannot
    type -- B, Fe, As, Ru, Se are the ones seen in LP-PDBBind -- raises partway
    through and leaves a truncated file holding PROTEIN ONLY. The grid stage then
    dies with 'No ligand atoms found', or worse, silently builds a grid from a
    ligand that lost some of its atoms. Compare the counts.
    """
    atoms = F.read_atomtypes(atomtypes)
    lig = [a for a in atoms if a.chain == F.LIGAND_CHAIN]
    want = sum(1 for ln in Path(complex_pdb).read_text(errors="replace").splitlines()
               if ln[:6] == "HETATM" and len(ln) > 21 and ln[21] == F.LIGAND_CHAIN
               and (ln[76:78].strip() or "C").upper() != "H")
    if not lig:
        die("arpeggio wrote no chain-'%s' ligand atoms. The usual cause is a "
            "ligand element outside its LIGAND_ELEMENT_TYPES (B, Fe, As, Ru, Se)."
            % F.LIGAND_CHAIN)
    if want and len(lig) != want:
        die("ligand atom count mismatch: the complex PDB has %d heavy ligand "
            "atoms, the .atomtypes has %d. Arpeggio died partway through; the "
            "grid would be built from a truncated molecule." % (want, len(lig)))
    return len(atoms), len(lig)


def build_grids(F, atomtypes, name, work, num_aug, seed):
    """.atomtypes -> the original grid, plus num_aug rotated views."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    aug_dir = (work / "aug") if num_aug else None
    if aug_dir:
        aug_dir.mkdir(parents=True, exist_ok=True)
    st = F.featurise_complex(
        atomtypes, 0.0, work / (name + ".npz"),
        pocket_cutoff=POCKET_CUTOFF, grid_length=GRID_LENGTH, resolution=RESOLUTION,
        num_aug=num_aug, aug_out_dir=aug_dir, seed=seed,
        max_translation=1.0, pdb_id=name, compress=False)
    grids = [work / (name + ".npz")]
    if aug_dir:
        grids += sorted(aug_dir.glob(name + "_augmented_*.npz"))
    return grids, st


# shufflenet_v3.py does `from utilies import init_params`, and the zip's
# utilies.py drags in matplotlib, pandas, tensorboardX and a dir_info module that
# hard-codes an absolute weights path. Only init_params is needed to construct
# the network, so a minimal stub is pre-registered -- byte-identical to the
# zip's function. This is what deepatom_lp_protocol.import_deepatom_model does,
# and the architecture is imported unmodified.
_INIT_PARAMS_SRC = """
import torch.nn as nn
import torch.nn.init as init


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)
"""

# trainV3.py's model switch. The paper's reported model is x2.0 (channels
# 244/488/976); the others exist because width is DeepAtom's capacity knob.
WIDTHS = {"ShuffleNetV3x2": 2.0, "ShuffleNetV3x1": 1.0, "ShuffleNetV3x0.5": 0.5}


def import_deepatom_model(where):
    """ShuffleNetV3 from a directory holding shufflenet_v3.py, or from a FILE.

    A file is the useful case when two copies of the release disagree: the one
    the checkpoint was trained against can be dropped in under any name --
    shufflenet_v3_blue.py, say -- and pointed at directly, without disturbing
    whichever shufflenet_v3.py the rest of the pipeline uses.
    """
    import importlib
    import types
    where = Path(where).expanduser()

    # Two spellings exist in the wild: the cluster copy does
    # `from utilies import init_params`, the elion copy `from utils import ...`.
    # Stub both, so either file imports without dragging in matplotlib/pandas/
    # tensorboardX or shadowing a real `utils` on the path.
    for name in ("utilies", "utils"):
        if name not in sys.modules or not hasattr(sys.modules[name], "init_params"):
            stub = types.ModuleType(name)
            exec(compile(_INIT_PARAMS_SRC, "<%s-stub>" % name, "exec"), stub.__dict__)
            sys.modules[name] = stub

    if where.is_file():
        import importlib.util
        # Its own directory goes on sys.path first: the file may import a
        # sibling (net_config, for instance), and those have to resolve.
        d = str(where.resolve().parent)
        if d not in sys.path:
            sys.path.insert(0, d)
        modname = "deepatom_net_" + re.sub(r"[^A-Za-z0-9_]", "_", where.stem)
        spec = importlib.util.spec_from_file_location(modname, str(where.resolve()))
        if spec is None or spec.loader is None:
            die("cannot import %s as a python module" % where)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        if not hasattr(mod, "ShuffleNetV3"):
            die("%s defines no ShuffleNetV3 class" % where)
        return mod.ShuffleNetV3

    d = str(where.resolve())
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module("shufflenet_v3").ShuffleNetV3


def load_model(repo_dir, ckpt_path, device, model_name=None, input_feature=24,
               dropout_prob=0.0):
    """Rebuild the network the checkpoint was trained with, and load it.

    The constructor is ShuffleNetV3(input_channel, dropout_prob,
    width_multiplier) -- the signature deepatom_lp_protocol.build_model uses.
    The WIDTH is not guessed: each candidate is tried and the one whose
    state_dict loads with no missing, unexpected or mis-shaped key wins. A
    checkpoint trained at x2 silently loaded into an x1 network would predict
    confident nonsense, and nothing downstream could detect it.
    """
    import torch
    ShuffleNetV3 = import_deepatom_model(repo_dir)

    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    order = [model_name] if model_name else ["ShuffleNetV3x2", "ShuffleNetV3x1",
                                             "ShuffleNetV3x0.5"]
    want_depth = _stage_depths(sd)          # what the WEIGHTS were trained with
    tried = []
    for name in order:
        if name not in WIDTHS:
            die("unknown model %r; choose from %s" % (name, list(WIDTHS)))
        # Two attempts per width: the release's own default depth, then the
        # depth the checkpoint's parameter names imply. Rebuilding at the
        # checkpoint's depth is only ever accepted when the resulting network
        # matches it EXACTLY -- zero missing, unexpected or mis-shaped keys --
        # so this can correct a block-repeat difference between two copies of
        # the release without ever loading weights into the wrong network.
        for depth in (None, list(want_depth.values()) if want_depth else None):
            built = _build(ShuffleNetV3, input_feature, dropout_prob,
                           WIDTHS[name], depth)
            if built is None:
                continue
            model, used_depth = built
            ref = model.state_dict()
            missing = [k for k in ref if k not in sd]
            unexpected = [k for k in sd if k not in ref]
            bad = [k for k in ref if k in sd and tuple(ref[k].shape) != tuple(sd[k].shape)]
            if not (missing or unexpected or bad):
                model.load_state_dict(sd, strict=True)
                model.to(device).eval()
                log("model    : %s (width %.1f, input_channel %d%s)"
                    % (name, WIDTHS[name], input_feature,
                       ", repeat_block %s rebuilt from the checkpoint" % used_depth
                       if used_depth else ""))
                return (model,
                        ck.get("epoch") if isinstance(ck, dict) else None,
                        ck.get("val_rmse") if isinstance(ck, dict) else None)
            if depth is None:
                tried.append((name, len(missing), len(unexpected), len(bad),
                              (bad or missing or unexpected or [""])[0], ref))
            if not want_depth:
                break

    lines = ["this checkpoint does not fit the shufflenet_v3.py in --repo-dir.",
             "       Tried (missing / unexpected / mis-shaped keys):"]
    for name, m, u, b, first, _ in tried:
        lines.append("         %-16s %3d / %3d / %3d   first offender: %s"
                     % (name, m, u, b, first))

    # Same width, different DEPTH is the common case and looks nothing like a
    # wrong checkpoint: the closest width has NO unexpected keys (the release
    # is a superset of what was trained) and the stage block counts differ.
    best = min(tried, key=lambda r: r[1] + r[3])
    ck_depth = _stage_depths(sd)
    md_depth = _stage_depths(best[5])
    if ck_depth and md_depth and ck_depth != md_depth:
        lines.append("")
        lines.append("       This is a DEPTH mismatch, not a width one. %s is the closest"
                     % best[0])
        lines.append("       fit (%d unexpected keys), and the block repeats differ:"
                     % best[2])
        lines.append("         checkpoint was trained with  repeat_block %s"
                     % _fmt_depth(ck_depth))
        lines.append("         this shufflenet_v3.py builds  repeat_block %s"
                     % _fmt_depth(md_depth))
        if best[3]:
            lines.append("       There are ALSO %d shape mismatches (first: %s), so depth is"
                         % (best[3], best[4]))
            lines.append("       not the only difference -- the out_block width differs too.")
        lines.append("       A rebuild was attempted and could not be made to fit. In the")
        lines.append("       DeepAtom release these constants are LOCAL VARIABLES inside")
        lines.append("       ShuffleNetV3.__init__, so there is no knob to set from outside.")
        lines.append("       Point --repo-dir / pose.deepatom_repo_dir at the shufflenet_v3.py")
        lines.append("       the checkpoint was TRAINED against -- a different copy of the")
        lines.append("       release is a different network, and its weights do not transfer.")
    else:
        lines.append("       Is %s really a DeepAtom checkpoint, and is --repo-dir the"
                     % ckpt_path)
        lines.append("       shufflenet_v3.py it was trained against?")
    die("\n".join(lines))


def _build(ShuffleNetV3, input_feature, dropout_prob, width, depth):
    """Construct the network, optionally forcing the stage block repeats.

    The release's __init__ is (input_channel, dropout_prob, width_multiplier),
    so `depth` is applied by whichever of these the copy in use supports:
      1. a repeat_block= keyword, if the constructor accepts one
      2. a module-level list of three ints (what the printed banner reads)
    Returns (model, depth_actually_forced_or_None), or None if depth was asked
    for and neither route exists.
    """
    if depth is None:
        return ShuffleNetV3(input_channel=input_feature, dropout_prob=dropout_prob,
                            width_multiplier=width), None
    try:
        return ShuffleNetV3(input_channel=input_feature, dropout_prob=dropout_prob,
                            width_multiplier=width, repeat_block=list(depth)), list(depth)
    except TypeError:
        pass
    # The constant is often NOT in shufflenet_v3 itself -- net_config.py is the
    # usual home, imported at module load. So every module whose file lives
    # under the repo directory is a candidate, and a name is only touched when
    # its current value is a same-length list/tuple of ints. Patch, build,
    # restore -- the process must not be left mutated.
    home = sys.modules.get(ShuffleNetV3.__module__)
    mods = [home] if home else []
    try:
        repo = str(Path(home.__file__).resolve().parent) if home and home.__file__ else None
    except (OSError, AttributeError):
        repo = None
    if repo:
        for m in list(sys.modules.values()):
            if m is None or m is home:
                continue
            f = getattr(m, "__file__", None)
            if not f:
                continue
            try:
                if str(Path(f).resolve()).startswith(repo):
                    mods.append(m)
            except OSError:
                continue

    for mod in mods:
        for attr in dir(mod):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(mod, attr)
            except Exception:
                continue
            if (isinstance(val, (list, tuple)) and len(val) == len(depth)
                    and all(isinstance(v, int) for v in val)
                    and list(val) != list(depth)):
                setattr(mod, attr, type(val)(depth))
                try:
                    built = ShuffleNetV3(input_channel=input_feature,
                                         dropout_prob=dropout_prob,
                                         width_multiplier=width)
                except Exception:
                    setattr(mod, attr, val)
                    continue
                setattr(mod, attr, val)
                got = _stage_depths(built.state_dict())
                if list(got.values()) == list(depth):
                    return built, list(depth)
    return None


def _stage_depths(sd):
    """{stage -> number of blocks} read off parameter names like stage3.2.foo.

    Works on a checkpoint's state_dict and on a live model's, so the two can be
    compared without importing anything or trusting a printed banner.
    """
    out = {}
    for k in sd:
        m = re.match(r"^(stage\d+)\.(\d+)\.", k)
        if m:
            out[m.group(1)] = max(out.get(m.group(1), -1), int(m.group(2)))
    return {k: v + 1 for k, v in sorted(out.items())}


def _fmt_depth(d):
    return "[%s]  (%s)" % (", ".join(str(v) for v in d.values()),
                           ", ".join("%s=%d" % (k, v) for k, v in d.items()))


def read_grid(npz_path):
    """One .npz -> the (C,G,G,G) float32 tensor the model expects.

    featurise_complex writes three arrays -- `pocket` (G,G,G,C), `label`, `pdb`
    -- and the training dataloader does exactly
        pocket = data["pocket"].astype("float32").transpose((3, 0, 1, 2))
    so that transpose is reproduced here rather than inferred from the shape:
    a 32^3 x 24 cube and a 24 x 32^3 one are both 4-D, and guessing between them
    would silently feed the network a transposed molecule.
    """
    import numpy as np
    z = np.load(str(npz_path))
    if "pocket" not in z.files:
        die("%s has no 'pocket' array (found: %s)" % (npz_path, ", ".join(z.files)))
    return np.ascontiguousarray(z["pocket"].astype("float32").transpose((3, 0, 1, 2)))


def predict(model, grids, device):
    import numpy as np
    import torch
    preds = []
    with torch.no_grad():
        for g in grids:
            x = torch.from_numpy(read_grid(g)[None, ...]).to(device)
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds.append(float(np.asarray(out.detach().cpu()).ravel()[0]))
    return preds


# =============================================================================
# self-test -- exercises everything except arpeggio, torch and the checkpoint
# =============================================================================

def selftest(args):
    import tempfile
    import numpy as np
    tools, F = find_tools(args.featurize_dir)
    log("featurize: %s  (%s)" % (tools, getattr(F, "__version__", "?")))

    root = Path(tempfile.mkdtemp(prefix="da_pose_selftest_"))
    ds = root / "Dataset_VS" / "probe"
    ds.mkdir(parents=True)

    # a synthetic .atomtypes straight from the featuriser's own generator, so the
    # grid path is exercised with no arpeggio present
    at = ds / "probe_complex.atomtypes"
    at.write_text("".join(F._synthetic_atomtypes(n_prot=150, n_lig=20, seed=3)))

    grids, st = build_grids(F, at, "probe", root / "npz", num_aug=2, seed=1)
    log("grids    : %d  (%s)" % (len(grids), ", ".join(p.name for p in grids)))
    assert len(grids) == 3, "expected the original plus 2 augmented"

    raw = np.load(str(grids[0]))["pocket"]
    a = read_grid(grids[0])
    log("on disk  : %s      as fed to the model: %s" % (raw.shape, a.shape))
    log("occupancy: nonzero %.2f%%   max %.4f   dtype %s"
        % (100.0 * (a != 0).mean(), float(a.max()), a.dtype))
    assert raw.shape == (32, 32, 32, 24), raw.shape
    assert a.shape == (24, 32, 32, 32), a.shape
    assert a.dtype == np.float32
    assert 0.0 <= float(a.max()) <= 1.0, "occupancy must be in [0,1]"
    # the transpose must be a relabelling, not a reshuffle
    assert np.array_equal(a[7, 3, 4, 5], raw[3, 4, 5, 7])

    # the augmented views must differ from the original and from each other,
    # or the "average of 12 rotations" is 12 copies of one number
    b, c = read_grid(grids[1]), read_grid(grids[2])
    assert not np.array_equal(a, b) and not np.array_equal(b, c), \
        "augmented grids are identical -- augmentation is not happening"
    log("augment  : original vs aug0 vs aug1 all differ  OK")

    # the two output formats the route parses
    csv = write_outputs(root, "probe", 6.5, [6.4, 6.6], "selftest")
    txt = csv.read_text()
    log("csv      : %s\n%s" % (csv, txt.rstrip()))
    assert "deltaG_kcal_mol" in txt and "probe" in txt

    import shutil
    shutil.rmtree(root, ignore_errors=True)
    log("\nselftest OK -- grid, augmentation and both output formats verified.")
    log("Not covered here (needs the host): arpeggio, torch, the checkpoint.")
    return 0


# =============================================================================
# output
# =============================================================================

def write_outputs(root, name, pk, preds, how):
    """Both formats the route accepts.

    _parse_deepatom_output skips everything before a '=====' line and then reads
    '<id> <float>'; _read_deepatom_csv wants PDB + deltaG_kcal_mol and derives
    pK = -dG/1.36. Writing both means a conda wrapper that decorates stdout
    cannot cost us the answer.
    """
    dg = -float(pk) * KCAL_PER_PK
    csv = Path(root) / "vs_pose.csv"
    csv.write_text("PDB,deltaG_kcal_mol,pred_pK,views,how\n%s,%.4f,%.4f,%d,%s\n"
                   % (name, dg, pk, len(preds), how))
    log("=" * 64)
    log("%s  %.4f" % (name, pk))
    # Everything after the score line is commented out with '#'.
    # _parse_deepatom_output reads every "<id> <float>" line past the separator
    # as a result, so a bare "pred_pK  6.5000" summary line would be returned as
    # a SECOND compound named "pred_pK". The '#' is outside its id character
    # class, which makes these lines unambiguously prose.
    log("")
    log("# pred_pK      %.4f" % pk)
    log("# deltaG       %.4f kcal/mol   (-1.36 x pK)" % dg)
    log("# views        %d  (%s)" % (len(preds), how))
    if len(preds) > 1:
        import statistics
        log("# spread       %.4f min  %.4f max  %.4f sd"
            % (min(preds), max(preds), statistics.stdev(preds)))
    log("# csv          %s" % csv)
    return csv


# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--test-type", default="vs",
                    help="the route always sends 'vs'; accepted and ignored")
    ap.add_argument("-d", "--data-dir", default=".", help="<root> staged by the route")
    ap.add_argument("--model", default=None, help="checkpoint (.pth.tar)")
    ap.add_argument("--repo-dir", default=None, help="pytorch/ dir with shufflenet_v3.py")
    ap.add_argument("--model-py", default=None,
                    help="the network definition FILE, under any name (e.g. "
                         "shufflenet_v3_blue.py). Wins over --repo-dir; use it when "
                         "the copy matching your checkpoint is not the one the rest "
                         "of the pipeline uses")
    ap.add_argument("--featurize-dir", default=None, help="dir with deepatom_featurize.py")
    ap.add_argument("--arpeggio", default=None, help="arpeggio.py")
    ap.add_argument("--arpeggio-python", default=None,
                    help="interpreter for arpeggio.py. arpeggio_mod2 is PYTHON 2; "
                         "leave unset and a working one is searched for and cached")
    ap.add_argument("--avg-num", type=int, default=12,
                    help="rotated views to average when AUGMENT=1 (default 12, "
                         "matching --avg_num_test in the retrain protocol)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if present)")
    ap.add_argument("--keep", action="store_true", help="keep the .atomtypes and .npz")
    ap.add_argument("--rescan", action="store_true",
                    help="discard the cached shufflenet_v3.py / arpeggio.py locations")
    ap.add_argument("--where", action="store_true",
                    help="resolve and print every path, then exit (no scoring)")
    ap.add_argument("--selftest", action="store_true",
                    help="exercise the grid path and output formats; no model needed")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest(a)

    if a.where:
        tools, F = find_tools(a.featurize_dir)
        log("featurize: %s  (%s)" % (tools, getattr(F, "__version__", "?")))
        log("repo     : %s" % (find_model_py(a.model_py)
                                or find_repo(a.repo_dir, a.rescan)))
        _arp = find_arpeggio(a.arpeggio, a.rescan)
        log("arpeggio : %s" % _arp)
        log("arp py   : %s" % arpeggio_python(_arp, a.arpeggio_python, a.rescan))
        log("model    : %s" % find_ckpt(a.model))
        log("cache    : %s" % _cache_path())
        log("\nAll resolved. This is what a scoring run will use.")
        return 0

    t0 = time.time()
    root = Path(a.data_dir).expanduser().resolve()
    log("deepatom_pose.py  root=%s" % root)

    # Resolve EVERY path before doing any work. arpeggio and the grid build cost
    # seconds each; discovering a missing shufflenet_v3.py after paying for them
    # wastes the run and buries the real cause under a stack of output.
    tools, F = find_tools(a.featurize_dir)
    repo = find_model_py(a.model_py) or find_repo(a.repo_dir, a.rescan)
    arp = find_arpeggio(a.arpeggio, a.rescan)
    arp_py = arpeggio_python(arp, a.arpeggio_python, a.rescan)
    ckpt = find_ckpt(a.model)
    log("featurize: %s  (%s)" % (tools, getattr(F, "__version__", "?")))
    log("repo     : %s" % repo)
    log("arpeggio : %s" % arp)
    log("model    : %s" % ckpt)

    # AUGMENT=1 is what the route sets; honour it, and let --avg-num size it.
    num_aug = a.avg_num if os.environ.get("AUGMENT", "0") not in ("", "0") else 0

    pdb, name = find_complex(root)
    log("complex  : %s" % pdb)

    at = run_arpeggio(arp, pdb, py=arp_py)
    n_all, n_lig = check_ligand_present(F, at, pdb)
    log("atomtypes: %d atoms, %d ligand" % (n_all, n_lig))

    grids, st = build_grids(F, at, name, root / "npz", num_aug, a.seed)
    clipped = getattr(st, "ligand_atoms_outside_box", 0) or 0
    if clipped:
        log("WARNING  : %d ligand atom(s) fall outside the 32 A box -- the grid "
            "is missing part of the molecule." % clipped)
    log("grids    : %d" % len(grids))

    device = a.device or ("cuda" if _cuda() else "cpu")
    model, epoch, vrmse = load_model(repo, ckpt, device)
    log("loaded   : epoch %s (val RMSE %s) on %s" % (epoch, vrmse, device))

    preds = predict(model, grids, device)
    if not preds:
        die("no prediction produced")
    pk = sum(preds) / len(preds)
    # num_aug=N yields N+1 samples -- the original grid plus N rotations -- and
    # all N+1 are averaged. That is what the retrain protocol's --avg_num_test N
    # does too (4,453 test complexes x 13 = 57,889 samples at N=12), so the
    # number printed here is comparable to the reported rotation-averaged row.
    how = ("mean of the original grid + %d rotations" % (len(preds) - 1)) \
        if len(preds) > 1 else "single view (AUGMENT unset)"

    write_outputs(root, name, pk, preds, how)
    log("elapsed      %.1fs" % (time.time() - t0))

    if not a.keep:
        for p in [at] + grids:
            try:
                p.unlink()
            except OSError:
                pass
    return 0


def _cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
