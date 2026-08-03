# =============================================================================
# routes/tools_routes.py
# /tools/* endpoints:
#   pdb_to_pdbqt  — PDB → PDBQT conversion (RDKit + obabel fallback)
# Also registers deepatom_routes via configure().
# =============================================================================

import os, shutil, tempfile, math
from pathlib import Path
from flask import jsonify, request
from uiapp import app

from uiapp.routes.shared import (
    logger, CONVERTED_ROOT, _INPUT_ROUTES_YML, sanitize_pdbqt,
    MGLTOOLS_PYTHON, PREPARE_LIGAND4, PREPARE_RECEPTOR4,
)

# ==============================================================================
# PDB → PDBQT Converter
# ==============================================================================

def _pdb_to_pdbqt_rdkit(pdb_path: str, out_path: str, mol_type: str = "ligand") -> dict:
    """
    Convert a PDB file to PDBQT format.

    For ligands : uses RDKit — adds Hs, Gasteiger charges, AutoDock atom types.
    For receptors: uses direct PDB-line passthrough — preserves all original atom
                   names/residues/coordinates, computes Gasteiger charges via RDKit,
                   writes the exact PDBQT column layout Vina expects.
                   No ROOT/ENDROOT/TORSDOF — those are ligand-only.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdPartialCharges

    try:
        if mol_type == "receptor":
            return _pdb_to_pdbqt_receptor_passthrough(pdb_path, out_path)

        # ── Ligand path (RDKit full pipeline) ────────────────────────────────
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        if mol is None:
            return {"status": "error", "message": "RDKit could not parse the PDB file."}

        mol = Chem.AddHs(mol, addCoords=True)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)

        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            for atom in mol.GetAtoms():
                atom.SetDoubleProp("_GasteigerCharge", 0.0)

        _write_pdbqt_ligand(mol, out_path)
        return {"status": "success", "message": "Converted successfully (ligand).",
                "output_path": out_path}

    except Exception as exc:
        logger.exception("pdb_to_pdbqt_rdkit failed")
        return {"status": "error", "message": str(exc)}


def _pdb_to_pdbqt_receptor_passthrough(pdb_path: str, out_path: str) -> dict:
    """
    Receptor PDB → PDBQT by direct line passthrough.

    Strategy:
    1. Parse ATOM/HETATM lines from the PDB preserving original text (atom names,
       residue names, chain, sequence numbers, coordinates, B-factor).
    2. Load the same PDB via RDKit to compute per-atom Gasteiger charges.
       Map charges by atom serial number.
    3. Write output: same record/serial/name/resName/chain/resSeq/coords/occup/B
       but replace columns 66-76 with charge and 77-78 with AutoDock atom type.
    4. No ROOT/ENDROOT/TORSDOF — Vina treats those as errors in a receptor.
    """
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges
    import re

    # ── Step 1: read raw PDB lines ────────────────────────────────────────────
    with open(pdb_path) as fh:
        raw_lines = fh.readlines()

    atom_lines = [(i, l) for i, l in enumerate(raw_lines)
                  if l.startswith(('ATOM  ', 'HETATM'))]

    if not atom_lines:
        return {"status": "error", "message": "No ATOM/HETATM records found in PDB file."}

    # ── Step 2: compute Gasteiger charges via RDKit ───────────────────────────
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    charge_by_serial: dict[int, float] = {}
    if mol is not None:
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
            rdPartialCharges.ComputeGasteigerCharges(mol)
            for atom in mol.GetAtoms():
                ri = atom.GetPDBResidueInfo()
                if ri:
                    serial = ri.GetSerialNumber()
                    q = atom.GetDoubleProp("_GasteigerCharge") \
                        if atom.HasProp("_GasteigerCharge") else 0.0
                    charge_by_serial[serial] = 0.0 if (q != q) else q  # NaN → 0
        except Exception as e:
            logger.warning("Gasteiger charge computation failed: %s — using 0.0", e)

    # ── AutoDock atom type from element + residue context ────────────────────
    _AD_TYPES = {
        'C':  'C',  'N':  'NA', 'O':  'OA', 'S':  'SA',
        'H':  'HD', 'P':  'P',  'F':  'F',  'CL': 'Cl',
        'BR': 'Br', 'I':  'I',  'FE': 'Fe', 'ZN': 'Zn',
        'CA': 'Ca', 'MG': 'Mg', 'MN': 'Mn', 'CU': 'Cu',
    }
    def _ad_type(elem: str, atom_name: str) -> str:
        e = elem.upper().strip()
        # Distinguish polar H (HD) from non-polar H by whether the heavy atom name
        # suggests attachment to N/O/S — a rough but standard heuristic
        if e == 'H':
            return 'HD'
        return _AD_TYPES.get(e, e[:2] if len(e) > 1 else e)

    # ── Step 3: write PDBQT ───────────────────────────────────────────────────
    out_lines = []
    serial_re = re.compile(r'^\w{6}(.{5})')  # cols 6-10

    for _, line in atom_lines:
        # Ensure line is exactly padded to at least 80 cols
        line = line.rstrip('\n').rstrip('\r')
        line = line.ljust(80)

        # Parse serial (cols 6-10)
        try:
            serial = int(line[6:11].strip())
        except ValueError:
            serial = -1

        # Parse element from cols 76-77 (PDB standard) or infer from atom name
        elem = line[76:78].strip()
        if not elem:
            # Infer from atom name (cols 12-15): strip digits, take alpha chars
            aname = line[12:16].strip()
            elem  = re.sub(r'[^A-Za-z]', '', aname)
            elem  = elem[0].upper() + (elem[1:2].lower() if len(elem) > 1 else '')
            # Common 2-char elements
            if elem[:2].upper() in ('CL','BR','FE','ZN','CA','MG','MN','CU'):
                elem = elem[:2]
            else:
                elem = elem[0]

        atype  = _ad_type(elem, line[12:16].strip())
        charge = charge_by_serial.get(serial, 0.0)

        # Build output line: cols 0-65 verbatim, then charge (10 chars), space, atype (2 chars)
        base = line[:66]  # record..B-factor
        out_lines.append(f"{base}{charge:+9.3f} {atype:<2s}\n")

    out_lines.append("TER\n")
    Path(out_path).write_text("".join(out_lines))
    return {"status": "success", "message": "Converted successfully (receptor).",
            "output_path": out_path}


# AutoDock atom-type mapping from RDKit element symbol
_AUTODOCK_TYPES = {
    "C":  "C",  "N":  "NA", "O":  "OA", "S":  "SA",
    "H":  "H",  "P":  "P",  "F":  "F",  "Cl": "Cl",
    "Br": "Br", "I":  "I",  "Fe": "Fe", "Zn": "Zn",
    "Ca": "Ca", "Mg": "Mg", "Mn": "Mn",
}


def _autodock_type(atom) -> str:
    sym = atom.GetSymbol()
    # Carbons bonded only to C/H are non-polar (type A in some force fields; keep C for Vina)
    return _AUTODOCK_TYPES.get(sym, sym[:2] if len(sym) > 1 else sym)


def _write_pdbqt_ligand(mol, out_path: str):
    """Write a PDBQT file for a small-molecule ligand."""
    conf  = mol.GetConformer()
    lines = ["ROOT\n"]
    for i, atom in enumerate(mol.GetAtoms()):
        pos     = conf.GetAtomPosition(i)
        charge  = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if charge != charge:          # NaN guard
            charge = 0.0
        atype   = _autodock_type(atom)
        elem    = atom.GetSymbol()
        name    = f"{elem}{i+1}"
        line = (
            f"ATOM  {i+1:5d} {name:<4s} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {charge:+8.3f} {atype}\n"
        )
        lines.append(line)
    lines.append("ENDROOT\n")
    lines.append("TORSDOF 0\n")
    Path(out_path).write_text("".join(lines))


def _write_pdbqt_receptor(mol, out_path: str):
    """
    Write a PDBQT file for a rigid receptor.
    Receptors must use plain ATOM/HETATM records — no ROOT/ENDROOT/TORSDOF tags
    (those are ligand-only; Vina will reject the receptor with a parsing error otherwise).
    Hydrogens are stripped; partial charges set to 0.0.
    """
    conf  = mol.GetConformer()
    lines = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:   # skip explicit H (already removed by RemoveHs, but guard anyway)
            continue
        pos    = conf.GetAtomPosition(i)
        atype  = _autodock_type(atom)
        elem   = atom.GetSymbol()
        res    = atom.GetPDBResidueInfo()
        # Use PDB residue info when available (preserves original atom names / residue numbering)
        if res:
            aname  = res.GetName().strip() or elem
            resn   = res.GetResidueName().strip() or "UNK"
            resi   = res.GetResidueNumber()
            chain  = res.GetChainId().strip() or "A"
            ins    = res.GetInsertionCode().strip() or " "
        else:
            aname  = f"{elem}{i+1}"
            resn   = "UNK"
            resi   = i + 1
            chain  = "A"
            ins    = " "
        std_res = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
                   "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
        record = "ATOM  " if resn in std_res else "HETATM"
        line = (
            f"{record}{i+1:5d} {aname:<4s} {resn:<3s} {chain}{resi:4d}{ins}   "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {0.0:+8.3f} {atype}\n"
        )
        lines.append(line)
    lines.append("END\n")
    Path(out_path).write_text("".join(lines))


def _pdb_to_pdbqt_mgltools(pdb_path: str, out_path: str, mol_type: str = "ligand") -> dict:
    """
    Convert PDB → PDBQT using MGLTools / AutoDockTools prepare_*4.py.

    These scripts must run under the MGLTools interpreter (pythonsh or a python
    env with AutoDockTools installed) — see shared.MGLTOOLS_PYTHON. We invoke
    them as a subprocess with cwd set to the input's directory and basenames for
    -l/-r/-o, because prepare_ligand4.py in particular mishandles absolute paths.
    """
    import subprocess

    if not MGLTOOLS_PYTHON:
        return {"status": "error",
                "message": "MGLTools not found. Install AutoDockTools (or set the "
                           "MGLTOOLS_PYTHON / ADT_PYTHON environment variable to your "
                           "pythonsh), or use the obabel engine instead."}

    script = PREPARE_RECEPTOR4 if mol_type == "receptor" else PREPARE_LIGAND4
    if not Path(script).is_file():
        return {"status": "error",
                "message": f"MGLTools script not found: {script}. Place prepare_"
                           f"{'receptor' if mol_type == 'receptor' else 'ligand'}4.py there."}

    work_dir = os.path.dirname(pdb_path) or "."
    in_name  = os.path.basename(pdb_path)
    out_name = os.path.basename(out_path)

    # -r for receptor, -l for ligand; both take -o for the output filename.
    in_flag = "-r" if mol_type == "receptor" else "-l"
    cmd = [MGLTOOLS_PYTHON, script, in_flag, in_name, "-o", out_name]

    try:
        proc = subprocess.run(cmd, cwd=work_dir, capture_output=True,
                              text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "MGLTools conversion timed out (120s)."}
    except Exception as exc:
        return {"status": "error", "message": f"MGLTools invocation failed: {exc}"}

    produced = os.path.join(work_dir, out_name)
    if not Path(produced).is_file():
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return {"status": "error",
                "message": f"MGLTools did not produce an output file. "
                           f"Log: {tail or '(no output)'}"}

    # Move to the requested destination if the script wrote it beside the input.
    if os.path.abspath(produced) != os.path.abspath(out_path):
        shutil.move(produced, out_path)

    return {"status": "success",
            "message": f"Converted with MGLTools prepare_{'receptor' if mol_type=='receptor' else 'ligand'}4 ({mol_type}).",
            "output_path": out_path}


def _pdbqt_center(path: str) -> dict:
    """Geometric centre + extent of a converted PDBQT, for the Vina docking box.

    The centre is the HEAVY-ATOM CENTROID — hydrogens excluded. That is the
    definition the rest of the app already uses: recomputing it for the reference
    ligand G001a_IL-6_gp130_SP_Score_-4.214 reproduces input_routes.yml's 1P9M
    centre, (-0.127, 2.204, -12.08), to the digit. The alternatives do not —
    including hydrogens gives (-0.285, 2.361, -12.061) and the bounding-box centre
    gives (-0.136, 1.780, -11.343) — so which one you print matters.

    For a RECEPTOR this is the centroid of the whole protein, i.e. a blind-docking
    starting point, not a pocket. The caller labels it as such.

    Returns {} if the file has no parseable atoms; callers merge it into their
    response, so a failure here never breaks a conversion that otherwise worked.
    """
    pts, n_all = [], 0
    try:
        with open(path, errors='ignore') as fh:
            for line in fh:
                if line.startswith('ENDMDL'):
                    break                       # first pose only
                if line[:6].rstrip() not in ('ATOM', 'HETATM'):
                    continue
                n_all += 1
                if line[77:].strip().upper() in ('H', 'HD', 'HS'):
                    continue
                try:
                    pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    except OSError:
        return {}
    if not pts:
        return {}

    n   = len(pts)
    cen = [round(sum(p[i] for p in pts) / n, 3) for i in range(3)]
    ext = [round(max(p[i] for p in pts) - min(p[i] for p in pts), 2) for i in range(3)]
    # A cube holding the molecule with ~4 Å clearance on every side, never below the
    # 20 Å every entry in the protein library uses.
    side = max(20, int(math.ceil(max(ext))) + 8)
    return {
        'center_x': cen[0], 'center_y': cen[1], 'center_z': cen[2],
        'extent_x': ext[0], 'extent_y': ext[1], 'extent_z': ext[2],
        'n_heavy_atoms': n, 'n_atoms': n_all, 'suggested_size': side,
    }


@app.route('/tools/pdb_to_pdbqt', methods=['POST'])
def tools_pdb_to_pdbqt():
    """
    POST /tools/pdb_to_pdbqt
    Accepts a multipart/form-data upload with:
        file      – the .pdb file
        mol_type  – "ligand" (default) or "receptor"
        engine    – "obabel" (default) or "mgltools"
        out_dir   – optional absolute output directory override

    Converted files are stored persistently under:
        uiapp/converted_pdbqt/{mol_type}/

    Returns JSON:
        { "status": "success"|"error",
          "message": str,
          "output_path": str,   # absolute path on server (ready to paste into Vina fields)
          "filename":    str,   # basename only
          "mol_type":    str,   # "ligand" or "receptor"
          "engine":      str,   # "obabel" or "mgltools"
          "pdbqt_text":  str }  # file content for optional browser download
    """
    import tempfile, shutil

    # Persistent storage root — uses global constant
    _CONVERTED_ROOT = CONVERTED_ROOT

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in request."}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({"status": "error", "message": "Empty filename."}), 400
    if not f.filename.lower().endswith('.pdb'):
        return jsonify({"status": "error",
                        "message": "Only .pdb files are accepted."}), 400

    mol_type = request.form.get('mol_type', 'ligand').strip().lower()
    if mol_type not in ('ligand', 'receptor'):
        mol_type = 'ligand'

    engine = request.form.get('engine', 'obabel').strip().lower()
    if engine not in ('obabel', 'mgltools'):
        engine = 'obabel'

    # Persistent output dir: .../converted_pdbqt/ligand/ or .../converted_pdbqt/receptor/
    out_dir = request.form.get('out_dir', '').strip()
    if not out_dir:
        out_dir = os.path.join(_CONVERTED_ROOT, mol_type)
    os.makedirs(out_dir, exist_ok=True)

    # Save upload to a temp file for processing
    tmp_dir = tempfile.mkdtemp(prefix="elion_pdb2pdbqt_")
    try:
        stem     = Path(f.filename).stem
        pdb_path = os.path.join(tmp_dir, f.filename)
        f.save(pdb_path)

        out_filename = stem + ".pdbqt"
        out_path     = os.path.join(out_dir, out_filename)

        # ── Engine: MGLTools (AutoDockTools prepare_*4.py) ────────────────────
        if engine == "mgltools":
            res = _pdb_to_pdbqt_mgltools(pdb_path, out_path, mol_type)
            if res["status"] == "success" and Path(out_path).exists():
                sanitize_pdbqt(out_path, mol_type)
                res["filename"]   = out_filename
                res["mol_type"]   = mol_type
                res["engine"]     = "mgltools"
                res["pdbqt_text"] = Path(out_path).read_text()
                res.update(_pdbqt_center(out_path))     # centre for the docking box
                return jsonify(res)
            # MGLTools was explicitly requested — surface its error rather than
            # silently substituting obabel output the user didn't ask for.
            res.setdefault("engine", "mgltools")
            return jsonify(res), 500

        # ── Engine: obabel (default), with RDKit fallback ─────────────────────
        # Try obabel first (more accurate atom typing for receptors)
        obabel = shutil.which("obabel")
        if obabel:
            import subprocess as _sp
            # For receptors: -xr suppresses torsion tree (ROOT/ENDROOT/TORSDOF)
            ob_flags = [obabel, pdb_path, "-O", out_path, "--partialcharge", "gasteiger"]
            if mol_type == "receptor":
                ob_flags += ["-xr"]   # rigid — no flexibility tree
            else:
                ob_flags += ["-h"]    # add hydrogens for ligands
            result = _sp.run(ob_flags, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and Path(out_path).exists():
                # obabel copies the source PDB's TITLE/REMARK header (incl.
                # Maestro "REMARK 888 WRITTEN BY MAESTRO" + "TITLE <name>")
                # straight into the PDBQT. Vina rejects those → best=None.
                # Strip them before the file ever leaves this route.
                n_stripped = sanitize_pdbqt(out_path, mol_type)
                msg = f"Converted with obabel ({mol_type})."
                if n_stripped:
                    msg += f" Stripped {n_stripped} Vina-illegal header line(s)."
                _resp = {
                    "status":      "success",
                    "message":     msg,
                    "output_path": out_path,
                    "filename":    out_filename,
                    "mol_type":    mol_type,
                    "engine":      "obabel",
                    "pdbqt_text":  Path(out_path).read_text(),
                }
                _resp.update(_pdbqt_center(out_path))   # centre for the docking box
                return jsonify(_resp)
            logger.warning("obabel failed (%s), falling back to RDKit", result.stderr)

        # RDKit fallback
        res = _pdb_to_pdbqt_rdkit(pdb_path, out_path, mol_type)
        if res["status"] == "success":
            if Path(out_path).exists():
                sanitize_pdbqt(out_path, mol_type)
            res["filename"]   = out_filename
            res["mol_type"]   = mol_type
            res["engine"]     = "obabel"
            res["pdbqt_text"] = Path(out_path).read_text()
            res.update(_pdbqt_center(out_path))         # centre for the docking box
        return jsonify(res)

    except Exception as exc:
        logger.exception("tools_pdb_to_pdbqt error")
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        # Always clean up the temp upload dir (output is already in CONVERTED_ROOT)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# DeepAtom routes are registered via routes/__init__.py → deepatom_routes.py