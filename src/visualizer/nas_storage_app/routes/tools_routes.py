# =============================================================================
# routes/tools_routes.py
# /tools/* endpoints:
#   pdb_to_pdbqt  — PDB → PDBQT conversion (RDKit + obabel fallback)
# Also registers deepatom_routes via configure().
# =============================================================================

import os, shutil, tempfile
from pathlib import Path
from flask import jsonify, request
from nas_storage_app import app

from nas_storage_app.routes.shared import logger, CONVERTED_ROOT, _INPUT_ROUTES_YML

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


@app.route('/tools/pdb_to_pdbqt', methods=['POST'])
def tools_pdb_to_pdbqt():
    """
    POST /tools/pdb_to_pdbqt
    Accepts a multipart/form-data upload with:
        file      – the .pdb file
        mol_type  – "ligand" (default) or "receptor"
        out_dir   – optional absolute output directory override

    Converted files are stored persistently under:
        nas_storage_app/converted_pdbqt/{mol_type}/

    Returns JSON:
        { "status": "success"|"error",
          "message": str,
          "output_path": str,   # absolute path on server (ready to paste into Vina fields)
          "filename":    str,   # basename only
          "mol_type":    str,   # "ligand" or "receptor"
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
                return jsonify({
                    "status":      "success",
                    "message":     f"Converted with obabel ({mol_type}).",
                    "output_path": out_path,
                    "filename":    out_filename,
                    "mol_type":    mol_type,
                    "pdbqt_text":  Path(out_path).read_text(),
                })
            logger.warning("obabel failed (%s), falling back to RDKit", result.stderr)

        # RDKit fallback
        res = _pdb_to_pdbqt_rdkit(pdb_path, out_path, mol_type)
        if res["status"] == "success":
            res["filename"]   = out_filename
            res["mol_type"]   = mol_type
            res["pdbqt_text"] = Path(out_path).read_text()
        return jsonify(res)

    except Exception as exc:
        logger.exception("tools_pdb_to_pdbqt error")
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        # Always clean up the temp upload dir (output is already in CONVERTED_ROOT)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══ DeepAtom Property Estimators ══════════════════════════════════════════════
# Implementation moved to deepatom_routes.py
from nas_storage_app import deepatom_routes as _deepatom_routes_mod
_deepatom_routes_mod.configure(input_routes_yml=_INPUT_ROUTES_YML)