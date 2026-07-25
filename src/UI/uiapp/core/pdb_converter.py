# ==============================================================================
# pdb_converter.py — PDB → PDBQT conversion + Ligand Center Calculator
#
# Bug fixes vs original routes.py:
#   1. Strip forbidden header records (TITLE, REMARK, MODEL, etc.) from both
#      receptor AND ligand PDBQT — Vina rejects them in both.
#   2. Ensure chain ID (col 21) is never blank — default to 'A' when missing.
#
# Flask routes registered:
#   POST /tools/pdb_to_pdbqt
#   POST /tools/ligand_center
# ==============================================================================

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from flask import jsonify, request
from uiapp import app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — set from routes.py after import via configure()
# ---------------------------------------------------------------------------
_CONVERTED_ROOT: str = ""


def configure(converted_root: str) -> None:
    """Call once from routes.py after _VIZ is resolved."""
    global _CONVERTED_ROOT
    _CONVERTED_ROOT = converted_root


# ==============================================================================
# AutoDock atom-type tables
# ==============================================================================

_AD_TYPES_RECEPTOR: dict[str, str] = {
    'C':  'C',  'N':  'NA', 'O':  'OA', 'S':  'SA',
    'H':  'HD', 'P':  'P',  'F':  'F',  'CL': 'Cl',
    'BR': 'Br', 'I':  'I',  'FE': 'Fe', 'ZN': 'Zn',
    'CA': 'Ca', 'MG': 'Mg', 'MN': 'Mn', 'CU': 'Cu',
}

_AUTODOCK_TYPES: dict[str, str] = {
    "C":  "C",  "N":  "NA", "O":  "OA", "S":  "SA",
    "H":  "H",  "P":  "P",  "F":  "F",  "Cl": "Cl",
    "Br": "Br", "I":  "I",  "Fe": "Fe", "Zn": "Zn",
    "Ca": "Ca", "Mg": "Mg", "Mn": "Mn",
}

# Records allowed in receptors (no torsion-tree records)
_RECEPTOR_ALLOWED = ('ATOM  ', 'HETATM', 'TER', 'END')

# Records allowed in ligands — no END, ligands terminate with TORSDOF
_LIGAND_ALLOWED = ('ATOM  ', 'HETATM', 'ROOT', 'ENDROOT', 'TORSDOF', 'BRANCH', 'ENDBRANCH')


# ==============================================================================
# Sanitizer — strips forbidden records + fixes blank chain IDs
# ==============================================================================

def _sanitize_pdbqt(pdbqt_path: str, mol_type: str = "receptor") -> None:
    """
    In-place cleanup of any PDBQT file produced by obabel or RDKit.

    obabel copies TITLE, REMARK, COMPND, … from the source PDB into the
    output PDBQT unconditionally.  Vina rejects these records in both
    receptors and ligands:
      - Receptor: "Unknown or inappropriate tag found in rigid receptor"
      - Ligand:   "Unknown or inappropriate tag found in flex residue or ligand"
    In both cases best affinity is reported as None.

    Allowed records:
      Receptor — ATOM, HETATM, TER, END
      Ligand   — ATOM, HETATM, ROOT, ENDROOT, TORSDOF, BRANCH, ENDBRANCH

    Also fixes blank chain ID (column 21) → defaults to 'A'.
    """
    allowed = _LIGAND_ALLOWED if mol_type == "ligand" else _RECEPTOR_ALLOWED

    clean: list[str] = []
    with open(pdbqt_path) as fh:
        for line in fh:
            s = line.rstrip('\n').rstrip('\r')

            if not any(s.startswith(p) for p in allowed):
                continue

            # Fix blank chain ID on coordinate lines (PDB column 21, 0-indexed)
            if s.startswith(('ATOM  ', 'HETATM')):
                s = s.ljust(80)
                if s[21] == ' ':
                    s = s[:21] + 'A' + s[22:]

            clean.append(s + '\n')

    # Receptors must end with END; ligands terminate with TORSDOF (already present)
    if mol_type == "receptor":
        if clean and not clean[-1].rstrip().startswith('END'):
            clean.append('END\n')

    Path(pdbqt_path).write_text(''.join(clean))


# ==============================================================================
# Receptor conversion — direct PDB-line passthrough (RDKit charges)
# ==============================================================================

def _pdb_to_pdbqt_receptor_passthrough(pdb_path: str, out_path: str) -> dict:
    """
    Receptor PDB → PDBQT by direct line passthrough.

    Strategy:
    1. Parse ATOM/HETATM lines from the PDB preserving original text.
    2. Load via RDKit to compute per-atom Gasteiger charges.
    3. Write: same record/serial/name/resName/chain/resSeq/coords/occup/B
       but replace columns 66-76 with charge and 77-78 with AutoDock atom type.
    4. No ROOT/ENDROOT/TORSDOF — Vina rejects those in a receptor.
    5. Chain ID is never blank (Bug 2 fix).
    """
    import re
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges

    with open(pdb_path) as fh:
        raw_lines = fh.readlines()

    atom_lines = [(i, l) for i, l in enumerate(raw_lines)
                  if l.startswith(('ATOM  ', 'HETATM'))]

    if not atom_lines:
        return {"status": "error", "message": "No ATOM/HETATM records found in PDB file."}

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
                    charge_by_serial[serial] = 0.0 if (q != q) else q
        except Exception as e:
            logger.warning("Gasteiger charge computation failed: %s — using 0.0", e)

    def _ad_type(elem: str) -> str:
        e = elem.upper().strip()
        if e == 'H':
            return 'HD'
        return _AD_TYPES_RECEPTOR.get(e, e[:2] if len(e) > 1 else e)

    out_lines: list[str] = []
    for _, line in atom_lines:
        line = line.rstrip('\n').rstrip('\r').ljust(80)

        try:
            serial = int(line[6:11].strip())
        except ValueError:
            serial = -1

        if line[21] == ' ':
            line = line[:21] + 'A' + line[22:]

        elem = line[76:78].strip()
        if not elem:
            aname = line[12:16].strip()
            elem  = re.sub(r'[^A-Za-z]', '', aname)
            elem  = elem[0].upper() + (elem[1:2].lower() if len(elem) > 1 else '')
            if elem[:2].upper() in ('CL', 'BR', 'FE', 'ZN', 'CA', 'MG', 'MN', 'CU'):
                elem = elem[:2]
            else:
                elem = elem[0]

        atype  = _ad_type(elem)
        charge = charge_by_serial.get(serial, 0.0)
        base   = line[:66]
        out_lines.append(f"{base}{charge:+9.3f} {atype:<2s}\n")

    out_lines.append("TER\n")
    out_lines.append("END\n")
    Path(out_path).write_text("".join(out_lines))
    return {"status": "success", "message": "Converted successfully (receptor).",
            "output_path": out_path}


# ==============================================================================
# Ligand conversion — full RDKit pipeline
# ==============================================================================

def _autodock_type(atom) -> str:
    sym = atom.GetSymbol()
    return _AUTODOCK_TYPES.get(sym, sym[:2] if len(sym) > 1 else sym)


def _write_pdbqt_ligand(mol, out_path: str) -> None:
    """Write a PDBQT file for a small-molecule ligand."""
    conf  = mol.GetConformer()
    lines = ["ROOT\n"]
    for i, atom in enumerate(mol.GetAtoms()):
        pos    = conf.GetAtomPosition(i)
        charge = atom.GetDoubleProp("_GasteigerCharge") \
                 if atom.HasProp("_GasteigerCharge") else 0.0
        if charge != charge:
            charge = 0.0
        atype = _autodock_type(atom)
        elem  = atom.GetSymbol()
        name  = f"{elem}{i + 1}"
        line  = (
            f"ATOM  {i+1:5d} {name:<4s} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {charge:+8.3f} {atype}\n"
        )
        lines.append(line)
    lines.append("ENDROOT\n")
    lines.append("TORSDOF 0\n")
    Path(out_path).write_text("".join(lines))


def _pdb_to_pdbqt_rdkit(pdb_path: str, out_path: str,
                         mol_type: str = "ligand") -> dict:
    """
    Convert a PDB file to PDBQT format.

    For ligands : uses RDKit — adds Hs, Gasteiger charges, AutoDock atom types.
    For receptors: delegates to _pdb_to_pdbqt_receptor_passthrough.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdPartialCharges

    try:
        if mol_type == "receptor":
            return _pdb_to_pdbqt_receptor_passthrough(pdb_path, out_path)

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


# ==============================================================================
# Ligand Center Calculator — pure PDB parsing, no RDKit needed
# ==============================================================================

def _compute_ligand_center(pdb_path: str) -> dict:
    """
    Parse all HETATM records from a PDB file (excluding water HOH/WAT),
    compute the geometric center (centroid) of all heavy atoms, and also
    return the bounding-box dimensions — directly usable as Vina's
    center_x/y/z and as a starting suggestion for size_x/y/z.

    Returns:
        {
          "center_x": float, "center_y": float, "center_z": float,
          "size_x":   float, "size_y":   float, "size_z":   float,
          "n_atoms":  int,
          "residues": [{"resname": str, "chain": str, "resseq": str}, ...]
        }
    """
    xs, ys, zs = [], [], []
    residues_seen: dict[str, dict] = {}

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('HETATM'):
                continue
            resname = line[17:20].strip()
            # Skip water molecules
            if resname in ('HOH', 'WAT', 'H2O', 'SOL'):
                continue
            # Skip hydrogens
            atom_name = line[12:16].strip()
            element   = line[76:78].strip() if len(line) > 77 else ''
            if element == 'H' or (not element and atom_name.startswith('H')):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)

            chain  = line[21].strip() or 'A'
            resseq = line[22:26].strip()
            key    = f"{resname}_{chain}_{resseq}"
            if key not in residues_seen:
                residues_seen[key] = {"resname": resname, "chain": chain, "resseq": resseq}

    if not xs:
        # Fall back: try ATOM records (e.g. the ligand was labelled ATOM not HETATM)
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith('ATOM  '):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                xs.append(x)
                ys.append(y)
                zs.append(z)

    if not xs:
        return {"status": "error",
                "message": "No HETATM (non-water) or ATOM records found. "
                           "Is this a ligand PDB file?"}

    cx = round(sum(xs) / len(xs), 3)
    cy = round(sum(ys) / len(ys), 3)
    cz = round(sum(zs) / len(zs), 3)

    # Bounding box — add 8 Å padding on each side for a sensible search box
    _PAD = 8.0
    sx = round(max(xs) - min(xs) + _PAD, 3)
    sy = round(max(ys) - min(ys) + _PAD, 3)
    sz = round(max(zs) - min(zs) + _PAD, 3)

    return {
        "status":   "success",
        "center_x": cx, "center_y": cy, "center_z": cz,
        "size_x":   sx, "size_y":   sy, "size_z":   sz,
        "n_atoms":  len(xs),
        "residues": list(residues_seen.values()),
    }


# ==============================================================================
# Flask routes
# ==============================================================================

@app.route('/tools/pdb_to_pdbqt', methods=['POST'])
def tools_pdb_to_pdbqt():
    """
    POST /tools/pdb_to_pdbqt
    Accepts multipart/form-data:
        file      – the .pdb file
        mol_type  – "ligand" (default) or "receptor"
        out_dir   – optional absolute output directory override

    Converted files are stored persistently under:
        uiapp/converted_pdbqt/{mol_type}/

    Returns JSON:
        { "status":      "success" | "error",
          "message":     str,
          "output_path": str,
          "filename":    str,
          "mol_type":    str,
          "pdbqt_text":  str }
    """
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

    out_dir = request.form.get('out_dir', '').strip()
    if not out_dir:
        out_dir = os.path.join(_CONVERTED_ROOT, mol_type)
    os.makedirs(out_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="elion_pdb2pdbqt_")
    try:
        stem         = Path(f.filename).stem
        pdb_path     = os.path.join(tmp_dir, f.filename)
        out_filename = stem + ".pdbqt"
        out_path     = os.path.join(out_dir, out_filename)
        f.save(pdb_path)

        # ── Try obabel first ──────────────────────────────────────────────────
        obabel = shutil.which("obabel")
        if obabel:
            import subprocess as _sp
            ob_flags = [obabel, pdb_path, "-O", out_path,
                        "--partialcharge", "gasteiger"]
            if mol_type == "receptor":
                ob_flags += ["-xr"]
            else:
                ob_flags += ["-h"]

            result = _sp.run(ob_flags, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and Path(out_path).exists():
                _sanitize_pdbqt(out_path, mol_type)
                return jsonify({
                    "status":      "success",
                    "message":     f"Converted with obabel ({mol_type}).",
                    "output_path": out_path,
                    "filename":    out_filename,
                    "mol_type":    mol_type,
                    "pdbqt_text":  Path(out_path).read_text(),
                })
            logger.warning("obabel failed (%s), falling back to RDKit",
                           result.stderr[:200])

        # ── RDKit fallback ────────────────────────────────────────────────────
        res = _pdb_to_pdbqt_rdkit(pdb_path, out_path, mol_type)
        if res["status"] == "success":
            if Path(out_path).exists():
                _sanitize_pdbqt(out_path, mol_type)
            res["filename"]   = out_filename
            res["mol_type"]   = mol_type
            res["pdbqt_text"] = Path(out_path).read_text()
        return jsonify(res)

    except Exception as exc:
        logger.exception("tools_pdb_to_pdbqt error")
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route('/tools/ligand_center', methods=['POST'])
def tools_ligand_center():
    """
    POST /tools/ligand_center
    Accepts multipart/form-data:
        file – a ligand .pdb file

    Parses all HETATM heavy atoms (skipping water), computes the geometric
    centroid and bounding-box dimensions, and returns values ready to paste
    directly into Vina's center_x/y/z and size_x/y/z fields.

    Returns JSON:
        { "status":   "success" | "error",
          "center_x": float, "center_y": float, "center_z": float,
          "size_x":   float, "size_y":   float, "size_z":   float,
          "n_atoms":  int,
          "residues": [{"resname": str, "chain": str, "resseq": str}] }
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in request."}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({"status": "error", "message": "Empty filename."}), 400
    if not f.filename.lower().endswith('.pdb'):
        return jsonify({"status": "error",
                        "message": "Only .pdb files are accepted."}), 400

    tmp_dir = tempfile.mkdtemp(prefix="elion_ligcenter_")
    try:
        pdb_path = os.path.join(tmp_dir, f.filename)
        f.save(pdb_path)
        result = _compute_ligand_center(pdb_path)
        return jsonify(result), (200 if result["status"] == "success" else 400)
    except Exception as exc:
        logger.exception("tools_ligand_center error")
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)