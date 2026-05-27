"""
routes/tools.py — Standalone tools blueprint.
Routes served under /tools/* (url_prefix in __init__.py).
"""

import os, shutil, tempfile, logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from elion_config import CONVERTED_ROOT
from elion_shared import pdb_to_pdbqt_receptor, pdb_to_pdbqt_ligand, autodock_type

logger = logging.getLogger(__name__)
bp = Blueprint('tools', __name__)

@bp.route('/pdb_to_pdbqt', methods=['POST'])
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