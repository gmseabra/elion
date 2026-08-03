"""
ts_mol_render.py — reagent SMILES lookup + RDKit 2D SVG rendering.

Two responsibilities:
  1. Load reagent CSV files (id → SMILES) for a reaction's building blocks.
  2. Render a SMILES string to a compact, transparent-background SVG suitable
     for placing behind a stats bar in the TS UI.

The CSV format is auto-detected: the SMILES column is whichever column parses
as a valid molecule for the majority of sampled rows; the ID column is matched
to the reagent IDs elion emits (the bare building-block number).

Designed to be imported by ts_routes.py, but also runnable standalone for
testing:

    python ts_mol_render.py --smiles "CC(=O)O" --out test.svg
    python ts_mol_render.py --csv /path/rxn110_1.csv --id 920405
"""
import csv
import os
import threading

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # silence parse warnings
    _RDKIT_OK = True
except Exception:
    _RDKIT_OK = False


# ── id → SMILES maps, cached per CSV path ─────────────────────────────────────
_csv_cache: dict = {}          # csv_path -> { reagent_id(str): smiles(str) }
_csv_lock = threading.Lock()


def _looks_like_smiles(s: str) -> bool:
    if not s or len(s) < 2 or len(s) > 600:
        return False
    if not _RDKIT_OK:
        # crude fallback: SMILES usually contain these and rarely spaces
        return any(c in s for c in "CcNnOoSs[]()=#") and " " not in s.strip()
    return Chem.MolFromSmiles(s) is not None


def _detect_columns(rows: list) -> tuple:
    """Return (id_col_idx, smiles_col_idx) by sampling rows.
    SMILES column = the column whose values parse as molecules most often.
    ID column = the first column that is NOT the SMILES column and looks like
    short alphanumeric identifiers (the building-block number).
    """
    if not rows:
        return (0, 1)
    ncols = max(len(r) for r in rows)
    sample = rows[: min(40, len(rows))]

    # Score each column by how many sampled cells look like valid SMILES
    smiles_hits = [0] * ncols
    for r in sample:
        for ci in range(min(len(r), ncols)):
            if _looks_like_smiles(r[ci].strip()):
                smiles_hits[ci] += 1
    smiles_col = max(range(ncols), key=lambda c: smiles_hits[c])

    # ID column: prefer a column of short tokens (<=12 chars, mostly digits)
    def id_score(ci):
        if ci == smiles_col:
            return -1
        hits = 0
        for r in sample:
            if ci < len(r):
                v = r[ci].strip()
                if 0 < len(v) <= 12 and any(ch.isdigit() for ch in v):
                    hits += 1
        return hits
    id_col = max(range(ncols), key=id_score)
    if id_col == smiles_col:  # degenerate; fall back
        id_col = 0 if smiles_col != 0 else 1
    return (id_col, smiles_col)


def load_reagent_map(csv_path: str) -> dict:
    """Load {reagent_id: smiles} from a reagent CSV. Cached by path."""
    with _csv_lock:
        if csv_path in _csv_cache:
            return _csv_cache[csv_path]

    mapping: dict = {}
    if not os.path.exists(csv_path):
        with _csv_lock:
            _csv_cache[csv_path] = mapping
        return mapping

    try:
        with open(csv_path, newline="") as f:
            # Sniff delimiter
            head = f.read(4096)
            f.seek(0)
            delim = "\t" if head.count("\t") > head.count(",") else ","
            reader = csv.reader(f, delimiter=delim)
            rows = [r for r in reader if r]

        if not rows:
            with _csv_lock:
                _csv_cache[csv_path] = mapping
            return mapping

        # Drop a header row if its cells don't look like data
        first = rows[0]
        has_header = not any(_looks_like_smiles(c.strip()) for c in first) and \
                     not any(c.strip().isdigit() for c in first)
        data_rows = rows[1:] if has_header else rows

        id_col, smi_col = _detect_columns(data_rows)
        for r in data_rows:
            if len(r) > max(id_col, smi_col):
                rid = r[id_col].strip()
                smi = r[smi_col].strip()
                if rid and smi:
                    mapping[rid] = smi
    except Exception:
        pass

    with _csv_lock:
        _csv_cache[csv_path] = mapping
    return mapping


def build_smiles_index(bb_base: str, reagent_files: list) -> dict:
    """Merge id→SMILES maps across all building-block CSVs for a reaction.
    Returns a single dict { reagent_id: smiles }.
    """
    merged: dict = {}
    for fname in reagent_files:
        path = fname if os.path.isabs(fname) else os.path.join(bb_base, fname)
        merged.update(load_reagent_map(path))
    return merged


# ── Reaction product computation, cached per (smarts, smiles_a, smiles_b) ─────
# Given two building-block SMILES and a reaction SMARTS, apply the reaction and
# return the product SMILES. The SMARTS has two reactant templates (slot 1 and
# slot 2); we don't know which BB matches which template, so we try BOTH
# orderings and take the first that yields exactly one product set.
_rxn_obj_cache: dict = {}      # smarts -> compiled ChemicalReaction (or None)
_product_cache: dict = {}      # (smarts, smi_a, smi_b) -> product_smiles ('' = none)
_rxn_lock = threading.Lock()


def _get_reaction(smarts: str):
    """Compile (and cache) a reaction SMARTS into an RDKit reaction object.
    Returns None if rdkit is unavailable or the SMARTS won't compile."""
    if not smarts or not _RDKIT_OK:
        return None
    with _rxn_lock:
        if smarts in _rxn_obj_cache:
            return _rxn_obj_cache[smarts]
    rxn = None
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is not None:
            rxn.Initialize()
    except Exception:
        rxn = None
    with _rxn_lock:
        _rxn_obj_cache[smarts] = rxn
    return rxn


def react_product_smiles(smarts: str, smiles_a: str, smiles_b: str) -> str:
    """Apply a 2-reactant reaction SMARTS to two BB SMILES and return the
    canonical product SMILES. Tries both reactant orderings (A,B) and (B,A)
    since we don't know which BB matches reactant template 1 vs 2.

    Returns '' when:
      - rdkit unavailable / SMARTS won't compile
      - either SMILES won't parse
      - neither ordering produces exactly one clean, sanitizable product
    """
    if not smarts or not smiles_a or not smiles_b or not _RDKIT_OK:
        return ""
    key = (smarts, smiles_a, smiles_b)
    with _rxn_lock:
        if key in _product_cache:
            return _product_cache[key]

    result = ""
    try:
        rxn = _get_reaction(smarts)
        ma = Chem.MolFromSmiles(smiles_a)
        mb = Chem.MolFromSmiles(smiles_b)
        if rxn is not None and ma is not None and mb is not None:
            # Try both orderings; collect every unique valid product.
            products = set()
            for r1, r2 in ((ma, mb), (mb, ma)):
                try:
                    outcomes = rxn.RunReactants((r1, r2))
                except Exception:
                    outcomes = ()
                for prod_tuple in outcomes:
                    if not prod_tuple:
                        continue
                    prod = prod_tuple[0]   # single-product reactions
                    try:
                        Chem.SanitizeMol(prod)
                        smi = Chem.MolToSmiles(prod)
                        if smi:
                            products.add(smi)
                    except Exception:
                        continue
            # Accept only an unambiguous single product. Multiple distinct
            # products (regiochemistry ambiguity, symmetric BBs reacting at
            # several sites) → treat as "no clean product" so the UI shows the
            # fallback rather than picking one arbitrarily.
            if len(products) == 1:
                result = next(iter(products))
    except Exception:
        result = ""

    with _rxn_lock:
        _product_cache[key] = result
    return result


# ── SVG rendering, cached per (smiles, w, h) ──────────────────────────────────
_svg_cache: dict = {}
_svg_lock = threading.Lock()


def smiles_to_svg(smiles: str, width: int = 220, height: int = 90) -> str:
    """Render a SMILES to a transparent-background SVG string.
    Returns '' if rdkit is unavailable or the SMILES can't be parsed.
    Cached by (smiles, width, height).
    """
    if not smiles or not _RDKIT_OK:
        return ""
    key = (smiles, width, height)
    with _svg_lock:
        if key in _svg_cache:
            return _svg_cache[key]

    svg = ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            d = rdMolDraw2D.MolDraw2DSVG(width, height)
            opts = d.drawOptions()
            opts.clearBackground = False     # transparent — dark UI shows through
            opts.bondLineWidth = 1
            opts.padding = 0.08

            # Light theme for a DARK background. RDKit defaults draw carbon/bonds
            # in black (invisible on the app's near-black panel). The atom palette's
            # key -1 is the DEFAULT colour (used for carbon and bonds); other keys
            # are atomic numbers. All chosen bright enough to read on dark.
            light = (0.85, 0.88, 0.92)       # off-white default (C + bonds)
            palette = {
                -1: light,                   # default (carbon, bonds, text)
                6:  light,                   # C
                1:  (0.75, 0.78, 0.82),      # H
                7:  (0.45, 0.62, 1.0),       # N  — blue
                8:  (1.0, 0.45, 0.45),       # O  — red
                9:  (0.55, 0.9, 0.55),       # F  — green
                15: (1.0, 0.6, 0.3),         # P  — orange
                16: (1.0, 0.85, 0.35),       # S  — yellow
                17: (0.45, 0.85, 0.5),       # Cl — green
                35: (0.85, 0.55, 0.35),      # Br — orange/brown
                53: (0.7, 0.45, 0.85),       # I  — purple
            }
            opts.setAtomPalette(palette)
            opts.setBackgroundColour((0, 0, 0, 0))  # fully transparent

            d.DrawMolecule(mol)
            d.FinishDrawing()
            svg = d.GetDrawingText()
            # Strip the XML prolog so it can be inlined directly in HTML
            if svg.startswith("<?xml"):
                svg = svg.split("?>", 1)[1].lstrip()
    except Exception:
        svg = ""

    with _svg_lock:
        _svg_cache[key] = svg
    return svg


def render_to_disk(smiles: str, out_path: str, width: int = 220, height: int = 90) -> bool:
    """Render a SMILES to an SVG file on disk. Returns True on success.
    Skips rendering if the file already exists (idempotent cache).
    """
    if not smiles or not _RDKIT_OK:
        return False
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    svg = smiles_to_svg(smiles, width, height)
    if not svg:
        return False
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(svg)
        os.replace(tmp, out_path)
        return True
    except Exception:
        return False


def prerender_reaction(bb_base: str, reagent_files: list, out_dir: str,
                       reagent_ids=None, width: int = 220, height: int = 90) -> int:
    """Pre-render molecule SVGs for a reaction into out_dir.
    If reagent_ids is given, only those are rendered; otherwise ALL reagents in
    the CSVs are rendered (can be large). Returns the count rendered.
    Files are named <reagent_id>.svg.
    """
    index = build_smiles_index(bb_base, reagent_files)
    ids = reagent_ids if reagent_ids is not None else list(index.keys())
    n = 0
    for rid in ids:
        smi = index.get(str(rid), "")
        if smi and render_to_disk(smi, os.path.join(out_dir, f"{rid}.svg"), width, height):
            n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles")
    ap.add_argument("--csv")
    ap.add_argument("--id")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=220)
    ap.add_argument("--height", type=int, default=90)
    a = ap.parse_args()

    print(f"rdkit available: {_RDKIT_OK}")

    smi = a.smiles
    if a.csv and a.id:
        m = load_reagent_map(a.csv)
        print(f"loaded {len(m)} reagents from {a.csv}")
        smi = m.get(a.id)
        print(f"id {a.id} -> {smi}")
        # show a few sample entries
        for i, (k, v) in enumerate(m.items()):
            if i >= 3:
                break
            print(f"   sample: {k} -> {v}")

    if smi:
        svg = smiles_to_svg(smi, a.width, a.height)
        print(f"svg length: {len(svg)}")
        if a.out:
            with open(a.out, "w") as f:
                f.write(svg)
            print(f"wrote {a.out}")