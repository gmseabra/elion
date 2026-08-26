# =============================================================================
# uiapp/core/pose_chem.py
# Chemistry-preserving export for docked poses.
#
# WHY THIS MODULE EXISTS
# ----------------------
# The batch pipeline used PDB as the interchange format between every stage:
#
#     SMILES ──RDKit──▶ .pdb ──obabel──▶ .mol2 ──obabel──▶ .pdbqt ──Vina──▶ *_out.pdbqt
#                        ▲                                                     │
#                        └───────────── download button ───────────────────────┘
#
# PDB carries no formal charges, and PDBQT carries no bond orders at all, so
# every stage after the first had to GUESS the chemistry back from 3D geometry.
# The guesses are fine for plain rings and fail badly for:
#
#   · rings containing oxygen (4H-chromen-4-one, coumarin, flavone, γ-pyranone)
#     — the old geometry heuristic forced every planar 5-/6-ring of C/N/O/S into
#     alternating Kekulé doubles with no valence cap, so the ring ether O came
#     out TRIVALENT. Maestro cannot keep an impossible valence, repairs it by
#     demoting the real C4=O carbonyl to a single bond, and then fills that
#     oxygen's free valence with an H → the 4-hydroxy-chromene the user saw
#     instead of the 4-one.
#   · N-oxides (quinoline/pyridine 1-oxide) — the N ends up with four bonds and
#     no formal charge is written anywhere, so Maestro adds an H to the terminal
#     oxygen → N–OH instead of N⁺–O⁻. H-bond donor/acceptor assignment and the
#     net charge are then both wrong.
#
# THE FIX, in order of preference:
#
#   1. TEMPLATE  — the ligand's SMILES is the ground truth and the pipeline knows
#                  it. Match it onto the docked heavy-atom skeleton and copy bond
#                  orders AND formal charges across. Exact, by construction.
#   2. PERCEIVED — no SMILES available: RDKit's rdDetermineBonds (the xyz2mol
#                  algorithm) does a real valence-consistent assignment and gets
#                  N-oxides, pyranones and nitro groups right.
#   3. GEOMETRY  — no RDKit at all: the old planarity/bond-length heuristic, but
#                  now valence-capped (an O with two neighbours can never take a
#                  double bond) and with terminal C=O / C≡N claimed BEFORE ring
#                  Kekulé, so an exocyclic carbonyl is never starved by the ring.
#
# and then two output formats:
#
#   · .pdb — same atoms, same coordinates, same order as before; now HETATM,
#            with symmetric order-duplicated CONECT and the formal charge in the
#            standard PDB columns 79-80.
#   · .sdf — the lossless one. Bond orders and charges are first-class fields in
#            an MDL molfile, so Maestro reads the structure exactly as intended
#            with nothing left to perceive. Import this when the drawing matters.
#
# Nothing here imports Flask; it is plain library code so the routes, the
# standalone re-export CLI and the tests can all share it.
# =============================================================================
from __future__ import annotations

import math
import os
import re

try:                                            # RDKit is optional at import time
    from rdkit import Chem
    from rdkit.Chem import AllChem
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass
    RDKIT_OK = True
except Exception:                               # pragma: no cover - env without rdkit
    Chem = None
    AllChem = None
    RDKIT_OK = False

try:
    from rdkit.Chem import rdDetermineBonds     # RDKit >= 2022.09
    DETERMINE_OK = True
except Exception:                               # pragma: no cover
    rdDetermineBonds = None
    DETERMINE_OK = False


# ── periodic-table scraps ────────────────────────────────────────────────────
# Covalent radii for distance-based connectivity. Identical numbers to
# pose_routes._infer_bonds_py / pose.js inferBonds on purpose: the three code
# paths must agree on WHICH atoms are bonded even when they disagree about the
# order, otherwise a pose renders one way in the viewer and exports another.
_COV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'P': 1.07,
        'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39, 'B': 0.84, 'Se': 1.20,
        'Si': 1.11}

# Neutral valence — the total bond order an atom of this element carries when it
# has no formal charge. Used as the cap that the old heuristic was missing.
_VAL = {'H': 1, 'B': 3, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Si': 4, 'P': 3,
        'S': 2, 'Cl': 1, 'Se': 2, 'Br': 1, 'I': 1}

# Reference bond lengths (Å): single, then double. A bond is called a double
# when it is shorter than the midpoint of the two — a scale-free test that works
# for S=O (1.44 vs 1.57, only 0.13 Å apart) and C=O (1.23 vs 1.43) alike. The
# previous fixed "single − 0.13 Å" rule silently missed every sulfonyl and
# phosphoryl double bond, which is why sulfonamides exported as S(O⁻)(O⁻).
_REF = {'C-C': 1.54, 'C-N': 1.47, 'C-O': 1.43, 'N-O': 1.44, 'C-S': 1.82,
        'N-N': 1.45, 'O-P': 1.63, 'O-S': 1.57, 'N-S': 1.71, 'C-Se': 1.98}
_DBL = {'C-C': 1.34, 'C-N': 1.28, 'C-O': 1.23, 'N-O': 1.22, 'C-S': 1.60,
        'N-N': 1.25, 'O-P': 1.50, 'O-S': 1.44, 'N-S': 1.54, 'C-Se': 1.74}
_TRIP = {'C-C': 1.27, 'C-N': 1.24, 'N-N': 1.20}


def _dbl_cut(pk):
    """Midpoint between the single- and double-bond reference for a pair."""
    s, d = _REF.get(pk), _DBL.get(pk)
    if s is None:
        return None
    return (s + d) / 2.0 if d is not None else s - 0.13


# =============================================================================
# PDB text → atom records
# =============================================================================
def parse_atom_lines(lines):
    """[(element, x, y, z, serial)] from PDB ATOM/HETATM lines.

    Element comes from columns 77-78 where present (a real PDB always has it,
    and _pdbqt_atom_line writes it for converted PDBQT), falling back to the
    leading letters of the atom name. Serial comes from columns 7-11 so the
    CONECT records we emit point at the numbers actually printed on the atom
    lines rather than assuming they run 1..N.
    """
    out = []
    for ln in lines:
        if ln[:6].rstrip() not in ('ATOM', 'HETATM'):
            continue
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
        try:
            serial = int(ln[6:11])
        except ValueError:
            serial = len(out) + 1
        out.append((el, x, y, z, serial))
    return out


def infer_bonds(atoms):
    """Distance-based connectivity — same rule as pose_routes/pose.js."""
    n = len(atoms)
    bonds = []
    for i in range(n):
        ei, xi, yi, zi = atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]
        ri = _COV.get(ei, 0.76)
        for j in range(i + 1, n):
            ej = atoms[j][0]
            dx = xi - atoms[j][1]; dy = yi - atoms[j][2]; dz = zi - atoms[j][3]
            d2 = dx * dx + dy * dy + dz * dz
            mx = ri + _COV.get(ej, 0.76) + 0.45
            if 0.20 < d2 < mx * mx:
                bonds.append((i, j))
    return bonds


# =============================================================================
# Tier 3 — geometry heuristic, now valence-capped
# =============================================================================
def perceive_orders_charges(atoms, bonds):
    """Bond orders AND formal charges from 3D geometry alone.

    Last-resort tier: used when RDKit is unavailable. Three ordered passes, all
    of them respecting a per-atom spare-valence budget:

      cap[i] = neutral_valence(element) − degree(i)

    so an ether oxygen with two neighbours has cap 0 and can never be handed a
    double bond — the single line of defence the previous version was missing,
    and the whole reason chromen-4-one exported as 4-hydroxychromene.

      pass A  terminal multiple bonds (C=O, N=O, C≡N …). Highest confidence and
              claimed FIRST, so a ring can no longer starve an exocyclic
              carbonyl by consuming its carbon's last free valence.
      pass B  planar 5-/6-rings → alternating Kekulé doubles, skipping any atom
              with no spare valence. The starting offset that covers the most
              ring atoms wins, which keeps benzene at three doubles while a
              γ-pyranone gets only the two its valences allow.
      pass C  whatever short bonds are left over.

    Formal charges are then read off the completed valences, but only for the
    patterns that are unambiguous (N with four bonds → N⁺, and a terminal O
    hanging off such an N or off a sulfonyl/phosphoryl/carboxyl → O⁻). A
    carbonyl we simply failed to spot must not turn into an alkoxide, so
    everything else stays neutral.

    Returns (orders, charges), both parallel to `bonds` / `atoms`.
    """
    n = len(atoms)
    orders = [1] * len(bonds)
    charges = [0] * n
    if not bonds:
        return orders, charges

    bidx = {}
    adj = [[] for _ in range(n)]
    for k, (a, b) in enumerate(bonds):
        bidx[(a, b) if a < b else (b, a)] = k
        adj[a].append(b)
        adj[b].append(a)

    def dist(i, j):
        return math.sqrt((atoms[i][1] - atoms[j][1]) ** 2 +
                         (atoms[i][2] - atoms[j][2]) ** 2 +
                         (atoms[i][3] - atoms[j][3]) ** 2)

    def pair(i, j):
        return '-'.join(sorted((atoms[i][0], atoms[j][0])))

    # ── spare valence ────────────────────────────────────────────────────────
    cap = [0] * n
    for i in range(n):
        el = atoms[i][0]
        deg = len(adj[i])
        val = _VAL.get(el, 4)
        if el == 'N' and deg == 3:
            # An sp2 N with three σ-bonds and a terminal O (or S) neighbour is an
            # N-oxide / nitro N: it really does carry four bonds and a + charge.
            if any(len(adj[j]) == 1 and atoms[j][0] in ('O', 'S') for j in adj[i]):
                val = 4
        elif el == 'N' and deg == 4:
            val = 4                                  # quaternary ammonium
        elif el == 'S' and deg >= 3:
            val = 6                                  # sulfoxide / sulfone / sulfonamide
        elif el == 'P' and deg >= 4:
            val = 5
        cap[i] = max(0, val - deg)

    def take(k, order):
        """Raise bond k to `order` if both atoms can afford the extra bonds."""
        a, b = bonds[k]
        extra = order - orders[k]
        if extra <= 0 or cap[a] < extra or cap[b] < extra:
            return False
        orders[k] = order
        cap[a] -= extra
        cap[b] -= extra
        return True

    # ── pass A: terminal (exocyclic) multiple bonds ──────────────────────────
    rings = _small_rings(adj, atoms)
    in_planar_ring = set()
    for cyc in rings:
        in_planar_ring.update(cyc)

    cand = []
    for k, (a, b) in enumerate(bonds):
        if atoms[a][0] == 'H' or atoms[b][0] == 'H':
            continue
        if len(adj[a]) != 1 and len(adj[b]) != 1:
            continue
        term, core = (a, b) if len(adj[a]) == 1 else (b, a)
        # An oxygen hanging off a nitrogen that sits IN a planar ring is an
        # aromatic N-oxide: the correct form is N⁺–O⁻ with the ring keeping its
        # aromatic double, not a pentavalent N=O that would leave a ring carbon
        # with nowhere to go. Exocyclic C=O (the chromen-4-one carbonyl) is the
        # opposite case and must still be claimed here, before the ring runs.
        if atoms[term][0] in ('O', 'S') and atoms[core][0] == 'N' \
                and core in in_planar_ring:
            continue
        pk = pair(a, b)
        d = dist(a, b)
        ref = _REF.get(pk)
        if ref is None:
            continue
        cand.append((d - ref, k, d, pk))             # most-shortened first
    cand.sort()
    for _slack, k, d, pk in cand:
        tri = _TRIP.get(pk)
        if tri is not None and d < tri and take(k, 3):
            continue
        cut = _dbl_cut(pk)
        if cut is not None and d < cut:
            take(k, 2)

    # ── pass B: planar rings ─────────────────────────────────────────────────
    for cyc in rings:
        m = len(cyc)
        best_off, best_hits = None, -1
        for off in range(m):
            hits, used = 0, set()
            for t in range(0, m - 1, 2):
                i, j = cyc[(off + t) % m], cyc[(off + t + 1) % m]
                if i in used or j in used or cap[i] < 1 or cap[j] < 1:
                    continue
                k = bidx.get((i, j) if i < j else (j, i))
                if k is None or orders[k] != 1 or dist(i, j) > 1.62:
                    continue
                used.add(i); used.add(j)
                hits += 1
            if hits > best_hits:
                best_hits, best_off = hits, off
        if not best_hits:
            continue
        used = set()
        for t in range(0, m - 1, 2):
            i, j = cyc[(best_off + t) % m], cyc[(best_off + t + 1) % m]
            if i in used or j in used or cap[i] < 1 or cap[j] < 1:
                continue
            k = bidx.get((i, j) if i < j else (j, i))
            if k is None or orders[k] != 1 or dist(i, j) > 1.62:
                continue
            if take(k, 2):
                used.add(i); used.add(j)

    # ── pass C: leftovers ────────────────────────────────────────────────────
    left = []
    for k, (a, b) in enumerate(bonds):
        if orders[k] != 1 or atoms[a][0] == 'H' or atoms[b][0] == 'H':
            continue
        pk = pair(a, b)
        if pk not in _REF:
            continue
        left.append((dist(a, b) - _REF[pk], k, dist(a, b), pk))
    left.sort()                                      # shortest (most certain) first
    for _slack, k, d, pk in left:
        tri = _TRIP.get(pk)
        if tri is not None and d < tri and take(k, 3):
            continue
        cut = _dbl_cut(pk)
        if cut is not None and d < cut:
            take(k, 2)

    # ── pass D: cationic-nitrogen rescue ─────────────────────────────────────
    # An atom still short of its valence after all that is the signature of a
    # missing formal charge: amidinium, guanidinium, pyridinium, imidazolium.
    # The partner N looks "full" only because we assumed it neutral, so promote
    # it to N⁺ and let it take the double. Nothing fires unless an atom is
    # genuinely left under-valent, so ordinary amides and anilines are untouched.
    rescue = []
    for i in range(n):
        if cap[i] < 1 or atoms[i][0] not in ('C', 'N'):
            continue
        for j in adj[i]:
            if atoms[j][0] != 'N' or cap[j] != 0 or len(adj[j]) != 3:
                continue
            k = bidx.get((i, j) if i < j else (j, i))
            if k is None or orders[k] != 1:
                continue
            cut = _dbl_cut(pair(i, j))
            d = dist(i, j)
            if cut is not None and d < cut:
                rescue.append((d, k, j))
    rescue.sort()
    promoted = set()
    for _d, k, j in rescue:
        if j in promoted:
            continue
        cap[j] += 1
        if take(k, 2):
            promoted.add(j)
        else:
            cap[j] -= 1

    # ── formal charges from the finished valences ────────────────────────────
    tot = [0] * n
    for k, (a, b) in enumerate(bonds):
        tot[a] += orders[k]
        tot[b] += orders[k]
    for i in range(n):
        el = atoms[i][0]
        if el == 'N' and tot[i] == 4:
            charges[i] = 1
        elif el == 'O' and tot[i] == 1 and len(adj[i]) == 1:
            j = adj[i][0]
            ej = atoms[j][0]
            # O⁻ only where the neighbour demands it: an N⁺ (N-oxide, nitro), a
            # hypervalent S/P (sulfonate, phosphate), or a sibling terminal O
            # that already took the double bond (carboxylate).
            sib = any(len(adj[w]) == 1 and atoms[w][0] == 'O' and tot[w] == 2
                      for w in adj[j] if w != i)
            if (ej == 'N' and tot[j] == 4) or (ej in ('S', 'P') and tot[j] >= 4) or sib:
                charges[i] = -1
    return orders, charges


def _small_rings(adj, atoms):
    """Planar 5-/6-rings, returned as ordered cycles.

    Same bounded DFS + planarity test as before (all atoms within 0.35 Å of the
    ring's mean plane, all of them C/N/O/S), just factored out.
    """
    n = len(adj)
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

    out = []
    for ring in rings:
        if not all(atoms[i][0] in ('C', 'N', 'O', 'S') for i in ring):
            continue
        a, b, c = atoms[ring[0]], atoms[ring[1]], atoms[ring[2]]
        ux, uy, uz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
        vx, vy, vz = c[1] - a[1], c[2] - a[2], c[3] - a[3]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl < 1e-3:
            continue
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        if max(abs((atoms[i][1] - a[1]) * nx + (atoms[i][2] - a[2]) * ny +
                   (atoms[i][3] - a[3]) * nz) for i in ring) > 0.35:
            continue                                  # not planar → not aromatic
        cyc, prev, cur = [ring[0]], -1, ring[0]
        for _ in range(len(ring) - 1):
            nxt = None
            for w in adj[cur]:
                if w != prev and w in ring and w not in cyc:
                    nxt = w
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
            cyc.append(cur)
        if len(cyc) == len(ring):
            out.append(cyc)
    return out


# =============================================================================
# The result object
# =============================================================================
class PoseMol(object):
    """One docked pose with its chemistry restored.

    mol      RDKit molecule with the docked conformer (None when RDKit is absent)
    atoms    [(el, x, y, z, serial)] parallel to the source ATOM/HETATM lines
    bonds    [(i, j)] indices into `atoms`
    orders   bond orders parallel to `bonds`
    charges  formal charges parallel to `atoms`
    source   'template' | 'perceived' | 'geometry' | 'none'
    smiles   the SMILES actually used (template tier) or read back from `mol`
    note     short human-readable reason, surfaced in the manifest
    """

    __slots__ = ('mol', 'atoms', 'bonds', 'orders', 'charges', 'source',
                 'smiles', 'note')

    def __init__(self, mol=None, atoms=None, bonds=None, orders=None,
                 charges=None, source='none', smiles='', note=''):
        self.mol = mol
        self.atoms = atoms or []
        self.bonds = bonds or []
        self.orders = orders or []
        self.charges = charges or []
        self.source = source
        self.smiles = smiles or ''
        self.note = note or ''

    @property
    def exact(self):
        """True when the chemistry came from the ligand's own SMILES."""
        return self.source == 'template'


# =============================================================================
# RDKit plumbing
# =============================================================================
_BT = None


def _bt():
    global _BT
    if _BT is None:
        _BT = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
               3: Chem.BondType.TRIPLE}
    return _BT


def _rwmol(atoms, bonds, no_implicit=True):
    """RWMol with the docked coordinates and every bond single.

    no_implicit=True is the honest default: these files carry every hydrogen as
    a real atom, so RDKit must not quietly invent more — inventing hydrogens to
    patch up a bad valence is precisely the Maestro behaviour this module
    exists to stop. The perception tier turns it off because xyz2mol needs the
    freedom to place hydrogens itself.
    """
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, (el, x, y, z, _s) in enumerate(atoms):
        a = Chem.Atom(el)
        a.SetNoImplicit(no_implicit)
        rw.AddAtom(a)
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    for a, b in bonds:
        if rw.GetBondBetweenAtoms(a, b) is None:
            rw.AddBond(a, b, Chem.BondType.SINGLE)
    m = rw.GetMol()
    m.AddConformer(conf, assignId=True)
    return Chem.RWMol(m)


def _skeleton(mol):
    """Element-and-connectivity-only copy: no orders, charges or aromaticity.

    Substructure matching a real (aromatic, charged) template against a docked
    skeleton whose bonds are all single would fail on every aromatic ring, so
    both sides are flattened to this common denominator first — the same trick
    RDKit's own AssignBondOrdersFromTemplate uses internally.
    """
    m = Chem.RWMol(mol)
    for b in m.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    m = m.GetMol()
    Chem.FastFindRings(m)                          # matching needs ring info
    return m


def _heavy_view(atoms, bonds, orders=None):
    """Heavy-atom sub-molecule plus the index map back to the full atom list."""
    keep = [i for i, a in enumerate(atoms) if a[0] != 'H']
    pos = {i: k for k, i in enumerate(keep)}
    rw = Chem.RWMol()
    for i in keep:
        a = Chem.Atom(atoms[i][0])
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    for k, (a, b) in enumerate(bonds):
        if a in pos and b in pos and rw.GetBondBetweenAtoms(pos[a], pos[b]) is None:
            rw.AddBond(pos[a], pos[b],
                       _bt()[orders[k]] if orders else Chem.BondType.SINGLE)
    m = rw.GetMol()
    Chem.FastFindRings(m)
    return m, keep, pos


def _finalise(rw, atoms):
    """Sanitize, and hand back (mol, bonds, orders, charges) or None.

    Orders are read off a KEKULIZED copy, never off the aromatic form. An
    aromatic bond's order is 1.5, and a CONECT record can only say "once,
    twice, three times" — rounding 1.5 up would write every bond of every
    aromatic ring as a double, which is exactly the kind of impossible valence
    this module exists to stop a reader from having to repair.
    """
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    try:
        kek = Chem.Mol(m)
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    bonds, orders = [], []
    for b in kek.GetBonds():
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        o = b.GetBondTypeAsDouble()
        orders.append(int(o) if o in (1.0, 2.0, 3.0) else 1)
    charges = [a.GetFormalCharge() for a in m.GetAtoms()]
    return m, bonds, orders, charges


# ── tier 1: template ─────────────────────────────────────────────────────────
def _assign_from_template(atoms, bonds, smiles):
    """Copy bond orders + formal charges off the ligand's own SMILES.

    Returns (mol, bonds, orders, charges) or None when the SMILES does not
    describe this pose (wrong row in a CSV, a salt/counter-ion the docked file
    does not contain, a different protomer, …). Never guesses: a mismatch falls
    through to the next tier rather than exporting something plausible.
    """
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return None
    ref_noH = Chem.RemoveHs(ref)
    heavy, keep, pos = _heavy_view(atoms, bonds)
    if heavy.GetNumAtoms() != ref_noH.GetNumAtoms():
        return None

    match = _skeleton(heavy).GetSubstructMatch(_skeleton(ref_noH))
    if not match:
        return None
    # match[k] is the heavy-view index of ref atom k → full-atom-list index
    ref_to_full = {k: keep[match[k]] for k in range(len(match))}

    rw = _rwmol(atoms, bonds)
    for b in ref_noH.GetBonds():
        i = ref_to_full[b.GetBeginAtomIdx()]
        j = ref_to_full[b.GetEndAtomIdx()]
        bond = rw.GetBondBetweenAtoms(i, j)
        if bond is None:
            return None
        bond.SetBondType(b.GetBondType())
        bond.SetIsAromatic(b.GetIsAromatic())
    for k in range(ref_noH.GetNumAtoms()):
        src = ref_noH.GetAtomWithIdx(k)
        dst = rw.GetAtomWithIdx(ref_to_full[k])
        dst.SetFormalCharge(src.GetFormalCharge())
        dst.SetIsAromatic(src.GetIsAromatic())
        dst.SetNoImplicit(True)
        # Hydrogens already present as real atoms keep their coordinates; only
        # the shortfall (a polar-H-only PDBQT, say) becomes implicit for now and
        # gets real coordinates further down in build_pose().
        have = sum(1 for nb in dst.GetNeighbors() if nb.GetSymbol() == 'H')
        want = src.GetTotalNumHs()
        if have > want:
            return None                            # more H than the SMILES allows
        dst.SetNumExplicitHs(want - have)
    for i, a in enumerate(atoms):
        if a[0] == 'H':
            rw.GetAtomWithIdx(i).SetNoImplicit(True)

    res = _finalise(rw, atoms)
    if res is None:
        return None
    mol = res[0]
    # Belt and braces: the exported structure must be the same molecule we were
    # handed. If it is not, say so by failing rather than shipping it.
    #
    # Compared WITHOUT stereo on purpose. Constitution — which atoms, which bond
    # orders, which charges — is what the template supplies and what has to
    # match. Configuration is supplied by the docked coordinates, which are the
    # ground truth for this pose even when the SMILES was drawn flat (or drawn
    # with the other enantiomer). _stereo_note() reports that case rather than
    # silently discarding a perfectly good template match.
    try:
        if Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False) != \
                Chem.MolToSmiles(ref_noH, isomericSmiles=False):
            return None
    except Exception:
        return None
    try:
        Chem.AssignStereochemistryFrom3D(mol)      # the pose's real configuration
    except Exception:
        pass
    return res + (_stereo_note(mol, ref_noH),)


def _stereo_note(mol, ref):
    """'' unless the SMILES specified a configuration the pose does not have."""
    try:
        ref_iso = Chem.MolToSmiles(ref)
        if ref_iso == Chem.MolToSmiles(ref, isomericSmiles=False):
            return ''                              # template had no stereo to lose
        if Chem.MolToSmiles(Chem.RemoveHs(mol)) != ref_iso:
            return 'stereochemistry taken from the docked coordinates (differs ' \
                   'from the SMILES)'
    except Exception:
        pass
    return ''


# ── tier 2: RDKit perception ─────────────────────────────────────────────────
def _assign_by_perception(atoms, bonds):
    """rdDetermineBonds (xyz2mol) with a small net-charge sweep.

    Handles the two failure families that broke the old heuristic — ring
    oxygens and N-oxides — because it solves valences properly instead of
    pattern-matching planarity. Neutral ligands dominate, so charge 0 is tried
    first and the sweep only pays for itself on genuine ions.
    """
    if not DETERMINE_OK:
        return []
    had_h = any(a[0] == 'H' for a in atoms)
    out = []
    for chg in (0, -1, 1, -2, 2):
        rw = _rwmol(atoms, bonds, no_implicit=False)
        try:
            rdDetermineBonds.DetermineBondOrders(rw, charge=chg,
                                                 allowChargedFragments=True)
        except Exception:
            continue
        res = _finalise(rw, atoms)
        if res is None:
            continue
        # Hard filter: the docked file's hydrogens are data, not a starting
        # guess. An assignment that only balances by inventing extra H is the
        # very thing that made Maestro draw N–OH, so throw it away.
        if had_h and sum(a.GetNumImplicitHs() for a in res[0].GetAtoms()):
            continue
        out.append(res)
        if _score(res[0])[:3] == (0, 0, 0):            # neutral and clean → done
            break
    return out


def _score(mol):
    """How chemically plausible an assignment is — lower is better.

    Ordered by how badly each thing would misrepresent the ligand:
      radicals            an unpaired electron is almost always a failed guess
      charged C/halogen   a carbanion where an amidinium was meant
      total charge        prefer the least charge separation that works
      aromatic rings      NEGATED: prefer keeping rings aromatic. This is what
                          picks N⁺–O⁻ on an aromatic ring over a de-aromatized
                          N=O with the charge pushed onto a neighbouring amide.
      multiple bonds      NEGATED: prefer the more conjugated assignment
    """
    rad = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    odd = sum(1 for a in mol.GetAtoms()
              if a.GetFormalCharge() and a.GetSymbol() in ('C', 'F', 'Cl', 'Br', 'I'))
    tot = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
    try:
        rings = mol.GetRingInfo().AtomRings()
        arom = sum(1 for r in rings
                   if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    except Exception:
        arom = 0
    mult = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() > 1.0)
    return (rad, odd, tot, -arom, -mult)


# ── tier 3: geometry ─────────────────────────────────────────────────────────
def _assign_by_geometry(atoms, bonds):
    """The valence-capped geometric heuristic, as a sanitized molecule.

    Returns (mol, bonds, orders, charges) when RDKit accepts the result, else
    None — an assignment RDKit will not sanitize is not one to export.
    """
    if not RDKIT_OK:
        return None
    orders, charges = perceive_orders_charges(atoms, bonds)
    rw = _rwmol(atoms, bonds)
    for k, (a, b) in enumerate(bonds):
        bd = rw.GetBondBetweenAtoms(a, b)
        if bd is not None:
            bd.SetBondType(_bt()[orders[k]])
    for i, c in enumerate(charges):
        if c:
            rw.GetAtomWithIdx(i).SetFormalCharge(c)
    return _finalise(rw, atoms)


# =============================================================================
# Public entry point
# =============================================================================
def build_pose(atom_lines, smiles=None, want_hydrogens=True):
    """Restore the chemistry of one docked pose.

    `atom_lines` are the PDB ATOM/HETATM lines of a single pose, in file order —
    the atom order, coordinates and serial numbers are preserved exactly, so the
    exported file is comparable line-for-line with what the pipeline produced
    before. `smiles`, when given, is the ligand's ground truth and is tried
    first; anything else falls back through perception and then geometry.

    Always returns a PoseMol — export must never fail because a molecule is odd.
    """
    atoms = parse_atom_lines(atom_lines)
    if not atoms:
        return PoseMol(source='none', note='no ATOM/HETATM records')
    bonds = infer_bonds(atoms)

    if not RDKIT_OK:
        orders, charges = perceive_orders_charges(atoms, bonds)
        return PoseMol(None, atoms, bonds, orders, charges, 'geometry',
                       smiles or '', 'rdkit not installed')

    note = ''
    if smiles:
        res = _assign_from_template(atoms, bonds, smiles)
        if res is not None:
            mol, b, o, c, snote = res
            return _with_hydrogens(PoseMol(mol, atoms, b, o, c, 'template',
                                           smiles, snote), want_hydrogens)
        note = 'SMILES did not match the docked skeleton; perceived instead'

    # Two independent perceivers, then take the more plausible answer. xyz2mol
    # solves valences properly but will happily de-aromatize a ring to balance
    # one; the geometry heuristic keeps rings intact but is blunter. Scoring
    # both and picking the better one is strictly safer than trusting either.
    cands = [(_score(r[0]), 'perceived', r) for r in _assign_by_perception(atoms, bonds)]
    geo = _assign_by_geometry(atoms, bonds)
    if geo is not None and geo[0] is not None:
        cands.append((_score(geo[0]), 'geometry', geo))
    if cands:
        cands.sort(key=lambda t: t[0])
        _key, src, (mol, b, o, c) = cands[0]
        try:
            Chem.AssignStereochemistryFrom3D(mol)
        except Exception:
            pass
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            smi = ''
        return _with_hydrogens(PoseMol(mol, atoms, b, o, c, src, smi, note),
                               want_hydrogens)

    # Nothing sanitized — fall back to the raw geometric assignment, which at
    # least gives the viewer something to draw and the PDB something to write.
    orders, charges = perceive_orders_charges(atoms, bonds)
    return PoseMol(None, atoms, bonds, orders, charges, 'geometry', '',
                   note or 'valence perception failed; geometry heuristic used')


def _with_hydrogens(pose, want):
    """Give real coordinates to any H the docked file was missing.

    A PDBQT prepared with polar hydrogens only comes back with implicit Hs after
    the template step; an SDF with implicit hydrogens is legal but Maestro then
    has nothing to place, so materialise them here. Atom order for the atoms
    that DID come from the file is untouched — AddHs appends.
    """
    if not want or pose.mol is None:
        return pose
    try:
        need = sum(a.GetNumExplicitHs() + a.GetNumImplicitHs()
                   for a in pose.mol.GetAtoms())
        if need:
            pose.mol = Chem.AddHs(pose.mol, addCoords=True)
            pose.note = (pose.note + '; ' if pose.note else '') + \
                        ('%d hydrogen(s) added with coordinates' % need)
    except Exception:
        pass
    return pose


# =============================================================================
# Writers
# =============================================================================
def _charge_field(c):
    """PDB columns 79-80: '1+' / '2-' / blank. This is the field Maestro needs
    to see an N-oxide as N⁺–O⁻ instead of quietly capping the oxygen with an H."""
    if not c:
        return '  '
    c = max(-9, min(9, int(c)))
    return '%d%s' % (abs(c), '+' if c > 0 else '-')


def pose_pdb_lines(pose, atom_lines, remarks=(), het=True):
    """PDB block for one pose: original atoms, correct bonds, real charges.

    Columns 1-76 of every atom line are reused verbatim (name, altLoc, resName,
    chain, resSeq, x/y/z, occupancy, B-factor), so nothing about the geometry or
    the atom naming shifts. What changes:

      · record name → HETATM. UNL is not a standard residue and CONECT is
        defined for het groups; leaving it as ATOM invites a reader to treat the
        ligand as a polymer fragment.
      · element right-justified in columns 77-78 (unchanged where already there).
      · formal charge in columns 79-80 — the missing piece for N-oxides,
        carboxylates, nitro groups and amidinium/guanidinium ions.
      · CONECT written symmetrically, with the partner repeated once per bond
        order, which is the form Maestro honours.
    """
    src = [l for l in atom_lines if l[:6].rstrip() in ('ATOM', 'HETATM')]
    atoms = pose.atoms or parse_atom_lines(src)
    charges = pose.charges or [0] * len(atoms)

    serials = [a[4] for a in atoms]
    if len(set(serials)) != len(serials):          # duplicated/unparsable → renumber
        serials = list(range(1, len(atoms) + 1))

    out = list(remarks)
    for i, ln in enumerate(src):
        if i >= len(atoms):
            break
        body = ln.rstrip('\n').ljust(80)
        head = 'HETATM' if het else body[:6]
        el = atoms[i][0]
        el = (el[0].upper() + el[1:].lower()) if len(el) == 2 else el.upper()
        line = (head + body[6:76] + el.rjust(2) +
                _charge_field(charges[i] if i < len(charges) else 0))
        out.append(line.rstrip())

    if pose.bonds:
        adj = {}
        for k, (a, b) in enumerate(pose.bonds):
            if a >= len(serials) or b >= len(serials):
                continue
            o = pose.orders[k] if k < len(pose.orders) else 1
            ia, ib = serials[a], serials[b]
            for _ in range(max(1, int(o))):
                adj.setdefault(ia, []).append(ib)
                adj.setdefault(ib, []).append(ia)
        for a in sorted(adj):
            nb = sorted(adj[a])
            for k in range(0, len(nb), 4):         # ≤4 partners per line (PDB spec)
                out.append('CONECT%5d' % a + ''.join('%5d' % x for x in nb[k:k + 4]))
    return out


def pose_sdf_block(pose, name='', props=None):
    """One SDF record — the lossless format, and the one to open in Maestro.

    Bond orders and charges are explicit fields in an MDL molfile, so there is
    nothing left for the reader to perceive and nothing that can be perceived
    wrongly. Returns '' when RDKit is unavailable or the molecule could not be
    rebuilt (the caller then simply ships the PDB alone).
    """
    if pose.mol is None or not RDKIT_OK:
        return ''
    try:
        m = Chem.Mol(pose.mol)
        m.SetProp('_Name', str(name or ''))
        block = Chem.MolToMolBlock(m, kekulize=True)
        tail = ''.join('>  <%s>\n%s\n\n' % (k, v)
                       for k, v in (props or {}).items() if v not in (None, ''))
        return block + tail + '$$$$\n'
    except Exception:
        return ''


# =============================================================================
# Where the SMILES comes from
# =============================================================================
_SMI_COLS = ('smiles', 'smile', 'canonical_smiles', 'smiles_string', 'structure')
_ID_COLS = ('name', 'ligand_id', 'id', 'title', 'molecule_name', 'mol_id',
            'compound_id', 'ligand', 'cid')
_INDEX_DIR_HINTS = ('results', 'result', 'smiles', 'smi', 'csv', 'data', 'ts',
                    'input', 'inputs', 'ligand', 'ligands')


def stem_keys(path):
    """Every plausible ligand id for a docked file name.

    Batch runs name their output <id>_<timestamp>_out.pdbqt, so '479575',
    '479575_20260806132002' and the full stem must all be tried against a
    SMILES table keyed on the original Name column.
    """
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    keys = [stem]
    if stem.lower().endswith('_out'):
        stem = stem[:-4]
        keys.append(stem)
    m = re.match(r'^(.*?)[._-]\d{8,14}$', stem)     # trailing yyyymmddhhmmss stamp
    if m and m.group(1):
        keys.append(m.group(1))
    m = re.match(r'^(\d+)\b', stem)                 # leading numeric id
    if m and m.group(1) not in keys:
        keys.append(m.group(1))
    seen, out = set(), []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def read_smiles_table(path, limit_bytes=64 * 1024 * 1024):
    """{id: smiles} from a .csv/.tsv/.smi/.txt table. {} if it is not one."""
    out = {}
    try:
        if os.path.getsize(path) > limit_bytes:
            return out
    except OSError:
        return out
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in ('.smi', '.smiles', '.txt'):
            with open(path, 'r', errors='ignore') as fh:
                for ln in fh:
                    tok = ln.split()
                    if len(tok) >= 2 and _looks_like_smiles(tok[0]):
                        out[tok[1]] = tok[0]
            return out
        if ext not in ('.csv', '.tsv'):
            return out
        import csv
        with open(path, 'r', errors='ignore', newline='') as fh:
            sample = fh.read(8192)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
            except Exception:
                dialect = csv.excel if ext == '.csv' else csv.excel_tab
            rd = csv.DictReader(fh, dialect=dialect)
            cols = {(c or '').strip().lower(): c for c in (rd.fieldnames or [])}
            smi_col = next((cols[c] for c in _SMI_COLS if c in cols), None)
            id_col = next((cols[c] for c in _ID_COLS if c in cols), None)
            if not smi_col or not id_col:
                return out
            for row in rd:
                sid = (row.get(id_col) or '').strip()
                smi = (row.get(smi_col) or '').strip()
                if sid and smi:
                    out.setdefault(sid, smi)
    except Exception:
        return out
    return out


def _looks_like_smiles(tok):
    return bool(tok) and not tok.replace('.', '').replace('-', '').isdigit()


def build_smiles_index(paths, extra_tables=(), max_up=5, max_tables=40):
    """{id: smiles} gathered from wherever the batch keeps its SMILES.

    Looks, cheapest first, at: any table the caller names explicitly, then every
    ancestor directory of the docked files (up to `max_up` levels) and the
    obviously-named subfolders inside them (results*/smiles*/csv/data/…). Bounded
    on purpose — this runs inside a download request, so it may not go wandering
    over the whole filesystem.
    """
    index, tables, seen_dirs = {}, [], set()
    for t in extra_tables or ():
        if t and os.path.isfile(t):
            tables.append(t)

    roots = []
    for p in paths:
        d = os.path.dirname(os.path.abspath(str(p)))
        for _ in range(max_up):
            if d in seen_dirs:
                break
            seen_dirs.add(d)
            roots.append(d)
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd

    for d in roots:
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            if e.is_file() and os.path.splitext(e.name)[1].lower() in \
                    ('.csv', '.tsv', '.smi', '.smiles'):
                tables.append(e.path)
            elif e.is_dir() and any(h in e.name.lower() for h in _INDEX_DIR_HINTS):
                try:
                    for f in os.scandir(e.path):
                        if f.is_file() and os.path.splitext(f.name)[1].lower() in \
                                ('.csv', '.tsv', '.smi', '.smiles'):
                            tables.append(f.path)
                except OSError:
                    pass
        if len(tables) >= max_tables:
            break

    for t in tables[:max_tables]:
        for k, v in read_smiles_table(t).items():
            index.setdefault(str(k).strip(), v)
    return index


def smiles_from_text(text):
    """SMILES carried in a REMARK line of the file itself, if any."""
    for ln in text.splitlines()[:400]:
        if ln[:6].strip() in ('REMARK', 'COMPND') and 'SMILES' in ln.upper():
            m = re.search(r'SMILES[:=\s]+(\S+)', ln, re.IGNORECASE)
            if m:
                return m.group(1)
        if ln[:6].rstrip() in ('ATOM', 'HETATM'):
            break
    return None


def resolve_smiles(path, text='', index=None):
    """(smiles, where) for one docked file — sidecar, REMARK, then the index."""
    base = os.path.splitext(str(path))[0]
    for cand in (base + '.smi', base + '.smiles',
                 re.sub(r'_out$', '', base) + '.smi'):
        try:
            if os.path.isfile(cand):
                tok = open(cand, errors='ignore').read().split()
                if tok and _looks_like_smiles(tok[0]):
                    return tok[0], 'sidecar'
        except OSError:
            pass
    if text:
        s = smiles_from_text(text)
        if s:
            return s, 'remark'
    if index:
        for k in stem_keys(path):
            if k in index:
                return index[k], 'table'
    return None, ''
