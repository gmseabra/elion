/* =============================================================================
 * vina_engine.js — AutoDock Vina 1.2.7 scoring function + search, in JavaScript.
 * -----------------------------------------------------------------------------
 * A faithful transcription of the parts of AutoDock Vina 1.2.7 that determine a
 * number: XS atom typing, the five-term empirical potential, the conf-independent
 * Nrot penalty, curl capping, the out-of-box penalty, the intramolecular pair
 * set, analytic Cartesian + torsion-tree gradients, BFGS with Armijo line search,
 * mutate_conf, and the Monte-Carlo schedule with pose clustering.
 *
 * Provenance for every constant is cited to the C++ file it came from so this
 * can be re-checked against upstream.
 *
 * NOT implemented (deliberately, per scope): the ad4 and vinardo scoring
 * functions, and the macrocycle glue types (G0..G3 / CG0..CG3) with their
 * linearattraction term. Ligands containing macrocycle closure dummies will
 * score as ordinary carbons.
 *
 * No DOM, no imports. Safe to load in a Worker or in Node.
 * ============================================================================= */
(function (root) {
'use strict';

/* ===========================================================================
 * 1. Atom types  (atom_constants.h)
 * =========================================================================== */

// X-Score atom types, in upstream order (atom_constants.h:80-98).
// Glue/macrocycle types 19..31 are out of scope; XS_SIZE stops at Met_D.
var XS = {
  C_H: 0, C_P: 1, N_P: 2, N_D: 3, N_A: 4, N_DA: 5,
  O_P: 6, O_D: 7, O_A: 8, O_DA: 9, S_P: 10, P_P: 11,
  F_H: 12, Cl_H: 13, Br_H: 14, I_H: 15, Si: 16, At: 17, Met_D: 18
};
var XS_SIZE = 19;
var XS_NAME = ['C_H','C_P','N_P','N_D','N_A','N_DA','O_P','O_D','O_A','O_DA',
               'S_P','P_P','F_H','Cl_H','Br_H','I_H','Si','At','Met_D'];

// xs_vdw_radii[] — atom_constants.h:263-296. Verbatim.
var XS_RADIUS = [
  1.9, // C_H
  1.9, // C_P
  1.8, // N_P
  1.8, // N_D
  1.8, // N_A
  1.8, // N_DA
  1.7, // O_P
  1.7, // O_D
  1.7, // O_A
  1.7, // O_DA
  2.0, // S_P
  2.1, // P_P
  1.5, // F_H
  1.8, // Cl_H
  2.0, // Br_H
  2.2, // I_H
  2.2, // Si
  2.3, // At
  1.2  // Met_D
];

// atom_constants.h:359-365
function xsIsHydrophobic(t) {
  return t === XS.C_H || t === XS.F_H || t === XS.Cl_H || t === XS.Br_H || t === XS.I_H;
}
// atom_constants.h:367-372
function xsIsAcceptor(t) {
  return t === XS.N_A || t === XS.N_DA || t === XS.O_A || t === XS.O_DA;
}
// atom_constants.h:374-380 — note Met_D counts as a donor.
function xsIsDonor(t) {
  return t === XS.N_D || t === XS.N_DA || t === XS.O_D || t === XS.O_DA || t === XS.Met_D;
}
// atom_constants.h:382-387
function xsHBondPossible(t1, t2) {
  return (xsIsDonor(t1) && xsIsAcceptor(t2)) || (xsIsDonor(t2) && xsIsAcceptor(t1));
}
// optimal_distance — potentials.h:46-49
function optimalDistance(t1, t2) { return XS_RADIUS[t1] + XS_RADIUS[t2]; }

// Metals Vina recognises: ad_type table + non_ad_metal_names (atom_constants.h:347).
var METALS = {
  Mg:1, Mn:1, Zn:1, Ca:1, Fe:1, Cu:1, Na:1, K:1, Hg:1, Co:1, U:1, Cd:1, Ni:1
};
// Anything not C/H is a heteroatom for the C_P rule — ad_is_heteroatom()
// (atom_constants.h:218-222) excludes exactly A, C, H and HD.
function isHeteroatomEl(el) { return el !== 'C' && el !== 'H'; }

/* ===========================================================================
 * 2. Atom typing
 * ---------------------------------------------------------------------------
 * Vina reads acceptor status from the PDBQT AD type (OA/NA) and donor status
 * from an attached polar hydrogen (HD) — model.cpp:418-419. A plain .pdb has
 * neither, so we reconstruct both: for the receptor from residue+atom-name
 * chemistry, for the ligand from the bond graph plus implicit-hydrogen valence.
 * This is what closes the ~2.5 kcal/mol H-bond gap the element-only rule left.
 * =========================================================================== */

// Protein donor/acceptor table, keyed "RES:ATOM" then falling back to "*:ATOM".
// [isDonor, isAcceptor]. Derived from standard amino-acid chemistry at pH 7.
var PROT_DA = {
  '*:N'   : [1,0],  // backbone amide NH (proline handled below)
  '*:O'   : [0,1],  // backbone carbonyl
  '*:OXT' : [0,1],
  'SER:OG' : [1,1], 'THR:OG1': [1,1], 'TYR:OH' : [1,1],
  'CYS:SG' : [0,0],                        // X-Score ignores SA — model.cpp:418
  'ASN:OD1': [0,1], 'ASN:ND2': [1,0],
  'GLN:OE1': [0,1], 'GLN:NE2': [1,0],
  'ASP:OD1': [0,1], 'ASP:OD2': [0,1],
  'GLU:OE1': [0,1], 'GLU:OE2': [0,1],
  'LYS:NZ' : [1,0],
  'ARG:NE' : [1,0], 'ARG:NH1': [1,0], 'ARG:NH2': [1,0],
  'HIS:ND1': [1,1], 'HIS:NE2': [1,1],      // tautomer-agnostic: both D and A
  'TRP:NE1': [1,0],
  'HOH:O'  : [1,1], 'WAT:O'  : [1,1]
};

function protDonorAcceptor(resn, name, el) {
  var k = resn + ':' + name;
  if (PROT_DA[k]) return PROT_DA[k];
  if (name === 'N' && resn === 'PRO') return [0,0];   // proline N has no H
  if (PROT_DA['*:' + name]) return PROT_DA['*:' + name];
  // Unknown residue (ligand-like HETATM, modified residue, cofactor): fall back
  // to conservative element chemistry — O accepts, N donates and accepts.
  if (el === 'O') return [1,1];
  if (el === 'N') return [1,1];
  return [0,0];
}

/**
 * Type a receptor from parsed PDB atoms.
 * @param {Array} atoms  [{el,x,y,z,name,resn,resseq,chain,het}]
 * @returns {{xs:Int32Array, x:Float64Array, y:Float64Array, z:Float64Array, n:number, label:Array}}
 */
function typeReceptor(atoms) {
  var n = 0, i;
  for (i = 0; i < atoms.length; i++) if (atoms[i].el !== 'H') n++;
  var xs = new Int32Array(n), X = new Float64Array(n),
      Y = new Float64Array(n), Z = new Float64Array(n), label = new Array(n);
  var idx = new Int32Array(n);            // back-reference into `atoms`
  var k = 0;
  for (i = 0; i < atoms.length; i++) {
    if (atoms[i].el === 'H') continue;
    X[k] = atoms[i].x; Y[k] = atoms[i].y; Z[k] = atoms[i].z;
    idx[k] = i;
    label[k] = (atoms[i].name || '?') + '/' + (atoms[i].resn || '?') + (atoms[i].resseq || '');
    k++;
  }
  // Covalent neighbour test for the C_P rule (Vina walks the bond graph; a
  // receptor .pdb has no CONECT for the protein, so 1.85 A stands in for it).
  var COV2 = 1.85 * 1.85;
  for (k = 0; k < n; k++) {
    var a = atoms[idx[k]], el = a.el;
    if (el === 'C') {
      var polar = false;
      for (var j = 0; j < n && !polar; j++) {
        if (j === k) continue;
        if (!isHeteroatomEl(atoms[idx[j]].el)) continue;
        var dx = X[k]-X[j], dy = Y[k]-Y[j], dz = Z[k]-Z[j];
        if (dx*dx + dy*dy + dz*dz < COV2) polar = true;
      }
      xs[k] = polar ? XS.C_P : XS.C_H;
    } else if (el === 'N' || el === 'O') {
      var da = protDonorAcceptor(a.resn || '', a.name || '', el);
      var d = !!da[0], ac = !!da[1];
      if (el === 'N') xs[k] = (ac && d) ? XS.N_DA : (ac ? XS.N_A : (d ? XS.N_D : XS.N_P));
      else            xs[k] = (ac && d) ? XS.O_DA : (ac ? XS.O_A : (d ? XS.O_D : XS.O_P));
    }
    else if (el === 'S')  xs[k] = XS.S_P;
    else if (el === 'P')  xs[k] = XS.P_P;
    else if (el === 'F')  xs[k] = XS.F_H;
    else if (el === 'Cl') xs[k] = XS.Cl_H;
    else if (el === 'Br') xs[k] = XS.Br_H;
    else if (el === 'I')  xs[k] = XS.I_H;
    else if (el === 'Si') xs[k] = XS.Si;
    else if (METALS[el])  xs[k] = XS.Met_D;
    else                  xs[k] = XS.C_P;   // unknown heavy atom: polar carbon-ish
  }
  return { xs: xs, x: X, y: Y, z: Z, n: n, label: label, srcIndex: idx };
}

// Neutral heavy-atom valences, for deciding whether an N/O carries an implicit H.
var VALENCE = { C:4, N:3, O:2, S:2, P:3, F:1, Cl:1, Br:1, I:1, B:3, Si:4 };

/**
 * Type a ligand from its element list and bond graph.
 * @param {Array<string>} els      element symbol per atom
 * @param {Array} bonds            [{a,b,order}]
 * @returns {Int32Array} xs type per atom (H atoms get -1)
 */
function typeLigand(els, bonds) {
  var n = els.length, i;
  var adj = [], orderSum = new Float64Array(n), nH = new Int32Array(n);
  for (i = 0; i < n; i++) adj.push([]);
  for (i = 0; i < bonds.length; i++) {
    var b = bonds[i], o = b.order || 1;
    if (o === 4 || o === 1.5) o = 1.5;                     // aromatic
    adj[b.a].push(b.b); adj[b.b].push(b.a);
    orderSum[b.a] += o; orderSum[b.b] += o;
    if (els[b.a] === 'H') nH[b.b]++;
    if (els[b.b] === 'H') nH[b.a]++;
  }
  var xs = new Int32Array(n);
  for (i = 0; i < n; i++) {
    var el = els[i];
    if (el === 'H') { xs[i] = -1; continue; }
    if (el === 'C') {
      var polar = false;
      for (var j = 0; j < adj[i].length; j++) {
        if (isHeteroatomEl(els[adj[i][j]])) { polar = true; break; }
      }
      xs[i] = polar ? XS.C_P : XS.C_H;
    } else if (el === 'N' || el === 'O') {
      // Implicit hydrogens: neutral valence minus the bond orders already used.
      var v = VALENCE[el] || 0;
      var implicit = Math.max(0, Math.round(v - orderSum[i]));
      var hasH = (nH[i] + implicit) > 0;
      var acceptor, donor = hasH;
      if (el === 'O') {
        // Every oxygen accepts (OA in AutoDock typing) — carbonyl, hydroxyl,
        // ether, carboxylate alike.
        acceptor = true;
      } else {
        // Nitrogen accepts unless its lone pair is delocalised (amide/aromatic
        // pyrrole-type N) or it is quaternary. Approximate the same way
        // AutoDock's NA assignment does: sp3/sp2 N with a free lone pair.
        var heavyDeg = 0, arom = false, amide = false;
        for (var m = 0; m < adj[i].length; m++) {
          var nb = adj[i][m];
          if (els[nb] !== 'H') heavyDeg++;
          if (els[nb] === 'C') {
            for (var q = 0; q < adj[nb].length; q++) {
              var nn = adj[nb][q];
              if (nn !== i && els[nn] === 'O') {
                for (var bi = 0; bi < bonds.length; bi++) {
                  var bb = bonds[bi];
                  if (((bb.a === nb && bb.b === nn) || (bb.b === nb && bb.a === nn)) &&
                      (bb.order === 2)) { amide = true; }
                }
              }
            }
          }
        }
        for (var bi2 = 0; bi2 < bonds.length; bi2++) {
          var b2 = bonds[bi2];
          if ((b2.a === i || b2.b === i) && (b2.order === 4 || b2.order === 1.5)) arom = true;
        }
        var quaternary = (heavyDeg + nH[i]) >= 4;
        acceptor = !quaternary && !amide && !(arom && hasH);
      }
      if (el === 'N') xs[i] = (acceptor && donor) ? XS.N_DA : (acceptor ? XS.N_A : (donor ? XS.N_D : XS.N_P));
      else            xs[i] = (acceptor && donor) ? XS.O_DA : (acceptor ? XS.O_A : (donor ? XS.O_D : XS.O_P));
    }
    else if (el === 'S')  xs[i] = XS.S_P;
    else if (el === 'P')  xs[i] = XS.P_P;
    else if (el === 'F')  xs[i] = XS.F_H;
    else if (el === 'Cl') xs[i] = XS.Cl_H;
    else if (el === 'Br') xs[i] = XS.Br_H;
    else if (el === 'I')  xs[i] = XS.I_H;
    else if (el === 'Si') xs[i] = XS.Si;
    else if (METALS[el])  xs[i] = XS.Met_D;
    else                  xs[i] = XS.C_P;
    }
  return xs;
}

/* ===========================================================================
 * 3. The five terms  (potentials.h + scoring_function.h:50-54)
 * =========================================================================== */

var CUTOFF = 8.0;                 // every vina_* potential is constructed with 8.0
var CUTOFF_SQR = CUTOFF * CUTOFF;

// slope_step — potentials.h:30-40, for the x_bad > x_good branch both terms use.
function slopeStep(xBad, xGood, x) {
  if (xBad < xGood) { if (x <= xBad) return 0; if (x >= xGood) return 1; }
  else              { if (x >= xBad) return 0; if (x <= xGood) return 1; }
  return (x - xBad) / (xGood - xBad);
}

var DEFAULT_WEIGHTS = {           // vina.h:123-126
  gauss1: -0.035579, gauss2: -0.005156, repulsion: 0.840245,
  hydrophobic: -0.035069, hbond: -0.587439, rot: 0.05846
};

/** Raw (unweighted) five terms for a type pair at distance r. */
function rawTerms(t1, t2, r, out) {
  out[0] = out[1] = out[2] = out[3] = out[4] = 0;
  if (r >= CUTOFF) return out;
  var opt = optimalDistance(t1, t2), s = r - opt;
  var a = s / 0.5;                 out[0] = Math.exp(-a * a);            // vina_gaussian(0, 0.5, 8)
  var b = (s - 3.0) / 2.0;         out[1] = Math.exp(-b * b);            // vina_gaussian(3, 2.0, 8)
  out[2] = (s < 0) ? s * s : 0;                                          // vina_repulsion(0, 8)
  if (xsIsHydrophobic(t1) && xsIsHydrophobic(t2))
    out[3] = slopeStep(1.5, 0.5, s);                                     // vina_hydrophobic(0.5, 1.5, 8)
  if (xsHBondPossible(t1, t2))
    out[4] = slopeStep(0.0, -0.7, s);                                    // vina_non_dir_h_bond(-0.7, 0, 8)
  return out;
}

/** Weighted energy for a type pair at distance r. */
function pairEnergy(t1, t2, r, w, tmp) {
  rawTerms(t1, t2, r, tmp);
  return w.gauss1 * tmp[0] + w.gauss2 * tmp[1] + w.repulsion * tmp[2] +
         w.hydrophobic * tmp[3] + w.hbond * tmp[4];
}

/* ===========================================================================
 * 4. precalculate  (precalculate.h)
 * ---------------------------------------------------------------------------
 * Vina bins the potential on r^2 with factor = 32 and stores (e, dE/dr / r) per
 * bin, then linearly interpolates. Reproducing the binning (rather than
 * evaluating analytically) matters: it is what the binary's gradients actually
 * see, and analytic evaluation would give a subtly different search path.
 * =========================================================================== */

var PRECALC_FACTOR = 32;

function Precalculate(weights, factor) {
  this.factor = factor || PRECALC_FACTOR;
  this.cutoffSqr = CUTOFF_SQR;
  this.n = ((this.factor * CUTOFF_SQR) | 0) + 3;         // precalculate.h:132 -> 2051 at factor 32
  var nt = XS_SIZE, npair = nt * (nt + 1) / 2;
  this.npair = npair;
  this.e = new Float64Array(npair * this.n);
  this.dor = new Float64Array(npair * this.n);
  var rs = new Float64Array(this.n), i;
  for (i = 0; i < this.n; i++) rs[i] = Math.sqrt(i / this.factor);       // calculate_rs()
  var tmp = new Float64Array(5);
  for (var t1 = 0; t1 < nt; t1++) {
    for (var t2 = t1; t2 < nt; t2++) {
      var base = pairIndex(t1, t2) * this.n;
      for (i = 0; i < this.n; i++) this.e[base + i] = pairEnergy(t1, t2, rs[i], weights, tmp);
      // init_from_smooth_fst — precalculate.h:59-72: central difference / (delta*r)
      for (i = 0; i < this.n; i++) {
        if (i === 0 || i === this.n - 1) { this.dor[base + i] = 0; continue; }
        var delta = rs[i + 1] - rs[i - 1], r = rs[i];
        this.dor[base + i] = (this.e[base + i + 1] - this.e[base + i - 1]) / (delta * r);
      }
    }
  }
}
// triangular_matrix_index_permissive — triangular_matrix_index.h
function pairIndex(t1, t2) {
  return (t1 <= t2) ? (t1 + t2 * (t2 + 1) / 2) : (t2 + t1 * (t1 + 1) / 2);
}
/** eval_fast — energy only. */
Precalculate.prototype.evalFast = function (t1, t2, r2) {
  var base = pairIndex(t1, t2) * this.n;
  var i = (this.factor * r2) | 0;
  if (i >= this.n) i = this.n - 1;
  return this.e[base + i];
};
/** eval_deriv — returns [e, dE/dr / r] with linear interpolation. */
Precalculate.prototype.evalDeriv = function (t1, t2, r2, out) {
  var base = pairIndex(t1, t2) * this.n;
  var rf = this.factor * r2;
  var i1 = rf | 0, i2 = i1 + 1;
  if (i2 >= this.n) { i1 = this.n - 2; i2 = this.n - 1; rf = i1; }
  var rem = rf - i1;
  out[0] = this.e[base + i1]   + rem * (this.e[base + i2]   - this.e[base + i1]);
  out[1] = this.dor[base + i1] + rem * (this.dor[base + i2] - this.dor[base + i1]);
  return out;
};

/* ===========================================================================
 * 5. curl  (curl.h:29-40)
 * =========================================================================== */
function curlE(e, v) {
  if (e > 0 && isFinite(v)) { var t = (v < 1e-8) ? 0 : v / (v + e); return e * t; }
  return e;
}
/** Returns [e', scale] where scale multiplies the derivative (curl.h:27-33). */
function curlEDeriv(e, v, out) {
  if (e > 0 && isFinite(v)) {
    var t = (v < 1e-8) ? 0 : v / (v + e);
    out[0] = e * t; out[1] = t * t;
  } else { out[0] = e; out[1] = 1; }
  return out;
}

/* ===========================================================================
 * 6. Neighbour grid  (szv_grid.cpp equivalent)
 * =========================================================================== */
function NeighborGrid(rec, pad) {
  var n = rec.n, i;
  var mnx = Infinity, mny = Infinity, mnz = Infinity,
      mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (i = 0; i < n; i++) {
    if (rec.x[i] < mnx) mnx = rec.x[i]; if (rec.x[i] > mxx) mxx = rec.x[i];
    if (rec.y[i] < mny) mny = rec.y[i]; if (rec.y[i] > mxy) mxy = rec.y[i];
    if (rec.z[i] < mnz) mnz = rec.z[i]; if (rec.z[i] > mxz) mxz = rec.z[i];
  }
  this.cell = pad || CUTOFF;
  this.o = [mnx - 1e-3, mny - 1e-3, mnz - 1e-3];
  this.dim = [
    Math.max(1, Math.ceil((mxx - mnx + 2e-3) / this.cell)),
    Math.max(1, Math.ceil((mxy - mny + 2e-3) / this.cell)),
    Math.max(1, Math.ceil((mxz - mnz + 2e-3) / this.cell))
  ];
  var nc = this.dim[0] * this.dim[1] * this.dim[2];
  var counts = new Int32Array(nc + 1);
  var ci = new Int32Array(n);
  for (i = 0; i < n; i++) {
    var c = this._cellOf(rec.x[i], rec.y[i], rec.z[i]);
    ci[i] = c; counts[c + 1]++;
  }
  for (i = 0; i < nc; i++) counts[i + 1] += counts[i];
  this.start = counts;
  this.items = new Int32Array(n);
  var fill = new Int32Array(nc);
  for (i = 0; i < n; i++) { var c2 = ci[i]; this.items[counts[c2] + fill[c2]++] = i; }
  this.nc = nc;
}
NeighborGrid.prototype._cellOf = function (x, y, z) {
  var i = Math.min(this.dim[0]-1, Math.max(0, ((x - this.o[0]) / this.cell) | 0));
  var j = Math.min(this.dim[1]-1, Math.max(0, ((y - this.o[1]) / this.cell) | 0));
  var k = Math.min(this.dim[2]-1, Math.max(0, ((z - this.o[2]) / this.cell) | 0));
  return i + this.dim[0] * (j + this.dim[1] * k);
};
/** Append receptor-atom indices within `cell` of (x,y,z) into `out`; returns count. */
NeighborGrid.prototype.query = function (x, y, z, out) {
  var i0 = Math.min(this.dim[0]-1, Math.max(0, ((x - this.o[0]) / this.cell) | 0));
  var j0 = Math.min(this.dim[1]-1, Math.max(0, ((y - this.o[1]) / this.cell) | 0));
  var k0 = Math.min(this.dim[2]-1, Math.max(0, ((z - this.o[2]) / this.cell) | 0));
  var cnt = 0;
  for (var dk = -1; dk <= 1; dk++) {
    var k = k0 + dk; if (k < 0 || k >= this.dim[2]) continue;
    for (var dj = -1; dj <= 1; dj++) {
      var j = j0 + dj; if (j < 0 || j >= this.dim[1]) continue;
      for (var di = -1; di <= 1; di++) {
        var i = i0 + di; if (i < 0 || i >= this.dim[0]) continue;
        var c = i + this.dim[0] * (j + this.dim[1] * k);
        for (var p = this.start[c]; p < this.start[c + 1]; p++) out[cnt++] = this.items[p];
      }
    }
  }
  return cnt;
};

/* ===========================================================================
 * 6b. Cache — precomputed affinity maps  (cache.cpp + grid.cpp)
 * ---------------------------------------------------------------------------
 * This is the single most important performance decision in Vina, and it is
 * also what the binary actually optimises against. During global_search Vina
 * does NOT test ligand-receptor pairs; it reads one trilinear interpolation per
 * ligand atom from a precomputed 3D map per XS type (0.375 A spacing). Doing
 * pairwise distance tests instead costs ~200x more per energy evaluation and
 * makes a full-depth run take hours rather than a minute.
 *
 * `cache::populate` shares one neighbour query across every needed type, so the
 * build cost is one pass over voxels, not one pass per type.
 * =========================================================================== */
function Cache(rec, prec, box, granularity, neededTypes, slope, onProgress) {
  var i, j, k, t;
  this.slope = (slope !== undefined) ? slope : 1e6;
  this.gran = granularity || 0.375;
  var bc = box.center, bs = box.size;
  this.init = [bc[0]-bs[0]/2, bc[1]-bs[1]/2, bc[2]-bs[2]/2];
  this.range = [bs[0], bs[1], bs[2]];
  this.dim = [
    Math.max(2, Math.ceil(bs[0]/this.gran) + 1),
    Math.max(2, Math.ceil(bs[1]/this.gran) + 1),
    Math.max(2, Math.ceil(bs[2]/this.gran) + 1)
  ];
  this.dimFlMinus1 = [this.dim[0]-1, this.dim[1]-1, this.dim[2]-1];
  this.factor = [this.dimFlMinus1[0]/this.range[0],
                 this.dimFlMinus1[1]/this.range[1],
                 this.dimFlMinus1[2]/this.range[2]];
  this.factorInv = [1/this.factor[0], 1/this.factor[1], 1/this.factor[2]];

  var needed = [];
  var seen = {};
  for (i = 0; i < neededTypes.length; i++) {
    t = neededTypes[i];
    if (t >= 0 && t < XS_SIZE && !seen[t]) { seen[t] = 1; needed.push(t); }
  }
  this.needed = needed;
  var nv = this.dim[0]*this.dim[1]*this.dim[2];
  this.nv = nv;
  this.grids = {};
  for (i = 0; i < needed.length; i++) this.grids[needed[i]] = new Float64Array(nv);

  // One neighbour query per voxel, accumulating every needed type at once.
  var nbr = new NeighborGrid(rec, CUTOFF);
  var buf = new Int32Array(rec.n);
  var acc = new Float64Array(needed.length);
  var total = this.dim[0], done = 0;
  for (i = 0; i < this.dim[0]; i++) {
    var px = this.init[0] + i * this.factorInv[0];
    for (j = 0; j < this.dim[1]; j++) {
      var py = this.init[1] + j * this.factorInv[1];
      for (k = 0; k < this.dim[2]; k++) {
        var pz = this.init[2] + k * this.factorInv[2];
        acc.fill(0);
        var cnt = nbr.query(px, py, pz, buf);
        for (var q = 0; q < cnt; q++) {
          var a = buf[q];
          var dx = rec.x[a]-px, dy = rec.y[a]-py, dz = rec.z[a]-pz;
          var r2 = dx*dx + dy*dy + dz*dz;
          if (r2 > CUTOFF_SQR) continue;
          var t1 = rec.xs[a];
          for (var n = 0; n < needed.length; n++) acc[n] += prec.evalFast(t1, needed[n], r2);
        }
        var idx = i + this.dim[0]*(j + this.dim[1]*k);
        for (var n2 = 0; n2 < needed.length; n2++) this.grids[needed[n2]][idx] = acc[n2];
      }
    }
    done++;
    if (onProgress && (done & 7) === 0) onProgress(done / total);
  }
}

/**
 * grid::evaluate_aux — trilinear interpolation, curl, and the out-of-box slope
 * penalty. `deriv` (optional Float64Array(3)) receives dE/dx,dE/dy,dE/dz.
 * Note grid.cpp:149 zeroes the interpolated gradient in any CLAMPED dimension
 * and substitutes `slope * region`, which a naive pairwise implementation gets
 * wrong.
 */
var _cd2 = new Float64Array(2);
Cache.prototype.evaluate = function (t, lx, ly, lz, v, deriv) {
  var g = this.grids[t];
  if (!g) return 0;
  var s = [ (lx - this.init[0]) * this.factor[0],
            (ly - this.init[1]) * this.factor[1],
            (lz - this.init[2]) * this.factor[2] ];
  var miss = 0, region = [0,0,0], a = [0,0,0], i;
  for (i = 0; i < 3; i++) {
    if (s[i] < 0) { miss += -s[i] * this.factorInv[i]; region[i] = -1; a[i] = 0; s[i] = 0; }
    else if (s[i] >= this.dimFlMinus1[i]) {
      miss += (s[i] - this.dimFlMinus1[i]) * this.factorInv[i];
      region[i] = 1; a[i] = this.dim[i] - 2; s[i] = 1;
    } else { region[i] = 0; a[i] = s[i] | 0; s[i] -= a[i]; }
  }
  var penalty = this.slope * miss;
  var d0 = this.dim[0], d1 = this.dim[1];
  var x0 = a[0], y0 = a[1], z0 = a[2], x1 = x0+1, y1 = y0+1, z1 = z0+1;
  var i000 = x0 + d0*(y0 + d1*z0), i100 = x1 + d0*(y0 + d1*z0);
  var i010 = x0 + d0*(y1 + d1*z0), i110 = x1 + d0*(y1 + d1*z0);
  var i001 = x0 + d0*(y0 + d1*z1), i101 = x1 + d0*(y0 + d1*z1);
  var i011 = x0 + d0*(y1 + d1*z1), i111 = x1 + d0*(y1 + d1*z1);
  var f000=g[i000], f100=g[i100], f010=g[i010], f110=g[i110],
      f001=g[i001], f101=g[i101], f011=g[i011], f111=g[i111];
  var x=s[0], y=s[1], z=s[2], mx=1-x, my=1-y, mz=1-z;
  var f = f000*mx*my*mz + f100*x*my*mz + f010*mx*y*mz + f110*x*y*mz
        + f001*mx*my*z  + f101*x*my*z  + f011*mx*y*z  + f111*x*y*z;
  if (deriv) {
    var xg = -f000*my*mz + f100*my*mz - f010*y*mz + f110*y*mz
             -f001*my*z  + f101*my*z  - f011*y*z  + f111*y*z;
    var yg = -f000*mx*mz - f100*x*mz + f010*mx*mz + f110*x*mz
             -f001*mx*z  - f101*x*z  + f011*mx*z  + f111*x*z;
    var zg = -f000*mx*my - f100*x*my - f010*mx*y - f110*x*y
             +f001*mx*my + f101*x*my + f011*mx*y + f111*x*y;
    curlEDeriv(f, v, _cd2);
    var scale = _cd2[1];
    xg *= scale; yg *= scale; zg *= scale;
    deriv[0] = (region[0] === 0 ? this.factor[0]*xg : 0) + this.slope*region[0];
    deriv[1] = (region[1] === 0 ? this.factor[1]*yg : 0) + this.slope*region[1];
    deriv[2] = (region[2] === 0 ? this.factor[2]*zg : 0) + this.slope*region[2];
    return _cd2[0] + penalty;
  }
  return curlE(f, v) + penalty;
};

/* ===========================================================================
 * 7. mt19937 — Vina seeds boost::mt19937 (random.cpp). Math.random() cannot be
 *    seeded, so --seed reproducibility requires our own generator.
 * =========================================================================== */
function MT19937(seed) {
  this.mt = new Uint32Array(624); this.idx = 625;
  this.mt[0] = seed >>> 0;
  for (var i = 1; i < 624; i++) {
    var s = (this.mt[i-1] ^ (this.mt[i-1] >>> 30)) >>> 0;
    this.mt[i] = (((((s >>> 16) * 1812433253) << 16) >>> 0) + (s & 0xffff) * 1812433253 + i) >>> 0;
  }
  this.idx = 624;
  this._haveGauss = false; this._gauss = 0;
}
MT19937.prototype._gen = function () {
  for (var i = 0; i < 624; i++) {
    var y = ((this.mt[i] & 0x80000000) | (this.mt[(i+1) % 624] & 0x7fffffff)) >>> 0;
    this.mt[i] = (this.mt[(i + 397) % 624] ^ (y >>> 1)) >>> 0;
    if (y & 1) this.mt[i] = (this.mt[i] ^ 0x9908b0df) >>> 0;
  }
  this.idx = 0;
};
MT19937.prototype.u32 = function () {
  if (this.idx >= 624) this._gen();
  var y = this.mt[this.idx++];
  y = (y ^ (y >>> 11)) >>> 0;
  y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
  y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
  return (y ^ (y >>> 18)) >>> 0;
};
MT19937.prototype.f = function (a, b) { return a + (b - a) * (this.u32() / 4294967296); };
MT19937.prototype.i = function (a, b) { return a + (this.u32() % (b - a + 1)); };
/** random_inside_sphere() — random.cpp: rejection sample the unit ball. */
MT19937.prototype.insideSphere = function (out) {
  for (;;) {
    out[0] = this.f(-1,1); out[1] = this.f(-1,1); out[2] = this.f(-1,1);
    var d = out[0]*out[0] + out[1]*out[1] + out[2]*out[2];
    if (d < 1 && d > 1e-12) return out;
  }
};

/* ===========================================================================
 * 8. Quaternions  (quaternion.cpp)
 * =========================================================================== */
function qNormalize(q) {
  var n = Math.sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  if (n < 1e-12) { q[0]=1;q[1]=0;q[2]=0;q[3]=0; return q; }
  q[0]/=n; q[1]/=n; q[2]/=n; q[3]/=n; return q;
}
function qMul(a, b, o) {
  o[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
  o[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
  o[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
  o[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
  return o;
}
/** angle_to_quaternion(rotation) — quaternion.cpp. `rot` is axis*angle. */
function rotToQuat(rot, o) {
  var ang = Math.sqrt(rot[0]*rot[0]+rot[1]*rot[1]+rot[2]*rot[2]);
  if (ang < 1e-12) { o[0]=1;o[1]=0;o[2]=0;o[3]=0; return o; }
  var h = ang / 2, s = Math.sin(h) / ang;
  o[0] = Math.cos(h); o[1] = rot[0]*s; o[2] = rot[1]*s; o[3] = rot[2]*s;
  return o;
}
/** quaternion_increment(q, rotation) — quaternion.cpp. */
var _qtmp = new Float64Array(4), _qtmp2 = new Float64Array(4);
function qIncrement(q, rot) {
  rotToQuat(rot, _qtmp);
  qMul(_qtmp, q, _qtmp2);
  q[0]=_qtmp2[0]; q[1]=_qtmp2[1]; q[2]=_qtmp2[2]; q[3]=_qtmp2[3];
  return qNormalize(q);
}
function qToMat(q, m) {
  var a=q[0],b=q[1],c=q[2],d=q[3];
  m[0]=a*a+b*b-c*c-d*d; m[1]=2*(b*c-a*d);     m[2]=2*(b*d+a*c);
  m[3]=2*(b*c+a*d);     m[4]=a*a-b*b+c*c-d*d; m[5]=2*(c*d-a*b);
  m[6]=2*(b*d-a*c);     m[7]=2*(c*d+a*b);     m[8]=a*a-b*b-c*c+d*d;
  return m;
}

/* ===========================================================================
 * 9. Model — ligand torsion tree, intramolecular pairs, coordinates, gradients
 * =========================================================================== */

/**
 * @param {Object} lig  { els:[string], ref:[[x,y,z]], bonds:[{a,b,order}],
 *                        tors:[{from,to,moves}], root:[ids] }
 * @param {Object} rec  output of typeReceptor()
 * @param {Object} opt  { weights, box:{center:[3], size:[3]}, slope }
 */
function Model(lig, rec, opt) {
  var i, j;
  this.weights = opt.weights || DEFAULT_WEIGHTS;
  this.rec = rec;
  this.slope = (opt.slope !== undefined) ? opt.slope : 1e6;   // vina.cpp:339
  this.box = opt.box;
  this.prec = opt.precalculate || new Precalculate(this.weights);

  this.els = lig.els;
  this.nAll = lig.els.length;
  this.xsAll = typeLigand(lig.els, lig.bonds);
  this.ref = lig.ref;
  this.tors = lig.tors || [];
  this.root = lig.root || [];
  this.bonds = lig.bonds;

  // Heavy-atom subset — Vina never scores hydrogens intermolecularly
  // (non_cache.cpp: `if (t1 >= n) continue;`, and H has no XS type).
  this.heavy = [];
  for (i = 0; i < this.nAll; i++) if (this.els[i] !== 'H') this.heavy.push(i);
  this.nHeavy = this.heavy.length;

  // Bond-distance matrix up to 3 (model.cpp:543 `bonded_to(i, 3)`).
  var adj = []; for (i = 0; i < this.nAll; i++) adj.push([]);
  for (i = 0; i < lig.bonds.length; i++) { adj[lig.bonds[i].a].push(lig.bonds[i].b); adj[lig.bonds[i].b].push(lig.bonds[i].a); }
  this.adj = adj;
  var within3 = [];
  for (i = 0; i < this.nAll; i++) {
    var seen = {}; seen[i] = 0; var q = [i];
    while (q.length) {
      var u = q.shift(); if (seen[u] >= 3) continue;
      for (j = 0; j < adj[u].length; j++) { var v = adj[u][j]; if (seen[v] === undefined) { seen[v] = seen[u]+1; q.push(v); } }
    }
    within3.push(seen);
  }

  // Which torsion subtree each atom belongs to — used for the mobility test.
  // Two atoms have DISTANCE_VARIABLE mobility iff some torsion moves exactly
  // one of them (model.cpp:485-490).
  var inMoves = [];
  for (i = 0; i < this.tors.length; i++) {
    var s = {}; for (j = 0; j < this.tors[i].moves.length; j++) s[this.tors[i].moves[j]] = 1;
    inMoves.push(s);
  }

  // Intramolecular pair list — model.cpp:535-560.
  //   ligand_i - ligand_i : mobility DISTANCE_VARIABLE, and NOT within 3 bonds.
  this.intraPairs = [];
  for (i = 0; i < this.nAll; i++) {
    if (this.els[i] === 'H') continue;
    for (j = i + 1; j < this.nAll; j++) {
      if (this.els[j] === 'H') continue;
      if (within3[i][j] !== undefined) continue;             // excludes 1-2, 1-3, 1-4
      var variable = false;
      for (var t = 0; t < inMoves.length; t++) {
        var ai = !!inMoves[t][i], aj = !!inMoves[t][j];
        if (ai !== aj) { variable = true; break; }
      }
      if (!variable) continue;                               // same rigid fragment
      this.intraPairs.push(i, j);
    }
  }

  // Nrot for the conf-independent term — conf_independent.cpp:86-109.
  this.numTors = this._numTors();

  this.nbr = new NeighborGrid(rec, CUTOFF);
  this._nbrBuf = new Int32Array(rec.n);
  // XS types the ligand actually presents -- only these maps need building.
  var nt_ = {}; this.neededTypes = [];
  for (i = 0; i < this.nHeavy; i++) {
    var tt_ = this.xsAll[this.heavy[i]];
    if (tt_ >= 0 && !nt_[tt_]) { nt_[tt_] = 1; this.neededTypes.push(tt_); }
  }
  this.cache = opt.cache || null;
  this._d3 = new Float64Array(3);

  // Scratch
  this.coords  = new Float64Array(this.nAll * 3);
  this.forces  = new Float64Array(this.nAll * 3);
  this._m      = new Float64Array(9);
  this._tmp5   = new Float64Array(5);
  this._pd     = new Float64Array(2);
  this._cd     = new Float64Array(2);
  this.ndof    = 3 + 3 + this.tors.length;                   // conf.h: 6 + torsions

  // Gyration radius, for the orientation mutation amplitude (model.cpp).
  this.gyr = this._gyration();
}

/**
 * num_tors = sum over heavy ligand atoms of 0.5 * atom_rotors(i)
 * where atom_rotors counts bonds that are rotatable, to a heavy atom, whose
 * far end has more than one heavy neighbour ("not counting CH_3, etc").
 * conf_independent.cpp:56-69, 96.  This is why Nrot is often a half-integer.
 */
Model.prototype._numTors = function () {
  var i, j, rotatable = {};
  for (i = 0; i < this.bonds.length; i++) {
    if (this.bonds[i].type === 'rot' || this.bonds[i].rotatable) {
      rotatable[this.bonds[i].a + '_' + this.bonds[i].b] = 1;
      rotatable[this.bonds[i].b + '_' + this.bonds[i].a] = 1;
    }
  }
  for (i = 0; i < this.tors.length; i++) {                    // torsion tree is authoritative
    rotatable[this.tors[i].from + '_' + this.tors[i].to] = 1;
    rotatable[this.tors[i].to + '_' + this.tors[i].from] = 1;
  }
  var heavyDeg = new Int32Array(this.nAll);
  for (i = 0; i < this.nAll; i++)
    for (j = 0; j < this.adj[i].length; j++)
      if (this.els[this.adj[i][j]] !== 'H') heavyDeg[i]++;

  var total = 0;
  for (i = 0; i < this.nAll; i++) {
    if (this.els[i] === 'H') continue;
    var ar = 0;
    for (j = 0; j < this.adj[i].length; j++) {
      var nb = this.adj[i][j];
      if (this.els[nb] === 'H') continue;
      if (!rotatable[i + '_' + nb]) continue;
      if (heavyDeg[nb] > 1) ar++;
    }
    total += 0.5 * ar;
  }
  return total;
};

Model.prototype._gyration = function () {
  var i, c = [0,0,0], n = 0;
  for (i = 0; i < this.nAll; i++) { if (this.els[i]==='H') continue; c[0]+=this.ref[i][0]; c[1]+=this.ref[i][1]; c[2]+=this.ref[i][2]; n++; }
  if (!n) return 1;
  c[0]/=n; c[1]/=n; c[2]/=n;
  var s = 0;
  for (i = 0; i < this.nAll; i++) { if (this.els[i]==='H') continue;
    var dx=this.ref[i][0]-c[0], dy=this.ref[i][1]-c[1], dz=this.ref[i][2]-c[2];
    s += dx*dx+dy*dy+dz*dz; }
  return Math.sqrt(s / n) || 1;
};

/**
 * set_conf: apply torsions to the reference frame, then the rigid-body
 * rotation about the ROOT centroid, then the translation. Matches
 * PoseEngine.computePose exactly, so poses stay interchangeable with the UI.
 * Also records, per torsion, the world-frame axis and pivot needed for the
 * gradient (tree.h `derivative` / `sum_force_and_torque`).
 */
Model.prototype.setConf = function (conf) {
  var i, t, nAll = this.nAll;
  var c = this._work || (this._work = new Float64Array(nAll * 3));
  for (i = 0; i < nAll; i++) { c[3*i]=this.ref[i][0]; c[3*i+1]=this.ref[i][1]; c[3*i+2]=this.ref[i][2]; }

  var axes = this._axes || (this._axes = new Float64Array(Math.max(1,this.tors.length) * 6));
  for (t = 0; t < this.tors.length; t++) {
    var tt = this.tors[t], ang = conf.tors[t] || 0;
    var fx=c[3*tt.from], fy=c[3*tt.from+1], fz=c[3*tt.from+2];
    var kx=c[3*tt.to]-fx, ky=c[3*tt.to+1]-fy, kz=c[3*tt.to+2]-fz;
    var kn=Math.sqrt(kx*kx+ky*ky+kz*kz) || 1; kx/=kn; ky/=kn; kz/=kn;
    if (ang) {
      var ca=Math.cos(ang), sa=Math.sin(ang), om=1-ca;
      for (i = 0; i < tt.moves.length; i++) {
        var id = tt.moves[i]; if (id === tt.from) continue;
        var px=c[3*id]-fx, py=c[3*id+1]-fy, pz=c[3*id+2]-fz;
        var dot=px*kx+py*ky+pz*kz;
        c[3*id]  =fx + px*ca + (ky*pz-kz*py)*sa + kx*dot*om;
        c[3*id+1]=fy + py*ca + (kz*px-kx*pz)*sa + ky*dot*om;
        c[3*id+2]=fz + pz*ca + (kx*py-ky*px)*sa + kz*dot*om;
      }
    }
  }
  // ROOT centroid, rigid rotation, then place that centroid at conf.pos.
  // conf.pos is the ABSOLUTE world position of the rotation centre, matching
  // Vina's rigid_conf::position (which conf::randomize samples inside the box).
  var cx=0, cy=0, cz=0, nr=Math.max(1, this.root.length);
  for (i = 0; i < this.root.length; i++) { var r=this.root[i]; cx+=c[3*r]; cy+=c[3*r+1]; cz+=c[3*r+2]; }
  cx/=nr; cy/=nr; cz/=nr;
  var m = qToMat(conf.q, this._m);
  var ox = conf.pos[0], oy = conf.pos[1], oz = conf.pos[2];
  this.refCentroid = [cx, cy, cz];
  for (i = 0; i < nAll; i++) {
    var dx=c[3*i]-cx, dy=c[3*i+1]-cy, dz=c[3*i+2]-cz;
    this.coords[3*i]   = m[0]*dx+m[1]*dy+m[2]*dz + ox;
    this.coords[3*i+1] = m[3]*dx+m[4]*dy+m[5]*dz + oy;
    this.coords[3*i+2] = m[6]*dx+m[7]*dy+m[8]*dz + oz;
  }
  // World-frame torsion axes and pivots, read off the final coordinates.
  for (t = 0; t < this.tors.length; t++) {
    var tw = this.tors[t];
    var ax=this.coords[3*tw.to]  -this.coords[3*tw.from];
    var ay=this.coords[3*tw.to+1]-this.coords[3*tw.from+1];
    var az=this.coords[3*tw.to+2]-this.coords[3*tw.from+2];
    var an=Math.sqrt(ax*ax+ay*ay+az*az)||1;
    axes[6*t]  =ax/an; axes[6*t+1]=ay/an; axes[6*t+2]=az/an;
    axes[6*t+3]=this.coords[3*tw.from]; axes[6*t+4]=this.coords[3*tw.from+1]; axes[6*t+5]=this.coords[3*tw.from+2];
  }
  this.rotCenter = [ox, oy, oz];
  return this.coords;
};

/**
 * Intermolecular energy, optionally accumulating Cartesian forces.
 * Mirrors non_cache::eval / eval_deriv: per ligand atom, clamp to the box and
 * charge an out-of-bounds penalty, sum pair terms inside the 8 A cutoff, then
 * curl the atom's total against v before adding it in.
 */
Model.prototype.evalInter = function (v, wantForces) {
  var e = 0, rec = this.rec, prec = this.prec, buf = this._nbrBuf;
  var bc = this.box.center, bs = this.box.size;
  var lo0=bc[0]-bs[0]/2, hi0=bc[0]+bs[0]/2,
      lo1=bc[1]-bs[1]/2, hi1=bc[1]+bs[1]/2,
      lo2=bc[2]-bs[2]/2, hi2=bc[2]+bs[2]/2;
  if (wantForces) this.forces.fill(0);
  for (var h = 0; h < this.nHeavy; h++) {
    var i = this.heavy[h], t1 = this.xsAll[i];
    if (t1 < 0) continue;
    var x=this.coords[3*i], y=this.coords[3*i+1], z=this.coords[3*i+2];
    // out-of-bounds clamp + penalty (non_cache.cpp:100-106, slope 1e6)
    var ax=x, ay=y, az=z, oob=0;
    if (x < lo0) { ax=lo0; oob += lo0-x; } else if (x > hi0) { ax=hi0; oob += x-hi0; }
    if (y < lo1) { ay=lo1; oob += lo1-y; } else if (y > hi1) { ay=hi1; oob += y-hi1; }
    if (z < lo2) { az=lo2; oob += lo2-z; } else if (z > hi2) { az=hi2; oob += z-hi2; }
    oob *= this.slope;

    var thisE = 0, gx=0, gy=0, gz=0;
    var cnt = this.nbr.query(ax, ay, az, buf);
    for (var p = 0; p < cnt; p++) {
      var j = buf[p];
      var dx=ax-rec.x[j], dy=ay-rec.y[j], dz=az-rec.z[j];
      var r2 = dx*dx+dy*dy+dz*dz;
      if (r2 >= CUTOFF_SQR) continue;
      var t2 = rec.xs[j];
      if (wantForces) {
        prec.evalDeriv(t1, t2, r2, this._pd);
        thisE += this._pd[0];
        gx += this._pd[1]*dx; gy += this._pd[1]*dy; gz += this._pd[1]*dz;   // dE/dr/r * dvec
      } else {
        thisE += prec.evalFast(t1, t2, r2);
      }
    }
    if (wantForces) {
      curlEDeriv(thisE, v, this._cd);
      var s = this._cd[1];
      // Gradient of the clamped position wrt the true position is 1 inside the
      // box and 0 outside; the penalty supplies the restoring term there.
      // dor from precalculate is already (dE/dr)/r, so dE/dx = dor * dx.
      // No chain-rule factor of 2 (model.cpp: force = dor * r_vec).
      this.forces[3*i]   += gx*s + (x<lo0 ? -this.slope : (x>hi0 ? this.slope : 0));
      this.forces[3*i+1] += gy*s + (y<lo1 ? -this.slope : (y>hi1 ? this.slope : 0));
      this.forces[3*i+2] += gz*s + (z<lo2 ? -this.slope : (z>hi2 ? this.slope : 0));
      e += this._cd[0] + oob;
    } else {
      e += curlE(thisE, v) + oob;
    }
  }
  return e;
};

/**
 * Intermolecular energy from the precomputed maps — cache::eval / cache::eval_deriv.
 * This is the path Vina's global_search uses; the exact pairwise path below is
 * what Vina::score uses for the final reported affinity.
 */
Model.prototype.evalInterCache = function (v, wantForces) {
  var e = 0, C = this.cache, d = this._d3;
  if (wantForces) this.forces.fill(0);
  for (var h = 0; h < this.nHeavy; h++) {
    var i = this.heavy[h], t1 = this.xsAll[i];
    if (t1 < 0) continue;
    if (wantForces) {
      e += C.evaluate(t1, this.coords[3*i], this.coords[3*i+1], this.coords[3*i+2], v, d);
      this.forces[3*i] += d[0]; this.forces[3*i+1] += d[1]; this.forces[3*i+2] += d[2];
    } else {
      e += C.evaluate(t1, this.coords[3*i], this.coords[3*i+1], this.coords[3*i+2], v, null);
    }
  }
  return e;
};

/** Intramolecular ligand energy — model::evali / eval_interacting_pairs. */
Model.prototype.evalIntra = function (v, wantForces) {
  var e = 0, prec = this.prec;
  for (var p = 0; p < this.intraPairs.length; p += 2) {
    var i = this.intraPairs[p], j = this.intraPairs[p+1];
    var dx=this.coords[3*i]-this.coords[3*j],
        dy=this.coords[3*i+1]-this.coords[3*j+1],
        dz=this.coords[3*i+2]-this.coords[3*j+2];
    var r2 = dx*dx+dy*dy+dz*dz;
    if (r2 >= CUTOFF_SQR) continue;
    var t1 = this.xsAll[i], t2 = this.xsAll[j];
    if (t1 < 0 || t2 < 0) continue;
    if (wantForces) {
      prec.evalDeriv(t1, t2, r2, this._pd);
      var ee = this._pd[0];
      curlEDeriv(ee, v, this._cd);
      var s = this._cd[1] * this._pd[1];
      this.forces[3*i]   += s*dx; this.forces[3*i+1] += s*dy; this.forces[3*i+2] += s*dz;
      this.forces[3*j]   -= s*dx; this.forces[3*j+1] -= s*dy; this.forces[3*j+2] -= s*dz;
      e += this._cd[0];
    } else {
      e += curlE(prec.evalFast(t1, t2, r2), v);
    }
  }
  return e;
};

/**
 * Energy only, via eval_fast — the piecewise-constant table. This is what
 * `non_cache::eval` and therefore `Vina::score()` use, so it is the right path
 * for a reported affinity. It is NOT differentiable: consecutive r^2 bins are
 * flat, so a finite difference of this function is meaningless. Use evalSmooth
 * when you need the function BFGS actually descends.
 */
Model.prototype.eval = function (conf, v) {
  this.setConf(conf);
  return this.evalInter(v, false) + this.evalIntra(v, false);
};

/**
 * Energy via the interpolated (smooth) table — the same values eval_deriv
 * returns, without computing forces. This is the objective the BFGS gradient
 * is the true derivative of, and what the line search compares against.
 */
Model.prototype.evalSmooth = function (conf, v) {
  this.setConf(conf);
  if (this.cache) {
    var ec = this.evalInterCache(v, false);
    for (var pc = 0; pc < this.intraPairs.length; pc += 2) {
      var ic = this.intraPairs[pc], jc = this.intraPairs[pc+1];
      var cx2=this.coords[3*ic]-this.coords[3*jc],
          cy2=this.coords[3*ic+1]-this.coords[3*jc+1],
          cz2=this.coords[3*ic+2]-this.coords[3*jc+2];
      var cr2 = cx2*cx2+cy2*cy2+cz2*cz2;
      if (cr2 >= CUTOFF_SQR) continue;
      var a1c = this.xsAll[ic], a2c = this.xsAll[jc];
      if (a1c < 0 || a2c < 0) continue;
      this.prec.evalDeriv(a1c, a2c, cr2, this._pd);
      ec += curlE(this._pd[0], v);
    }
    return ec;
  }
  var e = 0, rec = this.rec, prec = this.prec, buf = this._nbrBuf, h, i, p, j;
  var bc = this.box.center, bs = this.box.size;
  var lo0=bc[0]-bs[0]/2, hi0=bc[0]+bs[0]/2,
      lo1=bc[1]-bs[1]/2, hi1=bc[1]+bs[1]/2,
      lo2=bc[2]-bs[2]/2, hi2=bc[2]+bs[2]/2;
  for (h = 0; h < this.nHeavy; h++) {
    i = this.heavy[h];
    var t1 = this.xsAll[i]; if (t1 < 0) continue;
    var x=this.coords[3*i], y=this.coords[3*i+1], z=this.coords[3*i+2];
    var ax=x, ay=y, az=z, oob=0;
    if (x < lo0) { ax=lo0; oob += lo0-x; } else if (x > hi0) { ax=hi0; oob += x-hi0; }
    if (y < lo1) { ay=lo1; oob += lo1-y; } else if (y > hi1) { ay=hi1; oob += y-hi1; }
    if (z < lo2) { az=lo2; oob += lo2-z; } else if (z > hi2) { az=hi2; oob += z-hi2; }
    oob *= this.slope;
    var thisE = 0;
    var cnt = this.nbr.query(ax, ay, az, buf);
    for (p = 0; p < cnt; p++) {
      j = buf[p];
      var dx=ax-rec.x[j], dy=ay-rec.y[j], dz=az-rec.z[j];
      var r2 = dx*dx+dy*dy+dz*dz;
      if (r2 >= CUTOFF_SQR) continue;
      prec.evalDeriv(t1, rec.xs[j], r2, this._pd);
      thisE += this._pd[0];
    }
    e += curlE(thisE, v) + oob;
  }
  for (p = 0; p < this.intraPairs.length; p += 2) {
    i = this.intraPairs[p]; j = this.intraPairs[p+1];
    var ex=this.coords[3*i]-this.coords[3*j],
        ey=this.coords[3*i+1]-this.coords[3*j+1],
        ez=this.coords[3*i+2]-this.coords[3*j+2];
    var er2 = ex*ex+ey*ey+ez*ez;
    if (er2 >= CUTOFF_SQR) continue;
    var a1 = this.xsAll[i], a2 = this.xsAll[j];
    if (a1 < 0 || a2 < 0) continue;
    prec.evalDeriv(a1, a2, er2, this._pd);
    e += curlE(this._pd[0], v);
  }
  return e;
};

/**
 * Energy + gradient. Fills `g` = {pos:[3], ori:[3], tors:[n]} with dE/dparam.
 * Cartesian forces are projected onto the conf parameters exactly as
 * tree.h/model.cpp do: translation is the force sum, orientation is the torque
 * about the rotation centre, and each torsion takes the component of its
 * subtree's torque along its own axis.
 */
Model.prototype.evalDeriv = function (conf, v, g) {
  this.setConf(conf);
  var e = (this.cache ? this.evalInterCache(v, true) : this.evalInter(v, true))
        + this.evalIntra(v, true);
  var F = this.forces, i, k;
  var sx=0, sy=0, sz=0, tx=0, ty=0, tz=0;
  var cx=this.rotCenter[0], cy=this.rotCenter[1], cz=this.rotCenter[2];
  for (i = 0; i < this.nAll; i++) {
    var fx=F[3*i], fy=F[3*i+1], fz=F[3*i+2];
    if (!fx && !fy && !fz) continue;
    sx+=fx; sy+=fy; sz+=fz;
    var rx=this.coords[3*i]-cx, ry=this.coords[3*i+1]-cy, rz=this.coords[3*i+2]-cz;
    tx += ry*fz - rz*fy; ty += rz*fx - rx*fz; tz += rx*fy - ry*fx;   // (r-c) x F
  }
  g.pos[0]=sx; g.pos[1]=sy; g.pos[2]=sz;
  g.ori[0]=tx; g.ori[1]=ty; g.ori[2]=tz;
  var axes = this._axes;
  for (k = 0; k < this.tors.length; k++) {
    var tt = this.tors[k];
    var kx=axes[6*k], ky=axes[6*k+1], kz=axes[6*k+2];
    var px=axes[6*k+3], py=axes[6*k+4], pz=axes[6*k+5];
    var mx=0, my=0, mz=0;
    for (i = 0; i < tt.moves.length; i++) {
      var id = tt.moves[i]; if (id === tt.from) continue;
      var ffx=F[3*id], ffy=F[3*id+1], ffz=F[3*id+2];
      if (!ffx && !ffy && !ffz) continue;
      var qx=this.coords[3*id]-px, qy=this.coords[3*id+1]-py, qz=this.coords[3*id+2]-pz;
      mx += qy*ffz - qz*ffy; my += qz*ffx - qx*ffz; mz += qx*ffy - qy*ffx;
    }
    g.tors[k] = kx*mx + ky*my + kz*mz;
  }
  return e;
};

/* ===========================================================================
 * 10. conf / change  (conf.h)
 * =========================================================================== */
function makeConf(nt) { return { pos:[0,0,0], q:new Float64Array([1,0,0,0]), tors:new Float64Array(nt) }; }
function makeChange(nt) { return { pos:[0,0,0], ori:[0,0,0], tors:new Float64Array(nt) }; }
function copyConf(c) { return { pos:c.pos.slice(), q:new Float64Array(c.q), tors:new Float64Array(c.tors) }; }
/**
 * normalize_angle — common.h. Reduce to (-pi, pi] ARITHMETICALLY, not by
 * looping: the out-of-box gradient is O(slope) = O(1e6), so a single BFGS step
 * can move a torsion by ~1e6 rad and a `while (x > pi) x -= 2pi` loop would
 * spin ~160k times (or forever on a non-finite value).
 */
var TWO_PI = 2 * Math.PI;
function normalizeAngle(x) {
  if (!isFinite(x)) return 0;
  if (x > Math.PI || x < -Math.PI) {
    x -= TWO_PI * Math.floor((x + Math.PI) / TWO_PI);
    if (x <= -Math.PI) x += TWO_PI;
    if (x >   Math.PI) x -= TWO_PI;
  }
  return x;
}
function confIncrement(c, ch, alpha) {                       // conf.h ligand_conf::increment
  var dx=alpha*ch.pos[0], dy=alpha*ch.pos[1], dz=alpha*ch.pos[2];
  if (isFinite(dx)) c.pos[0]+=dx;
  if (isFinite(dy)) c.pos[1]+=dy;
  if (isFinite(dz)) c.pos[2]+=dz;
  var rx=alpha*ch.ori[0], ry=alpha*ch.ori[1], rz=alpha*ch.ori[2];
  if (isFinite(rx) && isFinite(ry) && isFinite(rz)) qIncrement(c.q, [rx, ry, rz]);
  for (var i=0;i<c.tors.length;i++) {
    var d = alpha*ch.tors[i];
    if (isFinite(d)) c.tors[i] = normalizeAngle(c.tors[i] + d);
  }
}
function changeFlat(ch, out) {
  out[0]=ch.pos[0]; out[1]=ch.pos[1]; out[2]=ch.pos[2];
  out[3]=ch.ori[0]; out[4]=ch.ori[1]; out[5]=ch.ori[2];
  for (var i=0;i<ch.tors.length;i++) out[6+i]=ch.tors[i];
  return out;
}
function changeUnflat(out, ch) {
  ch.pos[0]=out[0]; ch.pos[1]=out[1]; ch.pos[2]=out[2];
  ch.ori[0]=out[3]; ch.ori[1]=out[4]; ch.ori[2]=out[5];
  for (var i=0;i<ch.tors.length;i++) ch.tors[i]=out[6+i];
  return ch;
}

/**
 * conf::randomize — position uniform INSIDE the box (absolute coordinates,
 * conf.h rigid_conf::randomize(corner1, corner2)), random orientation, every
 * torsion uniform on (-pi, pi).
 */
function randomizeConf(conf, box, rng) {
  var bc=box.center, bs=box.size;
  conf.pos[0]=rng.f(bc[0]-bs[0]/2, bc[0]+bs[0]/2);
  conf.pos[1]=rng.f(bc[1]-bs[1]/2, bc[1]+bs[1]/2);
  conf.pos[2]=rng.f(bc[2]-bs[2]/2, bc[2]+bs[2]/2);
  // random_orientation(): normalised gaussian-ish quaternion via rejection
  for (;;) {
    var a=rng.f(-1,1), b=rng.f(-1,1), c=rng.f(-1,1), d=rng.f(-1,1);
    var n=a*a+b*b+c*c+d*d;
    if (n > 1e-6 && n < 1) { n=Math.sqrt(n); conf.q[0]=a/n; conf.q[1]=b/n; conf.q[2]=c/n; conf.q[3]=d/n; break; }
  }
  for (var i=0;i<conf.tors.length;i++) conf.tors[i]=rng.f(-Math.PI, Math.PI);
  return conf;
}

/**
 * mutate_conf — mutate.cpp:24-56. Exactly ONE of (2 + n_torsions) entities:
 * translation by amplitude * random_inside_sphere, orientation by
 * amplitude/gyration_radius * random_inside_sphere, or one torsion set to
 * uniform(-pi, pi). Note the torsion is RANDOMIZED, not nudged — that is what
 * lets Vina cross ring-flip-scale barriers.
 */
var _sph = new Float64Array(3);
function mutateConf(conf, model, amplitude, rng) {
  var n = 2 + conf.tors.length;
  var which = rng.i(0, n - 1);
  if (which === 0) {
    rng.insideSphere(_sph);
    conf.pos[0]+=amplitude*_sph[0]; conf.pos[1]+=amplitude*_sph[1]; conf.pos[2]+=amplitude*_sph[2];
    return;
  }
  which--;
  if (which === 0) {
    var gr = model.gyr;
    if (gr > 1e-8) { rng.insideSphere(_sph);
      qIncrement(conf.q, [amplitude/gr*_sph[0], amplitude/gr*_sph[1], amplitude/gr*_sph[2]]); }
    return;
  }
  which--;
  if (which < conf.tors.length) conf.tors[which] = rng.f(-Math.PI, Math.PI);
}

/* ===========================================================================
 * 11. BFGS  (bfgs.h) — verbatim constants
 * =========================================================================== */
function lineSearch(model, n, x, g, f0, p, xNew, gNew, v, st) {
  var c0 = 0.0001, maxTrials = 10, multiplier = 0.5;   // bfgs.h:67-69
  var alpha = 1, pg = 0, i;
  for (i = 0; i < n; i++) pg += p[i] * g[i];
  var f1 = f0;
  for (var trial = 0; trial < maxTrials; trial++) {
    // x_new = x; x_new.increment(p, alpha)
    xNew.pos[0]=x.pos[0]; xNew.pos[1]=x.pos[1]; xNew.pos[2]=x.pos[2];
    xNew.q[0]=x.q[0]; xNew.q[1]=x.q[1]; xNew.q[2]=x.q[2]; xNew.q[3]=x.q[3];
    xNew.tors.set(x.tors);
    changeUnflat(p, st.chTmp);
    confIncrement(xNew, st.chTmp, alpha);
    f1 = model.evalDeriv(xNew, v, st.gTmp);
    changeFlat(st.gTmp, gNew);
    if (isFinite(f1) && f1 - f0 < c0 * alpha * pg) break;   // Armijo
    alpha *= multiplier;
  }
  st.f1 = f1;
  return alpha;
}

/** bfgs() — bfgs.h:96-145. Returns the final energy; x is modified in place. */
function bfgs(model, x, maxSteps, v, st) {
  var n = model.ndof, i, j;
  var h = st.h, g = st.g, gNew = st.gNew, p = st.p, y = st.y, hy = st.hy;
  h.fill(0); for (i = 0; i < n; i++) h[i*n+i] = 1;

  var f0 = model.evalDeriv(x, v, st.gTmp);
  changeFlat(st.gTmp, g);
  var fOrig = f0;
  var xOrig = copyConf(x), gOrig = g.slice();

  for (var step = 0; step < maxSteps; step++) {
    for (i = 0; i < n; i++) { var s = 0; for (j = 0; j < n; j++) s -= h[i*n+j]*g[j]; p[i] = s; }
    var alpha = lineSearch(model, n, x, g, f0, p, st.xNew, gNew, v, st);
    var f1 = st.f1;
    for (i = 0; i < n; i++) y[i] = gNew[i] - g[i];
    f0 = f1;
    x.pos[0]=st.xNew.pos[0]; x.pos[1]=st.xNew.pos[1]; x.pos[2]=st.xNew.pos[2];
    x.q.set(st.xNew.q); x.tors.set(st.xNew.tors);

    var gn = 0; for (i = 0; i < n; i++) gn += g[i]*g[i];
    if (!(Math.sqrt(gn) >= 1e-5)) break;                // bfgs.h:126 (also breaks on NaN)
    for (i = 0; i < n; i++) g[i] = gNew[i];

    if (step === 0) {
      var yy = 0, yp = 0;
      for (i = 0; i < n; i++) { yy += y[i]*y[i]; yp += y[i]*p[i]; }
      if (Math.abs(yy) > 1e-12) { var d = alpha*yp/yy; h.fill(0); for (i = 0; i < n; i++) h[i*n+i] = d; }
    }
    // bfgs_update — bfgs.h:35-62, transcribed literally.
    //   minus_hy = -H*y   (minus_mat_vec_product)
    //   yhy      = -y . minus_hy = y.H.y
    //   r        = 1 / (alpha * yp)
    //   h(i,j)  += alpha*r*(minus_hy_i*p_j + minus_hy_j*p_i)
    //            + alpha^2 * (r*r*yhy + r) * p_i*p_j
    // The alpha powers matter: folding them into r corrupts H after step 1.
    var yp2 = 0; for (i = 0; i < n; i++) yp2 += y[i]*p[i];
    if (alpha * yp2 >= 1e-12) {
      for (i = 0; i < n; i++) { var s2 = 0; for (j = 0; j < n; j++) s2 += h[i*n+j]*y[j]; hy[i] = -s2; }
      var yhy = 0; for (i = 0; i < n; i++) yhy -= y[i]*hy[i];
      var r = 1 / (alpha * yp2);
      for (i = 0; i < n; i++) for (j = i; j < n; j++) {
        var val = h[i*n+j]
          + alpha * r * (hy[i]*p[j] + hy[j]*p[i])
          + alpha * alpha * (r*r*yhy + r) * p[i]*p[j];
        h[i*n+j] = val; h[j*n+i] = val;
      }
    }
  }
  if (!(f0 <= fOrig)) {                                  // bfgs.h:137-141
    f0 = fOrig;
    x.pos = xOrig.pos.slice(); x.q.set(xOrig.q); x.tors.set(xOrig.tors);
  }
  return f0;
}

function makeBfgsState(model) {
  var n = model.ndof, nt = model.tors.length;
  return {
    h: new Float64Array(n*n), g: new Float64Array(n), gNew: new Float64Array(n),
    p: new Float64Array(n), y: new Float64Array(n), hy: new Float64Array(n),
    gTmp: makeChange(nt), chTmp: makeChange(nt), xNew: makeConf(nt), f1: 0
  };
}

/* ===========================================================================
 * 12. Monte Carlo  (monte_carlo.cpp) + output container
 * =========================================================================== */

function heavyCoords(model, out) {
  for (var h = 0; h < model.nHeavy; h++) {
    var i = model.heavy[h];
    out[3*h]=model.coords[3*i]; out[3*h+1]=model.coords[3*i+1]; out[3*h+2]=model.coords[3*i+2];
  }
  return out;
}
/** rmsd_lower_bound (symmetric-agnostic variant used for pose dedup). */
function rmsdLB(a, b, nh) {
  var s = 0;
  for (var i = 0; i < nh; i++) {
    var dx=a[3*i]-b[3*i], dy=a[3*i+1]-b[3*i+1], dz=a[3*i+2]-b[3*i+2];
    s += dx*dx+dy*dy+dz*dz;
  }
  return Math.sqrt(s / Math.max(1, nh));
}
/** add_to_output_container — keeps the list sorted, dedups within min_rmsd. */
function addToOutput(out, cand, minRmsd, maxSize, nh) {
  for (var i = 0; i < out.length; i++) {
    if (rmsdLB(out[i].coords, cand.coords, nh) < minRmsd) {
      if (cand.e < out[i].e) { out[i] = cand; out.sort(function(a,b){return a.e-b.e;}); }
      return;
    }
  }
  out.push(cand);
  out.sort(function (a, b) { return a.e - b.e; });
  if (out.length > maxSize) out.length = maxSize;
}

function metropolisAccept(oldF, newF, T, rng) {
  if (newF < oldF) return true;
  return rng.f(0,1) < Math.exp((oldF - newF) / T);
}

/**
 * One Monte-Carlo task — monte_carlo::operator().
 * @param {Model} model
 * @param {Object} p  { globalSteps, localSteps, temperature, mutationAmplitude,
 *                      huntCap, authenticV, numSavedMins, minRmsd }
 */
/* onStep(step, bestE, bestConf) — bestConf is the conformation that produced
   bestE, so a caller can render the running best while the search continues.
   It is null until the first improvement. */
function monteCarlo(model, p, rng, onStep) {
  var nt = model.tors.length;
  var st = makeBfgsState(model);
  var out = [];
  var tmp = makeConf(nt);
  randomizeConf(tmp, model.box, rng);
  var tmpE = bfgs(model, tmp, p.localSteps, p.huntCap, st);
  var bestE = Infinity, bestConf = null;   // bestConf shadows bestE so a progress
                                          // callback can draw the pose it belongs to

  for (var step = 0; step < p.globalSteps; step++) {
    var cand = copyConf(tmp);
    mutateConf(cand, model, p.mutationAmplitude, rng);
    var candE = bfgs(model, cand, p.localSteps, p.huntCap, st);

    if (step === 0 || metropolisAccept(tmpE, candE, p.temperature, rng)) {
      tmp = cand; tmpE = candE;
      if (tmpE < bestE || out.length < p.numSavedMins) {
        // Re-minimise against authentic_v before recording — monte_carlo.cpp:65
        var refined = copyConf(tmp);
        var rE = bfgs(model, refined, p.localSteps, p.authenticV, st);
        model.setConf(refined);
        var co = new Float64Array(model.nHeavy * 3); heavyCoords(model, co);
        addToOutput(out, { e: rE, conf: copyConf(refined), coords: co },
                    p.minRmsd, p.numSavedMins, model.nHeavy);
        if (tmpE < bestE) { bestE = tmpE; bestConf = copyConf(refined); }   // copy: `refined`
      }                                              // is rebuilt next iteration and `st` is reused
    }
    if (onStep && (step & 63) === 0) onStep(step, bestE, bestConf);
  }
  return out;
}


/**
 * Post-search refinement — vina.cpp:925-957. Each merged pose is re-minimised
 * against the exact pairwise potential (not the grid) with the out-of-box slope
 * escalated 100, 1e4, 1e6, 1e8, 1e10 until the pose sits inside the box, then
 * rescored at slope 1e6 so every pose is ranked on the same penalty. Vina then
 * re-sorts, because "order often changes after non_cache refinement".
 */
function refineAndRescore(model, merged, localSteps, weights, energyRange, numModes,
                          numTorsOverride) {
  // Vina always divides by its own conf_independent num_tors. The UI exposes that
  // as a dropdown (auto / rigid / manual), so the search has to honour whatever the
  // panel is showing or the reported affinity and the breakdown card disagree.
  var nTors = (typeof numTorsOverride === 'number' && isFinite(numTorsOverride) &&
               numTorsOverride >= 0) ? numTorsOverride : model.numTors;
  var savedCache = model.cache, savedSlope = model.slope;
  model.cache = null;                                  // exact pairwise from here on
  var st = makeBfgsState(model);
  var i, p;
  for (i = 0; i < merged.length; i++) {
    for (p = 0; p < 5; p++) {
      model.slope = 100 * Math.pow(10, 2 * p);
      bfgs(model, merged[i].conf, localSteps, 1000, st);
      if (within(model, merged[i].conf)) break;
    }
    model.slope = savedSlope;
    model.setConf(merged[i].conf);
    merged[i].e = model.evalInter(1000, false) + model.evalIntra(1000, false);
    heavyCoords(model, merged[i].coords);
  }
  merged.sort(function (a, b) { return a.e - b.e; });   // vina.cpp:957

  model.setConf(merged[0].conf);
  var intraBest = model.evalIntra(1000, false);
  var modes = [];
  for (i = 0; i < merged.length; i++) {
    model.setConf(merged[i].conf);
    var inter = model.evalInter(1000, false);
    var intra = model.evalIntra(1000, false);
    var raw = inter + intra - intraBest;
    var total = confIndependent(raw, nTors, weights.rot);
    if (modes.length && total > modes[0].affinity + energyRange) break;
    if (modes.length >= numModes) break;
    modes.push({
      affinity: total, inter: inter, intra: intra, raw: raw,
      confIndependent: total - raw,
      rmsdLB: rmsdLB(merged[0].coords, merged[i].coords, model.nHeavy),
      conf: copyConf(merged[i].conf),
      coords: merged[i].coords
    });
  }
  model.cache = savedCache; model.slope = savedSlope;
  return modes;
}
/** non_cache::within — is every heavy atom inside the box? */
function within(model, conf) {
  model.setConf(conf);
  var bc = model.box.center, bs = model.box.size;
  for (var h = 0; h < model.nHeavy; h++) {
    var a = model.heavy[h];
    for (var k = 0; k < 3; k++)
      if (Math.abs(model.coords[3*a+k] - bc[k]) > bs[k]/2) return false;
  }
  return true;
}

/**
 * conf_independent / num_tors_div — conf_independent.cpp:148-163.
 * total = x / (1 + weight_rot * num_tors)
 */
function confIndependent(x, numTors, wRot) {
  var denom = 1 + wRot * numTors;
  if (Math.abs(x) < 1e-12) return 0;
  if (Math.abs(denom) < 1e-12) return (x * denom > 0) ? Infinity : -Infinity;
  return x / denom;
}

/**
 * Full dock — the equivalent of Vina::global_search + Vina::score.
 * @param {Object} spec { lig, rec, box, weights, exhaustiveness, seed,
 *                        numModes, energyRange, minRmsd, globalSteps, localSteps }
 * @param {Function} [onProgress] (fraction, bestSoFar)
 * @returns {{modes:Array, numTors:number, params:Object}}
 */
function dock(spec, onProgress) {
  var weights = spec.weights || DEFAULT_WEIGHTS;
  var prec = new Precalculate(weights);
  var slope = (spec.slope !== undefined ? spec.slope : 1e6);
  var model = new Model(spec.lig, spec.rec, {
    weights: weights, box: spec.box, precalculate: prec, slope: slope
  });
  if (spec.useCache !== false) {
    model.cache = new Cache(spec.rec, prec, spec.box,
                            spec.granularity || 0.375, model.neededTypes, slope);
  }

  // vina.cpp:897-899 — the search-schedule heuristics, verbatim.
  var ndof = model.ndof;
  var heuristic = model.nHeavy + 10 * ndof;
  var globalSteps = spec.globalSteps || Math.floor(70 * 3 * (50 + heuristic) / 2);
  var localSteps  = spec.localSteps  || Math.floor((25 + model.nHeavy) / 3);
  var exhaustiveness = spec.exhaustiveness || 8;

  var p = {
    globalSteps: globalSteps,
    localSteps: localSteps,
    temperature: 1.2,                                  // monte_carlo.h:39
    mutationAmplitude: 2.0,                            // monte_carlo.h:39
    huntCap: (spec.huntCap !== undefined) ? spec.huntCap : 10,   // vina.cpp:903 vec(10,10,10)
    authenticV: 1000,                                  // monte_carlo.cpp:44
    numSavedMins: spec.numModes || 9,
    minRmsd: (spec.minRmsd !== undefined) ? spec.minRmsd : 1.0
  };

  var seed = (spec.seed !== undefined && spec.seed !== null) ? spec.seed : 1;
  var all = [];
  for (var task = 0; task < exhaustiveness; task++) {
    var rng = new MT19937((seed + task * 7919) >>> 0);
    var res = monteCarlo(model, p, rng, function (step, best) {
      if (onProgress) onProgress((task + step / p.globalSteps) / exhaustiveness, best);
    });
    for (var i = 0; i < res.length; i++) all.push(res[i]);
  }
  // merge_output_containers uses min_rmsd = 2 across tasks (parallel_mc.cpp:57)
  var merged = [];
  all.sort(function (a, b) { return a.e - b.e; });
  for (var m = 0; m < all.length; m++)
    addToOutput(merged, all[m], 2.0, p.numSavedMins, model.nHeavy);

  var modes = merged.length
    ? refineAndRescore(model, merged, localSteps, weights,
                       spec.energyRange || 3, p.numSavedMins)
    : [];
  return { modes: modes, numTors: model.numTors, model: model,
           params: { globalSteps: globalSteps, localSteps: localSteps,
                     exhaustiveness: exhaustiveness, seed: seed,
                     nHeavy: model.nHeavy, ndof: ndof,
                     intraPairs: model.intraPairs.length / 2 } };
}

/**
 * Score a single pose without searching — the equivalent of --score_only.
 * Returns the weighted term breakdown the UI card displays.
 */
function scorePose(model, conf, weights) {
  model.setConf(conf);
  var w = weights || model.weights, rec = model.rec, buf = model._nbrBuf;
  var tmp = new Float64Array(5);
  var sums = [0,0,0,0,0], npair = 0;
  for (var h = 0; h < model.nHeavy; h++) {
    var i = model.heavy[h], t1 = model.xsAll[i];
    if (t1 < 0) continue;
    var x=model.coords[3*i], y=model.coords[3*i+1], z=model.coords[3*i+2];
    var cnt = model.nbr.query(x, y, z, buf);
    for (var p = 0; p < cnt; p++) {
      var j = buf[p];
      var dx=x-rec.x[j], dy=y-rec.y[j], dz=z-rec.z[j];
      var r2 = dx*dx+dy*dy+dz*dz;
      if (r2 >= CUTOFF_SQR) continue;
      rawTerms(t1, rec.xs[j], Math.sqrt(r2), tmp);
      sums[0]+=tmp[0]; sums[1]+=tmp[1]; sums[2]+=tmp[2]; sums[3]+=tmp[3]; sums[4]+=tmp[4];
      npair++;
    }
  }
  var cg1=w.gauss1*sums[0], cg2=w.gauss2*sums[1], crp=w.repulsion*sums[2],
      chy=w.hydrophobic*sums[3], chb=w.hbond*sums[4];
  var inter = cg1+cg2+crp+chy+chb;
  var dg = confIndependent(inter, model.numTors, w.rot);
  return { cg1:cg1, cg2:cg2, crp:crp, chy:chy, chb:chb,
           raw:{g1:sums[0],g2:sums[1],rep:sums[2],hyd:sums[3],hb:sums[4]},
           inter: inter, dg: dg, nrot: model.numTors, npair: npair };
}

/* ===========================================================================
 * 13. Interop with the Pose Generation UI
 * ---------------------------------------------------------------------------
 * The UI stores a pose as {off, quat, tors} where the world coordinates are
 *     world = R(q)*(c - cen) + cen + ligCenter(off),
 *     ligCenter(off) = box.center + off * box.size * 0.32
 * i.e. `off` is a box-normalised DISPLACEMENT from the ligand's own reference
 * centroid. The engine stores the absolute centroid position. These convert
 * between the two so poses stay interchangeable in both directions.
 * =========================================================================== */

/** UI {off, quat, tors} -> engine conf. `cen` is model.refCentroid. */
function confFromUI(model, ui, box) {
  var conf = makeConf(model.tors.length);
  conf.q[0]=ui.quat[0]; conf.q[1]=ui.quat[1]; conf.q[2]=ui.quat[2]; conf.q[3]=ui.quat[3];
  for (var i = 0; i < conf.tors.length; i++) conf.tors[i] = (ui.tors && ui.tors[i]) || 0;
  // refCentroid depends on the torsions, so resolve it at these torsion values.
  var probe = makeConf(model.tors.length);
  probe.tors.set(conf.tors); probe.pos = [0,0,0];
  model.setConf(probe);
  var cen = model.refCentroid, s = boxScalar(box);
  conf.pos = [
    cen[0] + box.center[0] + ui.off[0] * s,
    cen[1] + box.center[1] + ui.off[1] * s,
    cen[2] + box.center[2] + ui.off[2] * s
  ];
  return conf;
}
/** engine conf -> UI {off, quat, tors}. */
function confToUI(model, conf, box) {
  model.setConf(conf);
  var cen = model.refCentroid, s = boxScalar(box) || 1;
  return {
    off: [ (conf.pos[0] - cen[0] - box.center[0]) / s,
           (conf.pos[1] - cen[1] - box.center[1]) / s,
           (conf.pos[2] - cen[2] - box.center[2]) / s ],
    quat: [conf.q[0], conf.q[1], conf.q[2], conf.q[3]],
    tors: Array.prototype.slice.call(conf.tors)
  };
}
function boxScalar(box) {
  var s = Array.isArray(box.size) ? box.size[0] : box.size;
  return s * 0.32;
}

/* =========================================================================== */
var API = {
  confFromUI: confFromUI, confToUI: confToUI,
  XS: XS, XS_SIZE: XS_SIZE, XS_NAME: XS_NAME, XS_RADIUS: XS_RADIUS,
  DEFAULT_WEIGHTS: DEFAULT_WEIGHTS, CUTOFF: CUTOFF,
  xsIsHydrophobic: xsIsHydrophobic, xsIsAcceptor: xsIsAcceptor,
  xsIsDonor: xsIsDonor, xsHBondPossible: xsHBondPossible,
  optimalDistance: optimalDistance, slopeStep: slopeStep, rawTerms: rawTerms,
  pairEnergy: pairEnergy, curlE: curlE,
  typeReceptor: typeReceptor, typeLigand: typeLigand,
  Precalculate: Precalculate, NeighborGrid: NeighborGrid, Model: Model, Cache: Cache,
  MT19937: MT19937, makeConf: makeConf, makeChange: makeChange,
  copyConf: copyConf, randomizeConf: randomizeConf, mutateConf: mutateConf,
  bfgs: bfgs, makeBfgsState: makeBfgsState, monteCarlo: monteCarlo,
  refineAndRescore: refineAndRescore, heavyCoords: heavyCoords,
  confIndependent: confIndependent, dock: dock, scorePose: scorePose,
  qIncrement: qIncrement, qToMat: qToMat, rmsdLB: rmsdLB,
  VERSION: 'vina_engine.js / AutoDock Vina 1.2.7 semantics'
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
root.VinaEngine = API;

})(typeof self !== 'undefined' ? self : this);