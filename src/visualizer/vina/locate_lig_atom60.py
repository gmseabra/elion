import math

LIG  = '/blue/lic/huangzihang/repos/AutoDock-Vina/build/linux/release/iag933/IAG933_out.pdbqt'
REC  = '/blue/lic/huangzihang/repos/AutoDock-Vina/build/linux/release/TEAD3/TEAD38P0M.pdbqt'

# Target distances from non_cache log (mode 1 scoring)
TARGET = {
    463: 7.91513,
    464: 7.37129,
    465: 6.53416,
}
TOLERANCE = 0.02   # Angstrom

# ── Load ligand: MODEL 1 only, fallback to whole file if no MODEL record ──────
lig_lines = []
with open(LIG) as f:
    raw = f.readlines()

# Check if file has MODEL records
has_model = any(l.startswith('MODEL') for l in raw)

if has_model:
    in_model = False
    for l in raw:
        if l.startswith('MODEL'):
            in_model = True
            continue
        if l.startswith('ENDMDL'):
            break          # stop after MODEL 1
        if in_model and l[:4] in ('ATOM', 'HEAT'):
            lig_lines.append(l)
else:
    # No MODEL record — take all ATOM lines
    lig_lines = [l for l in raw if l[:4] in ('ATOM', 'HEAT')]

print(f'Ligand atoms loaded: {len(lig_lines)}')
if lig_lines:
    print(f'  First: {lig_lines[0].rstrip()}')
    print(f'  Last : {lig_lines[-1].rstrip()}')
print()

# ── Load receptor heavy atoms ─────────────────────────────────────────────────
with open(REC) as f:
    rec_heavy = [l for l in f if l[:4] in ('ATOM', 'HEAT')
                 and l[77:].strip() not in ('HD', 'H')]

print(f'Receptor heavy atoms: {len(rec_heavy)}')

def coords(l):
    return (float(l[30:38]), float(l[38:46]), float(l[46:54]))

def dist(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

# Print the receptor atoms we're targeting
print()
print('Target receptor atoms (grid_atoms indices):')
for j, target_r in TARGET.items():
    rc = coords(rec_heavy[j])
    print(f'  grid_atoms[{j}]: {rec_heavy[j].rstrip()}')
    print(f'             coords={rc}  target_r={target_r}')
print()

# ── Search: find ligand atom matching ALL three distances ─────────────────────
rec_coords = {j: coords(rec_heavy[j]) for j in TARGET}

print(f'Searching {len(lig_lines)} ligand atoms (tolerance={TOLERANCE} A)...')
print()

for i, la in enumerate(lig_lines):
    lc = coords(la)
    atype = la[77:].strip()
    distances = {j: dist(lc, rc) for j, rc in rec_coords.items()}

    # Check if this atom matches any target
    matches = {j: d for j, d in distances.items() if abs(d - TARGET[j]) < TOLERANCE}

    if matches:
        print(f'[{i:3d}] {la.rstrip()}')
        print(f'       coords={lc}  atype={atype}')
        for j, d in distances.items():
            flag = ' *** MATCH' if j in matches else ''
            print(f'       d_to_rec[{j}]={d:.5f}  target={TARGET[j]:.5f}{flag}')
        print()

# ── Also print distances for all OA ligand atoms regardless ──────────────────
print('--- All OA ligand atoms and their distances to rec[463] ---')
for i, la in enumerate(lig_lines):
    if la[77:].strip() not in ('OA', 'O'):
        continue
    lc = coords(la)
    d = dist(lc, rec_coords[463])
    print(f'  [{i:3d}] serial={la[6:11].strip():5s}  {la[12:16].strip():4s}  '
          f'atype={la[77:].strip():4s}  '
          f'coords=({lc[0]:.3f},{lc[1]:.3f},{lc[2]:.3f})  '
          f'd_to_463={d:.5f}')