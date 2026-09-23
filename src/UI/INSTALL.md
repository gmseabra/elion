# Overlay — unzip OVER your UI, do not replace it

```bash
cd /home/huangzihang/repos/elion/src
cp -r UI UI.bak.$(date +%Y%m%d)      # keep a way back
unzip -o /path/to/UI_overlay.zip     # overwrites 3 files, adds 12, deletes nothing
```

Then restart Flask and **hard-reload** the browser (Ctrl-Shift-R) — `pose.js` is
cached, and a stale copy shows the old toolbar even after the file is replaced.

## Receptor double bonds — the fix in this build

The pocket drew every protein bond as a single stick. `PG.init.site()` built its
sticks from `inferBonds(draw)`, which returns bare `[i, j]` pairs, and
`stickTraces` reads an **array** bond as `order = 1`:

```js
const ord = Array.isArray(pr) ? 1 : (pr.order || 1);
```

So the PHE/TYR/TRP/HIS rings, every backbone C=O, every Asp/Glu carboxylate and
every Arg guanidinium rendered flat. The uploaded-**ligand** path never had the
problem, because `ligandFromPDB()` perceives orders and hands `stickTraces`
`{a, b, order}` objects. Same renderer, two bond shapes — the receptor was simply
never told what its bond orders were.

The merge had taken `UI.zip`'s side of the whole chemistry section, which cost
two generations of work, not one. All three pieces are restored:

| restored | what it does |
|---|---|
| `protBondOrders()` + `_RES_DBL` / `_RES_AA` / `_RES_STD` | **templates first, geometry second.** A pocket is a *truncated* selection — sliced by distance to 1400 atoms, which cuts residues in half — and a severed side chain has free valence that geometry perception will happily spend on a double bond that does not exist. Standard residues are a closed set with known Kekulé structures, so they are looked up. Anything the table does not know (a cofactor, a modified residue, a co-crystallised HETATM) falls through to the identical code path the ligand gallery uses. |
| `perceivePdbChemistry()` | the **valence-capped** successor to `perceivePdbBondOrders()`. Without the cap, "every planar 5-/6-ring gets alternating Kekulé doubles" hands a double bond to a ring ether oxygen that already has two neighbours — a trivalent oxygen. That is what made γ-pyranones read as 4-hydroxychromenes and quinoline N-oxides as N-hydroxy once a reader repaired the impossible valence. Also returns formal charges. |
| `_pose_block()` / `_pose_chem()` | the **server-side twin**. `uiapp/core/pose_chem.py` was already in the tree but nothing called it, so downloaded poses fell back to geometry-only perception with no formal charges. `UI.zip`'s `_pdb_block_with_conect` body is byte-identical to `_pose_block`'s legacy branch, so this is strictly additive: SMILES-exact → RDKit → geometry. |

Only **one** bond per aromatic ring pair is marked — a Kekulé structure, not a
1.5-order flag — because `stickTraces` draws order 2 as a second parallel line,
so a benzene ring reads the way it does in the ligand view. Inter-residue bonds
(peptide C–N, phosphodiester P–O) stay single, which is correct and also stops a
cut chain end inventing one.

### Check it yourself

```bash
cd UI
node tests/js/test_prot_bonds.js     # 28/28: PHE/TYR 3 ring doubles + backbone C=O,
                                     # TRP 4 indole, HIS 2, ASP CG=OD1 with CG-OD2
                                     # single, ARG CZ=NH1 with CZ-NH2 single,
                                     # peptide C-N single, 0 atoms over valence
python3 -m pytest tests/test_pose_chem.py -q      # 103/103
node tests/js/compare_pocket.js      # renders the same pocket twice:
                                     #   before  bonds=76  multiple=0
                                     #   after   bonds=76  multiple=25
```

`compare_pocket.js` needs `playwright-core` and `plotly.js-dist-min` in the
directory holding `UI/`; the other two need nothing but node / pytest (+ RDKit
for the exact-chemistry half of the pytest run).

## What it touches

Three files replaced, each = your `UI.zip` version **plus** the
`UI_model_picker.zip` additions. Verified programmatically that nothing in
either copy was removed:

| file | change |
|---|---|
| `web/static/js/pose.js` | + the 3D Builder (49 `PG.builder*`/`PG._bldr*` functions) and its 8 integration hooks, + the chemistry above. 116/116 function declarations, 219 `PG.*`/element ids, 62 module consts — **0 lost from either parent**. |
| `uiapp/routes/pose_routes.py` | + `_pose_score_models`, `_pose_resolve_model`, `_abs_repo`, `_scorer_status`, `_deepatom_cfg`, `GET /pose/score_models`, `model` handling in the three scorer routes, + `_pose_chem`/`_pose_block`. 122 names — **0 lost from either parent**. |
| `config/input_routes.yml` | + `deepatom_ckpt_dir`, `deepatom_arpeggio`, `deepatom_repo_dir`, `deepatom_model_py`. **No existing value changed** — it is your `UI.zip` yml with 4 keys appended. |

Added: `uiapp/core/pose_chem.py` (the server-side chemistry `_pose_block` now
calls), `tests/js/{test_prot_bonds.js, harness.js, pocket.pdb, compare_pocket.js}`,
`tests/test_pose_chem.py`, and at the root `deepatom_pose.py`, `compare_arch.py`,
`diagnose_models.py`, `MERGE_NOTES.md`, `MODEL_PICKER.md`,
`POSE_EXPORT_CHEMISTRY.md`. Nothing under `data/`, `.raytmp/`, `engines/` or any
other directory is touched.

## The BUILD button

`#poseBldrBtn` is byte-identical to the one in `UI_model_picker.zip` (1,878
chars, verified). Its markup came across in the first merge, but three of its
**call sites** did not, which is why the button opened a builder you could not
click in. All three are now in:

| hook | where | what it does |
|---|---|---|
| 1–5 | toolbar, `builderToggle`, `_bldrSync`, `_redraw3D`, panel HTML | already present in the first merge |
| 6 | `PG.open` | `PG._bldrBtnState()` — re-syncs the button's enabled state each time the panel is reopened |
| 7 | `_draw3D` | ball-and-stick radius while editing (7/14 vs 7/3.2), docking box suppressed so the ligand is alone on screen, and the selection/hover **restyle of the ligand's own marker trace** |
| 8 | `_draw3D`, after `PG._camBind()` | `PG._bldrBind()` — installs the `plotly_click` handler that does the atom picking at all |

Hook 7's restyle is not cosmetic. The obvious implementation — a translucent
halo trace drawn on top of the picked atom — makes that atom **unpickable**:
gl3d resolves a pick to the topmost object, and `hoverinfo:'skip'` suppresses
the hover rather than falling through to what is behind it. So the selected
atom's own halo swallowed the click that was supposed to deselect it.

## Three engines in the Model dropdown

```
GIGN           GIGN · Yupu_GIGN.pt               ->  POST /pose/gign_score
GatorAffinity  GatorAffinity · (script default)  ->  POST /pose/gatoraffinity_score
DeepAtom       DeepAtom · best_val_rmse.pth.tar  ->  POST /pose/deepatom_score
```

An engine is listed whenever it is **configured at all** — either its script or
its model key non-blank — and an unrunnable one is greyed with the reason rather
than hidden. `#poseGatorScoreBtn` (⚛ Score (GatorAffinity)) is also still where
`UI.zip` put it: the ③ Monte-Carlo Scorer card, `display:none` until the
**Scorer** select is set to `gatoraffinity`.

`pose.deepatom_ckpt_dir` is scanned for `*.pt / *.pth / *.pth.tar / *.ckpt`, and
each one becomes a dropdown entry. A path sent by the browser is never trusted —
the id must match an entry this server already published, and its engine must be
the one the endpoint runs.
