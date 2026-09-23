# ui6 + ui7 — merged

`git pull ui7` is done. This tree is **ui7 as the base**, with the three files
ui6 owned merged in. Both branches are still here:

```
main              the merge (this tree)
remotes/ui7/ui7   UI.zip as received
git log --oneline --all
git diff ui7/ui7 main -- <path>
```

## How each conflict was resolved

58 files conflicted (no common ancestor, so git flags every difference).
**55 took ui7 verbatim** — I never edited them; ui6's copies were simply an older
base. Three were merged by hand:

| file | resolution |
|---|---|
| `web/static/js/pose.js` | ui7's refactored file **+** ui6's 3D Builder block (1,162 lines) **+** its five integration hooks |
| `uiapp/routes/pose_routes.py` | ui7's routes **+** the Model-dropdown backend |
| `config/input_routes.yml` | ui7's config **+** the `deepatom_*` keys |

ui7's `default_script` (`deepatom_score_pipeline.sh`) is kept — the working one.

### The five hooks the Builder needs

The panel is self-contained, but it has to be reachable and its atoms clickable:

1. `stickTraces` — ligand atoms get `customdata` + `hoverinfo:'text'` while the
   panel is open. Plotly's `'skip'` suppresses `plotly_click` as well as the
   hover label, so without this the picker never fires.
2. `_draw3D` — ligand atoms carry `_i` (their index in the *unstripped* array,
   so a pick survives the hydrogen toggle) and a label.
3. `_ligandPdb(coords, els, bonds)` — gained the two optional arguments, so SAR
   can write a PDB for the pre-edit snapshot without swapping the global `L`,
   and so CONECT records carry bond orders.
4. the **BUILD** button in the import stepper (`#poseBldrBtn`).
5. `PG.close` — closes the panel, drops the snapshot/SAR/undo state and clears
   the plotly binding flag, so re-opening cannot bind a second listener.

`PG._camForce` came with it: a one-shot camera override consumed in `_draw3D`,
which is how the panel re-frames on the lone ligand and restores the view after.

## Three engines

The dropdown now lists all three, grouped, each routed to its own endpoint:

```
gign           -> POST /pose/gign_score            pose.gign_script / gign_model
gatoraffinity  -> POST /pose/gatoraffinity_score   pose.gatoraffinity_script / _model
deepatom       -> POST /pose/deepatom_score        pose.default_script + deepatom_ckpt_dir
```

`pose.gatoraffinity_model` is blank, which means "the checkpoint
`gatoraffinity_pose.py` picks for itself". That is a real choice, so it is
offered as **GatorAffinity · (script default)** rather than hidden.

Each route validates the `model` the browser sends against the same published
allow-list (`_pose_resolve_model`): the body carries an **id**, never a path, and
an id whose engine does not match the endpoint is refused by name.

SAR is a before/after comparison for every engine — it scores the pre-edit
ligand and the edited one through whichever endpoint the selection names.

## Verified

- `node --check web/static/js/pose.js`
- `python -m py_compile uiapp/routes/pose_routes.py`
- `yaml.safe_load(config/input_routes.yml)`
- `_pose_score_models()` driven against the merged yml — all three engines `OK`
- the dropdown menu rendered headlessly: three groups, correct ids and handlers
- ui7's modules (`pose_gator_dock.js`, `devmap.js`, …) and the `hub.html`
  script tags that load them are intact

## Not verified — needs the host

Whether each scorer actually runs. That is unchanged by this merge: the same
scripts, conda envs and checkpoints as before.

Still open from earlier: `best_val_rmse.pth.tar` is a `[3,3,3]` / 1024-wide
checkpoint, and the elion tree's `shufflenet_v3.py` builds `[3,4,4]` / 2048.
`deepatom_score_pipeline.sh` hard-codes its model directory (lines 38 / 63 /
140); the patched copy I sent makes `DEEPATOM_MODEL_DIR` and `DEEPATOM_PRED_DIR`
environment-overridable, and `pose.deepatom_env` in this tree can inject them.

## GatorAffinity in ③ Evaluate With (2026-09-23)

This tree's `hub.html` had no GatorAffinity entry in the ③ Monte-Carlo
**Evaluate With** dropdown, although `pose.js` (`PG.mc.scoreGator`, the
`'gatoraffinity'` branch of `setScorer`, the auto-score after ▶ Run search) and
`/pose/gatoraffinity_score` already supported it. Added, with the markup the
glycine branch uses:

| file | change |
|---|---|
| `web/templates/hub.html` | `<option value="gatoraffinity">GatorAffinity ΔG (server)</option>`, the inline `#poseGatorScoreBtn`, and `#poseGatorCard` (output dir, predicted pK, ΔG, ⚛ Score) after the GIGN card |
| `uiapp/config.py` | `POSE_GATORAFFINITY_SCRIPT` (env `UI_POSE_GATORAFFINITY_SCRIPT`, default blank): the fallback the route reads when `pose.gatoraffinity_script` is blank. Without it that case raised AttributeError (HTTP 500) instead of answering "script not found" |

The card's note names `pose.gatoraffinity_model` instead of one checkpoint,
since this tree's config points it at a fine-tuned `.ckpt`. At that point this
`hub.html` still did not load the dock / trajectory add-ons listed under
"Verified" above; the next section changes that.

## glycine's solutions merged in (2026-09-23)

The "ui7 as the base" merge above kept older (Aug 6) copies of seven wiring
files, so ui7's modules were on disk here but switched off: the add-on
`<script>` tags, the RL loop, the devmap overlay and two TS fixes. The glycine
branch (`UI_glycine.zip`) has the newer copies. A file-by-file scan of the two
trees found nothing else in glycine that this tree lacks, so exactly these
files were taken from glycine:

| file | what it brings |
|---|---|
| `web/templates/hub.html` | the **Predict Affinity Batch** card (Tools); `<script>` tags for the ten pose add-ons (score docks, ◀ ▶ stepper, per-pose trajectory scoring, spark hover), `pose_affinity_batch.js` and `devmap.js`; the same GatorAffinity option and card as above (the note keeps this tree's wording) |
| `web/static/js/hub.js` | the `affinity_batch` nav case; `ts_rl_loop.js`, `ts_rl_focus.js`, `ts_score_dist.js`, `ts_rl_debug.js` in the TS lazy loader |
| `web/static/js/ts/ts_rl.js` | `data-rlv` / `data-rlcat` tags that the verdict filter (`ts_rl_focus.js`) needs |
| `web/static/js/ts/ts_ui.js` | the job bars are replaced on every poll instead of kept; stale ranking rows are cleared when the ranking is empty |
| `uiapp/routes/ts_routes.py` | `/vina_visualization/ts_rl_bias`, `ts_rl_feedback`, `ts_rl_harvest`, `ts_rl_runs`, `ts_score_dist`, `ts_smiles_export`; `_load_sessions(include_finished)`, which keeps a finished session for a retention window instead of deleting it |
| `uiapp/routes/__init__.py` | loads `devmap_routes` |
| `uiapp/config.py` | `devmap_settings()`, hotkey parsing and `_env_flag`; the same `POSE_GATORAFFINITY_SCRIPT` as above |
| `NEW_FEATURES.md`, `README.md` | GatorAffinity, the source map and `UI_POSE_GATORAFFINITY_SCRIPT` documented (route counts corrected for this tree: 108 in all, 25 under `/pose/`) |

Kept from this tree because glycine's copy is older: `pose.js` (the 3D
Builder, residue-template pocket bond orders), `pose_routes.py` (export
chemistry via `uiapp/core/pose_chem.py`, `/pose/score_models`, model ids, the
DeepAtom flags), `uiapp/core/pdb_converter.py` (no `END` in receptor PDBQT,
which Vina rejects), `engines/vina/pdb_to_pdbqt_cpu_thread.py` and
`pipeline/active_learning/get_pdbs_from_results_TS_smiles.py` (SDF first),
`config/input_routes.yml` (this machine's paths and checkpoints) and
`ELION_UI_PLAYBOOK.md`. Glycine's `data/` (logs, converted receptors, ChemBERT
splits) and its log files are runtime output, not code, and were left out.

Two fixes are in neither branch and come from the `fixes.patch` shipped with
the tests. `pose_routes.py`: `stage.resolve()` / `root.resolve()` after the
staging `mkdir` of the GIGN, GatorAffinity and DeepAtom scorers, so a relative
output dir no longer ends in "OSError: Bad input file". `pose.js`: `PG.close`
passes `builderToggle(false, true)`, so closing the workspace with the 3D
Builder open no longer throws a late Plotly error.
