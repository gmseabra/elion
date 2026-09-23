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
