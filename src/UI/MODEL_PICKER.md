# Scoring-model picker — 3D Builder

A **Model** dropdown in builder row 4 (under Move / Add Fragments / Draw) chooses
which trained model scores the ligand: **GIGN · GPLA.pt** or any **DeepAtom**
checkpoint found on disk.

Already applied to this tree. `add_model_picker.py` is kept for reference and for
re-applying to another checkout (`--check`, `--revert`). `deepatom_pose.py` at the
root is the DeepAtom pose scorer — see below.

## Files changed

| file | what |
|---|---|
| `web/static/js/pose.js` | Model button, its menu, loader/label, engine routing in `_bldrGign`, six SAR status strings that used to hardcode "GIGN" |
| `uiapp/routes/pose_routes.py` | `GET /pose/score_models`, `_pose_score_models()`, `_pose_resolve_model()`, and `model` honoured in both `/pose/gign_score` and `/pose/deepatom_score` (which now appends `--model`) |
| `config/input_routes.yml` | `pose.deepatom_ckpt_dir`, `pose.default_script` |
| `deepatom_pose.py` | **new** — the DeepAtom pose scorer |

## Configure

```yaml
pose:
  gign_model:        "/home/huangzihang/repos/elion/src/elion/properties/Yupu_GIGN/GPLA.pt"
  deepatom_ckpt_dir: "/home/huangzihang/repos/user_data/deepatom/checkpoints"
  default_script:    "/home/huangzihang/repos/user_data/deepatom/checkpoints/deepatom_pose.py"
  deepatom_arpeggio: "/home/huangzihang/repos/elion/src/elion/properties/deepatom/model_split_data/arpeggio_mod2/arpeggio.py"
  deepatom_repo_dir: "/home/huangzihang/repos/elion/src/elion/properties/deepatom/model_split_data/02_pytorch"
  deepatom_model_py: "/home/huangzihang/repos/user_data/deepatom/checkpoints/shufflenet_v3_blue.py"
```

`deepatom_ckpt_dir` is a DIRECTORY. Every `*.pt`, `*.pth`, `*.pth.tar` and
`*.ckpt` inside it is listed, labelled by filename with its size.

## Verify

```bash
curl -s localhost:5000/pose/score_models | python -m json.tool
```

Each entry is `{id, label, engine, path, ok, why, note}`.

## The engine is part of the choice

GPLA.pt is a GIGN graph network; a DeepAtom `best_val_rmse.pth.tar` is a
24-channel 32³ voxel CNN. Neither loads in the other's scorer, so the picker
selects the engine too:

- `engine: gign`     → `POST /pose/gign_score`     (runs `pose.gign_script`)
- `engine: deepatom` → `POST /pose/deepatom_score` (runs `pose.default_script`)

`pose.default_script` now points at **`deepatom_pose.py`**, shipped at the root of
this tree. Install it:

```bash
mkdir -p /home/huangzihang/repos/user_data/deepatom
cp deepatom_pose.py               /home/huangzihang/repos/user_data/deepatom/checkpoints/
cp /path/to/deepatom_featurize.py /home/huangzihang/repos/user_data/deepatom/checkpoints/
chmod +x /home/huangzihang/repos/user_data/deepatom/checkpoints/deepatom_pose.py
python /home/huangzihang/repos/user_data/deepatom/checkpoints/deepatom_pose.py --selftest
python /home/huangzihang/repos/user_data/deepatom/checkpoints/deepatom_pose.py --where
```

`deepatom_featurize.py` must sit beside it (that is where the grid code lives).
### arpeggio.py is Python 2

`arpeggio_mod2/arpeggio.py` uses print statements. The `elion_backend` env is
Python 3, so running it there dies with *Missing parentheses in call to 'print'*
before a single atom is typed. The script therefore picks the interpreter for
that one subprocess separately: it asks each candidate to **parse** arpeggio.py
and uses the first that can — `$ARPEGGIO_PYTHON`, `python2`/`python2.7` on PATH,
then `~/repos/envs/*/bin/python*` and the usual conda env roots. If none can, it
lists what it tried and suggests `conda create -n arpeggio python=2.7 biopython -y`.
Override with `--arpeggio-python`.

It also FINDS `shufflenet_v3.py` and `arpeggio.py` itself — searching the script's own
directory, its parents, `~/repos`, `~/repos/elion/src` and `~` to depth 7 — and
caches the answer in `.deepatom_pose_paths.json` beside itself, so only the first
run pays for the search. `--where` prints every resolved path without scoring;
`--rescan` discards the cache. Override with `--repo-dir` / `--arpeggio` or
`$DEEPATOM_REPO` / `$ARPEGGIO` if the search picks the wrong copy. Until the file exists, DeepAtom entries stay listed but greyed —
never hidden, because a checkpoint that vanishes from the menu is
indistinguishable from one that was never copied into the directory.

## Which DeepAtom backend runs

`/pose/deepatom_score` runs `pose.default_script`, and falls back to
`deepatom.script` when that is blank. Both the **⚛ Score best pose** button and
the **SAR** button (when DeepAtom is selected) go through that one endpoint, so
they always use the same backend.

- `default_script: ""` -> the shell pipeline
  (`predict_binding_affinity_..._elion.sh`), invoked exactly as it always has
  been: `-t vs -d <root>` and nothing else. This is the shipped default.
- `default_script: ".../deepatom_pose.py"` -> the standalone scorer, which also
  receives `--model`, `--arpeggio`, `--repo-dir` and `--model-py` from the keys
  below.

The extra flags are appended **only when the script ends in `.py`**, because the
shell script's `getopts` accepts `-t` and `-d` alone.

SAR is a before/after comparison either way: it scores the pre-edit ligand and
the edited ligand through the same endpoint and reports the change. The button's
sub-label says which machinery is behind it -- `before -> after` for GIGN,
`Score best pose x 2` for DeepAtom.

## deepatom_model_py -- the network definition file

`deepatom_model_py` names the **file** that defines ShuffleNetV3, under any name,
and wins over `deepatom_repo_dir`. It exists because the two copies of the
release on lysine are different networks:

| | elion tree | cluster copy (blue) |
|---|---|---|
| `repeat_block` | `[3, 4, 4]` | `[3, 3, 3]` |
| `out_block_config` | `[2048, 1]` | `[1024, 1]` |
| params @ x2 | 4,899,739 | 3,277,727 |
| `OutBlock4` reshape | `view(N,-1,C)` | `view(N,C,-1).permute(0,2,1)` |
| `OutBlock4` eval returns | `fc1(out)` | `(fc1(out), latent_feat)` |
| init import | `from utils` | `from utilies` |

A checkpoint trained against the cluster copy misses 36 tensors and mis-shapes 6
against the elion one. The reshape difference would not even show up as a shape
error -- same dimensions, different element order -- so pointing at the right
file is the only correct fix. `deepatom_pose.py` stubs both `utils` and
`utilies`, so the file needs no companions; put it anywhere and name it anything.

`--model-py` is the flag; `$DEEPATOM_MODEL_PY` also works. Leave the key blank to
fall back to `deepatom_repo_dir`.

Use `compare_arch.py` to diff any checkpoint against any copy:

```bash
python compare_arch.py --model best_val_rmse.pth.tar --repo-dir <dir>
```

## Pinning the DeepAtom release (deepatom_arpeggio / deepatom_repo_dir)

Left unset, `deepatom_pose.py` SEARCHES for `arpeggio.py` and `shufflenet_v3.py`,
and the search can land on a different copy of the release than the
`⚛ Score best pose` pipeline uses. There are (at least) two copies on lysine:

- `~/repos/deepatom/model_split_data/...` — what the bare search finds first, and
  its `arpeggio.py` is **Python 2** (dies with *Missing parentheses in print*).
- `~/repos/elion/src/elion/properties/deepatom/model_split_data/...` — the tree
  `predict_binding_affinity_v4_2_data_split_ZccE_elion.sh` → `pipeline_VS.py:278`
  actually runs, under Python 3.

`deepatom_arpeggio` and `deepatom_repo_dir` pin both to the elion tree, so the
Model dropdown scores with the identical code path as the existing button. The
route passes them to the script as `--arpeggio` / `--repo-dir`; the script skips
its search entirely when they are given. Point them at whatever
`$DEEPATOM_ROOT_DIR/deepatom/model_split_data` is on your host — confirm with
`md5sum` if you have more than one copy.

## What deepatom_pose.py does

```
<root>/Dataset_VS/<name>/<name>_complex.pdb     staged by the route
  -> arpeggio.py                    11 interaction types per atom
  -> select_pocket_residues(5.0)    whole residues within 5 A of the ligand
  -> pcmax_grid                     32^3 x 24,  occ = 1 - exp(-(r_vdw/d)^12)
  -> ShuffleNetV3x2                 -> pK,  dG = -1.36 x pK
```

Every step is imported from `deepatom_featurize.py`, not reimplemented, so a pose
scored in the UI goes through the identical transform the training set did. With
`AUGMENT=1` (which the route sets) it averages the original grid plus
`--avg-num` rotations — 13 forward passes at the default 12, matching
`--avg_num_test 12` in the retrain protocol.

It refuses rather than guesses in three places worth knowing about: a truncated
`.atomtypes` (arpeggio dies partway on B/Fe/As/Ru/Se ligands, leaving protein
only) is caught by comparing ligand atom counts against the complex PDB; a
checkpoint whose keys do not fit the architecture is named as such; and a ligand
extending past the 32 A box is warned about rather than silently clipped.

## Security

The browser sends an **id** the server itself published, never a path. An id that
matches nothing, or whose engine does not match the endpoint, is refused with a
plain-language reason:

```
../../etc/passwd                -> unknown model id: ../../etc/passwd
deepatom:x.pt  to /gign_score   -> DeepAtom · x.pt is a deepatom checkpoint;
                                   it cannot be run by the gign scorer
```

An empty `model` means "use the configured default", which is exactly what every
call did before this picker existed — so a failed enumeration degrades to the old
behaviour instead of blocking the button.
