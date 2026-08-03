# What was added on top of the base `elion` tree

This tree is the original `elion.zip` with three new subsystems layered on and
seven existing files updated. Nothing from the base was removed.

Full rationale, algorithms, parameters and traps: **`ELION_PLAYBOOK.md`**
(§16 GIGN, §17 warm-up checkpointing, §18 the DeepAtom consolidation, and the
new §0 rows G56–G80).

---

## 1. ⬡ GIGN — geometric interaction graph pose scorer  `properties/Yupu_GIGN/`

Scores a *posed* protein–ligand complex — the first predictor in the tree that
needs 3-D coordinates and a receptor rather than a SMILES string.

| | |
|---|---|
| Model | `GIGN(node_dim=35, hidden=256, layers=3)`, 3 × heterogeneous interaction layers, sum pooling, 3-block FC head |
| Graph | ligand + 5 Å pocket heavy atoms; two edge sets (covalent bonds / ligand↔pocket pairs < 5 Å) with separate weights |
| Distances | 9 Gaussian RBF centres over [0, 6] Å, σ = 0.667 |
| Output | `pK` (higher = better) **and** `ΔG = −1.36 × pK` (more negative = better) |
| Invocation | subprocess, **one pose per call** — `yupu_GIGN_pose.py --stage <dir> --name <n>` (staging-dir contract) or `yupu_GIGN_backend.py --lig --rec --smiles …` (all-argv contract) |

**Two things to know before running it.** `pocket_cutoff <= 0` means "use the
whole receptor", and because the readout is a *sum* over nodes with an unbounded
head, that produces a physically impossible score — the shipped
`gign_pose_score.log` contains a pK of 37.5 from exactly this. And the sign is
the opposite of every other reward in this repo, so parse `GIGN_DELTAG`, not
`GIGN_PRED_PK`, if you wire it into `Estimators`. There is deliberately **no
`Property` subclass** — see the playbook for why one cannot work as-is.

The default checkpoint (`model/…​.pt`) is not in the repo and must be supplied.

## 2. ⟲ Warm-up checkpointing for Thompson Sampling

- `warmup_checkpoint_loader.py` — monkeypatch loader; import it *before* any
  engine module and set `TS_WARMUP_CHECKPOINT=<file.json>`.
- `_warmup_wrapper_rxn{101_amide,110_suzuki,208_snar}.py` — the generated launch
  wrappers (stale artifacts kept as reference; the UI regenerates them).
- `generators/TS/thompson_sampling.py` — restore path at the top of `warm_up()`,
  plus the stdout-volume fix (see below).

Skips the `num_warmup_trials × Σ|reagents|` evaluations that precede iteration 0
by restoring each reagent's Gaussian posterior from JSON.

**The checkpoint is keyed on the reaction short name and nothing else** — not the
SMARTS, not the reagent files, not the reward function. It also does not carry
the disallow tracker or the retired-reagent list. Read G66–G68 before reusing one
across a config change.

## 3. TS analysis tooling  `generators/TS/`

`ceiling_vs_live.py`, `why_eligible_changes.py`, `tests/probe_competitors.py` —
three fixed-seed simulations that answer "why does the eligible-reagent counter
go *up*?". Answer: it is a conditional quantity, not a bug. `probe_competitors.py`
exits non-zero on failure and is the one with assertions.
`thompson_sampling_bkp.py` is the abandoned batched-TS experiment, kept because
its diagnosis was wrong in an instructive way.

## 4. DeepAtom pipeline, consolidated  `properties/deepatom/`

- `00_preprocess/generate_atomtypes.py` — stages 1–6 behind one `--stages` flag
- `01_generate_channels/generate_npz.py` — both grid builders behind `--stage`
- `bin/deepatom_score_pipeline.sh` — single-pose driver / batch dispatcher
- `bin/predict_binding_affinity_v4_2_data_split_{8P0M,PRMT5,ZccE_elion}.sh`
- `arpeggio_mod2/arpeggio_updated.py`

**Voxel channel semantics are unchanged** — verified function-by-function, so
existing model weights stay valid. The atom-typer fix means halogenated ligands
now produce atom types *at all* (previously a `KeyError` aborted the typer).

---

## Updated base files

| File | Change |
|---|---|
| `elion.py` | three-way `sys.path` pin (the host may have more than one `properties` package) |
| `generators/TS/thompson_sampling.py` | checkpoint restore; removed ~10 unparsed DEBUG lines/iteration that were back-pressuring the stdout pipe and stalling long runs; timing probe to a separate file |
| `generators/TS/ts_main.py` | repaired — it imported a module that does not exist and was dead on import |
| `input_TS.yml` | `visualizer.output_dir` + `visualizer.port` / `visualizer.host` (the UI reads its bind address from here — `$UI_PORT` > `visualizer.port` > 5000); merged vina/deepatom/pose config; `SAScore.rew_coeff 0.05→0`, `CHEMBERT_BE.rew_coeff 0.9→0.95`, `num_ts_iterations 5000→5` (a debug leftover — raise it for real runs) |
| `properties/CHEMBERT/chembert.py` | inference batch 16 → 128, `num_workers` 4 → 0, `model.eval()` now called |
| `properties/deepatom/.../arpeggio.py` | ligand-element capitalisation fix |
| `properties/deepatom/bin/predict_…_data_split.sh` | Chimera stage → `pipeline_VS.py` |

## Known-inherited state

Five Python files in `properties/deepatom/` are unmigrated Python 2 and do not
parse (`*_ind_prep.py`, `arpeggio_original.py`, `show_contacts.py`,
`bin/My_Data/test_binding_affinity.py`). They were like that in the base tree and
are left untouched. `properties/Yupu_GIGN/verify_gign_repro.sh` is byte-identical
to `yupu_GIGN_pose.py` — it is Python with a `.sh` extension, not a shell script.
