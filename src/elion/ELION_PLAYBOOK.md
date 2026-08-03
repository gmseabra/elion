# Elion — Distillation Playbook

**What this is.** A build spec for **Elion**, a config-driven de-novo drug-design engine: a molecular
*generator* is biased by reinforcement learning (or steered by Thompson Sampling over a combinatorial
reaction library) against a weighted sum of *property* rewards, where the properties are pluggable
scorers ranging from cheap RDKit descriptors to a fine-tuned SMILES transformer and a 3-D CNN. It is
written so a *fresh session on a new platform* can rebuild every subsystem quickly — each section gives
the **goal**, the **approach/algorithm**, the **key parameters**, the **gotchas that cost real debugging
time**, and **how to verify**.

**Substrate assumption.** Python 3.10, RDKit ≥ 2022.03, PyTorch ≥ 2.1, NumPy, pandas, PyYAML. The
*algorithms* are library-agnostic; the gotchas marked `[RDKIT]`, `[TORCH]` and `[PY2→3]` are specific to
those substrates and are the traps most likely to burn a reimplementation. Nothing here requires a
server; everything is a CLI run over local files.

**How to read it.** Start with §0 (the gotcha quick-reference — the highest-value page). Then build in
the order of §1→§16; later subsystems depend on earlier contracts. Constants are collected in the
Appendix.

> **Scope note.** This playbook describes the code **as shipped**, including the parts that do not run.
> Roughly a third of the tree is dead, half-migrated, or broken in ways that fail *silently* — and that
> is precisely the information a rebuild needs, because the failure modes are indistinguishable from
> "the science didn't work". Wherever a subsystem is non-functional, it is marked and the *reason* is
> given, because the reason is usually the reusable lesson.

> **The one-line summary of every hard bug below:** Elion's contracts are *conventions*, not types. A
> YAML key is a filename is a class name; a dict's iteration order is a CSV's column order; a property's
> return list must be index-aligned with the mol list it never sees. Nothing checks any of this. Every
> §0 entry marked **bold** is a convention that drifted.

> **v2 note — §16, §17 and §18 were added, and they are the three new subsystems.**
> **§16 ⬡ GIGN**, a geometric interaction graph network that scores a *posed* complex rather than a
> SMILES string — the first predictor in the tree that needs 3-D coordinates and a receptor, which is
> exactly why it does not fit the `Property` contract and is invoked out-of-process instead (**G56**).
> It brought eleven gotchas, of which two are load-bearing: sum pooling with an unbounded head makes the
> score **extensive in pocket size**, so feeding it a whole receptor produces a physically impossible
> pK — and the shipped log contains that number (**G57**); and the whole thing predicts pK, *higher is
> better*, which is the opposite polarity to every existing reward in the system (**G63**).
>
> **§17 ⟲ Warm-up checkpointing**, which lets a Thompson-Sampling rerun skip the
> `num_warmup_trials × Σ|reagents|` evaluations that precede iteration 0. The mechanism is sound and the
> *key* is not: a checkpoint is keyed on the reaction's short name and nothing else, so the same commit
> that added it also changed two reward coefficients while leaving the checkpoints reusable (**G66**).
> Three further gotchas follow from what the checkpoint does *not* carry: the disallow tracker (**G67**)
> and the retired-reagent list (**G68** — dead reagents return as the *most attractive* candidates,
> because Thompson sampling is optimistic under uncertainty).
>
> **§18 — the DeepAtom pipeline consolidated**, five shell stages collapsed into two Python entry points
> plus a dispatcher. The voxel channel semantics are **unchanged** — verified function-by-function, so
> existing weights stay valid (**G80**) — but the dispatcher's branch order lets a leftover single pose
> hijack a batch run (**G74**), and the pose path skips a stage whose substitute keeps receptor
> hydrogens, which silently changes the atom typing the same weights were trained on (**G78**).
>
> **Also changed in this drop:** the standalone TS CLI was repaired (it had been dead on import against a
> module that does not exist); `elion.py` gained a three-way `sys.path` pin because the box has more than
> one `properties` package; the ligand-element capitalisation fix in the atom typer means halogenated
> ligands now produce atom types **at all** (**G73**); and the ChemBERT inference batch went 16 → 128
> with `model.eval()` finally called.

---

## 0. Gotcha quick-reference (read this first)

These are the non-obvious things that were wrong on the first try, or that a reimplementation will get
backwards. None is guessable from documentation.

| # | Trap | Right answer |
|---|---|---|
| **G1** | **Every consumer of `estimate_properties()` crashes with `TypeError: 'int' object is not subscriptable`** | `Estimators.estimate_properties` injects `pred['__n_mols__'] = self.n_mols` — an **int** — as the last key of a dict whose values are otherwise all lists. `utils.print_results` (`results[prop][ind]`) and `utils.save_smi_file` (`zip(*[predictions[x] for x in predictions])`) both die on it; `print_stats` survives only because `np.max(200) == 200`, and emits a bogus column. It was added for re-entrancy (`evaluate()` scores one mol at a time and resets `self.n_mols`), which is a real problem — but the fix must be **out-of-band**: return `(preds, n_mols)`, or strip `__`-prefixed keys at every boundary. Proof it is a regression: the shipped `example_smiles.smi` header has no such column. |
| **G2** | A typo in a YAML key produces `ModuleNotFoundError` with no mention of the config file | Both plug-in systems resolve **by convention, not registry**: `importlib.import_module(f'properties.{key}')` then `getattr(module, key)`, and the same for `generators.{name}`. The YAML key must equal the **filename** *and* the **class name**, case-sensitively. (And `input_reader` compares `name.lower() == "release"` for the checkpoint default — so the default lookup is case-*insensitive* while the import is not.) Validate the whole key set at startup, before a 5000-iteration run dies on molecule 1. |
| **G3** | A property class constructs fine in isolation and `AttributeError`s inside `Estimators` | `Property.__init__` unconditionally prints `self.CITATION`, which the ABC never declares. `Vina_Score`, `Prop1`, `Prop2` have none and are therefore unconstructible. Either declare it abstract or default it. |
| **G4** | **Tuning `max_reward` / `reward_hook` / `allowed_threshold_jumps` in the YAML does nothing** | They are accepted into `**kwargs`, discarded, and hard-coded in `Property.__init__` as `15 / 1 / 0.3 / True`. `input_TS.yml` carries 13 lines of comments explaining how to tune them. With `max_reward: 1.0` on every property the config *looks* normalised to 1.0; the actual `Estimators.max_reward` is `15 × Σrew_coeff`. Accept them as named `__init__` params or delete the keys. |
| **G5** | **The `threshold` machinery has no effect on the reward for the properties that are actually configured** | `SAScore`, `QED_Score` and `CHEMBERT_BE` all **override `reward()`** with a continuous transform (`-v/10`, `v`, `-v`). Only properties that *don't* override it (e.g. `Scaffold_Match`) see the base step function. So `threshold`/`threshold_limit`/`threshold_step` and the whole percentile curriculum mutate state that nothing reads. A reimplementation naturally assumes thresholds gate the reward. Decide one model and enforce it. |
| G5b | The base reward is not a ramp | `Property.reward` is a **hard step**: `min_reward` (1) or `max_reward` (15), gated by `sign*value >= sign*threshold` where `sign = np.sign(thresh_step)`. There is no linear ramp, no sigmoid, no [0,1] normalisation. `rew_class: 'soft'` is accepted, validated, and then ignored — the body is `# TO-DO : Implement soft rewards`. |
| G5c | A `visualizer:` section in `input_TS.yml` that elion itself never reads | `visualizer.output_dir`, `visualizer.port` and `visualizer.host` are consumed by the **UI**, not by `elion.py` — they configure where the UI writes TS session state and which address it binds. They live in the engine's yml so one file drives both processes. Elion ignores unknown top-level sections, so a typo here fails silently on the elion side and shows up only in the UI's boot banner. |
| G6 | Rewards from different properties are summed raw and one term dominates | `TOTAL[i] = Σ_p reward_p[i] × rew_coeff_p`, no normalisation, no clamping. With `input_TS.yml`'s weights the objective is `0.9·(−BE) + 0.05·QED + 0.05·(−SA/10)` ⇒ BE contributes ≈ +4…+11 while the other two together span 0.05. And **`SAScore`'s reward is always negative** (`−v/10` ∈ [−1.0, −0.1]) — it can only subtract. Put the terms on comparable scales before weighting them. |
| **G7** | **`n_approved_mols` is permanently 0 and the generator silently never re-trains** | `np.sum(rew_np == estimator.max_reward)` — exact float equality against `Σ rew_coeff × 15`. It works only while every property returns *exactly* `max_reward` and the two sums accumulate in identical dict order. With the continuous `reward()` overrides it is never true, so the elite bucket never grows, the `new_mols_since_last_train > n_best` trigger never fires, and nothing errors. Use `>= max_reward - 1e-9`, or better, a quantile. |
| G8 | A fatal error cannot be caught, and vanishes under `python -S` | `quit(msg)` is used for every fatal path (`utils`, `input_reader`, `Estimators`, `Property.bomb_input`, `AbstractGenerator.bomb_input`). `quit` is injected by `site` — absent in frozen/embedded interpreters — and raises `SystemExit`, which is **not** caught by `except Exception`. That matters here: `TS.py`'s evaluator adapter wraps scoring in `except Exception`, so a `quit()` inside a property kills the whole search instead of NaN-ing one molecule. Raise a real exception. |
| **G9** | **A checkpoint loads cleanly and the model emits fluent nonsense** | The generator's 45-token vocabulary is an **explicit, unsorted list**, and that list order *is* the embedding index map. Regenerating tokens from data (`tokenize(smiles, None)`) returns them **sorted**; if you happen to get 45 characters, every tensor shape matches, `load_state_dict` succeeds, and the model is silently re-indexed. Ship `gen_tokens` beside the checkpoint and assert `all_characters == gen_tokens` at load. |
| G10 | Generated SMILES have a suspiciously low validity rate | `evaluate(...)[1:-1]` strips `'<'` and assumes the last char is `'>'`. When sampling exhausts `predict_len=120` without emitting the end token, the last **chemically meaningful** character is chopped, usually giving an unbalanced-parenthesis string that RDKit rejects. Never errors, just depresses validity. Discard truncated samples: `s = raw[1:]; s if s.endswith('>') else None`. |
| G11 | Two identical runs with every seed fixed produce different results | `generate()` ends with `return list(set(generated))` — Python string hashing is salted per process, so the *order* differs run to run. That order feeds `argsort` tie-breaking in the elite bucket and trajectory order in the policy gradient. Use `list(dict.fromkeys(...))`. (There is no seeding anywhere in the generator subsystem: Python `random` samples training chunks, torch samples tokens, NumPy does argsort tie-breaks, and `PYTHONHASHSEED` reorders the output. All four must be pinned.) |
| **G12** | **The generator never learns to stop emitting invalid SMILES** | Invalid samples are filtered *inside* `generate()`, so they never reach the reward function and the policy gradient sees only valid molecules — zero signal on the invalid rate. The older `reinforcement.py_working` at least resampled until reward ≠ 0. If you want validity pressure, the reward must see the failures. |
| G13 | `generate(n)` hangs forever with a frozen progress bar | The loop is `while total_unique < n_to_generate` with **no attempt cap**. A mode-collapsed policy that emits one molecule (or only invalid strings) wedges the entire run. Budget attempts (e.g. `50·n`) and return short. |
| **G14** | **Raising `n_batch` to `batch_size` OOMs at ~15 GB** | `policy_gradient` accumulates `rl_loss` across **all** trajectories and calls `backward()` once. Peak memory ≈ `n_batch × L × (stack 1.2 MB + GRU activations)`; at the default `n_batch=10, L≈60` that is already ~0.7 GB. This is why the RL update uses 10 molecules while `batch_size: 200` in the YAML governs only *evaluation* batches — a genuine footgun, since the YAML reads as if 200 trajectories per gradient step. Call `backward()` per trajectory and accumulate gradients. |
| G15 | The tail of long SMILES never learns | `discounted_reward` starts at `R_i` for token 1 and is multiplied by `γ=0.97` each step, so weight *decays forward*: position 48 carries 0.232, position 98 carries 0.051. The reward is a property of the *finished* molecule, so standard REINFORCE weights all tokens equally (or discounts backward, `γ^(T−p)`). Upstream ReLeaSE does the same thing — keep it only if you are reproducing. |
| G16 | The policy gradient is biased and you cannot see it | `generate()` returns RDKit **canonical** SMILES; `policy_gradient` re-wraps *that* string as the trajectory. The token sequence the network actually sampled is discarded, so `log π(canonical) ≠ log π(sampled)`. The `std_smiles` flag that used to control this is passed `True` and never read. |
| G17 | Every sampled molecule is reinforced, good or bad | `Property.min_reward = 1`, so the worst molecule still scores `Σ rew_coeff × 1 > 0`, and there is no baseline. Without an advantage the update raises the likelihood of *every* sample, differing only in magnitude — the classic high-variance REINFORCE failure. Subtract a moving-average baseline. |
| G18 | `n_best: 20` in the YAML validates and does nothing | `self.n_best = self.batch_size` is assigned first; the YAML value is read into a **local** `n_best`, range-checked, and dropped. The retrain trigger then needs 200 approved molecules instead of 20. Assign after the check. |
| G19 | The shipped default config crashes before doing any work | `print(f"SMILES seed: {self.seed_smi:<63s}")` with `seed_smi` defaulting to `False` → `TypeError: unsupported format string passed to bool.__format__`. Two lines later, `if seed_smi is not None` is the wrong falsy test for a `False` default → `'<' + False + '>'`. Use `str(...)` and truthiness. |
| G20 | The history CSV is off by one iteration | `history[key][reinforcement_iteration - gen_start]` — but `gen_start` was already incremented after the unbiased batch, so index 0 is the *unbiased* row and every subsequent row reports the previous iteration's numbers. Use `history[key][-1]`. |
| **G21** `[TORCH]` | **`n_layers > 1` silently destroys the augmented stack** | `hidden_2_stack = hidden_.squeeze(0)` assumes a single layer. With 2 layers the hidden is `(2,1,H)`, `squeeze(0)` is a no-op, the controls come out `(2,1,3)`, and `F.softmax(..., dim=1)` normalises over the **size-1** dimension — every control becomes exactly `1.0`, i.e. simultaneous full PUSH + POP + NO-OP. No error, garbage memory. Assert `n_layers == 1` whenever `has_stack`. |
| G22 `[TORCH]` | A GPU-saved checkpoint refuses to load on a CPU box | `torch.load(path)` with no `map_location`. Use `map_location='cpu', weights_only=True`. Related: `torch.save(model.state_dict())` **after** an `nn.DataParallel` wrap writes `module.`-prefixed keys that will not load into the bare model — so multi-GPU and single-GPU checkpoints are not interchangeable. Save `model.module.state_dict()` when wrapped. |
| G23 `[TORCH]` | Generation is 2–3× slower than it should be and spikes memory | `evaluate()` builds the full 120-step autograd graph (including 120 fresh `(1,200,1500)` stacks ≈ 144 MB) and discards it, for every molecule. Wrap in `torch.no_grad()`. Related: `change_lr` rebuilds the optimizer **without** `weight_decay` and throws away Adadelta's accumulators — mutate `param_group['lr']` instead. |
| G24 | "Losses" from iteration 50 include everything since process start | `def fit(self, data, n_iterations, all_losses=[], ...)` — a mutable default argument, hit by both call sites. |
| **G25** | **The training corpus silently loads as empty, then dies on an unrelated line** | `GeneratorData(..., cols_to_read=[])` is the default (upstream used `[0]`). With it, the reader returns `[]`, the wrap loop never runs, `file_len == 0`, and **no error** — until the first `random.randint(0, -1)` → `ValueError: empty range`. The concrete generator passes `cols_to_read=[0]` explicitly, which is the only reason anything works. Key insight for a rebuild: the corpus is **not needed** for generation or biasing (tokens are passed explicitly and the elite bucket replaces `gen_data.file` wholesale) — a one-line tab-delimited file satisfies the loader. |
| G26 `[TORCH]` | `RuntimeError: Expected all tensors to be on the same device` after forcing CPU | The data object and the model **auto-detect CUDA independently**. Forcing the model to CPU on a CUDA box leaves `char_tensor` returning CUDA tensors. Pass one device down from a single source. |
| **G27** | **The Thompson-Sampling disallow sentinels are documented backwards** | The inline comments on `Empty = -1` / `To_Fill = None` are swapped; the authoritative description is the `update()` docstring. Real semantics: `To_Fill = None` marks **the one slot currently being chosen** (exactly one required, enforced); `Empty = -1` means **not yet chosen** (wildcard); an `int ≥ 0` is a committed reagent. The mask dict is keyed on the full selection tuple and its value is the set of indices forbidden **at the `None` position**. Get this backwards and the sampler either forbids everything or nothing. |
| **G28** | **The search dies with `ValueError: All-NaN slice encountered` when the space runs out** | `np.nanargmax` over a fully-disallowed column raises. There is **no exhaustion guard on the search path**: `_n_sampled` is only maintained by the standalone `sample()` and by reagent retirement, never by `update()`, so the tracker's own guard never fires. Check `len(mask) == n_reagents_at_site` before picking, and end the run cleanly. |
| **G29** | **A docking failure becomes the global optimum** | Nothing in the TS package inverts a sign — maximise-vs-minimise is *entirely* the evaluator's problem. The shipped `FredEvaluator` returns a raw docking score (**lower is better**) *and* uses `score = 1000.0` as its failure sentinel; run it with the shipped `"ts_mode": "maximize"` and every failed docking is accepted (`np.isfinite(1000.0)` is `True`) and ranks first. `DBEvaluator` shows the correct pattern: map the failure sentinel to `np.nan`. In Elion the negation lives in the property (`rew = -value  # the less the better`), applied **once** — so `maximize` is right, but only by that convention. |
| G30 | Most of the compute is thrown away | Every warm-up combination is committed to the disallow tracker (so the search can never revisit it) and then `warm_up()`'s return value is **discarded** by the adapter. With `num_warmup_trials × Σ|reagents|` evaluations that can be the majority of the run — including the "top score found during warmup" the log prints. Merge warm-up results into the output. |
| G31 | One unparseable reagent SMILES kills a 5000-iteration run | `RunReactants` sits **outside** the `try` block. A bad SMILES makes `MolFromSmiles` return `None` and RDKit throws a Boost `ArgumentError` that propagates. Also `prod[0][0]` keeps only match 0, product 0 — a reagent with two matching sites silently loses the others, and *which* one survives depends on RDKit's substructure-match ordering. |
| G32 | The reagent files load with SMILES and names swapped | Column order is keyed on `db_name`: `SYNPLE` → `name,smiles,price`; `eXplore` → `smiles,name,price`. Wrong value ⇒ all-`None` mols; unknown value ⇒ `UnboundLocalError`. And the **reactant-template order is load-bearing**: `reagent_file_list[0]` must match reactant template 0 of the SMARTS. If the pair order is swapped, `RunReactants` returns `()` for every pair and warm-up dies later at `np.mean([])`. Assert `rxn.GetNumReactantTemplates() == len(reagent_file_list)` and that one trial pair yields ≥ 1 product, *before* the run. |
| G33 | Retiring a dead reagent doesn't actually retire it | `_retire_synthon_mask` never resets a deeper slot back to `Empty` after its recursive sweep returns, so for `n_cycles ≥ 3` the reagent stays selectable in almost every pairing. Two-component reactions are unaffected, which is why it survives. |
| G34 | Memory grows unbounded on long runs | `return self._disallow_mask[tuple(selection)]` is a `defaultdict` read — every pattern ever *queried* inserts an empty set. Use `.get(key, EMPTY_SET)`. Relatedly, `np.prod(reagent_counts)` for the total space is int64 and **wraps silently** past 9.2e18, corrupting the exhaustion roll-up; `math.prod` (used elsewhere in the same subsystem) is exact. Two disagreeing sources of truth for one quantity. |
| **G35** `[TORCH]` | **Every reported loss and "RMSE" from the transformer fine-tune is wrong** | `output[:,0]` is `(B,)` and the label is `(B,1)`; `MSELoss` **broadcasts to `(B,B)`** — the mean of the full pairwise error matrix. PyTorch emits only a `UserWarning`, the loss still decreases, and the model still "trains". Downstream, `reduction='none'` yields `(B,B)` matrices that `torch.cat` only accepts when every split size is an exact multiple of the batch size, and the code then indexes `[0]` — a workaround baked into the reported metrics. Separately, `sqrt(sq_err)` averaged is **MAE**, not RMSE. Assert `output.shape == target.shape` before every criterion call. |
| G36 | The golden regression fixture has one fewer row than its input | `csv.Sniffer().has_header(f.read(1024))` on a headerless SMILES file guesses `True`, so `read_csv(header=0)` **eats molecule 1** — the shipped `validation_100.smi` has 100 rows and `predictions_orig.dat` has 99. Detect the header explicitly (`'SMILES' in first_line.upper()`), and never trust a fixture whose row count you haven't checked. |
| G37 | Inference is stochastic after someone raises dropout | `model.eval()` is never called on the inference path. Harmless *only* because `dropout_rate=0`. |
| **G38** | **The 3-D CNN's predictions degrade silently when a feature list is reordered** | The voxel tensor's 24 channels are `[protein 11 arpeggio bits | protein excluded-volume | ligand 11 bits | ligand excluded-volume]`, and the network's first layer is a **1×1×1 `Conv3d(24→32)`** — a per-voxel linear map over channels. Permuting the atom-type tuple permutes input channels with **no shape change and no error**. Snapshot the exact 11-tuple in a test. (Beware: the same config file holds a *different* 10-element `FEATURE_SIFT` list describing *contacts*, not atom types. Do not conflate them.) |
| G39 | The 3-D pipeline's augmentation is not what the weights were trained on | Rotation/translation is applied to **atomic coordinates**, before typing and gridding — never to the voxel tensor. The original Chimera script transformed a live model without resetting, so its 36 samples **accumulate** into a random walk; the NumPy port re-reads pristine coordinates each iteration, giving 36 *independent* samples. That is a real distribution shift with respect to the checkpoint. |
| G40 | A tool writes its output somewhere other than where you told it | The atom-typer builds `os.path.join(outdir, pdb_filename.replace('.pdb','.atomtypes'))` where `pdb_filename` is **absolute** — so `join` discards `outdir` and the file lands next to the input. The orchestrator documents this and moves the file itself. Whenever `os.path.join` takes a caller-supplied path, normalise it to a basename first. |
| G41 | Protein atoms silently vanish from the typed output | One of the four atom-typer variants shadows the 40-entry protein lookup with a *local* 9-entry ligand-only dict; every protein atom then falls through to the `"@@"` unmapped code and is dropped at write time, producing ligand-only files. Pick one variant, and assert both a protein and a ligand atom survive typing. |
| G42 `[PY2→3]` | Half a subsystem won't even import, the other half returns garbage | The unmigrated files are still present beside the migrated ones. The five breakages worth knowing: tuple-parameter unpacking in `def f(a, (b,c)):` is a **SyntaxError**; `np.asarray(zip(...))` gives a 0-d object array; `map(...)` objects have no `+`; `open(..., 'rb')` then `float(line[39:49])` is a bytes/str `TypeError`; and `p.map_async(...)` whose result is never retrieved **swallows every worker exception**. Also `collections.Iterable` (removed in 3.10) and `np.NaN` (removed in NumPy 2.0) are still live. |
| G43 | A batch run reports success having computed nothing | Shipped cluster logs show `ls: cannot access .../pdbqt: No such file or directory` followed by `Done.` — no `set -e`, no exit-code check, and the real output redirected to a separate file. Worse, each batch writes the **same** output CSV name, so only the last batch's predictions survive; and the cleanup step `rm -rf`s every intermediate unconditionally. Treat "job completed" as meaningless; assert `len(outputs) == len(inputs)`. |
| **G44** | **A vendored `.patch` file is reversed** | Its header is `--- modified` / `+++ orig` (produced by `diff modified orig`), so `-` lines are the *local* changes and `+` lines are upstream. Applying it forward **reverts** the fixes. Always read the header before applying a vendored patch, and prefer a real diff against the retained `*_orig` copy. |
| G45 | Protonation returns SMILES that RDKit refuses to parse | The vendored protonator appends `"\t" + tag` to every emitted SMILES; its own `main()` strips it with `split("\t")[0]`, but a library caller that iterates the generator directly gets `"CCO\t"` and `MolFromSmiles` returns `None`. Removing the tag is the single most load-bearing local modification — and it silently makes `label_states=True` a no-op. (The same patch exists because the module used to `parse_args()` on **the host program's** `sys.argv` at import, and because its stderr-suppression trick calls `sys.stderr.fileno()`, which does not exist under an IPython kernel.) |
| G46 `[RDKIT]` | An options object is built, passed, and ignored | `df['col'].apply(GetStereoisomerCount, opts)` — `Series.apply`'s second **positional** parameter is `convert_dtype`, not a forwarded argument. The `maxIsomers` cap silently never applies. Use `apply(lambda m: f(m, opts))`. |
| G47 | Two columns differ by one space and only one is the dedup key | The preprocessing pipeline creates **`'InChI Key'`** (with a space) and dedups on it; two other modules create **`'InChIKey'`** (no space). Run them together and the frame carries both, only one of which anything looks at. |
| G48 | The docstring says tautomers are canonicalised; they are not | `standardize_molecule(mol, canonicalize_tautomer=False)` and the only caller has no way to pass the flag. Combined with the `/FixedH` InChI option (which *guarantees* tautomers get different keys), tautomer pairs survive dedup as distinct compounds. |
| G49 | A "cleaning" pipeline drops molecules with no record | Failures are silent at five separate points: an element whitelist that predates metal disconnection; `standardize_molecule` whose empty-molecule guard is commented out; `EmbedMolecule`'s `-1` return assigned to a variable named `scratch` and never checked; a bare `except: continue` around a block that references an un-imported module; and a `how='inner'` merge. Count rows at every hop. |
| G50 | A helper that skips 100 % of its input and returns empty, with no error | `if line.startswith("#") or "Smiles" in line or "SMILES" or "smiles" in line:` — the bare `"SMILES"` is a **truthy constant**, so the whole chain is always `True` and every line is `continue`d. The same function's headerless branch calls `readline()` inside a loop over `readlines()`, reading from an exhausted handle. Two bugs, one function, always returns `[]`. |
| G51 | The SA-score cache silently looks for `None.pkl.gz` | `if name is None: name == 'fpscores'` — `==` instead of `=`. The default-argument path works; passing `None` explicitly does not. (The path itself is the one thing in the tree done right: `op.join(op.dirname(__file__), name)`, so it is cwd-independent. Almost every other path in the config is cwd-relative and breaks the moment you run from anywhere but the package directory.) |
| G52 | An `import` resolves to a different file depending on how you launched | `__init__.py` does `sys.path.append(parent)` — **append**, not `insert(0, …)` — and there are **three** `utils.py` in the tree with incompatible APIs (two of them define a different `read_smi_file`). Meanwhile the top-level module uses flat `import utils` while a submodule uses `from elion import utils`; run `python elion.py` from inside the package and the latter resolves to *the entry-point script itself*, working only because it happens to bind the name `utils`. Use absolute package imports throughout and install the package. |
| G53 | Top-level YAML sections are silently dropped | Only `control`, `generator` and `reward_function` are read. The shipped TS config has an `evaluator_class_name` key and **four whole pipeline sections** (`smiles_to_pdb`, `pdb_to_pdbqt`, `split_pdbqt`, `vina_docking`) that never reach the config object. Reject unknown top-level keys loudly. |
| G54 | The example config destroys the example fixture | `smiles_file` is an **input** for `run_type: calculate_properties` and an **output** for `run_type: generate`. Running the shipped example in `generate` mode overwrites the shipped 200-molecule reference file. Separate the two keys. |
| G55 | The docking property computes nothing | `Vina_Score.predict` returns the literal `30` for every molecule. Real docking happens in an external pipeline configured in the YAML sections that `input_reader` drops (G53), and the docking signal reaches the model only *indirectly*, through a transformer fine-tuned on Vina labels. Know which numbers in your objective are real. |
| **G56** | **A new predictor that cannot be a `Property`, and no one wrote down why** | GIGN needs 3-D coordinates and a receptor; a `Mol` built from SMILES carries neither. So there is no `properties/Yupu_GIGN.py`, no subclass, no `CITATION` — it is a **subprocess, one pose per call**, driven from a web endpoint. That is the right call, but it means the length/order invariant of §5 does not apply and the caller owns the bookkeeping. If you ever do wrap it, the subclass must return a sentinel (never a short list) for complexes that fail to build a graph, or the length check kills the run. |
| **G57** | **A predicted pK of 37.5 — and the shipped log contains it** | Readout is `global_add_pool` (a **sum** over nodes) into an unbounded linear head, so the score is **extensive in the number of atoms**. Score a 5 Å pocket and you get 4–7; score a whole receptor and you get 37.5 against a label of −9.9. Both wrappers accept `pocket_cutoff <= 0` as "use the full receptor", which reproduces it exactly. Forbid it, and treat the recorded 37.5 as a *negative* control: reproducing it proves you rebuilt the mismatch faithfully, not that the model works. |
| G58 | A cutoff with two different values, one of them latent | `cut_pocket(..., cutoff=10.0)` is the **signature default**; every other site — both preprocessing scripts, the argparse default, the meta fallback, the YAML — says **5**. Both current callers pass it explicitly, so the 10.0 is dormant until someone calls it positionally-short and silently gets a 4–8× larger pocket (and, per G57, an inflated score). |
| G59 | Two `.pyg` graph caches for the same complex, same name, different sizes | The cache key is `{prefix}-{id}_{dis}A` — it encodes **only the edge cutoff**. Not the featurisation version, not `removeHs`, not the pocket-cut method, and not the *pocket* cutoff (which lives inside the pickled `.rdkit` and is invisible to the filename). Change `atom_features` and every existing graph silently keeps the old features. The tree ships three vintages of one complex side by side, and a debug script exists solely to load two of them and print them next to each other. |
| G60 | A missing staging directory produces a confident number from defaults | `meta = json.load(...) if os.path.isfile(meta_path) else {}` — a renamed or stale stage yields an empty dict, so the run uses the default cutoff, the default checkpoint and **no SMILES**, succeeds, and prints a pK. A second wrapper exists specifically because of a production incident where a trailing-underscore rename triggered exactly this; its fix is to take every parameter from argv and never read a meta file back. |
| G61 | Omitting SMILES silently changes the featurisation | A PDB encodes connectivity but not bond order, so without `AssignBondOrdersFromTemplate` **no ring is perceived aromatic**: the aromaticity feature is 0 for every atom and hybridisation shifts SP2→SP3. Training data came from files with real bond orders. One wrapper warns; the other skips silently. Template-assignment failures are swallowed in both. |
| G62 `[TORCH]` | `--device cpu` still crashes at load | The shared checkpoint loader is `model.load_state_dict(torch.load(ckpt))` — **no `map_location`** — and every tensor in the checkpoint is on `cuda:0`. Both wrappers *prefer* that helper and only fall back to a correct load if the *import* fails, which it won't. Note also `os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')`: an inherited empty value wins and silently forces CPU. |
| **G63** | **Wiring the new scorer into the existing reward would maximise weak binders** | GIGN predicts **pK — higher is better**. Every existing reward in this system is written for a Vina-like score where **more negative is better** (`rew = -value`). The wrappers print *both* markers, `GIGN_PRED_PK` and `GIGN_DELTAG = -1.36 × pK`; which one a future property parses determines the sign of the whole objective. Parse `GIGN_DELTAG` to match the house convention, or override `reward()` and say so. |
| G64 | Dead layers in the model definition lock the checkpoint to that exact class | Three `HIL` layers are constructed and never called (their forward lines are commented out) — but they **are in the state dict**, and the loader is `strict=True`. Delete them and the checkpoint stops loading with `Unexpected key(s)`. A second copy of the model class that omits them fails for exactly this reason. |
| G65 | The pinned requirements are archaeology | `requirements.txt` pins torch 1.10 / PyG 2.0.3; the code passes `weights_only=False` (a kwarg that did not exist in 1.10 and is only *needed* on ≥ 2.6), calls `add_safe_globals` (≥ 2.4) with a symbol that only exists in PyG ≥ 2.5, and imports `DataLoader` from both its old and new locations in different files. The `__pycache__` says Python 3.11. Also the `pymol` on PyPI is not PyMOL. Treat the pins as history and re-derive the working set. |
| **G66** | **A warm-up checkpoint is keyed on the reaction name and nothing else** | The resolver globs `<short_name>_*_warmup.json` and takes the newest; the consumer checks `os.path.isfile`. No reaction SMARTS, no reagent-file identity or mtime or hash, no evaluator or reward-function fingerprint — and the stored `rxn_key` is never read back. The very commit that added checkpointing also changed two `rew_coeff` values, i.e. moved the objective while leaving every checkpoint reusable. Store and compare the SMARTS, the reagent file list with (size, mtime), and a hash of the reward block; **refuse on mismatch**. |
| **G67** | **The posterior is restored; the sampling-without-replacement state is not** | Warm-up commits every combination it builds to the disallow tracker — `num_warmup_trials × Σ|reagents|` combinations permanently marked as sampled. The restore path builds **no tracker state at all**, so a restored run re-draws combinations the original warm-up already consumed, re-scores them, and Bayes-updates the same observation into the same reagents **twice**. One-line regression test: compare the tracker's mask size at the start of `search()` for a fresh vs a restored run. |
| **G68** | **Reagents that were retired for being dead come back as the most attractive candidates** | A reagent whose every reaction failed raises during prior initialisation and is retired — so it never emits the per-reagent log line, so it is **absent from the checkpoint**. On restore it falls into the "not found" branch and receives the prior mean with **maximum uncertainty**, and is not re-retired. Thompson sampling is optimistic under uncertainty, so it is drawn preferentially, fails, is skipped by the finite-score filter, never updates, and stays maximally uncertain — a permanent iteration sink. Persist the retired list and replay the retirements. |
| G69 | Two reagents with the same name in different slots collapse | The restore builds `belief_by_name[name] = record` flattened **across components**, discarding the component index the file already contains. Any name present in both slots collapses; the higher index wins and the other slot silently receives the wrong posterior. |
| G70 | The checkpoint is scraped out of stdout, not serialised from the objects | Two regexes parse `%.6f`-formatted log lines. Three consequences: at INFO the per-reagent lines do not exist, so you get a **structurally valid checkpoint with an empty component map** that resets every reagent to the flat prior, indistinguishable from a good one; every value is rounded to 6 decimals; and changing either log format string breaks checkpoint *creation* with no error. There is no pickle here, so no unpickling hazard — the fragility is log-format coupling instead. |
| G71 | "Clear the warm-up cache" clears one file | The clear endpoint deletes `_warmup_cache_path(k)`, which resolves to the **newest** match. With several timestamped checkpoints, one click silently promotes the next-oldest. There is no force-fresh flag anywhere. |
| **G72** | **A long run slows to a crawl at ~iteration 4000, and the cause is log volume** | Under DEBUG the sampler emitted ~10 lines per iteration that the dashboard never parses. Once the browser consumer lags, the stdout pipe backs up, `write()` blocks, and the run stalls. The fix was to **delete the unparsed lines** — keeping only the three the UI actually consumes — not to make the sampler faster. A previous hypothesis (batch-of-1 underutilisation) produced a whole batched-TS code path that was then abandoned; it survives in a `_bkp` file and is the more interesting artifact, because its diagnosis was wrong. Instrument to a **separate file**, never the measured pipe. |
| **G73** | **Halogenated ligands produced no atom types at all** | The ligand element was read as `atom.name[:2]` raw, so a PDB atom named `CL7` yielded `"CL"`, which is not a key in a table written with chemical capitalisation (`"Cl"`) — an uncaught `KeyError` that aborted the typer, so any ligand containing Cl or Br produced **no `.atomtypes` file**. The fix is `.strip()` + `.capitalize()`/`.upper()`, and it also removes a `try/except` that had been returning an unbound local. Grids built before and after are not interchangeable for such ligands. |
| **G74** | **A leftover single pose hijacks a batch screening run** | The new dispatcher tests the *pose* condition first: "does `Dataset_VS/*/*_complex.pdb` exist?" Pose runs persist exactly that file into the same directory a per-target batch script uses as its scratch root. So once any pose has been scored there, a later batch invocation takes the pose branch, scores **only the leftover pose**, exits 0, and writes a one-row CSV. Order the branch on the *requested* mode, not on what happens to be on disk. |
| G75 | Re-scoring a different pose under the same name returns the previous score | The grid builder deletes from its work list any entry whose `.npz` already exists. The pose output directory is persistent and pose names are user-supplied, so the inputs refresh (atom types are overwritten, the complex is re-copied) while the **tensor does not**. Add a force flag, or key the cache on the input hash. |
| G76 | A driver that calls three scripts which do not exist, and reports success | The base orchestrator invokes three shell stages absent from the tree. With no `set -e` anywhere and no exit-code checks, it runs to completion producing zero atom types and zero grids, and only the final inference step fails — on an empty directory. Only the per-target variants (which route through the Python pipeline) work. Do not resurrect the base script. Related: non-`vs` test types are a **silent no-op** in all three variants, and two shipped dataset entries declare exactly those types. |
| **G77** | **The pose path and the batch path disagree about receptor hydrogens** | The pose caller reimplements the complex-assembly stage but filters the receptor by record type only. The real stage additionally strips waters, non-`A` altlocs, blanks the altloc column, and **drops protein hydrogens**. Retained hydrogens flip the typer's `input_has_hydrogens` flag, which suppresses its own hydrogen addition and therefore changes the interaction bits. Same weights, different features — and no error. Altlocs are the harder half: duplicate serials raise inside the typer. |
| G78 | The scratch cleanup was removed, and that is now load-bearing | Three `rm -rf` lines were commented out, so per-run temp trees accumulate forever. This is not merely untidy: a downstream endpoint **globs those directories** to recover cached atom types. Deleting them re-breaks a feature. Document the dependency or make the cache explicit. |
| G79 | An inference batch size changed by deletion | Dropping `--batch_size=128` from the command line silently promoted the config default of 256. No numerics change (no BatchNorm, dropout 0), but it is an 8× memory jump on the CPU fallback path for a workload that was already CPU-bound. |
| **G80** | **The voxel channel semantics did not change — and that is the claim to verify first** | The pipeline consolidation merged two grid builders into one file; every function that touches the tensor is byte-identical modulo a removed `print`. Channel order, grid geometry, KD-tree radii, the occupancy formula and the output directory name all match. **Existing model weights remain valid.** Verify it the cheap way: run old and new builders over the same atom-type file into clean directories and assert the arrays are *bitwise* identical — the non-augmented path is fully deterministic. |

---

## 1. Architecture & state model

**One CLI, one config dict, two plug-in planes.**

```
elion.py (CLI)
  └─ input_reader.read_input_file(yml)  →  config
       config['Control']          run-loop knobs
       config['Generator']        → Generator(cfg).generator   (generators/<Name>.py, class <Name>)
       config['Reward_function']  → Estimators(cfg)            (properties/<Prop>.py, class <Prop>)
       config['elion_root_dir']   written, never read
```

- `config` has **exactly four** possible top-level keys. Note the case flip: YAML sections are lowercase
  (`control`, `generator`, `reward_function`); config keys are Capitalised.
- `run_type` ∈ `{calculate_properties, generate, bias_generator, post_process}`; anything else raises
  `ValueError`. `post_process` is `pass` — accepted by the dispatcher, silently does nothing.
- **`elion.py` owns no loop.** `bias_generator` is three lines: build the generator, build the estimator,
  call `generator.bias_generator(config['Control'], estimator)`. The entire RL/search loop lives inside
  the generator. This is the single most important structural fact: *the generator drives, the estimator
  is a service.*
- The two plug-in planes are symmetric and both resolve by naming convention (G2). Neither validates that
  the loaded class implements the expected interface.
- **State that is mutated in place and shared:** the single `Estimators` object is handed to the generator
  and its `Property` objects carry live `threshold` / `converged` state that the curriculum advances
  during the run. Both generators share one instance deliberately. Anything that snapshots a property's
  threshold must copy it.

**Build order.** input reader → property contract → estimator → one trivial property → the generator
contract → one generator → then the expensive predictors. Each verifies before the next.

---

## 2. Input reader — the YAML schema

**Goal.** Parse YAML into `config`, apply `Control` defaults and one generator checkpoint default, and
pass everything else through verbatim ("delegates error handling to the class being loaded").

**Approach.** `yaml.safe_load` → echo the raw input between two `"="*60` rules → set `elion_root_dir`
→ build the `ctrl` defaults dict then blanket-override from `cfg_input['control']` (**no whitelist, no
type coercion, no validation**) → restart handling → generator `initial_state` default → straight
assignment of `reward_function` (**the same object, not a copy**).

**`control:` — the only section with defaults**

| key | default | consumed by |
|---|---|---|
| `history_file` | `'biasing_history.csv'` | restart logic, RL driver |
| `n_iterations` | `1_000` | **nothing — dead** |
| `max_iter` | `1_000` | **nothing — dead** (the shipped example sets it anyway) |
| `gen_start` | `0` | **nothing — dead** (the driver recomputes its own) |
| `restart` | `False` | restart logic |
| `verbosity` | `0` | CLI, RL driver |
| `run_type` | *none* | dispatcher |
| `smiles_file` | *none* | input **or** output depending on `run_type` (G54) |
| `output_smi_file` | *none* | `calculate_properties` — **`KeyError` out of the box**, in neither shipped YAML |
| `comment` | *none* | RL driver banner — `KeyError` if absent |

**`generator:`** — passed through untouched except `initial_state`, which is auto-filled only when absent,
keyed on `name.lower()`. Note the semantics flip: the auto-default is an **absolute** `Path` built from
the module dir; specifying "the same" value in YAML makes it a **relative string** resolved against cwd.

**`reward_function:`** — a mapping of `<PropertyName>: {kwargs}`; see §4 for the kwargs.

**Gotchas.** G53 (unknown top-level keys dropped), G54, plus:

- **No type coercion anywhere.** `verbosity: "1"` from YAML gives `"1" > 0` → `TypeError`. Contrast
  `Property.__init__`, which *does* `float()`-coerce.
- **Restart is implemented twice** and only the second copy is used: the reader computes `gen_start` from
  the history file, and the RL driver recomputes it from the same file and never reads the config value.
  Two implementations that can drift (one uses `quit()`, the other `assert`).
- The `verbosity > 2` debug block is dead on arrival: `pprint` is imported only under `__main__`
  (`NameError`), and the line is `f"{'FINAL CONFIGURATION'}:^80s"` — the format spec is **outside** the
  braces, so it prints the literal `:^80s`.

**Verify.** `read_input_file(example)['Control']` must have exactly the 9 expected keys with `max_iter`
overridden and `n_iterations` surviving as a default. `set(config) == {'elion_root_dir','Control','Generator','Reward_function'}`
— which also documents that the pipeline sections were dropped.

---

## 3. The generator contract

**Goal.** Turn `config['Generator']['name']` into a live object with two methods.

**Factory** (3 lines): `importlib.import_module(f'generators.{name}')` → `getattr(module, name)` →
`cls(generator_properties)`. `Generator` is a *wrapper*; callers use `Generator(cfg).generator`.

**Subclass contract**

| Member | In the ABC | Actually required |
|---|---|---|
| `generate_mols(self)` | `@abstractmethod` | must return `list[rdkit.Chem.Mol]` — the CLI calls `Chem.MolToSmiles` on each |
| `bias_generator(self, prop_values, **kwargs)` | `@abstractmethod` | **really `(self, ctrl_opts: dict, estimator: Estimators)`** |
| `generate_smis(self)` | not in the ABC | required de facto by both implementations |
| `bomb_input(self, generator, msg)` | concrete helper | prints and `quit()`s (G8) |

**The ABC's `bias_generator` signature is a lie** — Python's ABC enforces only the *name*. Write the real
signature down, because a fresh implementation will follow the ABC and break at call time, deep inside
the CLI.

**Estimator surface a generator may use.** `estimate_properties(mols) -> dict`,
`estimate_rewards(pred) -> dict` (adds `'TOTAL'`), `smiles_reward_pipeline(smis, kwargs) -> list[float]`,
`check_and_adjust_thresholds(pred)`, `.properties` (name → `Property`), `.max_reward`, `.all_converged`.

**Return values are discarded.** One implementation returns `None`, the other a DataFrame; the CLI throws
both away. If you want the results, write them inside `bias_generator`.

**Verify.** `AbstractGenerator.__abstractmethods__ == frozenset({'generate_mols','bias_generator'})`;
`isinstance(Generator(cfg).generator, AbstractGenerator)`.

---

## 4. The property contract

**Goal.** A pluggable scorer: raw value → reward, with an optional self-tightening threshold.

**Must implement**

1. `predict(self, mols, **kwargs)` — the only `@abstractmethod`. Receives a **list of RDKit `Mol`** (never
   SMILES; properties that need SMILES convert internally) and returns `list[float]` of the **same length
   and order**. This index alignment is the load-bearing invariant of the whole system and nothing checks
   it (see G1's neighbourhood and §5).
2. A class attribute **`CITATION`** — not declared in the ABC but dereferenced unconditionally in
   `__init__` (G3).
3. Optionally `__init__(self, prop_name, **kwargs)` calling `super()`, and optionally an override of
   `reward(self, prop_values, **kwargs)`.

There is **no `value()` and no `score()`** anywhere; the split is `predict()` → raw physical value,
`reward()` → optimiser scalar.

**Constructor kwargs** (all `float()`-coerced where numeric)

| key | default | notes |
|---|---|---|
| `rew_coeff` | `1.0` | **forced to `0.0` when `optimize` is false** |
| `rew_class` | `'hard'` | `'soft'` validated then ignored (G5b) |
| `rew_acc` | `None` | validated `0 < x ≤ 1` only when `rew_class == 'soft'` |
| `optimize` | `False` | |
| `threshold` | `0.0` | |
| `threshold_limit` | `None` | required if `optimize`; must differ from `threshold` |
| `threshold_step` | `None` | required if `optimize`; ≠ 0 and same sign as `(limit − initial)` |

Direction is inferred from the sign of `threshold_step` and only exists when `optimize` is true.
`optimize: False` zeroes the property **twice** (coeff forced to 0 *and* rewards returned as zeros) — the
property is still *predicted*, so you pay the cost and get nothing.

**The base reward** (a step, not a ramp — G5b):

```python
sign = np.sign(self.thresh_step if self.optimize else self.threshold)
rew  = self.min_reward                             # 1
if (sign * value) >= (sign * self.threshold):
    rew = self.max_reward                          # 15
```

The `sign` trick makes it `value >= threshold` for increasing properties and `value <= threshold` for
decreasing ones. **Set `threshold_step` positive on a negative-is-better property and the comparison
silently inverts** — this is the one place the sign convention can bite inside the framework rather than
inside an evaluator (cf. G29).

**The threshold curriculum** (`check_and_adjust_property_threshold`, only when `optimize`):

```
increasing:  above = mean(values > threshold)
             if above > reward_hook and threshold < limit:
                 threshold = min(limit, percentile(values, 100 - reward_hook*100, method='higher'))
decreasing:  mirror with <, max(), percentile(values, reward_hook*100, method='lower')
converged:   threshold crosses the limit → clamp + converged = True
```

With `reward_hook = 0.3` that is the 70th / 30th percentile. When jumps are disabled it instead does
`threshold += thresh_step`. Both branches have a "step back" path for when too few molecules clear an
already-at-limit threshold. **This entire mechanism is dead for any property that overrides `reward()`**
(G5).

**Verify.** With `optimize=True, threshold=8.0, threshold_step=-0.1`: `reward([3.0]) == [15]` and
`reward([9.0]) == [1]`; the output must only ever be in `{1, 15}` — anything else means a subclass
override is in play. Monotonicity: a decreasing property's threshold must never increase across calls.

---

## 5. Estimators — batch scoring and the reward sum

**Goal.** Instantiate every configured property, run each over a batch, and assemble a weighted total.

**Discovery** is dynamic import by convention (G2). `properties/__init__.py` is empty; the package root
must be importable, which is what the `sys.path` hack in `elion/__init__.py` is for (G52).

**Batch run.** One call per property over the whole list; no parallelism, no chunking, **no invalid-mol
filtering**. `pred[prop] = cls.predict(mols)` for each, plus the `__n_mols__` sentinel (G1).

**Reward assembly.** For each property: `optimize` → `cls.reward(values)` after a strict length check that
`quit()`s on mismatch (G8); else `[0.0]*n`. Then:

```
TOTAL[i] = Σ_p  reward_p[i] × rew_coeff_p          # optimized properties only
max_reward = Σ_p rew_coeff_p × 15                  # the hard-coded per-property max
```

Plain weighted linear sum. **Not multiplicative, not a geometric mean, no normalisation, no clamping**
(G6).

**Three independent ordering contracts, none asserted**

1. Each `predict()` must return values **in input order**. True for the RDKit properties (index-aligned
   loops); *not* structurally guaranteed for properties that shell out to an external binary and
   reconstruct order from whatever row order it wrote.
2. The output CSV's column order is `predictions.keys()` — i.e. **YAML declaration order**. Reordering the
   YAML silently reorders the columns.
3. Batched evaluators zip a score array against the mol list they were given. Any property that drops or
   adds an element shifts **every subsequent score onto the wrong molecule**; the length check only fires
   on the total count, and only for `optimize: True` properties.

**Invalid molecules are handled six different ways.** `MolFromSmiles` returning `None` is not checked on
the hot path, and each property does something different:

| Property | behaviour on a `None` mol |
|---|---|
| `SAScore` | guarded → worst score `10` |
| `QED_Score` | `except Exception` → **`-1.0`** |
| `Scaffold_Match` | bare `except:` → **`0.0`** (the comment claims `-1`) |
| `Similarity_Score` | bare `except:` → **`0.0`** |
| `CHEMBERT_BE` | `MolToSmiles(None)` **raises** → whole batch dies |
| `ADMET_*` | **raises** → whole batch dies |

Then the sentinels poison everything downstream: batch means include them, **and so do the percentiles the
threshold curriculum uses**, so a batch full of invalid molecules *de-tightens* the curriculum. Decide one
policy — drop-with-a-count, or a documented sentinel excluded from all statistics — and apply it in
`Estimators`, not per property.

**Verify.** `len(predict(mols)) == len(mols)` and `predict(mols)[i] == predict([mols[i]])[0]` for every
property — the single most important invariant in the codebase. Then arithmetic: with weights
`SAScore 0.05 / QED 0.05 / CHEMBERT_BE 0.9` and `(SA=3.0, QED=0.8, BE=−7.5)`,
`TOTAL = 0.05·(−0.3) + 0.05·(0.8) + 0.9·(7.5) = 6.775`. And `max_reward == 15.0` for that config —
asserting that the YAML's `max_reward: 1.0` was ignored (G4).

---

## 6. The shipped properties

| Property | raw value | direction | `reward()` | range |
|---|---|---|---|---|
| `SAScore` | Ertl synthetic accessibility | LOW good | **override** `-v/10` | **[−1.0, −0.1]** — always negative |
| `QED_Score` | RDKit drug-likeness | HIGH good | **override** `v` | [0,1], or `−1.0` invalid |
| `CHEMBERT_BE` | transformer surrogate for a docking score, kcal/mol | **NEGATIVE good** | **override** `-v` | ≈ [+4, +11] |
| `Scaffold_Match` | MCS coverage of a SMARTS scaffold | HIGH good | base step | {1, 15} |
| `Similarity_Score` | Tanimoto vs a reference | HIGH good | base step | {1, 15} |
| `ADMET_Risk`, `Intestinal_Absorption`, `Oral_Bioavailability` | external predictor columns | mixed | base step | {1, 15} |
| `Vina_Score` | **returns the literal `30`** (G55) | — | — | — |
| `Prop1`, `Prop2` | return `10` / `20` — test stubs | — | — | — |

**SAScore** — Ertl fragment contributions. Morgan **radius 2 sparse count** fingerprint; per-fragment
scores from a gzipped pickle keyed by bit ID, **default `−4`** for unseen fragments, then `/= Σcounts`.
Complexity penalties: `size = nAtoms^1.005 − nAtoms`, `log10(nChiral+1)`, `log10(nSpiro+1)`,
`log10(nBridge+1)`, and a macrocycle term for any ring **> 8** atoms — note this last one is
`log10(2)` flat, a deliberate deviation from the paper's `log10(n+1)`. Symmetry term
`log(nAtoms/len(fps))·0.5` when `nAtoms > len(fps)`. Rescale `min=-4.0, max=2.5` →
`11 − (s − min + 1)/(max − min)·9`, a `log` knee above 8, clamp to **[1,10]**. Degenerate cases return 10.
*Two implementation notes worth keeping:* the fingerprint generator is rebuilt **per molecule** instead of
once in `__init__`, and the gzip handle is never closed.

**CHEMBERT_BE** — see §8. **The single negation lives here**, and it is the only sign flip in the entire
objective chain (G29).

**Scaffold_Match** — no fingerprint; it is MCS-based. Scaffold from the **first line only** of a SMARTS
file; `score = FindMCS([scaffold, mol]).numAtoms / scaffold.GetNumAtoms()`. Custom atom comparator
(atomic number must match unless either is a wildcard, then valence/chirality/charge/ring/query checks);
`CompleteRingsOnly` and `RingMatchesRingOnly` on both atoms and bonds; `BondCompare.CompareAny`. **A
`print` loop over ring atoms in `__init__` is load-bearing** — it exists to force RDKit's ring perception,
without which `RingMatchesRingOnly` matching changes. Missing `scaffold_file` fails *late*, at predict
time, with `AttributeError` (there is no `else` branch).

**Similarity_Score** — default RDKit topological FP (2048 bits); `fingerprint: morgan` switches to a
Morgan generator with `radius` (default 2), nBits left at the generator default. Metric is hardcoded
Tanimoto. `region_selector` ∈ {identity, `murcko`, `generic_molecule`, `generic_murcko`}. Note it
references `Chem.Scaffolds.*` without importing the subpackage — any non-default selector raises
`AttributeError`.

**ADMET properties** — shell out to an external predictor with a list-form `subprocess.run`, parse a
named column out of a TSV, and return a sentinel on failure. Three real traps: `check=False` with the
return code **never inspected** (a crash surfaces only as a missing file); **no timeout on any subprocess
call**; and the column-index variable is either unbound or `None` when the header is missing, raising
outside the `except ValueError` that was meant to catch it. Header matching is substring-based, so a wider
column name containing the target string matches first. A value is appended for **every** non-header line
including blank trailing ones, so `len(values) != n_mols` is likely — which then hits the `quit()` in
`Estimators` and kills the process (G8).

**Verify.** SAScore ∈ [1,10] for every valid mol, with anchors from the shipped 200-molecule fixture
(a peracetylated nucleoside = 3.97, a simple aromatic = 2.14). Every `Scaffold_Match` value must be an
exact multiple of `1/scaffold.GetNumAtoms()` — with the shipped 24-atom scaffold the observed set is
`{0,1,2,5,6,7,9,15}/24`, a perfect fixture. `Similarity_Score(reference) == 1.0` exactly. Reward signs:
`CHEMBERT_BE.reward([-9.0]) == [9.0]` (single negation), `SAScore.reward([1.0]) == [-0.1]` (documents that
it is a pure penalty), `QED_Score.reward([0.9]) == [0.9]`.

---

## 7. Utilities and the on-disk formats

**`read_smi_file(path) -> (mols, smis)`.** Skip a line if it starts with `#` or contains `Smiles`/`SMILES`;
delimiter is `,` if the line contains a comma else whitespace; take column 0; `MolFromSmiles`; warn and
drop on `None`. Missing file → `quit()`.

**`save_smi_file(path, smiles, predictions)`** — the canonical output format:

```
SMILES,Name,<prop1>,<prop2>,...
<smiles>,Gen-00001,0.25,3.97,0.59,-5.97
```

Names are `Gen-{i:05d}`, **1-indexed**, 5-digit zero pad; values `%.2f`; no quoting, no index column.
Column order is dict-insertion order (§5, contract 3).

**No canonicalisation and no deduplication anywhere in these helpers.** The reader returns raw input text
while the `generate` path emits `Chem.MolToSmiles` output — so round-tripping a file through the two paths
changes the strings. The only dedup in the whole system is a `set()` in the generator (G11), applied to
raw text rather than canonical SMILES.

**Gotchas.** G1 (the sentinel breaks both live writers), G50 (`read_smi_file_with_properties` always
returns empty), plus: warning messages quote the `enumerate` index over `readlines()`, which is 0-based
and counts skipped comment lines — not the line number a user sees in an editor. Seven of the eleven
helpers are dead code with no callers.

**Verify.** Round-trip: `save_smi_file` then `read_smi_file` returns `len(smis)` molecules; header is
exactly `SMILES,Name,` + property keys; row *i* is named `Gen-{i+1:05d}`. Regression guard for G1:
`assert '__n_mols__' not in open(out).readline()`. Regex every data row against
`^[^,]+,Gen-\d{5}(,-?\d+\.\d{2})+$`.

---

## 8. CHEMBERT — the SMILES transformer surrogate

**Goal.** Regress a docking-like binding energy from SMILES, cheaply enough to sit inside an RL loop. This
is what makes the whole system tractable: real docking is the ground truth, the transformer is the
in-loop proxy trained on it (G55).

**Architecture** — a character-level BERT with an *adjacency* side-channel:

```
Smiles_embedding = Embedding(vocab, 1024, padding_idx=0) + Embedding(256, 1024)
                   + adj_mask.unsqueeze(2) * Adjacency_embedding(adj_mat) broadcast over 256 positions
TransformerEncoder(TransformerEncoderLayer(1024, nhead=16, ff=1024, activation='gelu', dropout=0), nlayers=8)
head = Linear(1024, 1)
prediction = out[:, 0]        # the <start> token is the pooled representation
```

`batch_first` is **not** set, hence the `transpose(1,0)` in and out. Padding mask is `(src == 0)`.
`Adjacency_embedding` maps a `(256,256)` adjacency to one 1024-vector per molecule
(`(A @ weight_h).transpose(1,2) @ weight_a + bias`), gated per position by the atom mask.

**Tokenizer** — character-level. Specials `pad=0, mask=1, unk=2, start=3, end=4`. Two-letter elements are
folded to single characters **before** tokenisation (`Br→R`, `Cl→L`, `Sn→X`, `Na→A`), in that order, and
`R/L/X/A` are themselves real vocabulary tokens — so the substitution is lossy and order-dependent.
Crucially the *original* SMILES is retained for RDKit's adjacency matrix while the substituted string
feeds the tokenizer; **that split is correct and must be preserved**. Sequence is
`[start] + tokens + [end]`, truncated to 256 and zero-padded — with **no warning on truncation**, and the
`[end]` token is dropped along with the tail.

**Vocabulary size is not one number.** Inference builds a 48-token vocab; the fine-tuning script's copy has
**50**. They must each match their own checkpoint's embedding-matrix row count exactly. Adding a token
shifts every index; *re-ordering* one keeps the shape and silently garbles predictions (same failure shape
as G9).

**Checkpoint formats are not interchangeable.** Inference loads a `BERT_base` state dict (keys `bert.*`,
`linear.*`); fine-tuning loads a *pretrained* checkpoint into the bare encoder **before** wrapping. Add
`nn.DataParallel` and every key gains a `module.` prefix (G22).

**Fine-tuning.** `KBinsDiscretizer(3, strategy='kmeans')` on the labels → stratified `train_test_split`
80 %, then the remainder 50/50 → **80/10/10**, dumped as three `.smi` files. `Adam(lr=1e-5)`, `batch 64`,
`MSELoss` (regression) or `BCEWithLogitsLoss`, 15 epochs, 12-hour wall clock, checkpoint on validation
improvement only.

**Gotchas.** G35 (the `(B,)` vs `(B,1)` broadcast — the headline bug; and its two knock-ons: `torch.cat`
requires every split size to be a multiple of the batch size, and the reported "RMSE" is actually MAE),
G36 (Sniffer eats molecule 1), G37 (`model.eval()` never called), G22, plus:

- **dtype asymmetry between train and eval**: training casts the output to `double`, validation and test
  do not, and labels arrive as float64 from NumPy → `RuntimeError: Found dtype Double but expected Float`.
- **The `-m/--model` CLI flag is parsed and thrown away** — the function default is always used, and it is
  a *different* path from the CLI default. Neither exists in the checkout.
- `json.dump` crashes at the very end of a successful regression run because `best_score` is a NumPy
  float32. Only the regression path; the classification metric returns a Python float.
- The output directory is created only under `__main__`, so calling the function as a library raises
  `FileNotFoundError` on the first write.
- **Split leakage, two kinds**: the discretizer is fit on **all** labels before the split (minor — it only
  affects bin edges), and there is **no deduplication on canonical SMILES** before splitting (material —
  for a congeneric series this inflates the test score substantially). Deduplicate, or group-split on
  Murcko scaffold, and fit the discretizer on training labels only.

**Verify.** `len(Vocab()) == <expected>` and it must equal the checkpoint's
`bert.embedding.token.weight.shape[0]`. `predict([m])[0] == predict([m]*16)[0]` — batch-size invariance is
an excellent detector of pooling and mask bugs. All predictions negative (they are Vina-like). Before any
training run, print `output.shape` and `label.shape` at the criterion call — `(64,)` vs `(64,1)` proves
G35 immediately.

---

## 9. ReLeaSE — the stack-augmented generator

**Goal.** A character-level generative RNN with unbounded differentiable memory, so it can learn the
long-range structure SMILES needs (ring closures, bracket matching).

**Architecture** (Joulin & Mikolov stack-augmented RNN): a 1-layer GRU whose input at each step is
`[embedding(token) ‖ stack_top]`, plus two heads off the hidden state that drive a differentiable stack.

```
stack_controls_layer : Linear(H, 3)          # softmax over PUSH, POP, NO-OP  (in that index order)
stack_input_layer    : Linear(H, W)          # tanh
encoder              : Embedding(V, H)
rnn                  : GRU(H + W, H, 1)
decoder              : Linear(H, V)          # raw logits; CrossEntropyLoss consumes them directly
```

**The stack update** — the entire mechanism, in five lines:

```python
stack_down = cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
stack_up   = cat((input_val,        prev_stack[:, :-1]),   dim=1)
new_stack  = a_no_op*prev_stack + a_push*stack_up + a_pop*stack_down
```

Move-to-GPU happens **before** the optimizer is constructed — correct order, easy to get wrong.

**Sampling.** Start token `'<'`, end token `'>'`, `predict_len=120`, free-running (each step is fed its own
sample), `torch.multinomial` over the softmax. **Temperature is not a parameter** — upstream had
`output.div(temperature)`, here it is fixed at 1.0 with no knob.

**Supervised step.** Teacher forcing; hidden and stack re-initialised per sequence; one `loss.backward()`
over the whole sequence (full BPTT, no truncation); returns mean cross-entropy per token.

**The concrete generator** *is* the network (it subclasses the RNN) and adds `generate(n)`: loop sampling
until *n* unique, RDKit-valid, filter-passing **canonical** SMILES. The only filter is `len(s) >= 6`.
Candidates are parsed by RDKit **three times** (once raw, once after canonicalisation, once in the
filter) and `set(generated)` is rebuilt from scratch on every acceptance — O(n²).

**Data.** `GeneratorData` reads a corpus, keeps entries with `len(raw) <= max_len=120`, wraps each as
`'<' + smiles + '>'`, and hands out `(input, target)` pairs as `chunk[:-1]` / `chunk[1:]` — the correct
one-position shift, where the start token is never a target and the end token is never an input.
Consequence: **`'<'` and `'>'` must occupy the vocabulary indices the checkpoint expects** (0 and 1).

**Gotchas.** G9, G10, G11, G12, G13, G21, G22, G23, G24, G25, G26, plus:

- `raise InvalidArgumentError(...)` — a name that **is not defined or imported anywhere**. Passing an
  unsupported layer type raises `NameError` with a useless message.
- `char2idx` is built and never used; `char_tensor` does a `list.index()` scan per character (O(V) × 120 ×
  batch), and a character outside the vocab raises a bare `ValueError: 'X' is not in list` with no context
  — reachable whenever RDKit canonicalisation introduces a symbol the 45-token vocab lacks.
- **Instrumentation is the dominant runtime cost at small scale**: a `grad_norm` line per parameter per
  training step (9 lines/step) and one line per generated token (~24 000 lines per 200-molecule batch),
  none of it gated by `verbosity`. The `verbose` parameter and the class-level `verbosity` attribute are
  both inert.
- The reported per-step entropy is in **nats** while the comment claims bits. Uniform over 45 tokens is
  `ln 45 = 3.807` nats = `log2 45 = 5.49` bits.

**Verify.** Parameter shapes at the shipped config: `encoder [45,1500]`, `stack_controls [3,1500]`,
`stack_input [1500,1500]`, `rnn.weight_ih_l0 [4500,3000]`, `rnn.weight_hh_l0 [4500,1500]`,
`decoder [45,1500]` ⇒ **22 650 048 parameters = 90.60 MB fp32**. `stack_controls.sum() == 1.0` per step.
A fresh untrained model's mean CE per token ≈ `ln 45 = 3.807`; a loaded pretrained checkpoint should sit
well below 1.0 — *sustained 3.8 means the checkpoint did not load or the vocab order is wrong* (G9).
`len(generate(n)) == n`, all unique, all round-tripping through RDKit; validity rate ≈ 90–95 % on a healthy
checkpoint, and 0–20 % is the signature of G10 or G9.

---

## 10. REINFORCE — the policy-gradient update

**Goal.** One gradient step: sample a batch, score it, push up the log-likelihood of each trajectory in
proportion to its reward.

```python
rl_smiles = self.generator.generate(n_batch)          # n_batch = 10
rewards   = self.get_reward(rl_smiles, kwargs)
for ind, smi in enumerate(rl_smiles):
    trajectory = '<' + smi + '>'
    discounted_reward = rewards[ind]
    hidden, stack = init()                            # per trajectory
    for p in range(len(trajectory) - 1):
        output, hidden, stack = generator(traj_input[p], hidden, stack)
        log_probs = F.log_softmax(output, dim=1)
        rl_loss  -= log_probs[0, traj_input[p+1]] * discounted_reward
        discounted_reward *= gamma                    # 0.97 — forward decay
rl_loss = rl_loss / n_batch
rl_loss.backward()
```

In closed form, for trajectory *i* of length `L_i`:

$$\mathcal{L} = -\frac{1}{n_{batch}}\sum_i R_i \sum_{p=0}^{L_i-2} \gamma^{\,p}\, \log \pi_\theta\!\left(x^{(i)}_{p+1} \mid x^{(i)}_{\le p}\right)$$

**No baseline, no advantage, no entropy bonus, no importance ratio.** `grad_clipping` defaults to `None`,
so gradients are never clipped in practice.

**Gotchas.** G14 (whole-batch graph before one `backward()` — the reason `n_batch` is 10 and not 200),
G15 (forward discount), G16 (off-policy: canonical, not sampled, token sequence), G17 (every trajectory
reinforced), plus:

- `rl_loss` starts as the **int** `0`; an empty batch gives `0.0` and `backward()` raises
  `AttributeError`. Latent today only because `generate(n)` either returns *n* or hangs (G13).
- **The RL step and the supervised step share one Adadelta instance.** Its accumulators carry across the
  two objectives, so the RL step's effective learning rate depends on how much supervised fitting just
  happened.
- `char_tensor` can raise mid-batch, **after** `zero_grad()` has already run, if canonicalisation emits a
  character outside the vocab.

**The historical variant is worth reading once.** An older copy of this file sampled **one molecule at a
time and resampled until reward ≠ 0**, and called the reward function with a single SMILES expecting a
scalar back. It is incompatible with the current list-based estimator — but the rejection policy it
implements is the one behavioural idea the live version lost (G12).

**Verify.** `len(rl_smiles) == n_batch`; `total_reward == np.mean(rewards)`; **the returned loss is
negative** when rewards are positive (`rl_loss -= positive`) — a positive value means rewards went
negative. With `gamma=1.0` and constant reward `R`, `rl_loss ≈ -R × mean_total_NLL`.

---

## 11. The RL driver loop

**Goal.** Alternate REINFORCE steps with supervised re-fitting on an elite bucket, advancing the property
thresholds until everything converges.

1. **Banner + history header** — `Iteration,{prop}_thr,{prop}_avg,…` for every property. On restart, read
   the last line of the history file and resume from `int(field0) + 1`.
2. **Unbiased batch** — generate, score, record thresholds and averages, write row 0.
3. **Bucket** — `argsort(-TOTAL)[:n_best]`, collect SMILES and per-property predictions, dump to disk.
4. **Seeding** — if `seed_smi` is set, **every** bucket slot is overwritten with it; otherwise each bucket
   SMILES is wrapped `'<'…'>'`.
5. **Bucket → training data** — `gen_data.file = bucket_smiles`. This *replaces* the corpus in place (and
   is why G25's empty corpus never surfaces here).
6. **Warm fit** — 100 supervised next-token steps on the bucket.
7. **Outer loop** ×`max_iterations`, inner loop ×`n_policy`:
   a. one `policy_gradient` step (10 molecules — G14);
   b. generate a batch, score, count "approved" molecules (G7);
   c. append approved molecules to `gen_data.file`;
   d. when `new_mols_since_last_train > n_best`, re-score **all** of `gen_data.file`, take the top
      `n_best`, force the seed into the last slot, re-fit 100 steps, reset the counter;
   e. `check_and_adjust_thresholds`; break when `all_converged`.
8. Checkpoint on **even** iterations; final batch + final checkpoint at the end.

**Gotchas.** G18, G19 (both of which crash the shipped default config before any work happens), G20, G7
(which makes step 7d unreachable in practice), G1 (which crashes at step 2 — the *first* thing that runs
after the unbiased batch, and so the practical blocker), plus: the average-time report divides accumulated
time by the **absolute** iteration index rather than iterations completed this run, so it is wrong after a
restart.

**Verify.** The history CSV has `1 + 2×len(properties)` columns and `max_iterations + 1` data rows on a
clean run. After the bucket step, `len(gen_data.file) == n_best` and every element matches `^<.*>$`. The
checkpoint directory contains only even indices. With `seed_smi` set, all bucket entries are
byte-identical before the warm fit.

---

## 12. Thompson Sampling — the combinatorial searcher

**Goal.** Search a make-on-demand reaction library (N₁ × N₂ × … reagents) for high-reward products while
evaluating a tiny fraction of it. Each *reagent at each site of diversity* is an arm of a multi-armed
bandit with a Gaussian posterior over "the reward of products containing this reagent"; Thompson Sampling
runs over arms independently per site.

*Provenance note:* this is vendored from an upstream MIT-licensed reference implementation, with local
additions (batched evaluation, an adapter to the shared `Estimators`, heavy DEBUG instrumentation). Two
standalone CLI drivers and their JSON configs are **dead** — they import a module that does not exist —
so the adapter is the only working entry point.

**The posterior is an exact Normal-Normal conjugate update with known observation variance**, not a
running mean/std:

```
μ' = (σ²·x + σ₀²·μ) / (σ² + σ₀²)
σ' = sqrt(σ²·σ₀² / (σ² + σ₀²))
```

The prior is **shared by every reagent** and set at the end of warm-up from the pooled warm-up scores:
`prior_mean = mean(all)`, `prior_std = std(all)`, `known_var = prior_std²`. Because σ starts at σ₀, after
*n* updates `σ² = σ₀²/(n+1)` exactly — uncertainty decays as `1/√(n+1)`, and that is what drives
exploration → exploitation. Reagents buffer their warm-up scores and **replay** them through the real
update when the prior arrives.

**Warm-up.** For every site, every reagent, `num_warmup_trials` times: pin that reagent, fill the other
sites at random (respecting the disallow mask), commit the selection to the tracker, and batch-score.
Selection stays strictly sequential for tracker consistency; only *scoring* is batched. Reagents that
never scored are **retired**. Cost = `num_warmup_trials × Σ|reagents|`, and realised per-reagent counts
are uneven because a reagent also picks up scores whenever it is drawn as a random partner.

**Search.** Per cycle, visit the sites in **random order**; at each site draw one Gaussian sample per arm
(`rng.normal(size=n) * stds + mus`), NaN out the disallowed indices, and take `nanargmax` / `nanargmin` /
a Boltzmann-reweighted pick. Commit, build the product, evaluate, and flush posterior updates in batches.

**Product construction.** `reaction.RunReactants([r.mol for r in selected])`, `prod[0][0]`, `SanitizeMol`,
`MolToSmiles`; the product name is `"_".join(reagent_names)`. Failure → `"FAIL"` / `np.nan`.

**Gotchas.** G27 (sentinels), G28 (exhaustion), G29 (sign), G30 (warm-up discarded), G31 (`RunReactants`
outside the `try`, and dropped products), G32 (column order and reactant order), G33, G34, plus:

- **The disallow mask barely constrains anything except the last-picked site.** `_update` only registers
  *fully committed* patterns, while the first site of each cycle is queried with an all-wildcard pattern
  that only ever receives entries via the exhaustion roll-up. Sampling-without-replacement holds for
  *combinations*, not for *diversity* — TS re-picks its favourite reagents indefinitely.
- **Duplicate products are never deduplicated on the way out.** Different reagent combinations can give
  the same SMILES; every duplicate also re-updates the same reagents' posteriors. Dedup is applied only to
  the printed top-10.
- **Exceptions are used for control flow** in the batch path (`raise AttributeError("use per-mol path")`),
  so *any* `AttributeError` raised inside the evaluator silently re-runs the whole batch per-molecule —
  double-counting the evaluation counter and doubling GPU work, with no log line at all.
- **One bad molecule discards the entire batch**: the adapter's `except Exception` returns
  `[nan] * len(mols)`.
- Boltzmann modes are landmines: the warm-up std is `None` until warm-up finishes (`TypeError` if search
  runs first), and a small std overflows `exp` to `inf` → all-NaN probabilities zeroed without
  renormalising → `probabilities do not sum to 1`.
- The reagent-level DEBUG instrumentation **can never fire**: the logger factory calls `basicConfig`
  without `level=`, leaving root at WARNING, and no logger is threaded down to the reagent objects. Also
  `basicConfig` is a no-op after the first call, so whichever logger is created first wins the filename.
- **A dead config key that reads as live**: the shipped YAML sets `generator.batch_size: 200` *above* the
  `TS:` block, while the adapter reads `generator.TS.batch_size` (default **1**). Two keys, same name,
  200× apart. Easy to "fix" the wrong one.

**The adapter contract.** What Elion hands the vendored package, in order: `ThompsonSampler(mode, db_name,
log_level)` → `set_hide_progress` → `set_evaluator(adapter)` → `read_reagents_csv(files, num_to_select)`
→ `set_reaction(smarts)` → `warm_up(trials, eval_batch_size)` → `search(num_cycles, batch_size)`.
The adapter must expose `counter`, `evaluate(mol) -> float` (returning **`nan`, not raising**, on failure)
and optionally `evaluate_batch(mols) -> list[float]` — same order, same length. Note that `evaluate_batch`
is used by **warm-up only**; the search phase is unbatched, one molecule per call. `search()` returns
`[[score, smiles, name], …]`, one row per iteration whose score was finite.

**Verify (pure, no RDKit).** `DisallowTracker([2,2])`: commit all four combinations, then assert the mask
for the open slot is `{0,1}` (the roll-up fired) and that a fifth pick raises (G28). `[2,2,2]` plus
`retire_one_synthon(0,0)` should mask all four pairings — it does not (G33). Posterior: feed one reagent
`init_given_prior(0.0, 1.0)` and N identical scores; assert `σ² ≈ 1/(N+1)` and `μ → score`. Reaction
sanity **before** any run: `rxn.GetNumReactantTemplates() == len(reagent_file_list)` and one trial pair
yields ≥ 1 product (G32). Then a tiny end-to-end: two 10-line reagent CSVs, `num_warmup_trials=1`,
`num_ts_iterations=20`, a dependency-free evaluator — 100 products, so exhaustion is reachable, which is
the cheapest way to observe G28.

---

## 13. DeepAtom — the 3-D CNN affinity predictor

**Goal.** Given one protein and N docked ligand poses, predict a binding affinity per pose. The model
regresses pKd/pKi; the driver converts with `ΔG = −1.36 × pK`.

**The mechanism.** Build a **32×32×32×24** voxel cube centred on the ligand, where each channel is an
*interaction-propensity* bit from a CREDO-style atom typer, split into protein-side and ligand-side
halves; feed it to a 3-D ShuffleNet-V2 regressor. Test-time augmentation: 36 rotated/translated copies per
pose, predictions averaged.

**Stage 0 — complex assembly.** Strip waters and all altlocs except `'A'`; strip **all** hydrogens; rebuild
`TER` records; renumber ligand serials as `last_protein_serial + old_serial`; set `resName="LIG"` and
**`chainID='y'`**. That single character is the **protein/ligand discriminator for the entire rest of the
pipeline** — every downstream stage keys on `line[0] == 'y'`.

**Stage 0b — augmentation.** 6 rotation angles × 6 random axes = **36 samples**, base angle stepped 60°
across −180…180 with a ±60° jitter, rotating about the ligand **bounding-box midpoint** (not the
centroid), plus a random translation of length `uniform(-1,1)` Å. Applied to **coordinates**, before
typing and gridding (G39). The grid-building variant for augmented samples adds a further independent
±1 Å jitter of the grid centre.

**Stage 1 — atom typing.** OpenBabel adds hydrogens at **pH 7.4**, then typing runs in two passes:
SMARTS-driven for everything, then a **dictionary override for standard residues** — for any standard
residue, all SMARTS-assigned types are discarded and re-assigned from a hand-curated `resname+atomname`
lookup. Ligand atoms keep their SMARTS types. Output is a fixed-width `.atomtypes` record, and **that
layout is the contract** between the typer and the grid maker:

| cols (0-based) | width | content |
|---|---|---|
| `[0:22]` | 22 | atom id; **`[0] == 'y'` ⇔ ligand atom** |
| `[22:24]` | 2 | vdW-radius type code |
| `[24:28]` | 4 | padding |
| `[28:39]` | 11 | the interaction bit string |
| `[39:49]`, `[49:59]`, `[59:69]` | 10 each | x, y, z as `%10.3f` |

The 11 bits, **in exact channel order**: `hbond acceptor · hbond donor · xbond acceptor · weak hbond
acceptor · weak hbond donor · pos ionisable · neg ionisable · hydrophobe · carbonyl oxygen · carbonyl
carbon · aromatic`. Atoms are dropped if they are water or if the type code is unmapped.

**Stage 2 — voxelisation.** Ligand centre = bbox midpoint of the `'y'` lines; box = centre ± 16 Å; voxel
centres via `mgrid` with `num_points = ceil(32/1.0) = 32` ⇒ 32³ centres spanning 31 Å at exactly 1.000 Å
spacing, indexed by a `cKDTree`. Per atom, query neighbours within **2 × vdW** and accumulate

```
occ = 1 - exp( -(r_vdw / dist)**12 )
cube[voxel] = np.maximum(mask_vector * occ, cube[voxel])
```

element-wise max (hence the mode name `pcmax`). The **24-channel `mask_vector`** is:

```
[0:11]  protein interaction bits   (zeros if ligand atom)
[11]    protein excluded volume    (1 protein / 0 ligand)
[12:23] ligand interaction bits    (zeros if protein atom)
[23]    ligand excluded volume     (1 ligand / 0 protein)
```

**Stage 3 — the network.** `(N, 24, 32, 32, 32)` after a `(3,0,1,2)` transpose in the dataset's
`ToTensor`. Input block is `Conv3d(24→32, k=1)` + BN + LeakyReLU + `MaxPool3d(3, s=2, p=1)` (G38), then
three ShuffleNet-V2 stages `[3,4,4]` blocks with the first of each at stride 2, then dropout →
`Conv3d(→2048, k=1)` → `AdaptiveAvgPool3d(2)` → `Linear(2048→1)`. **The output shape differs between
modes**: training gives `(N, 8)`, eval averages the 8 positions and gives `(N, 1)`. Predictions for a
pose's 1 + 36 samples are grouped by PDB code and averaged.

**Gotchas.** G38, G39, G40, G41, G42, G43, plus:

- **A cutoff that appears twice with different multipliers**: the `pcmax` neighbour query uses `2 × vdW`,
  the (unused) `binary` mode uses `1 × vdW`. Switching modes silently halves the smearing radius.
- **argparse `choices` that contradict the shipped defaults** (`default=24, choices=[60,11]`). It works
  only because argparse validates supplied values, not defaults — so passing the default *explicitly* on
  the command line **fails**.
- **The driver reads its first five arguments strictly positionally** (`sys.argv[1].split('=')[-1]`), never
  checking the flag names, while a *second* independent `parse_args()` runs over the same `sys.argv`.
  Reordering the flags silently swaps arguments.
- **Weights are loaded from `os.listdir(model_dir)[0]`** — unsorted, so any stray file changes what you
  load.
- **Underscores in ligand filenames corrupt the augmentation grouping** (`aug_sample.split('_')[0]`), which
  the cluster script papers over with a `rename -- _ - *_*`.
- Normalisation is a one-way trip: enabling it normalises the *label* but nothing de-normalises the
  *output* before the ΔG conversion.
- Multiprocessing pickling shapes the code in three visible ways: a `Manager()` as a **mutable default
  argument evaluated at import time** (so importing the module spawns a manager process — it exists because
  a raw `Lock`/`Value` is not picklable through `Pool.map`); every worker hoisted to module level with an
  explicit docstring saying so; and an `os.chdir` **inside a worker**, making CWD per-process global state
  that every subsequent relative filename depends on.

**Verify.** `np.load(...)['pocket'].shape == (32,32,32,24)`, dtype float32, `0 ≤ v ≤ 1` everywhere (the
occupancy formula is bounded). Recompute the grid bounds independently and assert 1.000 Å spacing.
Channel semantics: both excluded-volume channels non-zero, and channels 0:12 zero wherever channel 23 is
saturated. `.atomtypes` alignment: every line exactly 69 chars, `[28:39]` matches `^[01]{11}$`, and
`line[0]=='y'` ⇔ the atom is a ligand record. Model shapes: `train()` → `(2,8)`, `eval()` → `(2,1)`.
Augmentation: exactly 36 files per complex, same atom count and element sequence as the source, ~0 RMSD
after optimal superposition (rigid transform only). And the count guard for G43:
`len(npz) == n_ligands` and `len(npz_aug) == 36 × n_ligands`.

---

## 14. vsdb — compound-library preparation

**Goal.** Turn a raw SMILES list into a clean, deduplicated, filtered, isomer-expanded, optionally
3-D-embedded library suitable for docking.

**There is no SQL database.** No `sqlite`, no ORM, no DDL. The "database" is an in-memory pandas
DataFrame and the schema is its column set. There are also **no writers** — persistence is the caller's
job.

**The pipeline**, in this exact order:

1. `dropna(subset=['SMILES'])`.
2. **Drop and rebuild the `ROMol` column from SMILES** — a documented workaround for RDKit's
   `AddMoleculeColumnToFrame` losing stereo information.
3. Drop invalid molecules: element whitelist `C N O H S P As Se F Cl Br I`, plus `None` and zero-atom
   rejects.
4. `standardize_molecule` per molecule: `SaltRemover(dontRemoveEverything=True)` → `LargestFragmentChooser`
   → `molvs.Standardizer` (RemoveHs, sanitize, metal disconnect, normalize, reionize, assign stereo) →
   `Uncharger` → `TautomerCanonicalizer` **(disabled — G48)**.
5. Regenerate `SMILES` from the standardized mol.
6. `InChI Key = MolToInchiKey(mol, options="/FixedH")`.
7. Dedup on that key, `keep='first'`; the full duplicate set is returned separately.
8. `reset_index(drop=True)`.

Then `prepare_2D_db` layers on molecular-weight and stereoisomer-count filters and stereoisomer
enumeration (`onlyUnassigned=True`, so molecules that already carry stereo are left alone); `prepare_3D_db`
adds `AddHs` + `EmbedMolecule`.

**Protonation** is delegated to a vendored Dimorphite-DL. Its algorithm, worth restating because the pH
semantics are easy to invert (G-note below):

1. **Neutralise** via a fixed-point loop of 6 SMARTS reactions (terminal `O⁻` gets an H, `N⁺` with an H
   loses it, bridging `O⁻` neutralised, trivalent `N⁺` neutralised, divalent `N⁻` gets an H, bad azide
   rewritten), then sanitize; anything that fails is skipped.
2. **Decide each rule's target state from the pH window**:
   ```
   min_pka = mean - std ;  max_pka = mean + std        # std already scaled by pka_precision
   if min_pka <= max_ph and min_ph <= max_pka:  BOTH
   elif mean > max_ph:                          PROTONATED
   else:                                        DEPROTONATED
   ```
   `pka_precision` is therefore a **number of standard deviations**, not a tolerance.
3. **Find sites, first-match-wins**: rules are tried **in file order**, and every atom of a match is marked
   protected so later, more generic rules cannot re-claim it. This is why the file order matters
   (`Phenyl_carboxyl` before `Carboxyl`, `Phenol` before `Alcohol`).
4. **Enumerate the cartesian product** — each `BOTH` site doubles the output.
5. **Set the charge**, with a `+1` offset for nitrogen (the protonated form is cationic) — **undone for
   rule names prefixed `*`**, which mark the groups whose *acidic* species is neutral (amides, imides,
   sulfonamides, azide, indole/pyrrole, protonated aromatic N). That is the entire meaning of the `*`
   prefix, and it is not documented anywhere else.
6. Dedup on canonical SMILES, drop anything that won't round-trip, and fall back to the input unchanged if
   nothing survives.

**Gotchas.** G44 (the reversed patch), G45 (why the patch exists), G46, G47, G48, G49, plus:

- **pH window semantics invert easily**: `min_ph`/`max_ph` define a *range to enumerate across*, not a
  single pH. Widening it or raising `pka_precision` grows output **multiplicatively** (2^n_BOTH).
- `DataFrame.append` was removed in pandas 2.0 — three call sites raise, and row-by-row appending in a
  loop is O(n²) anyway.
- **A module that only works because someone else imported a submodule**: it uses
  `Chem.EnumerateStereoisomers.*` while importing only `PandasTools, Descriptors`. `rdkit.Chem` gains that
  attribute only once *some* module imports it — which happens to be true in the normal call order and
  false when the module is imported standalone.
- Inconsistent package imports (half relative, half absolute top-level), so the package must be importable
  under **two** names simultaneously.
- The vendored protonator **prints a 6-line citation banner to stdout at import**, unconditionally.
- The element whitelist rejects all metals, boron and silicon **before** metal disconnection gets a chance
  — deliberate, but it silently deletes legitimate boron-containing drugs.

**Verify.** Run the protonator's own `--test` (it exercises every rule at pH −10⁷ and +10⁷, plus the
two-stage phosphate cases) — **but note it opens its rule file by relative path, so you must `cd` into its
directory first**, unlike the production loader which resolves against `__file__`. Then: assert emitted
SMILES contain **no tab character** (G45), all round-trip through RDKit, the rule table has the expected
line count with 5 or 8 tab fields each and every SMARTS parsing, and exactly the expected set of starred
names. Unit-test the pH decision directly at the inclusive boundary. Pipeline invariants:
`df['InChI Key'].is_unique`, `df['ROMol'].notna().all()`, `df['MolWT'].between(min,max).all()`, and
`all(m.GetNumConformers() > 0)` after 3-D preparation — the check the embedding step omits (G49).

---

## 16. ⬡ GIGN — the geometric interaction graph scorer

**Goal.** Score a *posed* protein–ligand complex. The model regresses **pK = −log(Kd/Ki)**, higher is
better; the wrappers also emit `ΔG = −1.36 × pK` for convention compatibility (G63).

**Why it is different from the voxel CNN (§13).** No grid, no box, no resolution, no rotational
augmentation. The input is a **sparse atom graph**, so cost scales with atom count rather than box
volume³, and rotation/translation invariance is **structural, not learned** — geometry enters only as
the scalar `‖pos_i − pos_j‖`. The protein and ligand form **one merged graph** with two explicitly typed
edge sets processed by separate weights, rather than separate channels.

### 16.1 Graph construction

**Pocket.** Whole protein residues with any heavy atom within **5 Å** of any heavy ligand atom; waters
dropped. Note there are **two independent implementations** — a PyMOL `byres … around 5` used at
training time, and a pure-Python heavy-atom reimplementation used at inference — and they are not
identical (G58 is the cutoff discrepancy; the H handling differs too).

**Nodes.** All heavy atoms of (ligand ∪ pocket), **ligand first**, hydrogens removed. A `split` mask
records which is which — and the model never reads it. Ligand-vs-protein identity reaches the network
*only* through which edge set an atom participates in.

**Atom features — exactly 35 dims, and the order is the contract:**

```
[0:10]  element one-hot   C N O S F P Cl Br I + Unknown
[10:17] degree            0..6
[17:24] implicit valence  0..6
[24:29] hybridization     SP SP2 SP3 SP3D SP3D2
[29]    is aromatic
[30:35] total num H       0..4
```

Out-of-set values map to the **last** slot, and only the element list has an explicit `Unknown`
sentinel — so degree 7 encodes as 6 and an unusual hybridisation encodes as SP3D2, silently.

**Edges — two sets, both symmetrised:**

- **intra (covalent):** RDKit bonds only, no distance cutoff, **bond order discarded**; ligand-internal
  and protein-internal bonds pooled into one tensor.
- **inter (non-covalent):** all ligand×pocket pairs with `dist < 5.0` Å, strict. No protein–protein or
  ligand–ligand non-covalent edges, and no stored edge features — the distance is recomputed inside the
  layer from `pos`.

### 16.2 HIL — the heterogeneous interaction layer

A `MessagePassing` layer with `aggr='add'`. Distances are embedded as **Gaussian RBFs**, 9 centres over
[0, 6] Å (σ = 6/9 = 0.667), with no envelope and the unusual `exp(−((d−μ)/σ)²)` form. Per edge set:

```
m_i = Σ_{j∈N(i)}  x_j ⊙ SiLU(W_coord · RBF(‖pos_i − pos_j‖))
```

— an **elementwise gate of the neighbour's features by the distance embedding**; the receiving node's own
features are accepted by `message()` and unused. Then the one line that makes it heterogeneous:

```python
out = mlp_node_cov(x + out_intra) + mlp_node_ncov(x + out_inter)
```

Covalent and non-covalent get their **own** distance encoder *and* their **own** update MLP — four
parameter blocks per layer — the residual is added **before** each MLP and therefore twice, and the two
branches are summed after the nonlinearity rather than concatenated.

**Forward:** `Linear(35→256)+SiLU` → 3 × HIL → `global_add_pool` → 3 × `Linear(256,256)+Dropout+
LeakyReLU+BatchNorm` → `Linear(256,1)`. The head is unbounded, and the pooling is a **sum** — together
that is G57.

**Training:** MSE on pK, `Adam(5e-4, wd 1e-4)`, `ReduceLROnPlateau(patience 40, factor 0.1, min 1e-5)`,
300 epochs, early stop 30, 3 repeats, seed 2024. Note the early-stopping and checkpoint-selection call
passes the **test** loader, not the validation one — test-set leakage into model selection.

**Cached intermediates.** `Pocket_{d}A.pdb` (text), `{id}_{d}A.rdkit` (a **pickle** of a `(ligand,
pocket)` Mol tuple), `Graph_GIGN-{id}_{d}A.pyg` (a `torch.save` of one PyG `Data`). The inference
wrappers **bypass both caches** and build the `Data` in memory. See G59 for why the cache key is not
sufficient.

### 16.3 The integration contract

Out-of-process, **one pose per invocation**, launched from a web endpoint:

```
python <script> --stage <dir> --name <name>          # cwd must be the script's own dir
  reads   <stage>/<name>_meta.json  {smiles, pocket_cutoff, drop_water, model, …}
          <stage>/<name>_{ligand,protein}.pdb
  writes  <stage>/<name>_pocket.pdb
  stdout  GIGN_PRED_PK <float>   GIGN_DELTAG <float>
  stderr  GIGN_ERROR <msg>, exit 1
```

A second, newer wrapper takes every parameter from argv instead (`--lig --rec --smiles --name --workdir
--model --pocket-cutoff --keep-water --dis-threshold --device`) and exists specifically to remove the
silent meta-file fallback (G60). Both share the same pocket-cut, graph-build and load code.

**Gotchas.** G56–G65, plus: four near-duplicate `predict*.py` research scripts differing only in paths
and the dataset module — the one that *ran* is identifiable from the log, and one of the others **deletes
the dataset directory it is about to read**. The dataset module used by the wrappers is the one with the
empty-inter-graph guard; the one the training scripts import crashes when no ligand atom is within 5 Å of
the pocket. The training entry points cannot run at all (two missing packages, and they unpack **three**
return values from a forward that returns **two** — so the checkpoint came from a different class
version, which the wrappers defensively handle).

**Verify.** Shape identity first: `GIGN(35, 256, 3)`, strict load succeeds, `in_features == 35`. Then the
featurisation golden test against the two feature rows printed in the shipped log — that pins the 35-d
layout. Then RBF centres and σ. Then end-to-end against the four (prediction, label) pairs in the log,
**deleting the stale caches first** — and treat the 37.5 row as the negative control (G57). Then the
gotcha assertions: pocket vs full receptor must differ wildly; with vs without SMILES must differ at all
(if identical, bond-order assignment is silently failing); three identical runs must be bit-identical;
CPU parity currently crashes (G62); an inter-graph with the ligand translated 100 Å away must return a
number rather than raise.

---

## 17. ⟲ Warm-up checkpointing for Thompson Sampling

**Goal.** Warm-up costs `num_warmup_trials × Σ|reagents|` evaluations before iteration 0 of the search —
on a real library, tens of thousands of surrogate calls. A checkpoint persists the post-warm-up state so
a rerun of the *same reaction* jumps straight into `search()`.

**Scope, precisely.** It checkpoints **warm-up only**. A restored run always restarts the search from
iteration 0, and a run that *loads* a checkpoint never *writes* one — so posterior improvements learned
during the search are discarded.

**Format.** JSON, `<output_dir>/Warmup_TS/<short_name>_<YYYYMMDD_HHMMSS>_warmup.json`:

```json
{ "rxn_key", "timestamp", "prior_mean", "prior_std", "known_var",
  "components": { "<comp_idx>": [ {"reagent_name","current_mean","current_std",
                                   "known_var","num_scores"}, … ] },
  "n_components", "n_reagents" }
```

Per-reagent posterior mean/std/count — **yes**. Raw warm-up scores — no. Disallow-tracker state — **no**
(G67). Retired reagents — **no** (G68).

**Restore.** Guarded by an env var naming the file; on hit, write the five fields onto each `Reagent`,
set its phase to `search`, clear its buffered scores, restore the pooled `warmup_std`, and **return a
synthetic sentinel row before any of the warm-up body runs**. This is mathematically sufficient for the
posterior — the conjugate update reads only mean, std and known variance — and it correctly skips the
score *replay* because the stored values are already post-replay. What it does **not** recompute is the
prior itself, so if the objective's scale moved, the restored `known_var` (the assumed observation noise,
shared by every reagent) is wrong and every subsequent update is mis-weighted.

**Restore is implemented twice** — inline at the top of `warm_up()`, and again as a monkeypatch module
that wraps `__init__` and assigns an *instance* attribute. The loader wins when present; the inline copy
is the fallback. They will drift. (The loader's comment records why `types.MethodType` was removed: a
double-`self` bug.)

**The launch wrappers.** Machine-generated per reaction, structurally identical, differing only in two
hardcoded path strings. Each forces DEBUG logging **to stdout** (so the dashboard's stream sees the lines
it parses), rewrites `sys.argv`, sets the checkpoint env var, imports the loader **before any engine
module**, and then `exec`s the entry point rather than using `runpy` "to avoid module caching issues" —
which is exactly why the entry point needed a cwd fallback in its path pin. Note the shipped wrappers are
**stale generated artifacts** pointing at temp directories that no longer exist; they are not source.

**The analysis tooling** answers one support question: *why does the dashboard's "eligible reagents"
counter go up as well as down when reagents should only ever retire?* Two simulations drive the real
tracker and show that the displayed number is **conditional on the partner already fixed this
iteration**, while the true retired-set ceiling is monotone. A third script asserts five claims about
that and exits non-zero on failure — including that two tracker instances do not cross-contaminate, i.e.
multiprocessing is not the cause. **The answer was "not a bug in the sampler; the UI is displaying a
conditional quantity."** That is worth keeping: a large fraction of "sampler is broken" reports are
display-semantics reports.

**Gotchas.** G66–G72.

**Verify.** The three analysis scripts are self-verifying with fixed seeds. Then the round-trip: run
fresh at DEBUG, confirm one prior line and one per-reagent line per *surviving* reagent, and assert
`n_reagents == Σ|reagents| − (retired count)` — **a gap here is exactly G68, and the missing names should
all appear in "skipping reagent" lines**. Rerun with the checkpoint and dump
`(name, mean, std, known_var)` for every reagent at the moment `search()` starts: fresh and restored must
match to 1e-6 (the log-format precision). Then the one-liner for G67: compare the tracker's mask size at
that same moment — fresh is large, restored is zero. The cache-key negative tests (change the SMARTS,
change a reward coefficient, swap a reagent file, and confirm the checkpoint is *still* reused) are the
specification for the fix.

---

## 18. The DeepAtom pipeline, consolidated

**Goal.** Replace five shell-orchestrated stages with two Python entry points plus a dispatcher, and add
a single-pose path that skips the batch assembly stages.

**The three shapes now in the tree:**

```
OLD batch   00_dirs.sh → 01_preprocess.py → 02a.sh → 02b_augment.sh → 03a.sh → 03b.sh
            → make_grid_mp.py → make_grid_for_aug_mp.py → test.py
NEW batch   00_dirs.sh → 01_preprocess.py → pipeline_VS.py --stages … → make_grid*.py → test.py
NEW pose    generate_atomtypes.py --stages 3,5 → generate_npz.py --stage non_augmented → test.py
```

`generate_atomtypes.py` is a *second-generation* consolidation — it swallows the earlier
`pipeline_VS.py` (whose worker functions it reproduces byte-identically) plus two upstream stages, and
exposes them as a `--stages 1,2,3,4,5,6` selector. `generate_npz.py` merges the two grid builders behind
a `--stage both|non_augmented|augmented` flag. `deepatom_score_pipeline.sh` is a **dispatcher**, not a
replacement driver: it runs the Python path for the single-pose case and `exec`s the old per-target
orchestrator for batch.

**The per-target variants** (three of them) differ from each other in **exactly one line** — the scratch
root. There are no per-target box, grid, dataset-name or test-type settings; the dataset name is derived
from the input directory's basename at runtime and the grid geometry is hardcoded in the builder. The
docking box in the config is **Vina-only** and never reaches this pipeline.

**What the consolidation changed behaviourally** (as opposed to structurally): the Chimera dependency is
gone (the NumPy augmenter replaces it); the inference batch size went 128 → 256 by deletion (G79); the
entire non-`vs` branch was deleted (G76); and all scratch cleanup was removed, which a downstream
endpoint now depends on (G78).

**The atom-typer change is the one with data-level consequences** (G73). Everything else in the typer is
untouched, including every contact cutoff. One variant of the typer in the tree shadows the 40-entry
protein lookup with a 9-entry ligand-only dict and therefore drops **every protein atom**; a different
variant carries the OpenBabel-3 import fix and an atom-matching tolerance change. Three lineages, one
live file — identify it by hash before editing.

**Gotchas.** G73–G80.

**Verify.** The highest-stakes claim is G80, and it is cheap to check: run the old and new grid builders
over the same atom-type file into clean directories and assert `np.array_equal` on the tensors — the
non-augmented path has no randomness. Then the structural claims by hash and AST: the "updated" typer
should be byte-identical to the live one; the three per-target scripts should differ by one hunk each;
every merged grid function should be AST-identical to its original modulo a removed `print`. Then the
dispatcher behaviours: create both a pose artifact and batch inputs in one directory and confirm which
branch runs (G74); re-run with a changed pose under the same name and confirm the score does not change
(G75); run with a non-`vs` type and confirm exit 0 with no output (G76).

---

## 19. Reproducibility

There is **no seeding anywhere**. Four independent sources of nondeterminism drive one run:

| source | used by |
|---|---|
| Python `random` | training-chunk sampling |
| Torch | token sampling (`multinomial`) |
| NumPy | augmentation, `argsort` tie-breaks, the TS Gaussian draws |
| `PYTHONHASHSEED` | the *order* of `generate()`'s output (G11) |

A rebuild that wants reproducibility must pin all four **and** replace the `set()` with an order-preserving
dedup. Pinning three of four is worse than pinning none, because it looks reproducible until it isn't.

---

## Verification methodology (applies throughout)

- **Prefer an invariant to a threshold.** `len(predict(mols)) == len(mols)` and
  `predict(mols)[i] == predict([mols[i]])[0]` cannot drift; "RMSE < 0.4" can. The same goes for
  `0 ≤ occupancy ≤ 1`, `stack_controls.sum() == 1`, `σ² == σ₀²/(n+1)`, and "every `Scaffold_Match` value is
  a multiple of 1/n_scaffold_atoms".
- **Ground truth beats intuition.** SA scores → the shipped 200-molecule fixture, whose values were
  produced by this exact pipeline. Bond/valence questions → RDKit. Transformer predictions → the
  golden `.dat`, *after* you have reconciled its row count with its input (G36).
- **Test the property, not its proxy.** A loss that decreases proves nothing about G35; only comparing
  `criterion(out, target)` against `criterion(out, target.squeeze(-1))` does.
- **Count rows at every hop.** The characteristic failure of this codebase is silent shrinkage: molecules
  vanish at six independent points in the property layer, five in the library prep, and four in the 3-D
  pipeline. `len(output) == len(input)` at each boundary is the single highest-yield assertion available.
- **A shape check is worth more than a numeric check** for anything touching PyTorch. Print
  `output.shape` and `target.shape` before every criterion call; print the input-channel count before every
  `load_state_dict`.
- **Assert the conventions the code relies on but never checks.** At startup: every `reward_function` key
  resolves to a module *and* a same-named `Property` subclass *and* has a `CITATION` (catches G2 and G3
  before a 5000-iteration run); the generator's `all_characters == gen_tokens` (G9); the reaction's
  reactant-template count equals the reagent-file count (G32); the 11-tuple of interaction bits is exactly
  as expected (G38).
- **Run from a different working directory.** Almost every path in the shipped configs is cwd-relative and
  the SA-score cache is the only `__file__`-relative one (G51). `cd /tmp && python -m elion …` surfaces the
  whole class at once.
- **Do not trust "job completed".** No shell script in this tree uses `set -e` or checks an exit code, and
  the cluster wrapper redirects the real output elsewhere (G43).

**Suite layout that would work** (each runnable standalone, no GPU, no external binaries):

```
schema_check    input_reader defaults · unknown-key rejection · plug-in name resolution (G2,G3,G53)
reward_check    step function both directions · curriculum monotonicity · TOTAL arithmetic (G4,G5,G6)
order_check     length + index alignment for every property · CSV column order · sentinel absence (G1)
gen_check       vocab order · parameter count · generate() uniqueness/validity/termination (G9,G10,G13)
rl_check        policy_gradient shapes · sign of the returned loss · memory ceiling at n_batch (G14)
ts_check        DisallowTracker exhaustion + retirement · posterior decay · reaction sanity (G27,G28,G32)
grid_check      voxel geometry · channel semantics · .atomtypes fixed-width contract (G38)
prep_check      protonator round-trip + no tabs · dedup key uniqueness · conformer presence (G45,G47,G49)
```

---

## Appendix — key constants

```
── orchestration ────────────────────────────────────────────────────────────────
run_type            calculate_properties | generate | bias_generator | post_process(no-op)
config top keys     elion_root_dir · Control · Generator · Reward_function   (all others dropped)
plug-in resolution  YAML key == module filename == class name, CASE-SENSITIVE
control defaults    history_file 'biasing_history.csv' · n_iterations 1000 · max_iter 1000
                    gen_start 0 · restart False · verbosity 0
                    (n_iterations, max_iter, gen_start are DEAD — nothing reads them)

── property / reward ────────────────────────────────────────────────────────────
Property hardcoded  max_reward 15 · min_reward 1 · reward_hook 0.3 · allowed_threshold_jumps True
                    (the YAML keys of the same names are INERT — G4)
kwargs defaults     rew_coeff 1.0 · rew_class 'hard' · rew_acc None · optimize False · threshold 0.0
base reward         step: min_reward, or max_reward when sign(thresh_step)·value >= sign·threshold
curriculum          increasing: percentile(v, 100 - hook*100, 'higher');  decreasing: percentile(v, hook*100, 'lower')
TOTAL               Σ_p reward_p × rew_coeff_p     (optimized properties only; no normalisation)
Estimators.max      Σ_p rew_coeff_p × 15
reward overrides    SAScore -v/10 ∈ [-1.0,-0.1]  ·  QED_Score v ∈ [0,1]  ·  CHEMBERT_BE -v
invalid sentinels   SAScore 10 · QED -1.0 · Scaffold_Match 0.0 · Similarity 0.0 · CHEMBERT/ADMET raise

── SA score ─────────────────────────────────────────────────────────────────────
fingerprint         Morgan radius 2, SPARSE COUNT;  unseen-fragment score -4;  score1 /= Σcounts
penalties           size nAtoms^1.005 - nAtoms · log10(chiral+1) · log10(spiro+1) · log10(bridge+1)
                    macrocycle: ring size > 8 ⇒ flat log10(2)   (deviates from the paper)
rescale             min -4.0 max 2.5 → 11 - (s-min+1)/(max-min)*9 ; log knee above 8 ; clamp [1,10]
asset               fpscores.pkl.gz, resolved via op.dirname(__file__)  ← the only cwd-safe path

── generator (stack-augmented RNN) ──────────────────────────────────────────────
hidden 1500 · stack_width 1500 · stack_depth 200 · GRU · n_layers 1 · unidirectional
optimizer Adadelta lr 0.001 weight_decay 1e-5 (lr 0.01 is the base-class default, overridden)
vocab 45 tokens, EXPLICIT UNSORTED ORDER — index 0 '<', index 1 '>'   (order IS the embedding map, G9)
params 22,650,048 = 90.60 MB fp32 · stack tensor 1.20 MB
start '<' · end '>' · predict_len 120 · max_len 120 (applied to the RAW smiles, so stored ≤ 122)
temperature FIXED at 1.0 — no knob
min generated length 6 · corpus delimiter '\t' · cols_to_read [0] (the default [] silently empties it, G25)

── REINFORCE ────────────────────────────────────────────────────────────────────
n_batch 10 (NOT batch_size — G14) · gamma 0.97 FORWARD-decaying (G15) · grad_clipping None
no baseline · no entropy bonus · rewards strictly positive so every trajectory is reinforced (G17)
returned rl_loss is NEGATIVE when rewards are positive

── RL driver ────────────────────────────────────────────────────────────────────
batch_size 200 · n_best = batch_size ALWAYS (the YAML key is discarded, G18) · n_policy 15
max_iterations 100 · warm-fit and re-fit 100 supervised steps each
checkpoint every EVEN iteration · history separator ','
approved-molecule test: exact float equality against max_reward (G7)

── Thompson Sampling ────────────────────────────────────────────────────────────
posterior           Normal-Normal, KNOWN variance:  μ'=(σ²x+σ₀²μ)/(σ²+σ₀²)  σ'=√(σ²σ₀²/(σ²+σ₀²))
                    shared prior from pooled warm-up scores ⇒ σ² = σ₀²/(n+1) after n updates
sentinels           To_Fill = None  = the ONE slot being chosen (exactly one, enforced)
                    Empty   = -1    = not yet chosen (wildcard)      ← the inline comments are swapped (G27)
mask key            the full selection tuple; value = indices forbidden AT the None position
defaults            mode 'maximize' · num_warmup_trials 3 · num_cycles 25 · batch_size 1
                    eval_batch_size 256 · num_ts_iterations 5000 (adapter) · db_name 'eXplore' (hardcoded)
reagent CSV order   SYNPLE: name,smiles,price     eXplore: smiles,name,price      (G32)
reactant order      reagent_file_list[i] must match reaction template i           (G32)
product             prod[0][0] only — other matches and products are dropped      (G31)
sign                NOTHING in TS inverts anything; the evaluator owns it          (G29)

── CHEMBERT ─────────────────────────────────────────────────────────────────────
max_len 256 · feature_dim 1024 · nhead 16 (head_dim 64) · ff 1024 · nlayers 8 · gelu · dropout 0
adj=True · padding_idx 0 · head Linear(1024,1) · prediction from position 0 (the <start> token)
specials pad 0 · mask 1 · unk 2 · start 3 · end 4
halogen fold Br→R Cl→L Sn→X Na→A  (applied to the TOKENIZER input only; RDKit sees the original)
inference batch 16 / workers 4 · finetune batch 64 · Adam lr 1e-5 · 15 epochs · 720 min wall clock
split 80/10/10 stratified by KBinsDiscretizer(3, 'kmeans')

── DeepAtom ─────────────────────────────────────────────────────────────────────
grid 32 Å @ 1.0 Å ⇒ 32³ voxels (centres span 31 Å) · 24 channels · mode 'pcmax'
channels [0:11] protein bits · [11] protein ExVol · [12:23] ligand bits · [23] ligand ExVol   (G38)
11 bits  hbondA · hbondD · xbondA · weak-hbondA · weak-hbondD · posIon · negIon · hydrophobe
         · carbonylO · carbonylC · aromatic
occupancy occ = 1 - exp(-(r_vdw/d)^12), combined by np.maximum; KDTree radius 2×vdW (binary mode: 1×)
ligand discriminator  chainID 'y'  (line[0] of every .atomtypes record)
.atomtypes layout  [0:22] id · [22:24] type · [24:28] pad · [28:39] 11 bits · 3 × %10.3f coords = 69 chars
augmentation 6 angles × 6 axes = 36; step 60° over [-180,180]; jitter ±60°; translate U(-1,1) Å
             about the ligand BBOX MIDPOINT, applied to COORDINATES not voxels    (G39)
             augmented-grid variant adds an independent ±1 Å centre jitter
network ShuffleNetV2-3D, width 2.0 ⇒ (244,488,976), repeats [3,4,4]; input Conv3d(24→32,k=1)
output  train (N,8) · eval (N,1);  ΔG = -1.36 × pK
protonation pH 7.4 (OpenBabel AddHydrogens) · contact cutoffs: hbond 3.9 · weak 3.6 · aromatic 4.0
            · ionic 4.0 · hydrophobic 4.5 · carbonyl 3.6 · metal 2.8 · global max 4.5 · vdw-comp 0.1

── library prep (vsdb) ──────────────────────────────────────────────────────────
element whitelist   C N O H S P As Se F Cl Br I    (rejects metals/B/Si BEFORE metal disconnection)
MolWT filter 50–500 Da · max_stereoisomers 4 · maxIsomers 10 · onlyUnassigned True · tryEmbedding True
dedup key   'InChI Key' (WITH a space) via MolToInchiKey(options="/FixedH")   ← 'InChIKey' also exists (G47)
standardize SaltRemover → LargestFragmentChooser → molvs.Standardizer → Uncharger → [tautomer: OFF] (G48)
protonation min_ph 6.4 · max_ph 8.4 · pka_precision 1.0 (= σ multiplier, not a tolerance)
            39 SMARTS rules (37 single-pKa, 2 dual), FIRST-MATCH-WINS in file order
            10 names prefixed '*' = the acid is the NEUTRAL form ⇒ undo the +1 nitrogen offset
            6 neutralisation SMARTS run to a fixed point before anything else

── GIGN (⬡ pose scorer) ─────────────────────────────────────────────────────────
model       GIGN(node_dim 35, hidden 256, layers 3) · Linear(35→256)+SiLU · 3×HIL
            · global_add_pool (SUM ⇒ score is EXTENSIVE in atom count — G57)
            · FC 3×[Linear(256,256)+Dropout .1+LeakyReLU+BatchNorm] · Linear(256,1), NO activation
HIL         aggr 'add' · message = x_j ⊙ SiLU(W·RBF(‖pos_i−pos_j‖)) — x_i unused
            update  = mlp_cov(x + m_intra) + mlp_ncov(x + m_inter)   ← 4 param blocks/layer
RBF         9 centres over [0, 6] Å, σ = 6/9 = 0.6667, exp(−((d−μ)/σ)²), no envelope
            (the function's own defaults 0/20/16 are overridden at BOTH call sites)
features    35 = element 10 (C N O S F P Cl Br I + Unknown) · degree 7 · implicit valence 7
            · hybridization 5 (SP SP2 SP3 SP3D SP3D2) · aromatic 1 · totalNumH 5
            out-of-set → LAST slot, silently
edges       intra = RDKit bonds, bond ORDER DISCARDED, no cutoff (lig + prot pooled)
            inter = ligand×pocket pairs, strict dist < 5.0 Å, no prot–prot, no lig–lig
pocket      whole residues with a heavy atom within 5 Å of a heavy ligand atom; waters dropped
            cut_pocket() SIGNATURE default is 10.0 — every other site says 5    (G58)
training    MSE on pK · Adam 5e-4 wd 1e-4 · ReduceLROnPlateau(patience 40, ×0.1, min 1e-5)
            300 epochs · early stop 30 · 3 repeats · seed 2024 · valid split = last 1000
            (early stopping is driven by the TEST loader — leakage)
caches      Pocket_{d}A.pdb · {id}_{d}A.rdkit (PICKLE of (ligand,pocket) Mols)
            Graph_GIGN-{id}_{d}A.pyg (torch.save of one PyG Data)
            key encodes ONLY the edge cutoff — not featurisation, removeHs, or pocket method (G59)
contract    subprocess, ONE pose per call · stdout GIGN_PRED_PK / GIGN_DELTAG = −1.36 × pK
            pK HIGHER is better; ΔG MORE NEGATIVE is better — opposite polarity  (G63)
            pocket_cutoff <= 0 means FULL RECEPTOR and must be forbidden        (G57)
stack       torch ≥ 2.6 · torch_geometric ≥ 2.5 · Python 3.11  (the pinned reqs are archaeology — G65)

── TS warm-up checkpointing (⟲) ─────────────────────────────────────────────────
env var     TS_WARMUP_CHECKPOINT — absent/empty ⇒ normal warm-up
file        <output_dir>/Warmup_TS/<short_name>_<YYYYMMDD_HHMMSS>_warmup.json   (JSON, indent 2)
contents    prior_mean · prior_std · known_var · components{idx: [{reagent_name, current_mean,
            current_std, known_var, num_scores}]} · n_components · n_reagents
NOT stored  disallow-tracker state (G67) · retired reagents (G68) · raw warm-up scores
            · SMARTS · reagent-file identity · reward-function fingerprint      (G66)
cache key   the reaction short_name ONLY; glob <short>_*_warmup.json, newest by name
            consumer check = os.path.isfile. Nothing else is compared.          (G66)
restore     writes 5 fields per Reagent, phase → 'search', initial_scores = [],
            restores warmup_std, returns [[prior_mean,"checkpoint","checkpoint"]] and RETURNS
            before any warm-up body runs. Implemented TWICE (inline + monkeypatch loader).
source      regex-scraped from %.6f DEBUG log lines, not serialised from objects  (G70)
            at INFO the per-reagent lines don't exist ⇒ valid file, EMPTY components
reactions   amide · buchwald · sonogashira · suzuki · sulfonamide · snar (short_name = cache key)

── DeepAtom pipeline consolidation ──────────────────────────────────────────────
entry pts   generate_atomtypes.py --batch-dir --pre-dir --scripts-dir [--workers N] [--stages 1..6]
            generate_npz.py BATCH DATASET [--stage both|non_augmented|augmented]
            deepatom_score_pipeline.sh -t <type> -d <dir>   ← DISPATCHER, pose branch tested FIRST (G74)
stage map   1 assemble dirs · 2 build complex · 3 copy non-aug · 4 augment · 5 arpeggio · 6 arpeggio-aug
            pose path = --stages 3,5 (the caller has already built the complex)
AUGMENT=0   stages 3,5 + non_augmented (default)   AUGMENT=1 → 3,4,5,6 + both
exit codes  1 usage · 2 no inputs · 3 atomtypes failed · 4 npz failed · 5 orchestrator missing
            · 6 test failed · 7 zero .atomtypes · 8 zero .npz
per-target  the three variants differ in ONE line (the scratch root) and nothing else
UNCHANGED   channel order · grid geometry · KD-tree radii · occupancy formula · dir naming
            ⇒ existing weights remain valid                                      (G80)
changed     inference batch 128 → 256 (by deletion) · Chimera → NumPy augmenter
            · non-`vs` branch deleted · ALL scratch cleanup removed (and now depended on)
typer fix   ligand element: name[:2] raw → .strip() + .capitalize()/.upper()
            before the fix, any ligand with Cl or Br produced NO atomtypes at all (G73)

── I/O formats ──────────────────────────────────────────────────────────────────
smi output  SMILES,Name,<props…>  ·  Gen-{i:05d} 1-indexed  ·  values %.2f  ·  column order = YAML order
history csv 1 + 2×n_properties columns:  Iteration,{prop}_thr,{prop}_avg,…
TS results  score,SMILES,Name   (Name = "_".join(reagent_names))
GIGN out    stdout markers GIGN_PRED_PK / GIGN_DELTAG ; stderr GIGN_ERROR + exit 1
DeepAtom out <out_csv_dir>/<test_type>_<dataset>.csv  cols: PDB, deltaG_kcal_mol
```

*Build order recap:* input reader → property contract → estimator (+ one trivial property) → generator
contract → ReLeaSE + REINFORCE → the RL driver → Thompson Sampling → **warm-up checkpointing** → then the
expensive predictors (CHEMBERT, DeepAtom, **GIGN**) and library prep. The predictors go **last** because
they are the only parts that need weights, GPUs and external binaries — and because every one of their
failure modes is invisible until the cheap contracts above are already asserted.

Checkpointing goes **after** a working Thompson Sampling loop for the same reason superposition goes last
in a viewer: it is the piece that lets you *skip* previously-computed state, so it can only be validated
against a run that already produces that state correctly. And GIGN goes last among the predictors because
it is the only one that needs a pose — every other scorer in the tree takes a SMILES string, so GIGN is
the first thing that forces you to decide who owns 3-D coordinates.
