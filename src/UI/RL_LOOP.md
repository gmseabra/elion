# The closed loop — TS → Vina pose → DeepAtom/GIGN → `bias_generator`

What was added, why each piece is where it is, and the four traps that make a
naive version of this quietly wrong.

```
  🎲 Run TS  ──▶ results CSV  ──▶ top 30% by the ChemBERT score TS already computed
                                        │
                                        ▼
                        Vina 1.2.7 search (in-browser, per ligand)
                                        │
                                        ▼
                     DeepAtom ΔG  /  Yupu_GIGN pK  /  Vina ΔG only
                                        │
                                        ▼
                            RL_Feedback/<short>_<ts>_affinity.json
                                        │
                                        ▼
                       bias_generator  ──▶  Warmup_TS/<short>_<ts>_warmup.json
                                        │
                                        ▼
                        the NEXT 🎲 Run TS loads it and skips warm-up
```

---

## The constraint that shaped everything

**There is no channel into a running TS process.** `ts_routes._run_ts_job`
spawns `elion.py` with `stdout=PIPE, stderr=PIPE` and no `stdin`, and never
stores the `Popen` object — only the pid. Nothing in the Flask process could
write to the child even if a protocol existed. Grep confirms it: `stdin`
appears nowhere in `ts_routes.py`.

The one injection point the engine already honours is `TS_WARMUP_CHECKPOINT`,
read once inside `ThompsonSampler.warm_up()` at process start.

So **feedback biases the next round, not the one in flight.** That is a
property of the engine, not a shortcut. Making it live needs an IPC channel
added on the elion side — a poll inside `search()` that calls `add_score` on
named reagents between iterations.

---

## Files

| File | Status | What it is |
|---|---|---|
| `uiapp/core/rl_bias.py` | **new** | The algorithm. Selection + credit assignment + the checkpoint merge. **stdlib only.** |
| `uiapp/routes/ts_routes.py` | +4 routes | `ts_rl_harvest`, `ts_rl_feedback`, `ts_rl_bias`, `ts_rl_runs`. Appended; `time` added to the top-level import. |
| `web/static/js/ts/ts_rl_loop.js` | **new** | The loop driver. Self-installing into `#tsPaneRl`, like `ts_rl.js`. |
| `web/static/js/hub.js` | +1 line | `'ts/ts_rl_loop.js'` after `'ts/ts_rl.js'` in the serial loader. |
| `elion_bias_generator_TS.py` | **new** | Drop into `<elion>/src/elion/generators/TS/rl.py`. |
| `tests_rl_roundtrip.py` | **new** | 45 assertions, no dependencies. `python tests_rl_roundtrip.py`. |

### Why `rl_bias.py` is stdlib-only

It is loaded two ways: imported by the Flask route, and loaded **by absolute
path** by the elion shim — which runs in a different conda env. A shared file
with no dependencies cannot import cleanly on one side and fail on the other.

This is also why `tests_rl_roundtrip.py` loads it by path rather than as
`uiapp.core.rl_bias`: importing the package executes `uiapp/__init__.py`,
which imports every route module, which imports torch. A stdlib-only algorithm
should not need the ML stack to test.

---

## Wiring the elion side

```bash
cp elion_bias_generator_TS.py  <elion>/src/elion/generators/TS/rl.py
```

Then one method on whatever class `Generator('TS')` returns:

```python
from generators.TS.rl import bias_generator as _rl_bias

class TSGenerator:
    def bias_generator(self, control_cfg, estimator):
        return _rl_bias(control_cfg, estimator)
```

and `run_type: bias_generator` in `input_TS.yml`. `elion.py`'s existing
dispatch already calls `generator.bias_generator(config['Control'], estimator)`.

The shim finds `rl_bias.py` via `$ELION_RL_IMPL`, then `$ELION_UI_ROOT`, then a
sibling-layout probe — and raises with the paths it tried rather than falling
back to a vendored copy. A silent fallback is the drift this design exists to
prevent.

`estimator` is accepted and **deliberately unused**: the pK → reward-scale
mapping is arithmetic over the previous checkpoint's `prior_std`, so this step
loads no ChemBERT and touches no GPU. A commit stays seconds, not minutes.

### Two modes

| Mode | Trigger | Does |
|---|---|---|
| **bias** | `ELION_RL_AFFINITY` set | measurements → merged checkpoint. Prints `RL_BIAS_CHECKPOINT <path>` (the UI route greps for this line — keep it). |
| **selection-only** | `ELION_RL_RESULTS_CSV` set instead | applies the top-30% rule to a results CSV and writes `rl_worklist.json`. Use this if you want the *engine*, not the browser, to choose the batch. |

The UI's `mode: "local"` runs the same functions in-process; `mode: "elion"`
patches a copy of the yml to `run_type: bias_generator` and shells out. Both
produce an identical checkpoint, because both call the same file.

---

## The four traps

### 1. A partial checkpoint silently destroys the run's learning

The loader `ts_routes._write_warmup_loader` generates does this:

```python
belief = _belief_by_name.get(reagent.reagent_name)
if belief:  reagent.current_mean = belief["current_mean"] ...
else:       reagent.current_mean = _prior_mean      # ← every absent reagent
```

Write a checkpoint containing only the ~30 reagents you just measured, and the
other ~80,000 are **reset to the global prior** — a full run of Bayesian
updating, gone, from a file that looks correct and loads without error.

`merge_into_checkpoint()` therefore **requires a previous checkpoint to merge
into** and raises without one. `allow_no_prior=True` exists, and its error
message says what it costs. Verified by `tests_rl_roundtrip.py [4]`.

### 2. The floor on `current_std` must never *raise* it

`current_std` is what the Thompson draw samples with, so it is the exploration
budget. The obvious form is wrong:

```python
new_std = max(old_std * shrink, floor)          # ← WRONG
new_std = min(old_std, max(old_std * shrink, floor))   # ← what ships
```

A converged reagent's std is `prior_std/sqrt(n+1)` — already `0.32 × prior_std`
after nine observations. With the original `std_floor = 0.5`, the bare `max()`
*raised* a 0.30 std to 0.50: the round that measured your best building block
made the bandit **less** sure of it. Caught by the test, fixed by the outer
`min()`, and `DEFAULT_STD_FLOOR` dropped to `0.05`. The floor is a lower bound
on how far we tighten, never a licence to loosen.

### 3. Units — pK is not the reward

TS maximises the Elion reward (`CHEMBERT_BE` at rew_coeff 0.95, plus
SAScore/QED). DeepAtom and GIGN return a pK. Writing one into a field holding
the other is the easiest way to make this loop useless.

The default mapping, `zblend`, never uses the pK's absolute value — only its
position in the batch:

```
z_i   = (pk_i − mean(pk)) / std(pk)
shift = z_i × prior_std × strength
```

`prior_std` is the spread of the reward TS actually saw, so the shift is in the
units of the thing being biased. `strength` (default 0.5) means what it says: a
reagent one sigma above the batch moves half a reward-sigma.

`rank` ignores magnitudes entirely (robust to a miscalibrated critic). `raw`
writes the pK straight in and **warns**, because the scales do not match.

A batch where every pK is identical moves nothing, and says so — a critic that
cannot separate the batch carries no information about it.

### 4. `_modes` is not cleared between Vina runs

`pose.js`'s `_procReset` leaves `PG.vina.eng._modes` populated. A failed search
would otherwise report the **previous** ligand's affinity against the current
molecule. The driver clears `_modes` and `_dg` before every `run()`.

The same class of problem one level up: `PG.build()` resolves even when the
server rejected the SMILES — it falls back to the in-browser parser, and if
that also fails it leaves the *previous* ligand loaded. `eng.ready()` is
checked after every build for exactly that reason.

---

## Integration notes on `pose.js`

Three facts drove the driver's shape:

1. **No events.** No `CustomEvent`, no callback argument, no promise from
   `vina.eng.run()`. The only genuine promise is `PG.build()`. Everything else
   is a **bounded** poll on `_running` / `_modes` / `_dg` — bounded because a
   poll with no deadline against a worker that can die silently hangs the loop
   on molecule 7 of 30 forever.

2. **The receptor is private.** `let protein = null` at `pose.js:599`, no
   accessor. The driver wraps `PG._applyProtein` to mirror `raw` — the same
   wrap-don't-fork trick `ts_rl.js` already uses on `PG.open`/`PG.close`.

3. **The scorers are called directly**, not via `PG.mc.scoreDeepAtom()` /
   `scoreGign()`. Those are fire-and-forget (they return `undefined` and paint
   the DOM), and `scoreDeepAtom` dereferences `PG.mc.best` with **no fallback**
   — it throws if the MC stage was never entered. A loop needs values back, in
   order, with their errors attached.

### Auto-start on run completion

`_tsFinalise` is wrapped, **and** `_ts.running` is polled as a backstop.
Both, because `ts_run.js` calls `_tsFinalise` as a bare identifier from
`_tsCancel` and from the `__DONE__` handler — a bare call resolves the
top-level function declaration, not `window._tsFinalise`, and there is no way
to rebind a function declaration from another script. The wrapper catches
external callers; the poll catches the internal ones.

Success is read **after** a 900 ms delay, because `_tsFinalise` itself defers
its success/error decision (`ts_run.js:326`) — the error line arrives on a
~500 ms path and reading `_sawError` immediately races it.

---

## Honest reporting, by design

Every count that could hide a problem is surfaced rather than inferred:

- `n_fraction` vs `n_selected` vs `capped` — a capped list is never presented
  as "the top 30%". The panel says *"Posing 30 of 356 in the top 30% (1187
  unique products)"*.
- `n_no_score`, `n_unresolved_name` — records dropped and why.
- `n_applied` / `n_added` / `n_carried_unchanged` — a large `n_added` means the
  affinity file and the checkpoint came from different runs, and it says so.
- Scorer errors are shown **verbatim**. `pose.gign_script` and
  `pose.default_script` are both `""` in `config/input_routes.yml`, so an
  unconfigured scorer returns HTTP 200 `{ok:false, err:"...script not found"}`
  — which inside a 30-molecule loop is 30 silent nulls unless you print it. The
  dropdown also mirrors the pose tool's own options and labels anything absent
  *before* the run.

---

## Verify

```bash
python tests_rl_roundtrip.py            # 45 assertions, no dependencies
node --check web/static/js/ts/ts_rl_loop.js
python -m pyflakes uiapp/core/rl_bias.py elion_bias_generator_TS.py

# the elion path, end to end, without elion:
ELION_RL_IMPL=$PWD/uiapp/core/rl_bias.py \
ELION_RL_AFFINITY=<affinity.json> \
ELION_RL_WARMUP_DIR=<...>/Warmup_TS \
python elion_bias_generator_TS.py
```

`tests_rl_roundtrip.py [3]` includes a **drift guard**: it greps
`ts_routes.py` for the loader's actual field accesses
(`belief["current_mean"]`, `belief.get("known_var")`, …). If the generated
loader's schema ever changes, the test fails instead of the next run silently
losing its posteriors.

### ⚠ Two things this drop does *not* update

- **`tests/` is not in the uploaded tree**, so
  `test_import_smoke.py::test_every_frontend_endpoint_exists` and the route-count
  assertion (81) have not been updated. Four routes were added; the assertion
  will fail until you bump it.
- **`_RL_DIR` is created on first write**, not at import, so it will not appear
  under `visualizer.output_dir` until a round is committed.
