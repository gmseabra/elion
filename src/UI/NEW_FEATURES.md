# What was added on top of the base `UI` tree

This tree is the original `UI.zip` — its `uiapp/` package layout, its
env-var-driven `uiapp/config.py`, and its structural test — with five new
subsystems ported in from the `visualizer/` development tree.

**The Workflow and Handoff Manager features were deliberately excluded**, along
with all of their supporting code, and the **Get Ligand Center** tool and the
in-app **Reasoning gateway** have since been removed the same way.
`tests/test_import_smoke.py::test_excluded_features_absent` asserts all four
stay out. The Reasoning *button* is kept as an external link — see below.

Full rationale, algorithms, parameters and traps: **`ELION_UI_PLAYBOOK.md`**
(§19 sessions, §20 mini-chat + CoT, §21 pose generation, §22 the TS frontend
split, §23 the two removed tools, and the new §0 rows G67–G91).

---

## New subsystems

| | Where | What |
|---|---|---|
| **⧉ Sessions** | `uiapp/routes/session_routes.py` | SQLite chat history keyed on `(client, chat_type)`, plus a model registry. 6 endpoints under `/session/`. |
| **✎ Unified mini-chat** | `uiapp/routes/mini_chat_routes.py` | One SSE generator serves vina / attn / deepatom. Replaces the three near-duplicate per-tool implementations (`vina_chat_routes.py` is gone). |
| **⌁ Consolidated CoT** | `uiapp/routes/cot_routes.py` | One chain-of-thought router replacing four copy-pasted ones. The broken copy in `hub_routes.py` was deleted. |
| **⬢ Pose generation** | `uiapp/routes/pose_routes.py`, `web/static/js/pose.js` | SMILES → 3-D conformer + torsion tree server-side; all kinematics in the browser. Three scorers: DeepAtom, GIGN, and a pure-Python Vina five-term function. 15 endpoints under `/pose/`. |
| **⟐ TS frontend, split** | `web/static/js/ts/` (7 modules) | Was one file. Adds multi-job fan-out, worker pre-seeding, traceback capture, rAF frame budgets, and off-thread top-5 ranking with hysteresis. |
| **⌖ Source map** | `uiapp/routes/devmap_routes.py`, `web/static/js/devmap.js` | Hover any button → the file:line of its handler, its markup, and the Flask view behind every endpoint it calls. Development aid, so it ships **off**: enable it with **Alt+Ctrl/⌘+Shift+D** or the bottom-left badge. Default state, hotkey, pin modifier and badge visibility are configured under `visualizer.devmap` in the engine's `input_TS.yml` (see `add_devmap_config.py`). 4 endpoints under `/devmap/`. |

Frontend shell: `web/static/js/hub.js` now owns what used to be `hub.html`'s
inline script (nav, chat SSE, lazy feature loader, model picker, session panel).
`web/static/js/elion_mini_chat.js` drives one themed panel for four tools.

**Route count: 38 → 81 → 95** (devmap 4, the RL loop 4, devmap source 3, and the
TS/pose additions). OpenDock's 9 endpoints were added and later removed with the
engine; `engines/opendock/`, `uiapp/routes/opendock_routes.py`,
`web/static/js/opendock.js`, `web/templates/opendock_modal.html` and
`OPENDOCK.md` are gone from the tree — see git history.

## Changes I made during the port

These were required to make the ported features work inside the packaged layout,
rather than optional cleanups:

1. **Imports rewritten** to the `uiapp` namespace — `nas_storage_app.*` →
   `uiapp.*`, flat `qwen_client` / `elion_ui_router` / `kb_auto_learner` imports →
   their `uiapp.llm.*` / `uiapp.router.*` package paths.
2. **Paths routed through `uiapp/config.py`** instead of being hardcoded. The
   development tree had 41+ absolute `/home/<user>` paths; the base tree's own
   `test_no_hardcoded_user_paths` forbids them, and it now passes. New config
   entries: `SESSIONS_DB`, `POSE_*`, `TS_OUTPUT_DIR`, `TS_BB_BASE`,
   `TS_ENGINE_DIR`, `MGLTOOLS_*`, `DEEPATOM_ROOT`, `COT_*`.
3. **Six module-level `NameError`s fixed** — missing `Path` (attn, vina_chembert),
   `Chem` (vina_chembert), `sys` (deepatom), and the `DEEPATOM_*` constants that
   the development tree dropped when it removed the config module. Each of these
   made an endpoint 500 on *every* request; a missing import is not a feature.
4. **The fine-tune subprocess** now points at `uiapp/core/finetune_chembert.py`
   (it was looking in `routes/`, where no such file exists).
5. **No blueprints.** Every route module decorates the global `app` directly.
   (The development tree's one blueprint belonged to the reasoning gateway,
   which is not shipped.)
6. **`shared.py` extended** with the MGLTools probe and the PDBQT sanitiser
   (`sanitize_pdbqt`, `ensure_vina_safe_pdbqt`) — the fix for Vina reporting
   `affinity = None` when obabel copies `TITLE`/`REMARK` into a PDBQT.
7. **`config/input_routes.yml`** gained a `pose:` section and
   `deepatom.timeout_seconds`; `run.py` now absolutises the pose paths the same
   way it already did the vina ones.
8. **`ELION_CWD` is probed, not guessed.** The development tree hardcoded the
   engine path; the base tree guessed one relative layout. Neither works for
   both, and getting it wrong makes the TS routes answer **404** — which in the
   browser is indistinguishable from a missing route, with the Reactions picker
   silently absent. `config.resolve_elion_cwd()` now takes the first candidate
   directory that actually holds `input_TS.yml`, `run.py` reports the result at
   boot, and the TS modal shows a banner instead of nothing.
   (`tests/test_engine_paths.py`)
9. **`/vina_visualization/ts_gpu` now exists.** `ts_ui.js` has opened an
   EventSource on it since the TS frontend was split and the route was never
   written, so every TS open logged a 404 and the Monitor tab's GPU panel stayed
   blank. Implemented against pynvml with an `nvidia-smi` fallback; a host with
   neither reports `available: false` rather than erroring.
   `test_import_smoke.py::test_every_frontend_endpoint_exists` now walks the
   frontend's own URLs so the next one cannot go missing silently.
10. **The UI port is configurable from `input_TS.yml`** — `visualizer.port` /
   `visualizer.host`, overridable by `$UI_PORT` / `$UI_HOST`, defaulting to
   `0.0.0.0:5000`. Bad values fall through instead of raising.

Everything else was ported as-is. The defects documented in the playbook that are
*behavioural* rather than fatal — the renderer lock (G68), the IP-keyed sessions
(G69), SQLite without WAL (G70), the history-clear rebind (G71) — are **left in
place and documented**, not silently rewritten.

## Removed features

Four sidebar tools are not shipped. Each was removed with its whole vertical
slice — button, modal, JS, routes, clients, config keys and state directory —
not just hidden:

| Tool | What went with it |
|---|---|
| 🧪 **Workflow** | `workflow_routes.py`, `workflow.js`, the pipeline chip |
| ⇄ **Handoff Manager** | `handoff_routes.py`, `handoff_manager.js`, `product_context_routes.py` (its only consumer), `/tools/handoff_types` |
| ◎ **Get Ligand Center** | `ligand_center.js`, `ligand_center_modal.html`, the inlined `#ligandCenterModal` in `hub.html`, and `_compute_ligand_center` + `POST /tools/ligand_center` in `uiapp/core/pdb_converter.py` |
| ⌁ **Reasoning gateway** | `reasoning_routes.py` (the tree's only blueprint), `uiapp/llm/claude_client.py`, `gateway_probe.py`, `reasoning_modal.html`, `REASONING_STATE_DIR`, `data/reasoning/` |

**The Reasoning button itself is kept.** It is now a plain external link: its
own `onclick` opens `config.REASONING_URL` (env `UI_REASONING_URL`, default
`http://localhost:5001/`) in a new tab. It deliberately carries **no
`data-nav`**, because `hub.js`'s delegated dispatcher only routes to in-page
openers and there is no longer anything in-page to open. `hub_routes.hub()`
passes `reasoning_url` to the template — that is the page's only variable.

Losing Get Ligand Center costs nothing: `POST /tools/pdb_to_pdbqt` already
returns `center_x/y/z`, `extent_x/y/z` and `suggested_size` for the file it just
converted, computed from the *heavy-atom* centroid — the same definition the
rest of the app uses (see `_pdbqt_center` in `tools_routes.py`). The removed
tool used a different definition (all non-water HETATM, bounding box for size),
so the two disagreed on the same input.

⚠️ **`pose.js` and `pose_generation.html` contain `PG.init.ligCenter` /
`PG._ligCenter`.** That is pose-frame geometry — the ligand centroid used to
place and rotate a conformer — and has nothing to do with the removed tool. It
is untouched. Grep for `ligand_center`, never `ligCenter`.

## Optional backends

Every external resource is blank-by-default and reports as unconfigured rather
than failing: the sibling elion engine (`ELION_CWD`), DeepAtom
(`DEEPATOM_SCRIPT`), the GIGN scorer (`pose.gign_script`), MGLTools
(`UI_MGLTOOLS_DIR`), the sentence-transformer encoder (`UI_ENCODER_PATH`) and
the ChemBERT weights.

## Verify

```bash
python -m pytest tests/            # 70 tests, no ML stack required
python tests/test_structure.py     # layout + path hygiene, zero dependencies
python tests/test_import_smoke.py  # imports the app, asserts the 81-route table
python tests/test_engine_paths.py  # ELION_CWD auto-detection + UI host/port resolution
python -m pyflakes uiapp/          # catches the class of bug in item 3 above
```

`test_import_smoke.py` is new. It stubs torch/rdkit/faiss/etc., imports `uiapp`
for real, and asserts every route module loaded and every expected endpoint
registered — the check that a dependency-free structural test structurally
cannot make.
