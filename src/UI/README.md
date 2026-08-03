# Elion UI

A Flask web platform that fronts the Elion drug-design stack with interactive
visualizers: **AutoDock Vina-GPU docking**, **ChemBERT attention** inspection,
**DeepAtom** saliency, and a **Thompson-Sampling active-learning** driver — all
tied together by a RAG-powered *UI action router* that turns natural-language
requests in the mini-chat into concrete button actions.

No configuration lives in code: every path is resolved relative to this
repository (see `uiapp/config.py`) and every genuinely-external resource
(model weights, the sibling Elion engine, DeepAtom data) is overridable with an
environment variable.

## Layout

```
UI/
├── run.py                     # entry point: engine env → load config → app.run
├── run.sh                     # thin launcher (cd here && python run.py)
├── requirements.txt
├── config/
│   └── input_routes.yml       # vina targets + deepatom datasets (repo-relative paths)
├── uiapp/                     # the Flask application package
│   ├── __init__.py            # app factory (templates/static → ../web)
│   ├── config.py              # single source of truth for every path
│   ├── core/                  # domain logic
│   │   ├── pdb_converter.py
│   │   ├── deepatom_saliency.py
│   │   └── finetune_chembert.py
│   ├── llm/                   # LLM clients
│   │   ├── qwen_client.py     #   in-app chat + UI action routing
│   │   ├── qwen_server.py     #   standalone llama.cpp inference server
│   │   └── serve_qwen.sh
│   ├── router/                # RAG UI-action router
│   │   ├── ui_action_router.py
│   │   ├── kb_auto_learner.py
│   │   └── ui_action_kb/      # human-authored action knowledge base (*.md)
│   ├── routes/                # route modules (loaded in order by routes/__init__.py)
│   │   ├── shared.py          #   paths, job stores, PDBQT sanitiser, MGLTools probe
│   │   ├── cot_routes.py      #   the single chain-of-thought router
│   │   ├── session_routes.py  #   SQLite chat history + model registry
│   │   ├── mini_chat_routes.py#   ALL chat endpoints (vina / attn / deepatom)
│   │   ├── pose_routes.py     #   SMILES→3D pose + 3 scoring backends
│   │   ├── ts_mol_render.py   #   reagent SMILES index + 2D SVG (library)
│   │   └── hub/attn/vina_*/ts/tools/deepatom_routes.py
│   └── CHEMBERT/              # ChemBERT model package
├── web/                       # frontend
│   ├── templates/             # Jinja2 modals + hub
│   └── static/js/             # vanilla-JS visualizers
│       ├── hub.js             #   shell: nav, chat SSE, lazy feature loader
│       ├── elion_mini_chat.js #   one themed mini-chat panel, four tools
│       ├── pose.js            #   pose engine (pure) + pose UI controller
│       └── ts/                #   Thompson Sampling, split into 7 modules
├── engines/
│   └── vina/                  # vendored AutoDock Vina-GPU (binaries + OpenCL kernels)
├── pipeline/
│   └── active_learning/       # Thompson-Sampling → PDB → docking → distribution-shift scripts
├── data/                      # runtime data (gitkept, contents ignored)
│   ├── chembert/              # sample SMILES / vocab
│   ├── converted_pdbqt/       # prepared receptors + ligands
│   └── kb_shadow/             # auto-generated shadow KB + CoT logs
└── tests/
    ├── test_structure.py      # dependency-free layout / path-hygiene checks
    ├── test_import_smoke.py   # imports the app with the ML stack stubbed and
    │                          #   asserts the full route table registers
    ├── test_elion_vina.py     # protein-library selection logic
    └── parse_ts_session.py    # TS session-JSON inspector (CLI, not a test)
```

## Quickstart

```bash
# 1. Create and activate the environment
conda create -n elion-ui python=3.10 -y
conda activate elion-ui

# 2. Install the binary-heavy libraries from conda-forge (most reliable)
conda install -c conda-forge rdkit faiss-cpu -y

# 3. Install everything else from requirements.txt
cd UI                        # the unzipped folder
pip install -r requirements.txt
#   ^ rdkit/faiss are already satisfied by step 2, so pip skips them

# 4. Start the app
bash run.sh                  # or: python run.py
```

Endpoints: `/` (hub), `/attention_visualization/`, `/vina_visualization/`.

## Configuration

All paths default to sensible repo-relative locations. Override any of these via
environment variable when your layout differs (see `uiapp/config.py`):

| Variable | Purpose | Default |
|---|---|---|
| `UI_INPUT_ROUTES_YML` | docking / dataset config file | `config/input_routes.yml` |
| `UI_VINA_DIR` | AutoDock Vina engine directory | `engines/vina` |
| `UI_CONVERTED_PDBQT` | prepared receptor/ligand root | `data/converted_pdbqt` |
| `UI_ENCODER_PATH` | MiniLM sentence-transformer dir | `../LLM/all-MiniLM-L6-v2` |
| `UI_CHEMBERT_FINETUNED` / `UI_CHEMBERT_PRETRAINED` | ChemBERT checkpoints | `uiapp/CHEMBERT/*.pt` |
| `QWEN_MODEL` | Qwen `.gguf` weights | `models/…` |
| `ELION_CWD` / `ELION_VENV` | Elion engine dir (the one holding `input_TS.yml`) + its Python | auto-detected, current Python |
| `DEEPATOM_SCRIPT` / `DEEPATOM_DATA_DIR` / `DEEPATOM_ROOT` | external DeepAtom project | unset |
| `UI_SESSIONS_DB` | chat-history SQLite file | `data/elion_sessions.db` |
| `UI_POSE_ROOT` / `UI_POSE_UPLOAD_DIR` / `UI_POSE_DEBUG_DIR` | pose scratch + uploads | `data/pose/…` |
| `UI_POSE_GIGN_SCRIPT` | GIGN pose scorer | unset (scorer disabled) |
| `UI_TS_OUTPUT_DIR` | TS session + warm-up state | `data/ts` |
| `UI_TS_BB_BASE` / `UI_TS_ENGINE_DIR` | reagent building blocks / TS engine dir | derived from `ELION_CWD` |
| `UI_MGLTOOLS_DIR` / `MGLTOOLS_PYTHON` | AutoDockTools prep scripts + interpreter | `engines/vina/mgltools`, autodetected |
| `UI_REASONING_URL` | destination of the Reasoning sidebar link (external app, not served here) | `http://localhost:5001/` |
| `UI_HOST` / `UI_PORT` | server bind address; overrides `visualizer.host` / `visualizer.port` in the engine yml | `0.0.0.0` / `5000` |

`config/input_routes.yml` stores docking targets and DeepAtom datasets with
repo-relative paths; `run.py` resolves them to absolute paths at load time.

**`ELION_CWD` is auto-detected.** `uiapp/config.py` probes the layouts the UI is
normally checked out in — `../elion`, `../elion/src/elion`, `../../elion/src/elion`,
`./elion` — and takes the first that actually contains `input_TS.yml`; an explicit
`ELION_CWD` always wins and is never probed. `run.py` prints the result at boot and
says so loudly when nothing was found, because **the Thompson-Sampling routes answer
HTTP 404 when that file is missing** — in the browser that is indistinguishable from
an unregistered route, and the Reactions picker (built from the `/ts_config`
response) silently never appears. `TS_ENGINE_DIR`, `TS_BB_BASE` and `DEEPATOM_ROOT`
are all derived from it, so one override repoints the whole engine.

**The listening port lives in the engine yml.** `visualizer.port` (and
`visualizer.host`) in `input_TS.yml` set where the UI binds, so the port is
configured in the same file as `output_dir` rather than in code:

```yaml
visualizer:
  output_dir: "/path/to/user_data"
  port: 5000          # $UI_PORT > visualizer.port > 5000
  host: "0.0.0.0"     # $UI_HOST > visualizer.host > 0.0.0.0
```

A non-numeric or out-of-range value falls through to the next level instead of
raising — editing this file can never stop the server from starting. The boot
banner prints the resolved address and which of the three levels it came from.

## What's here

| Feature | Entry point | Notes |
|---|---|---|
| **Vina docking** | `/vina_visualization` | GPU docking, byte-level stdout streaming, per-atom energy decomposition |
| **ChemBERT** | `/attention_visualization` | per-atom weight view, A/B compare, fine-tuning jobs |
| **DeepAtom** | `/vina_visualization/deepatom_*` | 3-D CNN screening + per-atom saliency |
| **Thompson Sampling** | `/vina_visualization/ts_*` | live warm-up / belief monitor, multi-reaction jobs |
| **Pose Generation** | `/pose/*` | SMILES → 3-D conformer + torsion tree; DeepAtom / GIGN / pure-Python Vina scorers |
| **Sessions** | `/session/*` | SQLite chat history + model registry |
| **Mini-chat** | `/{vina,attention}_visualization/chat*` | one implementation for all three tools |

The **Reasoning** sidebar entry is a plain external link, not a feature of this
server — it opens `UI_REASONING_URL` in a new tab and there is no `/reasoning/*`
endpoint. **Get Ligand Center** is not shipped; use the `center_x/y/z` +
`suggested_size` fields that `POST /tools/pdb_to_pdbqt` already returns.

External resources — the sibling elion engine, DeepAtom, GIGN, MGLTools, the
sentence-transformer encoder and the ChemBERT weights — are all **optional**.
Each one is blank-by-default in `config/input_routes.yml` or `uiapp/config.py`;
a feature whose backend is unconfigured reports that instead of failing.

## Components

The **UI** package is the base application (originally the *visualizer*). The
**vina** engine and the **active-learning pipeline** are vendored alongside it:
docking runs shell out to `engines/vina`, and the `pipeline/active_learning`
scripts implement the Thompson-Sampling loop (extract PDBs from TS results →
dock → chart the vina score distribution shift) that the TS routes drive.

## Tests

```bash
python -m pytest tests/            # everything (55 tests, no ML stack needed)
python tests/test_structure.py     # layout + path hygiene, zero dependencies
python tests/test_import_smoke.py  # imports the app, asserts the route table
```

`test_structure.py` verifies the package layout, that the route loader agrees
with the files on disk, and that no user-specific absolute paths have returned.

`test_import_smoke.py` is the complement: it stubs torch/rdkit/faiss/etc., imports
`uiapp` for real, and asserts every route module loaded and every expected
endpoint registered. A missing `import` at module scope is invisible to a
structural test and fatal in production — this is what catches it. Pair it with
`python -m pyflakes uiapp/` in CI.
