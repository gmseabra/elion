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
│   ├── llm/                   # Qwen client + standalone inference server
│   │   ├── qwen_client.py
│   │   ├── qwen_server.py
│   │   └── serve_qwen.sh
│   ├── router/                # RAG UI-action router
│   │   ├── ui_action_router.py
│   │   ├── kb_auto_learner.py
│   │   └── ui_action_kb/      # human-authored action knowledge base (*.md)
│   ├── routes/                # Flask blueprints (hub, attn, vina_*, ts, tools, deepatom)
│   └── CHEMBERT/              # ChemBERT model package
├── web/                       # frontend
│   ├── templates/             # Jinja2 modals + hub
│   └── static/js/             # vanilla-JS visualizers
├── engines/
│   └── vina/                  # vendored AutoDock Vina-GPU (binaries + OpenCL kernels)
├── pipeline/
│   └── active_learning/       # Thompson-Sampling → PDB → docking → distribution-shift scripts
├── data/                      # runtime data (gitkept, contents ignored)
│   ├── chembert/              # sample SMILES / vocab
│   ├── converted_pdbqt/       # prepared receptors + ligands
│   └── kb_shadow/             # auto-generated shadow KB + CoT logs
└── tests/
    └── test_structure.py      # dependency-free layout / path-hygiene checks
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
| `ELION_CWD` / `ELION_VENV` | sibling Elion engine + its Python | `../elion/src/elion`, current Python |
| `DEEPATOM_SCRIPT` / `DEEPATOM_DATA_DIR` / `DEEPATOM_ROOT` | external DeepAtom project | unset |

`config/input_routes.yml` stores docking targets and DeepAtom datasets with
repo-relative paths; `run.py` resolves them to absolute paths at load time.

## Components

The **UI** package is the base application (originally the *visualizer*). The
**vina** engine and the **active-learning pipeline** are vendored alongside it:
docking runs shell out to `engines/vina`, and the `pipeline/active_learning`
scripts implement the Thompson-Sampling loop (extract PDBs from TS results →
dock → chart the vina score distribution shift) that the TS routes drive.

## Tests

```bash
python tests/test_structure.py     # or: python -m pytest tests/
```

These verify the package layout, that the route loader agrees with the files on
disk, and that no user-specific absolute paths have returned.
