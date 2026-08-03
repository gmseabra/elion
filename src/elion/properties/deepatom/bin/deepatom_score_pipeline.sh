#!/bin/bash
# =============================================================================
# deepatom_score_pipeline.sh   —   DeepAtom scoring entry-point + dispatcher
#
# Point  deepatom.script:  in input_routes.yml at THIS file.  It serves BOTH
# DeepAtom callers from a single script:
#
#   POSE endpoint  (POST /pose/deepatom_score)
#       pose_routes.py pre-builds  <root>/Dataset_VS/<name>/<name>_complex.pdb
#       (+ <name>_ligand.pdb) and runs us as:  <script> -t vs -d <root>.
#       We score that one pre-built complex end-to-end and write the results
#       CSV the card parses — see "POSE PATH" below.
#
#   BATCH endpoint (POST /vina_visualization/deepatom_estimate)
#       deepatom_routes.py passes a dataset dir that already contains
#       ligands_in_bound_pose/ + protein/.  We delegate UNCHANGED to the
#       original orchestrator, so the existing batch pipeline is untouched.
#
# POSE PATH writes, under the server output dir <root>:
#       <root>/atomtypes/<name>.atomtypes        generate_atomtypes.py 3,5
#       <root>/3d_32_24_pcmax/<name>.npz         generate_npz.py  non_augmented
#       <root>/3d_32_24_pcmax_aug/               (created empty — see note below)
#       <root>/vs_<dataset>.csv                  test.py  (cols: PDB, deltaG_kcal_mol)
#   pose_routes._read_deepatom_csv reads PDB + deltaG_kcal_mol -> pred_pk=-dG/1.36.
#
# WHY the empty 3d_32_24_pcmax_aug/ dir:
#   test.py requires avg_test=True (test_y_pred is only assigned under that
#   branch), and dataloader.py's vs branch then does
#   listdir(<batch_dir>/3d_32_24_pcmax_aug) — which raises if the dir is
#   missing.  An empty dir makes multi_avg() average the single original grid
#   (a valid one-pose score).  For augmentation-averaged scores that match the
#   batch pipeline, run with AUGMENT=1 (needs 02c_augment_no_Chimera_VS.py).
# =============================================================================

set -o pipefail

# ---- CONFIG: the two paths that must match your install ---------------------
DEEPATOM_ROOT_DIR="/home/huangzihang/repos/elion/src/elion/properties"
# Original batch orchestrator (keep it where it is; we delegate to it):
ORCHESTRATOR="$DEEPATOM_ROOT_DIR/deepatom/bin/predict_binding_affinity_v4_2_data_split_8P0M.sh"
# Set AUGMENT=1 in the environment to also build augmented grids (slower; needs 02c_augment).
AUGMENT="${AUGMENT:-0}"
# -----------------------------------------------------------------------------

usage() { echo "Usage: $0 -t <test_type> -d <server_output_dir>" >&2; }

input_type="vs"
input_dir=""
while getopts ":ht:d:" OPTION; do
    case "$OPTION" in
        h) usage; exit 1 ;;
        t) input_type="$OPTARG" ;;
        d) input_dir="$OPTARG" ;;
        ?) usage; exit 1 ;;
    esac
done
if [ -z "$input_dir" ]; then usage; exit 1; fi

SCRIPTS_DIR="$DEEPATOM_ROOT_DIR/deepatom/model_split_data"
PRE_DIR="$SCRIPTS_DIR/00_preprocess"
GEN_DIR="$SCRIPTS_DIR/01_generate_channels"
PRED_DIR="$SCRIPTS_DIR/02_pytorch"
MODEL_DIR="$SCRIPTS_DIR/model"

root="${input_dir%/}"
dataset_name="${root##*/}"

echo "[driver] root=$root  test_type=$input_type  dataset=$dataset_name  AUGMENT=$AUGMENT"

# ---- DISPATCH ---------------------------------------------------------------
if ls "$root"/Dataset_VS/*/*_complex.pdb >/dev/null 2>&1; then
    echo "[driver] pre-built *_complex.pdb found -> POSE path (single-compound scoring)"
elif [ -d "$root/ligands_in_bound_pose" ] && [ -d "$root/protein" ]; then
    echo "[driver] raw inputs found -> delegating to batch orchestrator (unchanged)"
    if [ ! -f "$ORCHESTRATOR" ]; then
        echo "[driver] ERROR: batch orchestrator not found: $ORCHESTRATOR" >&2; exit 5
    fi
    exec /bin/bash "$ORCHESTRATOR" -t "$input_type" -d "$root"
else
    echo "[driver] ERROR: under $root, found neither" >&2
    echo "                Dataset_VS/*/*_complex.pdb  nor  ligands_in_bound_pose/ + protein/" >&2
    exit 2
fi

# ============================ POSE PATH ======================================
if [ "$AUGMENT" = "1" ]; then
    ATOM_STAGES="3,4,5,6"      # copy non-aug, augment, arpeggio non-aug, arpeggio aug
    NPZ_STAGE="both"
else
    ATOM_STAGES="3,5"          # copy non-aug, arpeggio non-aug  (== pipeline_VS --stages 1,3)
    NPZ_STAGE="non_augmented"
fi

echo "[driver] POSE path   pwd=$(pwd)"
echo "[driver]   1) $PRE_DIR/generate_atomtypes.py --batch-dir $root --stages $ATOM_STAGES"
echo "[driver]   2) $GEN_DIR/generate_npz.py $root $dataset_name --stage $NPZ_STAGE"
echo "[driver]   3) $PRED_DIR/test.py --batch_dir=$root --test_type=$input_type --test_dataset=$dataset_name"

# ---- 1) atom types (arpeggio): -> <root>/atomtypes/<name>.atomtypes ---------
python "$PRE_DIR/generate_atomtypes.py" \
    --batch-dir   "$root" \
    --pre-dir     "$PRE_DIR" \
    --scripts-dir "$SCRIPTS_DIR" \
    --stages      "$ATOM_STAGES" || { echo "[driver] generate_atomtypes.py failed" >&2; exit 3; }

# fail loudly if atom-typing wrote nothing (almost always arpeggio missing in the env),
# so the error names the real cause instead of surfacing as a downstream test.py crash
_nat=$(find "$root/atomtypes" -name '*.atomtypes' 2>/dev/null | wc -l)
if [ "$_nat" -eq 0 ]; then
    echo "[driver] ERROR: atom-typing produced 0 .atomtypes in $root/atomtypes" >&2
    echo "[driver]   -> arpeggio is likely not on PATH / failing inside the elion_backend env." >&2
    echo "[driver]   -> check:  which arpeggio   and the arpeggio_mod2 scripts under $SCRIPTS_DIR" >&2
    exit 7
fi

# ---- 2) voxel grid npz: -> <root>/3d_32_24_pcmax/<name>.npz -----------------
python "$GEN_DIR/generate_npz.py"  "$root"  "$dataset_name"  --stage "$NPZ_STAGE" \
    || { echo "[driver] generate_npz.py failed" >&2; exit 4; }

# fail loudly if grid generation wrote nothing
_nnpz=$(find "$root/3d_32_24_pcmax" -name '*.npz' 2>/dev/null | wc -l)
if [ "$_nnpz" -eq 0 ]; then
    echo "[driver] ERROR: grid generation produced 0 .npz in $root/3d_32_24_pcmax" >&2
    echo "[driver]   -> check config.py (anolea_vdw_dict / ANOLEA_SIZE) next to generate_npz.py." >&2
    exit 8
fi

# ---- 3) guarantee the aug grid dir exists (dataloader avg_test lists it) ----
mkdir -p "$root/3d_32_24_pcmax_aug"

echo "[driver] atomtypes: $(find "$root/atomtypes" -name '*.atomtypes' 2>/dev/null | wc -l)  npz: $(find "$root/3d_32_24_pcmax" -name '*.npz' 2>/dev/null | wc -l)"

# ---- 4) CNN inference -> <root>/vs_<dataset>.csv (PDB, deltaG_kcal_mol) -----
#     test.py chdirs to its own dir and reads npz from <batch_dir>/3d_32_24_pcmax(_aug);
#     batch_dir is absolute so the chdir is harmless.  Writes <test_type>_<test_dataset>.csv.
python "$PRED_DIR/test.py" \
    --batch_dir="$root" \
    --test_type="$input_type" \
    --test_dataset="$dataset_name" \
    --model_dir="$MODEL_DIR" \
    --out_csv_dir="$root" || { echo "[driver] test.py (CNN inference) failed" >&2; exit 6; }

echo "[driver] done -> $(find "$root" -maxdepth 1 -name "${input_type}_${dataset_name}.csv")"