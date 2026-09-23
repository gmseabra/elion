# GatorAffinity

**GatorAffinity** is a geometric deep learning model for protein–ligand binding affinity prediction. 
It leverages large-scale synthetic structural data, including over 1.45 million protein–ligand complexes sourced from the jointly released [GatorAffinity-DB](https://huggingface.co/datasets/AIDD-LiLab/GatorAffinity-DB) (over 450,000 complexes with K<sub>d</sub>/K<sub>i</sub> values) and the SAIR dataset (over 1 million IC<sub>50</sub>-annotated complexes). 
The model is pre-trained on these synthetic complexes and subsequently fine-tuned using experimental structures from PDBbind, enabling accurate and generalizable affinity prediction.
For further details, please refer to the [GatorAffinity](https://www.biorxiv.org/content/10.1101/2025.09.29.679384v1) paper.

![](./assets/flowchart.png)

## Synthetic Dataset at Scale

![](./assets/dataset.png)

- **450K+ Kd/Ki complexes** generated using Boltz-1 [[4]](#references) structure prediction 
- **1M+ IC50 complexes** from SAIR database [[1]](#references)  
- **Total: 1.5M synthetic protein-ligand pairs for pre-training**
<div style="text-align: center;">
  <img src="./assets/scale.png" alt="GatorAffinity" width="400">
</div>

## Installation

### Environment:
```bash
git clone https://github.com/AIDD-LiLab/GatorAffinity.git
cd GatorAffinity
bash environment.sh
```

### Data Download

#### Original Structural Data
1. **[GatorAffinity-DB Complete Original Data](https://huggingface.co/datasets/AIDD-LiLab/GatorAffinity-DB)**
2. **[SAIR Complete Original Data](https://www.sandboxaq.com/sair)**

#### Preprocessed Data
1. **[Synthetic kd+Ki+IC50 data for GatorAffinity Pre-training](https://huggingface.co/datasets/AIDD-LiLab/GatorAffinity-Processed-For-Pretraining)**
2. **filtered LP-PDBbind For Fine-tuning** - `./LP-PDBbind`


## Model Checkpoints

### Pre-trained Models
- **Base model**: Pre-trained on IC50+Kd+Ki datasets  
  `./model_checkpoints/Kd+Ki+IC50_pretrain.ckpt`

- **Fine-tuned model** (best performance): Pre-trained on IC50+Kd+Ki, fine-tuned on experimental structures with LP-PDBbind split  
  `./model_checkpoints/Kd+Ki+IC50_experimental_fine_tuning.ckpt`

### ATOMICA Backbone
ATOMICA-Universal atomic scale molecular interaction representation model used as GatorAffinity's backbone.  
**[Download ATOMICA Checkpoints](https://huggingface.co/ada-f/ATOMICA/tree/main/ATOMICA_checkpoints/pretrain)**

**Note**: Our experiments show that ATOMICA backbone significantly improves performance with limited pre-training structures, though benefits diminish as synthetic training data increases.


## Usage


### Training
```bash
python train.py \
    --train_set_path LP-PDBbind/train.pkl \
    --valid_set_path LP-PDBbind/valid.pkl \
    --pretrain_ckpt model_checkpoints/Kd+Ki+IC50_pretrain.ckpt
```

### Inference
```bash
python inference.py \
    --model_ckpt model_checkpoints/Kd+Ki+IC50_experimental_fine_tuning.ckpt \
    --test_set_path LP-PDBbind/test.pkl
```

### Single-Pose Scoring (`gatoraffinity_pose.py`)

`inference.py` is a benchmark driver: it wants a pre-processed `.pkl`, a label
column and a batch, and it prints RMSE / Pearson / CI. When you have **one posed
complex** and want **one number** — a docking pose to rescore, an interactive
tool asking for an affinity — use `gatoraffinity_pose.py` instead:

```bash
python gatoraffinity_pose.py --stage /path/to/stage --name mypose
```

It reads three files from `--stage`, all named after `--name`:

| File | What it is |
|------|-----------|
| `<name>_ligand.pdb` | the posed ligand, HETATM records, heavy atoms |
| `<name>_protein.pdb` | the receptor (a whole protein is fine — see below) |
| `<name>_meta.json` | everything else: `smiles`, `pocket_cutoff`, `drop_water`, `model`, `device`, `fragmentation_method` |

and prints, on stdout:

```
GATOR_PRED_PK 6.8412
GATOR_DELTAG  -9.3040
```

Everything else it writes goes beside the inputs: `<name>_pocket.pdb` (exactly
what the model saw), `<name>_ligand_named.pdb`, and `<name>_result.json`.

Three things it does that `data/process_pdbs.py` does not, because they only
come up when the input is a live pose rather than a curated dataset:

1. **It cuts the pocket.** `process_pdbs.py` passes `protein_pdb` straight to
   `pdb_to_list_blocks` — correct there, because its input CSV already names
   pockets (`1a4h_pocket_5A.pdb`). A pose from a docking tool comes with the
   whole receptor, and feeding that in is not "slower and equivalent": the block
   set is a different distribution from the 5 Å pockets the model was fine-tuned
   on. Default `pocket_cutoff: 5`, byres, waters and hydrogens dropped. Set it
   to `0` to pass the full receptor and accept the mismatch.

2. **It gives ligand atoms unique names.** Biopython keys a residue's atoms by
   name and silently discards duplicates. Tools that name every atom after its
   element — `C`, `C`, `C`, `O`, … — therefore lose almost the whole ligand: a
   40-atom molecule parses as 3 atoms, fragmentation fails, `extract_pdb_ligand`
   falls back to per-atom blocks, and the model returns a confident number for
   the wrong molecule. The atom count is checked after parsing and a mismatch is
   refused rather than scored.

3. **It says when fragmentation failed.** `extract_pdb_ligand` catches that
   exception internally and degrades to one block per atom. That is a different
   input distribution from PS_300 fragments, so it is called out in the log
   instead of passing as a normal result.

The output is a **pK** (the LP-PDBbind `label` column, pKd/pKi), matching what
`inference.py` reports; ΔG is the conventional `-1.36 x pK`.

### Custom Data Processing

GatorAffinity supports processing your own PDB data for training and inference.

#### Example Data

We provide example data in `data/example/` to help you get started:
- `1a4h_pocket_5A.pdb`, `1a4h_ligand.pdb`: Example protein pocket and ligand structure
- `1bux_pocket_5A.pdb`, `1bux_ligand.pdb`: Example protein pocket and ligand structure
- `example.csv`: Example data index file
- `example.pkl`: Pre-processed example data

#### Data Format

Create a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `pdb_id` | PDB identifier | `1a4h` |
| `protein_pdb` | Path to protein pocket PDB file | `data/example/1a4h_pocket_5A.pdb` |
| `ligand_pdb` | Path to ligand PDB file | `data/example/1a4h_ligand.pdb` |
| `protein_chains` | Protein chain(s) for pocket | `A` or `A_B` for multiple chains |
| `lig_code` | Ligand residue name | `UNL`, `LIG`, `ATP` |
| `smiles` | Ligand SMILES string | `CCO`, `c1ccccc1` |
| `lig_resi` | Ligand residue number | `1`, `100` |
| `label` | Binding affinity label (pKd/pKi) | `5.92`, `4.85` |

#### Processing Your Data

```bash
python data/process_pdbs.py \
    --data_index_file your_data.csv \
    --out_path processed_data.pkl
```

#### Example with Provided Data

```bash
python data/process_pdbs.py \
    --data_index_file data/example/example.csv \
    --out_path data/example/example.pkl
```


## Performance

**State-of-the-art on filtered LP-PDBbind [[2]](#references):**

![](./assets/lp_pdbbind.png)

## License

This repository is licensed under two different licenses:

### Main Repository - MIT License
The source code, documentation, and most files are licensed under the [MIT License](./LICENSE).

### Model Checkpoints - CC BY-NC-SA 4.0
The model checkpoints in the `./model_checkpoints/` directory are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](./model_checkpoints/LICENSE).

**Model checkpoints:**
- `Kd+Ki+IC50_pretrain.ckpt`
- `Kd+Ki+IC50_experimental_fine_tuning.ckpt`

### Other Data
For the license of other data, please refer to the specific license file provided by the repository.


## References

[1] Lemos, P., Beckwith, Z., Bandi, S., Van Damme, M., Crivelli-Decker, J., Shields, B.J., Merth, T., Jha, P.K., De Mitri, N., Callahan, T.J., et al. (2025). SAIR: Enabling deep learning for protein-ligand interactions with a synthetic structural dataset. *bioRxiv*.

[2] Wang, Y., Sun, K., Li, J., Guan, X., Zhang, O., Bagni, D., Zhang, Y., Carlson, H.A., Head-Gordon, T. (2025). A workflow to create a high-quality protein–ligand binding dataset for training, validation, and prediction tasks. *Digital Discovery*, 4(5), 1209-1220.

[3] Fang, A., Zhang, Z., Zhou, A., and Zitnik, M. (2025). ATOMICA: Learning Universal Representations of Intermolecular Interactions. *bioRxiv*.

[4] Wohlwend, J., Corso, G., Passaro, S., Reveiz, M., Leidal, K., Swiderski, W., Portnoi, T., Chinn, I., Silterra, J., Jaakkola, T., et al. (2024). Boltz-1: Democratizing biomolecular interaction modeling. *bioRxiv*.

## Acknowledgments

This work builds upon [ATOMICA](https://github.com/mims-harvard/ATOMICA) framework. We thank the ATOMICA authors for making their codebase available.

## Citation
If you use the code or data in this package, please cite:
```bibtex
@article{wei2025gatoraffinity,
  title={GatorAffinity: Boosting Protein-Ligand Binding Affinity Prediction with Large-Scale Synthetic Structural Data},
  author={Wei, Jinhang and Zhang, Yupu and Ramdhan, Peter A and Huang, Zihang and Seabra, Gustavo and Jiang, Zhe and Li, Chenglong and Li, Yanjun},
  journal={bioRxiv},
  pages={2025--09},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
