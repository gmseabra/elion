# Tool: attention_visualization
# ─────────────────────────────────────────────────────────────────────────────
# ChemBERT / CHEM-BERT attention-weight visualizer
# Converted from old attn_action_kb.md format + auto-learned entries
# ─────────────────────────────────────────────────────────────────────────────

## action: open_visualizer
**btnId:** adjWeightBtn
**triggers:**
- open the ChemBERT visualizer
- show me ChemBERT
- what is ChemBERT
- what's ChemBERT
- open visualizer
- show 3d
- visualize compound
- single compound
- visualize smiles
- open modal
- open the visualizer
- show visualizer
- launch ChemBERT
- I want to visualize a molecule
- start ChemBERT
- open attention visualizer
- what is chem-bert
- chembert
- chem bert
- attention weight visualization
- open the attention weight tool
**response:** Click the flashing **🧠 ChemBERT** button to open the attention-weight visualizer.

## action: show_3d
**btnId:** runSingle
**triggers:**
- visualize this molecule
- show the 3D view
- run single molecule visualization
- show me the attention map
- generate the 3D attention
- click visualize
- run the visualization
- show 3D
- now what
- then what
- what next
- next step
- then
- and then
- go ahead and visualize
- run it
- hit visualize
- show result
- I entered the SMILES what do I do
- how do I see the results
- I have the SMILES ready
- visualize now
**response:** Enter your SMILES string in the input box, then click the flashing **Visualize** button to generate the 3D attention map.

## action: compare_molecules
**btnId:** runCompare
**triggers:**
- compare two molecules
- compare attention weights
- side by side comparison
- difference between two SMILES
- compare molecules
- run comparison
- show differences
- compare mode
- visualize both molecules
- two compounds
- side by side
**response:** Click **Compare** to load both SMILES strings and visualize their attention weights side by side.

## action: fine_tune_model
**btnId:** runFinetune
**triggers:**
- fine tune the model
- fine-tune ChemBERT
- train on my data
- adapt the model
- retrain
- custom training
- fine tuning
- train ChemBERT
- finetune
- train model
- custom model training
**response:** Click **Fine-Tune** to begin adapting the ChemBERT model on your custom molecular dataset.

## action: load_model
**btnId:** runLoadModel
**triggers:**
- load a model
- load my checkpoint
- load pretrained weights
- load fine-tuned model
- import model
- use saved model
- load weights
- model path
- custom model
- checkpoint
- load checkpoint
**response:** Click **Load Model** to import your saved or fine-tuned ChemBERT checkpoint.

## action: open_vina
**btnId:** vinaBtn
**triggers:**
- open vina docking
- I want to do docking
- switch to vina
- use autodock vina
- molecular docking
- show me the docking tool
- vina
- dock a molecule
- autodock
- open the docking panel
- what is vina
- what's vina
- vina docking
**response:** Click the flashing **🔬 Vina Docking** button to switch to the molecular docking interface.

## action: explain_ask_elion
**btnId:** askElionBtn
**triggers:**
- what is ask elion
- what's ask elion
- what does ask elion do
- what is the ask elion button
- what does the ask elion button do
- explain ask elion
- show me ask elion
- highlight ask elion
- flash ask elion button
- point to ask elion
- where is ask elion
- can you highlight ask elion button
**response:** Click the flashing **🚀 Ask Elion** button to ask Elion questions about your molecule, binding affinity, or attention weights.