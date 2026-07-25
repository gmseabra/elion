# Tool: vina_visualization
# ─────────────────────────────────────────────────────────────────────────────
# AutoDock Vina molecular docking visualizer
# Converted from old vina_action_kb.md format to ## action: format
# ─────────────────────────────────────────────────────────────────────────────

## action: open_vina
**btnId:** vinaBtn
**triggers:**
- open vina docking
- I want to dock a molecule
- start docking
- molecular docking
- autodock vina
- launch vina
- use vina
- what is vina
- what's vina
- how does vina work
- show me the docking panel
- open docking tool
- docking interface
- run docking
- dock my ligand
- protein-ligand docking
- binding pose prediction
- virtual screening
- I want to predict binding
- open the visualizer
- show the 3D view
- view molecule
- show molecule
**response:** Click the flashing **🔬 Vina Docking** button to open the AutoDock Vina interface.

## action: run_docking
**btnId:** vinaDockBtn
**triggers:**
- run the docking
- start docking now
- dock the ligand
- execute docking
- submit docking job
- go ahead and dock
- perform docking
- click dock
- hit dock
- run vina
- calculate binding
- now what
- what now
- next step
- what next
- what do I do now
- what do I click
- what should I do
- I'm ready
- proceed
- paths are set
- both paths are ready
- then what
- then
- and then
- ready to dock
**response:** Click the flashing **Dock** button to start the AutoDock Vina calculation. Results will stream in the terminal below.

## action: set_ligand_path
**btnId:** vinaLigandPath
**triggers:**
- set the ligand file
- enter ligand path
- specify ligand pdbqt
- where do I put the ligand file
- ligand path input
- set ligand
- ligand file location
- load ligand
- enter ligand
- load the pdbqt ligand
- load lig
- pdbqt ligand
**response:** Enter the path to your ligand **.pdbqt** file in the **Ligand Path** field.

## action: set_receptor_path
**btnId:** vinaReceptorPath
**triggers:**
- set the receptor file
- enter receptor path
- specify receptor pdbqt
- where do I put the protein file
- receptor path input
- set receptor
- protein file location
- load receptor
- enter receptor path
- load the protein
- receptor file
- pdbqt receptor
**response:** Enter the path to your receptor **.pdbqt** file in the **Receptor Path** field.

## action: set_config_path
**btnId:** vinaConfigPath
**triggers:**
- set the config file
- enter config path
- vina configuration file
- where is the config
- set config
- config file location
- my config is at
- configuration path
- set up the search box
- grid box config
- search space config
**response:** Enter the path to your Vina **config.txt** file in the **Config Path** field.

## action: open_chembert
**btnId:** adjWeightBtn
**triggers:**
- open ChemBERT
- switch to ChemBERT
- I want to visualize attention
- go to attention visualizer
- show chembert
- use the attention tool
- what is chembert
- what's chembert
**response:** Click the flashing **🧠 ChemBERT** button to switch to the attention-weight visualizer.