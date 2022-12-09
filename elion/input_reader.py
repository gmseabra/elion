# --Python
from typing import List, Dict
import yaml

from pathlib import Path

# Chemistry
import rdkit
from rdkit import Chem


# -----------------------------------------------------------------
# Reads an input file and returns a dictionary of the contents.
# -----------------------------------------------------------------

def read_input_file(input_file_name:str)-> Dict:

    with open(input_file_name,'r') as f:
        cfg = yaml.safe_load(f)
    print("="*60)
    print("Input File")
    print("="*60)
    print(yaml.dump(cfg,indent=5))
    print("="*60)

    # Module root path
    # ----------------
    module_dir = Path(__file__).parent
    cfg['elion_root_dir'] = module_dir

    # Molecule Generator
    # ------------------
    if 'generator' in cfg.keys():
        if 'initial_state' not in cfg['generator'].keys():
            if cfg['generator']['name'].lower() == "release":
                cfg['generator']['initial_state'] = Path(module_dir,"generators/release/checkpoints/generator/checkpoint_biggest_rnn")
            elif cfg['generator']['name'].lower() == "moler":
                cfg['generator']['initial_state'] = Path(module_dir,"generators/moler/PRETRAINED_MODEL")
    
    # Docking Score Predictor
    # -----------------------
    model_type = cfg['reward_properties']['prob_active']['model_type']
    model_file = cfg['reward_properties']['prob_active']['model_file']

    if model_type == "CHEMBERT":
        # CHEMBERT model
        from properties.activity.CHEMBERT.chembert import chembert_model

        print(f"\nInitializiing CHEMBERT with state from file {model_file} ... ", end='')
        activity_model = chembert_model(model_file)
        print("Done.")
    cfg['reward_properties']['prob_active']['predictor'] = activity_model

    # Scaffold
    # --------
    template_smiles_file = cfg['reward_properties']['scaffold_match']['scaffold_file']
    print(f"\nLoading scaffold from {template_smiles_file}. ")
    with open(template_smiles_file,'r') as tf:
        template = tf.readline().strip() 
    template = Chem.MolFromSmarts(template)

    # This prints info, but also forces the info about rings to be calculated.
    # It is necessary because (a bug?) in RDKit that does not calculate the
    # infor about rings until they are requested (or printed)
    print(f"Atom  AtNum  InRing?  Arom?")
    for idx, atom in enumerate(template.GetAtoms()):
        print(f"{idx:>4d}  {atom.GetAtomicNum():5d}  {str(atom.IsInRing()):>7}  {str(atom.GetIsAromatic()):>5}")
    cfg['reward_properties']['scaffold_match']['scaffold'] = template

    # ----------------
    #   JOB CONTROL
    # ----------------

    # Defaults
    if 'control' not in cfg.keys():
        cfg['control'] = {}
        
    if 'gen_start' not in cfg['control']:
        cfg['control']['gen_start'] = 0

    if 'restart' not in cfg['control']:
        cfg['control']['restart'] = False

    if 'verbosity' not in cfg['control']:
        cfg['control']['verbosity'] = 0

    if 'n_iterations' not in cfg['control']:
        cfg['control']['n_iterations'] = 1_000


    if cfg['control']['restart'] == True:

        # Tries to find the info from the biasing_history file
        history_file = Path(cfg['control']['history_file'])
        assert history_file.is_file(), f"File {history_file} does not exist !!"
        with open('biasing_history.csv','r') as f:

            # Finds the last line
            last_line = ""
            for line in f:
                last_line = line
            last_iteration = last_line.split(',')[0]
            cfg['control']['gen_start'] = int(last_iteration) + 1

        print("RESTARTING JOB FROM ITERATION", cfg['control']['gen_start'])

    # Verbosity
    # ---------
        
    return cfg