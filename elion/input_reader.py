"""
    # ----------------------------------------------------------------------- 
    # Reads the Input file and initializes needed objects.
    # Whenever possible, delegates error handling to the class being loaded.
    # ----------------------------------------------------------------------- 
Returns:
    config (dict): A dictionary with all the information.
"""
import importlib
from typing import Dict
from pathlib import Path
import yaml

# Chemistry
# import rdkit
# from rdkit import Chem


# -----------------------------------------------------------------
# Reads an input file and returns a dictionary of the contents.
# -----------------------------------------------------------------

def read_input_file(input_file_name:str)-> Dict:

    # Reads the input file
    # --------------------
    with open(input_file_name,'r') as f:
        cfg_input = yaml.safe_load(f)
    print("="*60)
    print("Input File")
    print("="*60)
    print(yaml.dump(cfg_input,indent=5))
    print("="*60)

    # ---------------------------------------------------------------
    #                   PROCESSING THE INPUT FILE
    # ---------------------------------------------------------------
    cfg = {}

    # Module root path
    # ----------------
    module_dir = Path(__file__).parent
    cfg['elion_root_dir'] = module_dir

    # ---------------
    # Control Options
    # ---------------
    # Define defaults
    ctrl = {
            'history_file': 'biasing_history.csv',
            'n_iterations': 1_000,
            'max_iter': 1_000,
            'gen_start': 0,
            'restart': False,
            'verbosity': 0,
           }

    # Get new / modified paramters from input file
    if 'control' in cfg_input.keys():
        for option, value in cfg_input['control'].items():
            ctrl[option] = value

    # IF this is a restart, sets the history acccordingly
    if ctrl['restart']:
        
        # Tries to find the info from the biasing_history file
        history_file = Path(ctrl['history_file'])
        if not history_file.is_file():
            # BOMB INPUT
            msg = (f"ERROR while reading CONTROL session:\n"
                   f"History file <{history_file.absolute()}> not found.")
            quit(msg)

        with open(history_file,'r') as f:
            # Finds the last line
            last_line = ""
            for line in f:
                last_line = line
            last_iteration = last_line.split(',')[0]
            ctrl['gen_start'] = int(last_iteration) + 1

        print("RESTARTING JOB FROM ITERATION", ctrl['gen_start'])
    cfg['Control'] = ctrl

    # ------------------
    # Molecule Generator
    # ------------------
    if 'generator' in cfg_input.keys():
        generator = cfg_input['generator']

        if 'initial_state' not in generator.keys():
            if generator['name'].lower() == "release":
                generator['initial_state'] = Path(module_dir,"generators/release/checkpoints/generator/checkpoint_biggest_rnn")
            elif generator['name'].lower() == "moler":
                generator['initial_state'] = Path(module_dir,"generators/moler/PRETRAINED_MODEL")
        cfg['Generator'] = generator

    # -------------------------------
    # Molecular Properties Prediction
    # -------------------------------
    # The 'Properties' dict will contain OBJECTS to calculate properties and rewards.
    # Each of those objects must implement at least 2 methods: 
    #   1) 'value':  Gets an RDKit Mol object and returns a property value; and
    #   2) 'reward': Gets a property value and returns a reward. 
    properties = {}
    if 'properties' in cfg_input.keys():
        for prop in cfg_input['properties'].keys():
            module = importlib.import_module(f'properties.{prop}')
            module = getattr(module, prop)
            properties[prop] = module(prop,**cfg_input['properties'][prop])
        print("Done reading properties.")
        print("="*80)
        print("\n")
    cfg['Properties'] = properties

    return cfg

if __name__ == "__main__":
    # Just for testing purposes
    
    import pprint
    root_dir = Path().cwd()
    input_file = Path(root_dir,"elion/input_example.yml")
    result = read_input_file(input_file)
    if result['Control']['verbosity'] > 0:
        pprint.pprint(result)
    for prop, cls in result['Properties'].items():
        print(prop, cls.value())

# OLD STUFF (REMOVED)
# Keeping it here just in case I need it back.

    # # Docking Score Predictor
    # # -----------------------
    # model_type = cfg['reward_properties']['prob_active']['model_type']
    # model_file = cfg['reward_properties']['prob_active']['model_file']

    # if model_type == "CHEMBERT":
    #     # CHEMBERT model
    #     from properties.activity.CHEMBERT.chembert import chembert_model

    #     print(f"\nInitializiing CHEMBERT with state from file {model_file} ... ", end='')
    #     activity_model = chembert_model(model_file)
    #     print("Done.")
    # cfg['reward_properties']['prob_active']['predictor'] = activity_model

    # # Scaffold
    # # --------
    # template_smiles_file = cfg['reward_properties']['scaffold_match']['scaffold_file']
    # print(f"\nLoading scaffold from {template_smiles_file}. ")
    # with open(template_smiles_file,'r') as tf:
    #     template = tf.readline().strip() 
    # template = Chem.MolFromSmarts(template)

    # # This prints info, but also forces the info about rings to be calculated.
    # # It is necessary because (a bug?) in RDKit that does not calculate the
    # # infor about rings until they are requested (or printed)
    # print(f"Atom  AtNum  InRing?  Arom?")
    # for idx, atom in enumerate(template.GetAtoms()):
    #     print(f"{idx:>4d}  {atom.GetAtomicNum():5d}  {str(atom.IsInRing()):>7}  {str(atom.GetIsAromatic()):>5}")
    # cfg['reward_properties']['scaffold_match']['scaffold'] = template
