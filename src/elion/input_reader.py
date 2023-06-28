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

    # If this is a restart, sets the history acccordingly
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

    # -------------------------
    # Molecular Rewards Funtion
    # -------------------------
    # We don't initialize anything here. The properties are initialized
    # by instantiating the 'Estimators' class.
    
    if 'reward_function' in cfg_input.keys():
        cfg['Reward_function'] = cfg_input['reward_function']

    # For DEBUG purposese, print the whole configuration
    if cfg['Control']['verbosity'] > 2:
        print("="*80)
        print(f"{'FINAL CONFIGURATION'}:^80s")
        print("="*80)
        pprint.pprint(cfg)
        print("="*80)

    return cfg

if __name__ == "__main__":
    # Just for testing purposes
    
    import pprint
    root_dir = Path().cwd()
    input_file = Path(root_dir,"elion/input_example.yml")
    result = read_input_file(input_file)
    for prop, cls in result['Reward_function'].items():
        print(prop, cls.value())