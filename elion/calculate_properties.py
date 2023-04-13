from pathlib import Path
from rdkit import Chem

import input_reader

if __name__ == "__main__":
    # Just for testing purposes
    import pprint
    example_mols = [
        "CCCCCCCCCCCCCCCN1N=C(CCC)C2=C1C(=O)NC(C1=CC(S(=O)(=O)N3CCN(CC4=CC=C(F)C=C4)CC3)=CC=C1OCC)=N2",
        "CCCCCCCCCCCCCCCN1N=C(CCC)C2=C1C(=O)NC(C1=CC(S(=O)(=O)N3CCN(CC4=CC=CC=C4)CC3)=CC=C1OCC)=N2",
        "CCCCCCC1=CC=C(CN2N=C(CCC)C3=C2C(=O)NC(C2=CC(S(=O)(=O)N4CCN(CCC5=CC=CS5)CC4)=CC=C2OCC)=N3)C=C1",
        "CCCCCCCCCCCCCCCCN1N=C(CCC)C2=C1C(=O)NC(C1=CC(S(=O)(=O)N3CCN(C(=O)C4=CC=CC=C4)CC3)=CC=C1OCC)=N2",
        "CCCCCOCCOCCOCCCN1N=C(CCC)C2=C1C(=O)NC(C1=CC(S(=O)(=O)N3CCN(C)CC3)=CC=C1OCC)=N2",
        "CCCCCCCCCCCCCCCCCCCCCCCN1N=C(CCC)C2=C1C(=O)NC(C1=CC(S(=O)(=O)N3CCN(C)CC3)=CC=C1OCC)=N2",
        "CCCC1=NN(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCO)C2=C1N=C(C1=CC(S(=O)(=O)N3CCN(C(O)=NO)CC3)=CC=C1OCC)NC2=O",
        "CCCCCCCCCCCCCCCCCCCCCCN1N=C(CCC)C2=C1C(=O)NC(C1=CC(S(=O)(=O)N3CCN(C(=N)S)CC3)=CC=C1OCC)=N2",
        "CCCCCCCC1=CC=C(C2=CSC(CCCCCCN3CCN(S(=O)(=O)C4=CC=C(OCC)C(C5=NC6=C(C(=O)N5)N(C)N=C6CCC)=C4)CC3)=N2)C=C1",
        "CCCCCCCCCCCCCN1N=C(CCC)C2=C1C(=O)NC(C1=CC(S(=O)(=O)N3CCN(CC4CCCN4C(=O)OC)CC3)=CC=C1OCC)=N2",
    ]

    root_dir = Path().cwd()
    input_file = Path(root_dir,"elion/input_example.yml")
    config = input_reader.read_input_file(input_file)
    if config['Control']['verbosity'] > 0:
        pprint.pprint(config)

    # For pretty-printing purposes, find the longest
    # SMILES string to print
    length_lim = 0
    for mol in example_mols:
        if len(mol) > length_lim:
            length_lim = len(mol)
    if length_lim > 30:
        length_lim = 30

    # Properties
    print(f"\nProperties")
    print(f"\t{'Molecule':{length_lim+3}s}", end="")
    for prop, cls in config['Properties'].items():
        print(f"\t{prop}", end="")
    print("")
    
    for ind, mol in enumerate(example_mols):
        print(f"{ind}\t{mol:{length_lim}.{length_lim}s}...", end="")

        for prop, cls in config['Properties'].items():
            title_len = len(prop)
            value = cls.value(Chem.MolFromSmiles(mol))
            print(f"\t{value:^{title_len}.2f}", end="")
        print("")

    # Rewards
    print(f"\nRewards")
    print(f"\t{'Molecule':{length_lim+3}s}", end="")
    for prop, cls in config['Properties'].items():
        print(f"\t{prop}", end="")
    print("")
    
    for ind, mol in enumerate(example_mols):
        print(f"{ind}\t{mol:{length_lim}.{length_lim}s}...", end="")

        for prop, cls in config['Properties'].items():
            title_len = len(prop)
            value = cls.value(Chem.MolFromSmiles(mol))
            reward = cls.reward(value)
            print(f"\t{reward:^{title_len}.2f}", end="")
        print("")