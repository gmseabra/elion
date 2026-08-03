import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from pymol import cmd
from rdkit import RDLogger
import os
import time
from multiprocessing import Pool
RDLogger.DisableLog('rdApp.*')


def convert_pdbqt_to_pdb(input_pdbqt, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_pdb = os.path.join(output_dir,f'molecule_0.pdb')
    command = f'obabel {input_pdbqt} -O {output_pdb}'
    
    os.system(command)
    # print(f"Converted {input_pdbqt} to {output_pdb}")
def split_pdbqt_to_pdb(input_pdbqt, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_pdbqt, 'r') as file:
        data = file.read()
    
    molecules = data.split('ENDMDL\n')
    
    for i, mol in enumerate(molecules):
        if mol.strip(): 
            pdbqt_filename = os.path.join(output_dir, f'molecule_{i+1}.pdbqt')
            pdb_filename = os.path.join(output_dir, f'molecule_{i+1}.pdb')
            
            with open(pdbqt_filename, 'w') as mol_file:
                mol_file.write(mol + 'ENDMDL\n')
            
            os.system(f'obabel {pdbqt_filename} -O {pdb_filename}')
        if i>10:
            break
def find_files_with_decoys(directory):
    decoy_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'decoys' in file:
                decoy_files.append(os.path.join(root, file))
    
    return decoy_files
def startProcess(i,path):
    root_path = "/blue/zhe.jiang/y.zhang1/PDBdata/"
    input_path = os.path.join(root_path,path)
    input_pdb=find_files_with_decoys(input_path)
    for file in input_pdb:
# input_pdbqt = "/blue/zhe.jiang/y.zhang1/PDBdata/10/10GS-VWW_decoys.pdbqt"
# output_dir = '/blue/zhe.jiang/y.zhang1/PDBdata/10/10GS-VWW/'   # 输出目录
        file_ligand = file.split('_')[0]+"_ligand.pdbqt"
        
        out_dir = file.split(os.path.sep)[-1][:8]
        out_dir = os.path.join(input_path,out_dir)
        # print(out_dir)
        convert_pdbqt_to_pdb(file_ligand,out_dir)
        # print(file)
        # print(out_dir)
        # split_pdbqt_to_pdb(file, out_dir)
def main():
    root_path = "/blue/zhe.jiang/y.zhang1/PDBdata/"
    paths = os.listdir(root_path)
    # for path in paths:
    with Pool(processes=80) as pool:
        pool.starmap(startProcess, enumerate(paths))
if __name__=='__main__':
    main()