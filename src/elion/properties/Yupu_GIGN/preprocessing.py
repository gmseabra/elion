# # %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import time
from multiprocessing import Pool
# %%
def generate_pocket(data_dir, distance=5,ligand_dir=None,protein_name=None):
    os.makedirs(ligand_dir+"_save", exist_ok=True)
    complex_id = os.listdir(ligand_dir)
    # # print(obj[-3:])
    # # complex_dir = os.path.join(ligand_dir, cid)
    # lig_native_path = os.path.join(ligand_dir,f'molecule_0.pdb')
    # # lig_native_path = os.path.join(complex_dir, f"{cid}")
    # protein_path= os.path.join(data_dir, f"{protein_name}_target.pdbqt")
    # save_path = ligand_dir+"_save"
    # if not os.path.exists(os.path.join(save_path, f'Pocket_{distance}A.pdb')):
        
    #     pymol.cmd.load(protein_path)
    #     pymol.cmd.remove('resn HOH')
    #     pymol.cmd.load(lig_native_path)
    #     pymol.cmd.remove('hydrogens')
    #     pymol.cmd.select('Pocket', f'byres {molecule_0}_ligand around {distance}')
    #     pymol.cmd.save(os.path.join(save_path, f'Pocket_{distance}A{0}.pdb'), 'Pocket')
    #     pymol.cmd.delete('all')
    for cid in complex_id:
        obj = cid.split('.')[0]
        save_num = obj.split('_')[-1]
        if int(save_num)>10:
            continue
        # print(obj[-3:])
        complex_dir = os.path.join(ligand_dir, cid)
        lig_native_path = complex_dir
        # lig_native_path = os.path.join(complex_dir, f"{cid}")
        protein_path= os.path.join(data_dir, f"{protein_name}_target.pdbqt")
        save_path = ligand_dir+"_save"
        if os.path.exists(os.path.join(save_path, f'Pocket_{distance}A.pdb')):
            continue
        
        print('save_path: %s' % save_path)
        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {obj} around {distance}')
        pymol.cmd.save(os.path.join(save_path, f'Pocket_{distance}A{save_num}.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2',protein_name='10GS-VWW'):
    os.makedirs(os.path.join(data_dir,protein_name)+"_savecomplex", exist_ok=True)
    cid, pKa = str(0), float(0)
    # print()
    complex_dir = os.path.join(data_dir,protein_name)
    pocket_path = os.path.join(complex_dir+'_save',f'Pocket_{distance}A{0}.pdb')
    if input_ligand_format != 'pdb':
        ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
        ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
        os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
    else:
        ligand_path = os.path.join(complex_dir, f'molecule_{cid}.pdb')
    save_path = os.path.join(complex_dir+'_savecomplex', f"{cid}_{distance}A.rdkit")
    ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
    if ligand == None:
        print(f"Unable to process ligand of {cid}")
        # continue

    pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
    if pocket == None:
        print(f"Unable to process protein of {cid}")
        # continue

    complex = (ligand, pocket)
    with open(save_path, 'wb') as f:
        pickle.dump(complex, f)
    # pbar = tqdm(total=10)
    
    for i, row in data_df.iterrows():

        cid, pKa = str(i+1), float(row['rmsd'])
        if i+1>10:
            break
        complex_dir = os.path.join(data_dir,protein_name)
        # complex_dir = data_dir
        # pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        pocket_path = os.path.join(complex_dir+'_save',f'Pocket_{distance}A{cid}.pdb')
        if input_ligand_format != 'pdb':
            ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(complex_dir, f'molecule_{cid}.pdb')
        save_path = os.path.join(complex_dir+'_savecomplex', f"{cid}_{distance}A.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        if ligand == None:
            print(complex_dir)
            print(f"Unable to process ligand of {cid}")
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        # pbar.update(1)
def find_files_with_decoys(directory):
    decoy_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            print('file: %s' % file)
            if 'decoys' in file:
                decoy_files.append(os.path.join(root, file))
    
    print('decoy_files: %s' % decoy_files)
    return decoy_files
def process(i,path):
    distance = 5
    input_ligand_format = 'pdb'
    # root_path = '/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/TEAD3/'
    root_path = '/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/'
    # data_root = os.path.join(root_path,path)
    data_root = root_path
    print('data_root: %s' % data_root)
    complex_path = find_files_with_decoys(data_root)
    for complex_name in complex_path:
        print('complex_name: %s' % complex_name)
        data_name = complex_name.split('_')[0]
        ligand_dir = os.path.join(data_root,data_name)
        # data_dir = os.path.join(data_root, 'toy_set')
        data_df = pd.read_csv(os.path.join(data_root, f'{data_name}_decoy_scores.csv'))
    
        ## generate pocket within 5 Ångström around ligand 
        generate_pocket(data_dir=data_root, distance=distance,ligand_dir=ligand_dir,protein_name=data_name)
        # print(data_df.index)
        generate_complex(data_root, data_df, distance=distance, input_ligand_format=input_ligand_format,protein_name=data_name)
if __name__ == '__main__':
    root_path = '/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/'
    paths = os.listdir(root_path)
    with Pool(processes=80) as pool:
        pool.starmap(process, enumerate(paths))
# from rdkit import Chem
# from rdkit.Chem import AllChem

# ligand_path = '/blue/zhe.jiang/y.zhang1/PDBdata/10/10GS-VWW/molecule_0.pdb'

# # 尝试读取 PDB 文件
# try:
#     ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)

#     # 检查是否成功读取分子
#     if ligand is not None:
#         print("Successfully read the ligand from PDB file.")
#         print(f"Number of atoms: {ligand.GetNumAtoms()}")

#         # 进行一些基本操作，例如生成 2D 坐标
#         AllChem.Compute2DCoords(ligand)

#         # 生成 SMILES 表示
#         smiles = Chem.MolToSmiles(ligand)
#         print(f"SMILES: {smiles}")
#     else:
#         print("Failed to read the ligand from PDB file. The ligand object is None.")
# except Exception as e:
#     print(f"An error occurred while reading the PDB file: {e}")

# %%
