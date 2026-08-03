# %%
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset,DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import  Data
import warnings
import time
from multiprocessing import Pool
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5.):

    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p = mol2graph(pocket)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
    y = torch.FloatTensor([label])
    pos = torch.concat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
    
    data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)

    torch.save(data, save_path)
    # return data

# %%
# class PLIDataLoader(DataLoader):
#     def __init__(self, data, **kwargs):
#         super().__init__(data, collate_fn=data.collate_fn, **kwargs)
def find_files_with_decoys(directory):
    decoy_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'decoys' in file:
                decoy_files.append(os.path.join(root, file))
    
    return decoy_files
def pre_process(i,path,dis_threshold=5):
    root_path = "/blue/zhe.jiang/y.zhang1/PDBdata"
    complexes = find_files_with_decoys(os.path.join(root_path,path))
    complex_name = [l.split(os.sep)[-1].split('_')[0] for l in complexes]
    complex_scores = [l.split('_')[0]+'_decoy_scores.csv' for l in complexes]
    save_path = "/blue/zhe.jiang/y.zhang1/proteinData/pretrain521/"
    for k in range(len(complex_name)):
        data_name = complex_name[k]
        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        dis_thresholds = []
        data_df = pd.read_csv(complex_scores[k])
        cid, pKa = data_name+'_0', float(0)
        data_dir = os.path.join(os.path.join(root_path,path),data_name+'_savecomplex')
        if(len(os.listdir(data_dir)))<11:
            continue
        complex_dir = os.path.join(save_path, data_name)
        if not os.path.exists(complex_dir):
            os.makedirs(complex_dir)
        # complex_dir = save_path+data_name
        graph_path = os.path.join(complex_dir, f"GIGN_Graph-{cid}_{dis_threshold}A.pyg")
        complex_path = os.path.join(data_dir, f"{0}_{dis_threshold}A.rdkit")

        complex_path_list.append(complex_path)
        complex_id_list.append(cid)
        pKa_list.append(pKa)
        dis_thresholds.append(dis_threshold)
        graph_path_list.append(graph_path)
        for i, row in data_df.iterrows():
            if i>9:
                break
            cid, pKa = data_name+f"_{i+1}", float(row['rmsd'])
            complex_dir = os.path.join(save_path, data_name)
            
            graph_path = os.path.join(complex_dir, f"GIGN_Graph-{cid}_{dis_threshold}A.pyg")
            complex_path = os.path.join(data_dir, f"{i+1}_{dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)
            dis_thresholds.append(dis_threshold)
        # pool = multiprocessing.Pool(10)
        # pool.starmap(mols2graphs,
        #                 zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
        # pool.close()
        # pool.join()
        for input_data1,input_data2,input_data3,input_data4 in zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds):
            mols2graphs(input_data1,input_data2,input_data3,input_data4)
class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()


    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx])

    # def collate_fn(self, batch):
    #     return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.data_df)
class PreGraphDatasetD(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False,transform=None, pre_transform=None, pre_filter=None, start = None, size = None):
        super().__init__(data_dir, transform, pre_transform, pre_filter)
        self.data_dir = data_dir
        self.data_name = os.listdir(self.data_dir)
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.transform = None
        # self._indices = None
        self.num_process = num_process
        
        self.datalist = []
        
        # self._pre_process()
    @property
    def processed_file_names(self):
        return self.complex_ids
    # def load_data(self,name):
    #     data_input = os.path.join(self.data_dir,name)
    #     datalist = []
    #     for i in range(11):
    #         # start = time.time()
    #         load_name = os.path.join(data_input,f"GIGN_Graph-{name}_{i}_5A.pyg")
    #         # load_name = os.path.join(data_input,f"GIGN_Graph-6VO6-NAI_{i}_5A.pyg")
    #         data = torch.load(load_name)
    #         datalist.append(data)
    #     return datalist
    #         # end = time.time()
    #         # print(f"epoch:{end - start}")
    def load(self):
        save_name = []
        for name in self.data_name:
            data_input = os.path.join(self.data_dir,name)
            data_name = os.listdir(data_input)
            if(len(data_name)==11):
                save_name.append(name)
        self.data_name = save_name
        # pool = Pool(36)
        # # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        # self.datalist = pool.map(self.load_data, self.data_name)
        # pool.close()
        # pool.join()
        
        # print(len(self.datalist))
        for idx in range(len(self.data_name)):
            data_input = os.path.join(self.data_dir,self.data_name[idx])
            datalist = []
            for i in range(11):
                
                load_name = os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{i}_5A.pyg")
                # load_name = os.path.join(data_input,f"GIGN_Graph-6VO6-NAI_{i}_5A.pyg")
                data = torch.load(load_name)
                datalist.append(data)
            data = datalist[0]
            datalist.append(data)
            for i in range(9):
                mean = torch.randn(1).item() 
                std = 0.001
                noise = mean + std * torch.randn_like(data.pos)
                data1 = data.clone().detach()
                data1.pos = data.pos+noise
                data1.y = torch.tensor([mean]).float()
                datalist.append(data1)
            self.datalist.append(datalist)

    def len(self):
        return len(self.data_name)
    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    # def get(self, idx):
    #     data_input = os.path.join(self.data_dir,self.data_name[idx])
    #     # data_input = os.path.join(self.data_dir,"6VO6-NAI")
    #     # data_name = os.listdir(data_input)
    #     # data_name = [os.path.join(data_input,l) for l in data_name]
        
    #     # data = torch.load(self.graph_paths[idx])
    #     # data.y = 0
    #     # datalist = [data]
    #     datalist = []
    #     # mean_range = [-1,1]
    #     # std_range = []
    #     # for i in range(9):
    #     #     mean = torch.randn(1).item() 
    #     #     std = 0.01
    #     #     noise = mean + std * torch.randn_like(data.pos)
    #     #     data1 = data.clone().detach()
    #     #     data1.pos = data.pos+noise
    #     #     data1.y = mean
    #     #     datalist.append(data1)
    #     # if len(data_name)<11:
    #         # print(torch.ones(1,1))
    
    #     for i in range(11):
    #         start = time.time()
    #         load_name = os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{i}_5A.pyg")
    #         # load_name = os.path.join(data_input,f"GIGN_Graph-6VO6-NAI_{i}_5A.pyg")
    #         data = torch.load(load_name)
    #         datalist.append(data)
    #         end = time.time()
    #         print(f"epoch:{end - start}")
    #     # else:
    #     # print(len(data_name))
    #     # for i in range(11):
    #     #     load_name = os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{i}_5A.pyg")
    #     #     # load_name = os.path.join(data_input,f"GIGN_Graph-6VO6-NAI_{i}_5A.pyg")
    #     #     data = torch.load(load_name)
    #     #     datalist.append(data)
    #     #     # data1 = torch.load(os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{0}_5A.pyg"))
    #     #     # datalist.append(data1)
    #     #     # for i in range(9):
    #     #     #     mean = torch.randn(1).item() 
    #     #     #     std = 0.01
    #     #     #     noise = mean + std * torch.randn_like(data1.pos)
    #     #     #     data2 = data1.detach().clone() 
    #     #     #     data2.pos = data2.pos+noise
    #     #     #     data2.y = torch.tensor([1])*mean
    #     #     #     datalist.append(data2)
    #     return datalist
    def collate_fn(self, batch):
        return Batch.from_data_list(batch)
    def get(self,idx):
        return self.datalist[idx]
    
    def __len__(self):
        return len(self.data_name)
class PreGraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False,transform=None, pre_transform=None, pre_filter=None, start = None, size = None):
        super().__init__(data_dir, transform, pre_transform, pre_filter)
        self.data_dir = data_dir
        self.data_name = os.listdir(self.data_dir)
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.transform = None
        # self._indices = None
        self.num_process = num_process
        
        self.datalist = []
        
        # self._pre_process()
    @property
    def processed_file_names(self):
        return self.complex_ids
    # def load_data(self,name):
    #     data_input = os.path.join(self.data_dir,name)
    #     datalist = []
    #     for i in range(11):
    #         # start = time.time()
    #         load_name = os.path.join(data_input,f"GIGN_Graph-{name}_{i}_5A.pyg")
    #         # load_name = os.path.join(data_input,f"GIGN_Graph-6VO6-NAI_{i}_5A.pyg")
    #         data = torch.load(load_name)
    #         datalist.append(data)
    #     return datalist
    #         # end = time.time()
    #         # print(f"epoch:{end - start}")
    def load(self):
        save_name = []
        for name in self.data_name:
            data_input = os.path.join(self.data_dir,name)
            data_name = os.listdir(data_input)
            if(len(data_name)==11):
                save_name.append(name)
        self.data_name = save_name
        # pool = Pool(36)
        # # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        # self.datalist = pool.map(self.load_data, self.data_name)
        # pool.close()
        # pool.join()
        
        # print(len(self.datalist))
        for idx in range(len(self.data_name)):
            data_input = os.path.join(self.data_dir,self.data_name[idx])
            datalist = []
            for i in range(11):
                
                load_name = os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{i}_5A.pyg")
                # load_name = os.path.join(data_input,f"GIGN_Graph-6VO6-NAI_{i}_5A.pyg")
                data = torch.load(load_name)
                datalist.append(data)
            self.datalist.append(datalist)

    def len(self):
        return len(self.data_name)
    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    # def get(self, idx):
    #     data_input = os.path.join(self.data_dir,self.data_name[idx])
    #     # data_input = os.path.join(self.data_dir,"6VO6-NAI")
    #     # data_name = os.listdir(data_input)
    #     # data_name = [os.path.join(data_input,l) for l in data_name]
        
    #     # data = torch.load(self.graph_paths[idx])
    #     # data.y = 0
    #     # datalist = [data]
    #     datalist = []
    #     # mean_range = [-1,1]
    #     # std_range = []
    #     # for i in range(9):
    #     #     mean = torch.randn(1).item() 
    #     #     std = 0.01
    #     #     noise = mean + std * torch.randn_like(data.pos)
    #     #     data1 = data.clone().detach()
    #     #     data1.pos = data.pos+noise
    #     #     data1.y = mean
    #     #     datalist.append(data1)
    #     # if len(data_name)<11:
    #         # print(torch.ones(1,1))
    
    #     for i in range(11):
    #         start = time.time()
    #         load_name = os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{i}_5A.pyg")
    #         # load_name = os.path.join(data_input,f"GIGN_Graph-6VO6-NAI_{i}_5A.pyg")
    #         data = torch.load(load_name)
    #         datalist.append(data)
    #         end = time.time()
    #         print(f"epoch:{end - start}")
    #     # else:
    #     # print(len(data_name))
    #     # for i in range(11):
    #     #     load_name = os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{i}_5A.pyg")
    #     #     # load_name = os.path.join(data_input,f"GIGN_Graph-6VO6-NAI_{i}_5A.pyg")
    #     #     data = torch.load(load_name)
    #     #     datalist.append(data)
    #     #     # data1 = torch.load(os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{0}_5A.pyg"))
    #     #     # datalist.append(data1)
    #     #     # for i in range(9):
    #     #     #     mean = torch.randn(1).item() 
    #     #     #     std = 0.01
    #     #     #     noise = mean + std * torch.randn_like(data1.pos)
    #     #     #     data2 = data1.detach().clone() 
    #     #     #     data2.pos = data2.pos+noise
    #     #     #     data2.y = torch.tensor([1])*mean
    #     #     #     datalist.append(data2)
    #     return datalist
    def collate_fn(self, batch):
        return Batch.from_data_list(batch)
    def get(self,idx):
        return self.datalist[idx]
    
    def __len__(self):
        return len(self.data_name)
class PreGraphDatasetV(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False,transform=None, pre_transform=None, pre_filter=None, start = None, size = None):
        super().__init__(data_dir, transform, pre_transform, pre_filter)
        self.data_dir = data_dir
        self.data_name = os.listdir(self.data_dir)
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.transform = None
        # self._indices = None
        self.num_process = num_process
        self._pre_process()
    @property
    def processed_file_names(self):
        return self.complex_ids
    
    def len(self):
        return len(self.graph_paths)
    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def get(self, idx):
        data_input = os.path.join(self.data_dir,self.data_name[idx])
        data_name = os.listdir(data_input)
        data_name = [os.path.join(data_input,l) for l in data_name]
        
        data = torch.load(self.graph_paths[idx])
        data.y = 0
        datalist = [data]
        # datalist = []
        mean_range = [-1,1]
        std_range = []
        for i in range(9):
            mean = torch.randn(1).item() 
            std = 0.01
            noise = mean + std * torch.randn_like(data.pos)
            data1 = data.clone().detach()
            data1.pos = data.pos+noise
            # data1.pos.requires_grad_()
            data1.y = mean
            
            datalist.append(data1)
        if len(data_name)==0:
            # print(torch.ones(1,1))
            return torch.ones(1,1)
        # for i in range(11):
        #     load_name = os.path.join(data_input,f"GIGN_Graph-{self.data_name[idx]}_{i}_5A.pyg")
        #     data = torch.load(load_name)
        #     datalist.append(data)
        return datalist
    # def collate_fn(self, batch):
    #     return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.data_df)
if __name__ == '__main__':
    # data_root = './data'
    # toy_dir = os.path.join(data_root, 'toy_set')
    # toy_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv"))
    # toy_set = GraphDataset(toy_dir, toy_df, graph_type='Graph_GIGN', dis_threshold=5, create=True)
    # train_loader = PLIDataLoader(toy_set, batch_size=128, shuffle=True, num_workers=4)
    root_path = "/blue/zhe.jiang/y.zhang1/PDBdata"
    paths = os.listdir(root_path)
    with Pool(processes=80) as pool:
        pool.starmap(pre_process, enumerate(paths))
    # pre_process('10')
# %%
