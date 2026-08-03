import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pandas as pd
import torch
import torch.serialization  # For add_safe_globals
import torch_geometric.data.data  # For DataEdgeAttr and DataTensorAttr classes
import numpy as np
from sklearn.metrics import mean_squared_error

# Add safe globals early (before DataLoader or torch.load calls)
# Include both classes for completeness
torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr
])

from GIGN import GIGN
from dataset_GIGN import GraphDataset, PLIDataLoader
from utils import *

# Fixed val function (unchanged from your update)
def val(model, dataloader, device):
    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred, y = model(data)  # Unpack tuple (pred, y)
            print(f'pred: {pred}')  # Print just the prediction

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(y.detach().cpu().numpy())  # Use returned y (same as data.y)

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    return rmse, coff

# Generate rdkit (unchanged)
# from rdkit import Chem
# ligand = Chem.MolFromPDBFile('/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/toy_set/10gs/10gs_ligand.pdb', removeHs=True)
# pocket = Chem.MolFromPDBFile('/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/toy_set/10gs/10gs_pocket.pdb', removeHs=True)
# complex = (ligand, pocket)
# with open('/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/toy_set/10gs/10gs_5A.rdkit', 'wb') as f:
#     pickle.dump(complex, f)

# TEAD
from rdkit import Chem
ligand = Chem.MolFromPDBFile('/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/toy_set/TEAD3/TEAD3_ligand.pdb', removeHs=True)
pocket = Chem.MolFromPDBFile('/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/toy_set/TEAD3/TEAD3_protein.pdb', removeHs=True)
complex = (ligand, pocket)
with open('/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/toy_set/TEAD3/TEAD3_5A.rdkit', 'wb') as f:
    pickle.dump(complex, f)

# Predict affinity value
device = torch.device('cuda:0')
model = GIGN(35, 256, 3).to(device)
load_model_dict(model, './model/epoch-165, train_loss-0.2755, train_rmse-0.5249, valid_rmse-1.1359, valid_pr-0.8530.pt')
model = model.cuda()  # Redundant if .to(device) already done, but OK
data_root = './data'
toy_dir = os.path.join(data_root, 'toy_set')
toy_df = pd.read_csv(os.path.join(toy_dir, "toy_examples.csv"))
toy_set = GraphDataset(toy_dir, toy_df, graph_type='Graph_GIGN', dis_threshold=5, create=True)
print('toy_set.graph_paths: %s' % toy_set.graph_paths)
toy_set_loader = PLIDataLoader(toy_set, batch_size=1, shuffle=True, num_workers=0)  # Keep 0 for debug; try 4 after success
toy_set_rmse, toy_set_coff = val(model, toy_set_loader, device)
print(toy_set_rmse, toy_set_coff)

# checkpoint_torch = torch.load('./model/epoch-165, train_loss-0.2755, train_rmse-0.5249, valid_rmse-1.1359, valid_pr-0.8530.pt')
# print(checkpoint_torch)