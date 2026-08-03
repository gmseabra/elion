#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
debug_toy_gpu.py
----------------
GPU inference on the toy_set with GIGN.
* Loads checkpoint on CPU → .to(cuda) once
* Re-creates every graph (create=True) and checks for NaNs / out-of-bounds
* Fixed val() that never touches an empty pred_list
* num_workers=0 → no fork-related segfaults
"""

import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from sklearn.metrics import mean_squared_error
from HIL import HIL

# ----------------------------------------------------------------------
# 1. CUSTOM LAYERS (HIL, FC) – copy exactly from your repo
# ----------------------------------------------------------------------
class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super().__init__()
        self.predict = nn.ModuleList()
        for i in range(n_FC_layer):
            if i == 0:
                self.predict.append(nn.Linear(d_graph_layer, d_FC_layer))
                self.predict.append(nn.Dropout(dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            else:
                self.predict.append(nn.Linear(d_FC_layer, d_FC_layer))
                self.predict.append(nn.Dropout(dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
        self.output = nn.Linear(d_FC_layer, n_tasks)

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return self.output(h), h


# ----------------------------------------------------------------------
# 2. GIGN MODEL
# ----------------------------------------------------------------------
class GIGN(nn.Module):
    def __init__(self, node_dim, hidden_dim, layer_num):
        super().__init__()
        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.gconv = nn.ModuleList([HIL(hidden_dim, hidden_dim) for _ in range(layer_num)])
        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    def forward(self, data):
        x = self.lin_node(data.x)
        for layer in self.gconv:
            x = layer(x, data.edge_index_intra, data.edge_index_inter, data.pos)
        x = global_add_pool(x, data.batch)
        pred, _ = self.fc(x)
        return pred.view(-1), data.y


# ----------------------------------------------------------------------
# 3. SAFE CHECKPOINT LOADER (CPU → GPU)
# ----------------------------------------------------------------------
def load_model_gpu(path: str, model: nn.Module, device: torch.device):
    print(f"\n[1] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"[1] Model on {device}")
    return model


# ----------------------------------------------------------------------
# 4. VALIDATION (GPU → CPU only for aggregation)
# ----------------------------------------------------------------------
def val(model: nn.Module, loader, device: torch.device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)

            # ---- sanity checks (optional but helpful) ----
            assert not torch.isnan(data.x).any(), f"NaN in x (batch {i})"
            assert data.edge_index_intra.max() < data.x.size(0), "intra edge out-of-bounds"
            assert data.edge_index_inter.max() < data.x.size(0), "inter edge out-of-bounds"

            pred, y = model(data)                 # (B,)  (B,)
            preds.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())

    pred  = np.concatenate(preds).ravel()
    label = np.concatenate(labels).ravel()

    rmse = np.sqrt(mean_squared_error(label, pred))
    pearson = np.corrcoef(pred, label)[0, 1] if len(pred) > 1 else float('nan')
    return rmse, pearson


# ----------------------------------------------------------------------
# 5. DATASET WRAPPER (force re-create + sanity)
# ----------------------------------------------------------------------
def build_toy_dataset(toy_dir, toy_csv):
    """
    Deletes any old .pt files and forces GraphDataset(..., create=True).
    Returns (dataset, loader).
    """
    from dataset_GIGN import GraphDataset, PLIDataLoader   # ← YOUR classes

    # ---- wipe old graphs ------------------------------------------------
    if os.path.exists(toy_dir):
        print(f"[2] Removing old toy_set folder: {toy_dir}")
        shutil.rmtree(toy_dir)
    os.makedirs(toy_dir, exist_ok=True)

    df = pd.read_csv(toy_csv)
    print(f"[2] Loaded {len(df)} toy examples from {toy_csv}")

    dataset = GraphDataset(
        root=toy_dir,
        df=df,
        graph_type='Graph_GIGN',
        dis_threshold=5,
        create=True                     # <-- forces .pt creation
    )
    print(f"[2] Created {len(dataset)} graphs → {dataset.graph_paths[:3]} ...")

    loader = PLIDataLoader(dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=0)   # <-- 0 workers = no fork segfault
    return dataset, loader


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------- device -------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Using {device} ===\n")

    # ------------------- model -------------------
    model = GIGN(node_dim=35, hidden_dim=256, layer_num=3)
    ckpt_path = './model/epoch-165, train_loss-0.2755, train_rmse-0.5249, valid_rmse-1.1359, valid_pr-0.8530.pt'
    model = load_model_gpu(ckpt_path, model, device)

    # ------------------- dataset -------------------
    toy_dir = os.path.join('./data', 'toy_set')
    toy_csv = os.path.join(toy_dir, 'toy_examples.csv')
    dataset, loader = build_toy_dataset(toy_dir, toy_csv)

    # ------------------- single-graph sanity -------------------
    print("\n[3] Single-graph test...")
    data = dataset[0].to(device)
    with torch.no_grad():
        p, y = model(data)
        print(f"   pred = {p.item():.4f}   true = {y.item():.4f}")

    # ------------------- full inference -------------------
    print("\n[4] Full toy-set inference...")
    rmse, r = val(model, loader, device)
    print(f"\n=== RESULTS ===")
    print(f"   RMSE    : {rmse:.4f}")
    print(f"   Pearson : {r:.4f}")

if __name__ == '__main__':
    main()