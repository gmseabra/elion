#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
utils.py: 
"""
import numpy as np
import os
from shutil import copy, rmtree
#from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from test_net_config import METRICS, DRUG_ROOT_DIR
import time
import sys
from collections import OrderedDict, defaultdict
import json
import re
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

__author__ = "Yanjun Li"
__license__ = "MIT"


def create_dir(root_dir, x):
    target_dir = os.path.join(root_dir, x)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def write_config(dest_dir, *args):
    if not os.path.exists(dest_dir):
        os.makedirs(os.path.dirname(dest_dir))

    item_list = os.listdir(dest_dir)
    num_config_files = 0
    for x in item_list:
        if re.match(r'config_\d\.json', x):
            num_config_files += 1

    new_config_name = os.path.join(dest_dir, 'config_{}.json'.format(num_config_files))
    print(num_config_files, new_config_name)
    data_dict_list = []
    for data_dict in args:
        order_dict = OrderedDict(sorted(data_dict.items(), key=lambda t: t[0]))
        data_dict_list.append(order_dict)
    with open(new_config_name, 'w') as fp:
        json.dump(data_dict_list, fp, indent=2)


def plot_label_value(stem_dir):
    train_df = pd.read_csv(os.path.join(stem_dir, "train.csv"))
    train_energy = train_df["pKd/pKi"].values

    test_df = pd.read_csv(os.path.join(stem_dir, "test.csv"))
    test_energy = test_df["pKd/pKi"].values

    n_bins = 50
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(train_energy, bins=n_bins)
    axs[0].set_title('train')
    axs[1].hist(test_energy, bins=n_bins)
    axs[1].set_title('test')
    plt.show()


def create_weight_dir(root_weight_dir):
    weight_dir_dict = {}
    for metric in METRICS:
        weight_dir_dict[metric] = create_dir(root_weight_dir, metric + '/')
    return weight_dir_dict


def print_metrics(ordered_metrics_dict):
    print(dict(ordered_metrics_dict))


def print_dict_byline(target_dict):
    for k, v in target_dict.items():
        print(str(k + ':').ljust(15) + str(v))


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def list_duplicate_indexes(seq):
    tally = defaultdict(list)
    print('seq: %s' % seq)
    for i, item in enumerate(seq):
        tally[str(item)].append(i)
    return tally


def multi_avg(y_name, y_true, y_pred):
    """
    y_true, y_pred have to be numpy array, with shape (L, 1)
    :param y_name:
    :param y_true:
    :param y_pred:
    :return:
    """
    duplicate_indexes = list_duplicate_indexes(y_name)
    print('y_name: %s' % y_name)
    print('duplicate_indexes: %s' % duplicate_indexes)
    avg_pdb, avg_y_true, avg_y_pred = [[], [], []]
    for pdb, indexes in duplicate_indexes.items():
        # print('y_pred: %s' % y_pred)
        # if len(y_pred[indexes])==1:
        #     print('real complex prediction: %s' % y_pred[indexes])
        #     avg_pdb.extend([pdb])
        #     continue
        # else:
        #     # print('augmented complexes prediction: %s' % y_pred[indexes])
        #     pass
        avg_pdb.extend([pdb])
        avg_y_true.extend([np.average(y_true[indexes])])        # y_true = [[1], [3], [5]]
        avg_y_pred.extend([np.average(y_pred[indexes])])
    avg_y_true = np.asarray(avg_y_true)
    avg_y_pred = np.asarray(avg_y_pred)
    print('avg_y_pred: %s' % avg_y_pred)
    print('avg_pdb: %s' % avg_pdb)
    return avg_pdb, avg_y_true, avg_y_pred


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Save dicts to csv file
def save_dict_to_csv(csv, pred_dict):
    """
    :param csv:
    :param pred_dict: The key will serve as column name and the value is a list
    :return:
    """
    print('pred_dict: %s' % pred_dict)
    df = pd.DataFrame.from_dict(pred_dict)
    csv_folder = os.path.dirname(csv)

    os.makedirs(csv_folder, exist_ok=True)
    df.to_csv(csv, float_format='%.3f', index=False, header=True, mode='w')    
    
    print('Save the results to the {}'.format(csv))


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)