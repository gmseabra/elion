#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
record_config.py: 
"""
import argparse
from test_net_config import *
from collections import namedtuple, OrderedDict
import re
import json
#import yaml
import os

__author__ = "Yanjun Li"
__license__ = "MIT"


def get_arguments(mode, batch_dir, test_type, test_dataset, model_dir, out_csv_dir):
    if mode == 'test':
        DEFAULT_CONFIG = DEFAULT_TEST_CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default=mode,
                        help='train/test')
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model'],
                        help='The name of dl model')
    parser.add_argument('--number', '-n', type=str, default=DEFAULT_CONFIG['number'],
                        help='The number of model.')
    parser.add_argument('--restore', '-r', type=bool, default=DEFAULT_CONFIG['restore'],
                        help='Whether restore the weight file or not.')
    parser.add_argument('--restore_criteria', '-rc', type=str, default=DEFAULT_CONFIG['restore_criteria'],
                        choices=['loss', 'pearson', 'spearmanr'])
    parser.add_argument('--debug', type=str, default=DEFAULT_CONFIG['debug'])
    parser.add_argument('--aug_train', type=bool, default=DEFAULT_CONFIG['aug_train'],
                        help='Whether use the augmented data for training.')
    parser.add_argument('--aug_num_train', type=int, default=DEFAULT_CONFIG['aug_num_train'],
                        help='Number of augmented data can be used for training one pdb.')
    parser.add_argument('--avg_valid', type=bool, default=DEFAULT_CONFIG['avg_valid'],
                        help='Whether use the multi_avg augmented for valid')
    parser.add_argument('--avg_test', type=bool, default=DEFAULT_CONFIG['avg_test'],
                        help='Whether use the multi_avg augmented for test')
    parser.add_argument('--num_gpus', type=int, default=DEFAULT_CONFIG['num_gpus'],
                        help='Whether use multi-gpu')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_CONFIG['num_workers'],
                        help='The number of workers to process data.')
    parser.add_argument('--manualSeed', type=int, default=DEFAULT_CONFIG['manualSeed'],
                        help='The random seed for variable initialization and python random.')
    # ===================================================================

    parser.add_argument('--data_source', type=str, default=DEFAULT_CONFIG['data_source'],
                        choices=['PDBbind_2016', 'vs', 'ind'])
    parser.add_argument('--data_kind', type=str, default=DEFAULT_CONFIG['data_kind'],
                        choices=['refined', 'general'],
                        help='datakind of refined or general.')
    parser.add_argument('--input_feature', type=int, default=DEFAULT_CONFIG['input_feature'],
                        choices=[60, 11],
                        help='The number of features of input data.')
    parser.add_argument('--num_grid', type=int, default=DEFAULT_CONFIG['num_grid'],
                        choices=[48, 64, 86],
                        help='The 3d pocket num_grid.')
    parser.add_argument('--occupancy', type=str, default=DEFAULT_CONFIG['occupancy'])

    parser.add_argument('--split_mode', type=str, default=DEFAULT_CONFIG['split_mode'],
                        help='the split ratio for the dataset.')
    parser.add_argument('--test_type', type=str, default=DEFAULT_CONFIG['test_type'],
                        help='virtual screen or independent test.')
    parser.add_argument('--test_dataset', type=str, default=test_dataset,
                        help='dataset of virtual screen or independent test dataset.')
    parser.add_argument('--batch_dir', type=str, default=None,
                        help='base temporary directory for the user to process the inputs.')
    parser.add_argument('--model_dir', type=str, default=model_dir,
                        help='directory where saved weights for trained model were saved.')
    parser.add_argument('--out_csv_dir', type=str, default=out_csv_dir,
                        help='directory where model predictions will be saved as a CSV file.')
    # ===================================================================
    parser.add_argument('--batch_size', '-b', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Number of pockets to process in a batch.')

    # ===================================================================
    parser.add_argument('--normalize', type=bool, default=DEFAULT_CONFIG['normalize'],
                        help='whether normalize the label.')
    parser.add_argument('--mean', type=float, default=DEFAULT_CONFIG['mean'],
                        help='mean value of target for train dataset.'
                             '(For refined: 6.38203640501, For general: 6.3927599094)'
                             '(New: refined: 6.37938088445, general: 6.3927599094)')
    parser.add_argument('--std', type=float, default=DEFAULT_CONFIG['std'],
                        help='std of target for the train dataset. '
                             '(For refined: 1.97930731751, For general: 1.92693220647)'
                             'New refined: 1.97857695779, general: 1.92693220647')
    parser.add_argument('--min',  type=float, default=DEFAULT_CONFIG['min'],
                        help='min value of target for train dataset.'
                             '(For refined: 2.07, For general: 1.26)')
    parser.add_argument('--max', type=float, default=DEFAULT_CONFIG['max'],
                        help='max value of target for train dataset.'
                             '(For refined: 11.52, For general: 15.0)')

    parser.add_argument('--decimal_precision', type=float, default=DEFAULT_CONFIG['decimal_precision'])
    return parser.parse_args()


def append_dicts(dict_a, dict_b):
    dest_dict = {}
    for k, v in dict_a.iteritems():
        v.extend([dict_b[k]])
        dest_dict[k] = v
    return dest_dict


def write_config(dest_dir, *args):
    if not os.path.exists(dest_dir):
        os.makedirs(os.path.dirname(dest_dir))

    item_list = os.listdir(dest_dir)
    num_config_files = 0
    for x in item_list:
        if re.match(r'config_\d+\.json', x):
            num_config_files += 1

    new_config_name = os.path.join(dest_dir, 'config_{}.json'.format(num_config_files))
    print(num_config_files, new_config_name)
    data_dict_list = []
    for data_dict in args:
        order_dict = OrderedDict(sorted(data_dict.items(), key=lambda t: t[0]))
        data_dict_list.append(order_dict)
    with open(new_config_name, 'wb') as fp:
        json.dump(data_dict_list, fp, indent=2)


#def load_config(dest_dir, index):
#    config_name = os.path.join(dest_dir, 'config_{}.json'.format(index))
#    assert os.path.exists(config_name) is not None, 'The config file does not exist.'
#    with open(config_name, 'rb') as fp:
#       configs = yaml.safe_load(fp)
#    return configs


def dict_to_namedtuple(dict_data):
    return namedtuple('GenericDict', dict_data.keys())(**dict_data)
