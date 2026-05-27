#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_net_config.py:
Consolidated config — includes global constants (formerly gl.py)
and directory info (formerly dir_info.py).
"""

import os
import getpass

__author__ = "Yanjun Li"
__license__ = "MIT"

# ============================================================================
# Directory info (formerly dir_info.py)
# ============================================================================

username = getpass.getuser()
DRUG_ROOT_DIR = os.path.join("/blue/lic/", username, "repos/deepatom_LP_tmp_root/")

# ============================================================================
# Global constants (formerly gl.py)
# ============================================================================

METRICS = ['mse']
# MIN_METRICS = ['mse', 'mae', 'logcosh', 'smoothl1']       # less is better
# MAX_METRICS = ['pearson_r2', 'spearman_r2']     # More is better

# ============================================================================
# Default test configuration
# ============================================================================

# (vs: 'vina_320', 'glide_327', 'vina_327', 'glide_504', 'vina_504')
DEFAULT_TEST_CONFIG = {
    'model':                'ShuffleNetV3x2',   # [MobileNet_v1, MobileNet_v2, ShuffleNet_g3]
    'number':               4.3,               # [MobileNet_v1: 8.3, ShuffleNet_g3: 11.11, ShuffleNet_g3x2: 15.11]
    'restore':              True,
    'restore_criteria':     'mse',             # [mse, pearson_r2, spearman_r2]
    'debug':                False,
    'aug_train':            True,
    'aug_num_train':        36,                # [1-36]
    'avg_valid':            True,
    'avg_test':             True,
    'num_gpus':             0,
    'num_workers':          0,
    'manualSeed':           None,
    # -------------------------------------------------------------------------
    'data_source':          'Bmoad_general_refined',  # Saved weight for [PDBbind_2016]
    'data_kind':            'kdkaki',          # (refined/virtual screen data, such as 'glide_504')
    'input_feature':        24,               # (60/11/24)
    'num_grid':             32,               # (32/64/86)
    'occupancy':            'pcmax',          # (binary/pcmax)
    'test_type':            'vs',             # [vs, ind]
    'batch_dir':            None,
    'pipeline_mode':        'dataset',
    'split_mode':           'None',
    # -------------------------------------------------------------------------
    'batch_size':           256,
    # -------------------------------------------------------------------------
    'normalize':            False,
    'mean':                 6.392,            # [Refined: 6.37938088445, General: 6.3927599094]
    'std':                  1.995,            # [Refined: 1.97857695779, General: 1.92693220647]
    'min':                  1.26,             # [Refined: 2.07, General: 1.26]
    'max':                  15.0,             # [Refined: 11.62, General: 15.0]
    # -------------------------------------------------------------------------
    'decimal_precision':    1,                # For prediction: digits after decimal point
}