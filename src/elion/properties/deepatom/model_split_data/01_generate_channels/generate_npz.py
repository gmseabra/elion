#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_npz.py
===============
Single-file replacement for the two DeepAtom grid-generation scripts:

    make_grid_mp.py            -> non-augmented .npz voxel grids
    make_grid_for_aug_mp.py    -> augmented (rotated/translated) .npz voxel grids

It is equivalent to running, in order:

    python make_grid_mp.py          BATCH_DIR  DATASET_NAME
    python make_grid_for_aug_mp.py  BATCH_DIR  DATASET_NAME

Usage
-----
    # Run BOTH (default), exactly like the two original scripts back-to-back:
    python generate_npz.py  BATCH_DIR  DATASET_NAME

    # Run only one half:
    python generate_npz.py  BATCH_DIR  DATASET_NAME  --stage non_augmented
    python generate_npz.py  BATCH_DIR  DATASET_NAME  --stage augmented

    # Override grid / feature settings (defaults match the originals):
    python generate_npz.py  BATCH_DIR  DATASET_NAME  --mode pcmax --num-feature 24 \
                            --grid-length 32 --resolution 1.0

Inputs / outputs (under BATCH_DIR)
----------------------------------
    Non-augmented:  reads  atomtypes/        ->  writes  3d_<grid>_<feat>_<mode>/
    Augmented:      reads  atomtypes_aug/    ->  writes  3d_<grid>_<feat>_<mode>_aug/

The two halves use disjoint input and output directories, so running them in a
single process (the default) is safe and matches the original behaviour.

This file must live next to config.py (the 01_generate_channels/ directory),
because - like the originals - it does `from config import *` to obtain
anolea_vdw_dict, ANOLEA_SIZE, LIG_ELEMENT_SIZE, etc.

Notes on the merge
------------------
The two originals shared ~10 byte-identical helpers (parsing, masking, channel
updates, KD-tree overlap, Counter); those are defined ONCE here.  The pieces
that genuinely differ are kept separate under distinct names:

    non-augmented              augmented
    -------------              ---------
    MakeGridInfoCollect        MakeGridInfoCollectAug
    make_grid                  make_grid_aug
    get_ligand_center          get_ligand_center_from_src_atomtypes + translation
    verify_npz                 verify_npz_aug

(The only behavioural trim: the stray `print(np.__version__)` that the
non-augmented `extract_binding_box` emitted once per complex has been dropped;
it never affected the saved grids.)
"""

from config import *                 # anolea_vdw_dict, ANOLEA_SIZE, LIG_ELEMENT_SIZE, ...

import argparse
import os
import sys
import math
import time
import shutil
import random
import re
from glob import glob
from collections import namedtuple, defaultdict
from functools import partial
from multiprocessing import Pool, Manager, Value, Lock, cpu_count

import numpy as np
import pandas as pd
from scipy import spatial
from scipy.spatial import distance

__author__ = "Yanjun Li"
__license__ = "MIT"


# ===========================================================================
# Shared helpers (identical across both original scripts)
# ===========================================================================

def print_message(pdb_name, current_num, total_num):
    print("===================================================================")
    print("{}:   complex {} (out of {})".format(pdb_name, current_num, total_num))


def create_ligand_file(src_atomtypes, lig_pdb):
    """Write only the ligand lines (chain 'y', assigned in 01_preprocess_complexes_VS.py)."""
    with open(src_atomtypes, 'r') as inFile:
        with open(lig_pdb, 'w') as outFile:
            for line in inFile:
                if line.startswith('y'):
                    outFile.write(line)


def extract_binding_box(src_atomtypes, bindingBox_atomtypes, greater_box_max, greater_box_min):
    """Copy only the atoms whose coordinates fall inside the greater box."""
    with open(src_atomtypes, 'r') as inComplex:
        with open(bindingBox_atomtypes, 'w') as outBindingSite:
            for line in inComplex:
                x_coord = float(line[39:49].strip())
                y_coord = float(line[49:59].strip())
                z_coord = float(line[59:69].strip())
                atom_coord = np.array([x_coord, y_coord, z_coord])

                if np.all(atom_coord < greater_box_max) & np.all(atom_coord > greater_box_min):
                    outBindingSite.write(line)


def parse_line_by_substr(line):
    LineTuple = namedtuple('LineTuple',
                           'atomid, anolea, arpeggio, xcoord, ycoord, zcoord')
    # This tuple returns strings only.
    # They need to be converted to other types wherever needed.
    ltpl = LineTuple(
        atomid=line[:22],
        anolea=line[22:24],
        arpeggio=line[28:39],
        xcoord=line[39:49].strip(),
        ycoord=line[49:59].strip(),
        zcoord=line[59:69].strip()
    )
    return ltpl


def generate_mask_channel(num_feature, _linetuple, config_dict):
    """
    Generate the mask channel for each atom according to the number of features.
    """
    if num_feature == 60:
        anolea_vector = list(map(int, [ix == int(_linetuple.anolea) - 1
                                       for ix in range(ANOLEA_SIZE + LIG_ELEMENT_SIZE)]))
        arpeggio_vector = list(map(int, list(_linetuple.arpeggio)))
        mask_vector = np.asarray(anolea_vector + arpeggio_vector)
    elif num_feature == 11:
        arpeggio_vector = list(map(int, list(_linetuple.arpeggio)))
        mask_vector = np.asarray(arpeggio_vector)

    elif num_feature == 24:
        if not _linetuple.atomid.startswith('y'):  # i.e. a protein atom
            arpeggio_vector_pr = list(map(int, list(_linetuple.arpeggio)))
            ExVol_pr = [1]   # Excluded volume for protein
            arpeggio_vector_lig = list(map(int, list('0' * config_dict['ARPEGGIO_LENGTH'])))
            ExVol_lig = [0]  # Excluded volume for ligand
        else:
            arpeggio_vector_pr = list(map(int, list('0' * config_dict['ARPEGGIO_LENGTH'])))
            ExVol_pr = [0]   # Excluded volume for protein
            arpeggio_vector_lig = list(map(int, list(_linetuple.arpeggio)))
            ExVol_lig = [1]  # Excluded volume for ligand

        mask_vector = np.asarray(arpeggio_vector_pr + ExVol_pr +
                                 arpeggio_vector_lig + ExVol_lig)
    return mask_vector


def update_channel_binary(_pocket_cube, _mask_vector, _update_indexes_flat, grid_shape):
    """Update the channel using the binary scheme."""
    x_index, y_index, z_index = np.unravel_index(_update_indexes_flat, grid_shape)
    overlap_indices_on_grid = list(zip(x_index, y_index, z_index))

    for ix in overlap_indices_on_grid:
        _pocket_cube[ix] = list(map(int, np.logical_or(_pocket_cube[ix], _mask_vector)))
    return _pocket_cube


def update_channel_pc(_pocket_cube, _mask_vector, _linetuple, _update_indexes_flat, points, grid_shape):
    """Update the channel using the dense (point-charge max) scheme."""
    try:
        vdw_radius = anolea_vdw_dict[_linetuple.anolea]
    except KeyError:
        vdw_radius = anolea_vdw_dict['01']  # Carbon as default atom

    atom_coord = [float(_linetuple.xcoord),
                  float(_linetuple.ycoord),
                  float(_linetuple.zcoord)]

    for ix in _update_indexes_flat:
        dist = distance.euclidean(atom_coord, points[ix])
        ratio = vdw_radius / dist
        occ = 1 - math.exp(-ratio ** 12)
        update_vect = _mask_vector * occ

        overlap_indices_on_grid = np.unravel_index(ix, grid_shape)
        _pocket_cube[overlap_indices_on_grid] = np.maximum(update_vect,
                                                           _pocket_cube[overlap_indices_on_grid])
    return _pocket_cube


def find_overlap_grid_for_binary(linetuple, grid_tree):
    atom_coord = [float(linetuple.xcoord),
                  float(linetuple.ycoord),
                  float(linetuple.zcoord)]

    try:
        vdw_radius = anolea_vdw_dict[linetuple.anolea]
    except KeyError:
        vdw_radius = anolea_vdw_dict['01']  # Carbon as default atom

    # returns the indices for the overlapping grid points
    overlap_indices_flat = grid_tree.query_ball_point(atom_coord, vdw_radius)
    return overlap_indices_flat


def find_overlap_grid_for_pc(linetuple, grid_tree):
    """Like find_overlap_grid_for_binary, but the KD-tree query is within 2 vdw distance."""
    atom_coord = [float(linetuple.xcoord),
                  float(linetuple.ycoord),
                  float(linetuple.zcoord)]

    try:
        vdw_radius = anolea_vdw_dict[linetuple.anolea]
    except KeyError:
        vdw_radius = anolea_vdw_dict['01']  # Carbon as default atom

    # returns the indices for the overlapping grid points (up to 2 vdw_radius)
    # on a flattened 1D vector
    overlap_indices_flat = grid_tree.query_ball_point(atom_coord, 2 * vdw_radius)
    return overlap_indices_flat


class Counter(object):
    def __init__(self, initval=0, mag=Manager()):
        self.val = mag.Value('i', initval)
        self.lock = mag.Lock()


# ===========================================================================
# NON-AUGMENTED  (was make_grid_mp.py)
# ===========================================================================

def get_ligand_center(lig_pdb):
    """Ligand center taken from a ligand-only PDB file (non-augmented path)."""
    lig_coords = []

    with open(lig_pdb, 'r') as ligFile:
        for line in ligFile:
            x_coord = float(line[39:49].strip())
            y_coord = float(line[49:59].strip())
            z_coord = float(line[59:69].strip())
            lig_coords.append((x_coord, y_coord, z_coord))

    if not lig_coords:
        raise ValueError(
            "No ligand atoms found in {}. "
            "The .atomtypes file may not contain lines starting with 'y' "
            "(the chain ID assigned to ligand atoms). "
            "Check the atomtypes file and the chain identifier used.".format(lig_pdb)
        )

    maxc = np.squeeze(np.max(lig_coords, axis=0))
    minc = np.squeeze(np.min(lig_coords, axis=0))
    lig_center = (maxc + minc) / 2.0

    return lig_center


def verify_npz(label_csv, src_dir):
    df = pd.read_csv(label_csv)
    for index, row in df.iterrows():
        real_pdb_code = row['pdb']
        real_label = np.float64(row['pKd/pKi'])

        npz = np.load(os.path.join(src_dir, real_pdb_code + '.npz'))
        # print('pdb:{}, {}'.format(real_pdb_code, npz['label'] == real_label))


class MakeGridInfoCollect(object):
    """Collect atom-type file info for the NON-augmented grids."""

    def __init__(self, **config_dict):
        """
        mode: 'binary' : only record whether this kind of atom exists or not;
              'gaussian': continuous contribution of the atom (TODO);
              'pcmax'   : point-charge max occupancy.
        """
        data_source = config_dict['data_source']
        dataset_name = config_dict['dataset_name']
        num_feature = config_dict['num_feature']
        grid_length = config_dict['grid_length']
        resolution = config_dict['resolution']
        self.mode = config_dict['mode']

        num_grid = int(np.ceil(grid_length / resolution))  # [48/86]
        self.dataset_name = dataset_name

        self.base_data_dir = config_dict['batch_dir']
        print("config_dict['batch_dir']: %s" % config_dict['batch_dir'])
        self.original_data_dir = os.path.join(self.base_data_dir, 'original')
        # changed from original_dir/src to base_dir/atomtypes !
        self.src_atomtypes_dir = os.path.join(self.base_data_dir, 'atomtypes')
        self.dest_atomtypes_dir = os.path.join(self.original_data_dir, 'dest')
        self.dest_np_dir = os.path.join(self.base_data_dir,
                                        '3d_' + str(num_grid) + '_' + str(num_feature) + '_' + self.mode)

        os.makedirs(self.src_atomtypes_dir, exist_ok=True)
        os.makedirs(self.dest_atomtypes_dir, exist_ok=True)

        # Legacy safety: if any *.atomtypes were left at the batch-dir root,
        # move them into atomtypes/.  In the current pipeline they already live
        # in atomtypes/, so this is a no-op.
        for f in glob(os.path.join(self.base_data_dir, '*.atomtypes')):
            shutil.move(f, os.path.join(self.src_atomtypes_dir, os.path.split(f)[-1]))

        os.makedirs(self.dest_np_dir, exist_ok=True)

    def collect_data_with_index(self):
        """
        Refer to a clean index CSV (used for PDBbind-style sources).
        """
        index_path = os.path.join(self.original_data_dir, 'index')
        os.makedirs(index_path, exist_ok=True)
        self.index_file = os.path.join(self.original_data_dir, 'index', self.dataset_name + '.csv')
        print('self.index_file: %s' % self.index_file)
        print('index_path: %s' % index_path)
        shutil.copy(self.index_file, index_path)
        print('Copy complete.')
        df = pd.read_csv(self.index_file)
        label_df = df[['pdb', 'pKd/pKi']]
        print('label_df: %s' % label_df)

        # Format: {file_name: {pdb: ..., label: ..., 'atomtypes': ...}}
        self.src_atomtypes_dict = dict()
        for idx, x in label_df.iterrows():
            file_name = x['pdb']                      # Eg. 1w4r
            pdb_dict = {'pdb': x['pdb'],
                        'label': x['pKd/pKi'],
                        'atomtypes': os.path.join(self.src_atomtypes_dir, x['pdb'] + '.atomtypes')}
            self.src_atomtypes_dict[file_name] = pdb_dict

    def collect_data_without_index(self):
        print("Running 'collect_data_without_index' ...")

        self.index_file = None
        self.src_atomtypes_dict = dict()
        print("(self.src_atomtypes_dir) = " + self.src_atomtypes_dir)

        # Format: {file_name: {pdb: ..., label: ..., 'atomtypes': ...}}
        for x in os.listdir(self.src_atomtypes_dir):
            file_name = x.split('.atomtypes')[0]        # Eg. 1w4r_PLM_282_B_MOD for TL
            pdb_dict = {'pdb': file_name,
                        'label': np.nan,
                        'atomtypes': os.path.join(self.src_atomtypes_dir, x)}
            self.src_atomtypes_dict[file_name] = pdb_dict

        print("Number of complex: {}".format(len(self.src_atomtypes_dict)))

    def exclude_exist(self):
        exist_np = os.listdir(self.dest_np_dir)
        processed_pdbs = [x.split('.npz')[0] for x in exist_np]

        for processed_pdb in processed_pdbs:
            del self.src_atomtypes_dict[processed_pdb]

        self.remaining_atomtpyes = len(self.src_atomtypes_dict)
        print("The pdb complexes which need to be generated: {}.".format(self.remaining_atomtpyes))


def make_grid(config, dest_atomtypes_dir, dest_np_dir, counter, total_num, file_info):
    """Build one NON-augmented voxel grid and save it as <file_name>.npz."""
    file_name, pdb_dict = file_info
    data_source = config['data_source']
    grid_length = config['grid_length']
    resolution = config['resolution']
    num_feature = config['num_feature']
    mode = config['mode']
    pdb_name = pdb_dict['pdb']
    label = pdb_dict['label']
    src_atomtypes = pdb_dict['atomtypes']
    print('src_atomtypes: %s' % src_atomtypes)

    with counter.lock:
        counter.val.value += 1
        atomtypes_counter = counter.val.value

    print_message(pdb_name, atomtypes_counter, total_num)

    dest_atomtypes_pdb_dir = os.path.join(dest_atomtypes_dir, pdb_name)
    if not os.path.exists(dest_atomtypes_pdb_dir):
        os.makedirs(dest_atomtypes_pdb_dir, exist_ok=True)

    # =======================================================================
    # 0. copy the src atomtype file to dest dir, and change the work dir to it.
    shutil.copy(src_atomtypes, dest_atomtypes_pdb_dir)  # work dir doesn't change
    os.chdir(dest_atomtypes_pdb_dir)

    # =======================================================================
    # 1. Generate the ligand file
    lig_pdb = '{0}_ligand.pdb'.format(pdb_name)

    # Recreate the ligand file if missing OR empty (stale cache guard)
    if os.path.exists(lig_pdb) and os.path.getsize(lig_pdb) > 0:
        print("{} file exists.".format(lig_pdb))
    else:
        create_ligand_file(src_atomtypes, lig_pdb)

    # =======================================================================
    # 2. calculate the center of ligand
    lig_center = get_ligand_center(lig_pdb)

    # =======================================================================
    # 3. extract the atoms inside the binding box (the greater box, not the
    #    final grid)
    greater_box_max = lig_center + (grid_length / 2.0)
    greater_box_min = lig_center - (grid_length / 2.0)

    bindingBox_atomtypes = '{0}_{1}_bindingBox.atomtypes'.format(pdb_name, int(grid_length))

    if os.path.exists(bindingBox_atomtypes):
        print("{} file exists.".format(bindingBox_atomtypes))
    else:
        extract_binding_box(src_atomtypes, bindingBox_atomtypes, greater_box_max, greater_box_min)

    # =======================================================================
    # 4. generate the grid
    pickled_channels = os.path.join(dest_np_dir, '{0}.npz'.format(file_name))

    grid_max = greater_box_max - resolution / 2.0
    grid_min = greater_box_min + resolution / 2.0

    num_points = int(math.ceil(grid_length / resolution))  # number of points, inclusive
    num_points_j = complex(0, num_points)

    # binding site grid (only one grid, used for all channels)
    bs_grid = np.mgrid[grid_min[0]:grid_max[0]:num_points_j,
                       grid_min[1]:grid_max[1]:num_points_j,
                       grid_min[2]:grid_max[2]:num_points_j]

    x, y, z = bs_grid
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_tree = spatial.cKDTree(points)
    grid_shape = (num_points, num_points, num_points)

    with open(bindingBox_atomtypes, 'r') as inBindingSite:
        if mode == 'binary':
            pocket_cube = np.zeros((num_points, num_points, num_points, num_feature), dtype=np.int8)
            for line in inBindingSite:
                linetuple = parse_line_by_substr(line)
                overlap_indices_flat = find_overlap_grid_for_binary(linetuple, grid_tree)
                mask_vector = generate_mask_channel(num_feature, linetuple, config)
                pocket_cube = update_channel_binary(pocket_cube, mask_vector, overlap_indices_flat, grid_shape)
        elif mode == 'gaussian':
            # TODO
            pocket_cube = np.zeros((num_points, num_points, num_points, num_feature), dtype=np.float32)
            pass
        elif mode == 'pcmax':
            pocket_cube = np.zeros((num_points, num_points, num_points, num_feature), dtype=np.float32)
            for line in inBindingSite:
                linetuple = parse_line_by_substr(line)
                overlap_indices_flat = find_overlap_grid_for_pc(linetuple, grid_tree)
                mask_vector = generate_mask_channel(num_feature, linetuple, config)
                pocket_cube = update_channel_pc(pocket_cube, mask_vector, linetuple,
                                                overlap_indices_flat, points, grid_shape)
        np.savez(pickled_channels, pdb=pdb_name, label=label, pocket=pocket_cube)
    print('pickled_channels: %s' % pickled_channels)


def run_non_augmented(batch_dir, dataset_name, num_feature=24, grid_length=32,
                      resolution=1.0, mode='pcmax', arpeggio_length=11,
                      data_source='vs'):
    """Equivalent of make_grid_mp.py's main()."""
    starter = time.time()

    config_dict = dict()
    config_dict['data_source'] = data_source        # [Bmoad_general_refined, PDBbind_2016, tl, vs, ind]
    config_dict['batch_dir'] = str(batch_dir)
    config_dict['dataset_name'] = dataset_name
    config_dict['num_feature'] = num_feature        # [ARPEGGIO_LENGTH + excluded volume] * 2
    config_dict['grid_length'] = grid_length
    config_dict['resolution'] = resolution          # [0.375/0.5/1.0]
    config_dict['mode'] = mode                      # [binary/gaussian/pcmax/pc_sum]
    config_dict['ARPEGGIO_LENGTH'] = arpeggio_length

    mgic = MakeGridInfoCollect(**config_dict)
    if config_dict['data_source'] in ['PDBbind_2016', 'Bmoad_general_refined']:
        mgic.collect_data_with_index()
    elif config_dict['data_source'] in ['tl', 'vs', 'ind']:
        print("second branch!")
        mgic.collect_data_without_index()
    mgic.exclude_exist()

    if mgic.remaining_atomtpyes > 0:
        m = Manager()
        shared_src_atomtypes_dict = m.dict(mgic.src_atomtypes_dict)
        counter = Counter(0)

        p = Pool()
        try:
            p.map(partial(make_grid,
                          config_dict,
                          mgic.dest_atomtypes_dir,
                          mgic.dest_np_dir,
                          counter,
                          mgic.remaining_atomtpyes),
                  list(shared_src_atomtypes_dict.items()))
        except Exception as exc:
            print("ERROR in worker: {}".format(exc))
            raise
        finally:
            p.close()
            p.join()

        end = time.time()
        print("During: {}s".format(end - starter))

    print('mgic.index_file: %s' % mgic.index_file)
    if mgic.index_file:
        verify_npz(mgic.index_file, mgic.dest_np_dir)


# ===========================================================================
# AUGMENTED  (was make_grid_for_aug_mp.py)
# ===========================================================================

def get_ligand_center_from_src_atomtypes(src_atomtypes):
    """Ligand center taken directly from the 'y' (ligand) lines of an .atomtypes file."""
    lig_coords = []
    with open(src_atomtypes, 'r') as inFile:
        for line in inFile:
            if line.startswith('y'):
                x_coord = float(line[39:49].strip())
                y_coord = float(line[49:59].strip())
                z_coord = float(line[59:69].strip())
                lig_coords.append((x_coord, y_coord, z_coord))

    if not lig_coords:
        raise ValueError(
            "No ligand atoms found in {}. "
            "The .atomtypes file contains no lines starting with 'y' "
            "(the chain ID assigned to ligand atoms). "
            "Check the atomtypes file and chain identifier.".format(src_atomtypes)
        )

    maxc = np.squeeze(np.max(lig_coords, axis=0))
    minc = np.squeeze(np.min(lig_coords, axis=0))
    lig_center = (maxc + minc) / 2.0
    return lig_center


def translation(center):
    """Translate the ligand center by a random vector in [-1, 1] along each of 3 dims."""
    translation_vector = 2 * np.random.random_sample(3) - 1
    return center + translation_vector


def verify_npz_aug(label_csv, src_dir):
    df = pd.read_csv(label_csv)
    for index, row in df.iterrows():
        real_pdb_code = row['pdb']
        real_label = np.float64(row['pKd/pKi'])

        matched_aug_pdb = [x.split('.')[0] for x in os.listdir(src_dir) if re.search(real_pdb_code, x)]

        for aug_sample in matched_aug_pdb:
            npz = np.load(os.path.join(src_dir, aug_sample + '.npz'))
            print("pdb:{}, {}".format(aug_sample, npz['label'] == real_label))


class MakeGridInfoCollectAug(object):
    """Collect atom-type file info for the AUGMENTED grids."""

    def __init__(self, **config_dict):
        data_source = config_dict['data_source']
        dataset_name = config_dict['dataset_name']
        num_feature = config_dict['num_feature']
        grid_length = config_dict['grid_length']
        resolution = config_dict['resolution']
        self.mode = config_dict['mode']
        self.keep_all_aug = config_dict['keep_all_aug']     # Whether keep all the augmented data
        if self.keep_all_aug:
            self.select_k = 36
        else:
            self.select_k = config_dict['select_k']         # Select k among all augmented data of one pdb code

        num_grid = int(np.ceil(grid_length / resolution))   # [48/86]
        self.dataset_name = dataset_name
        self.index_file = None

        self.base_data_dir = config_dict['batch_dir']
        self.original_data_dir = os.path.join(self.base_data_dir, 'original_aug')
        # changed from original_dir/src to base_dir/atomtypes_aug !
        self.src_atomtypes_dir = os.path.join(self.base_data_dir, 'atomtypes_aug')
        self.dest_atomtypes_dir = os.path.join(self.original_data_dir, 'dest')
        self.dest_np_dir = os.path.join(self.base_data_dir,
                                        '3d_' + str(num_grid) + '_' + str(num_feature) + '_' + self.mode + '_aug')

        os.makedirs(self.src_atomtypes_dir, exist_ok=True)
        os.makedirs(self.dest_atomtypes_dir, exist_ok=True)

        # (Unlike the non-augmented collector, the augmented one does NOT move
        #  *.atomtypes from the batch-dir root - they already live in atomtypes_aug/.)

        os.makedirs(self.dest_np_dir, exist_ok=True)

    def collect_aug_data_with_index(self):
        """Like the PDBbind path, but each pdb code has many augmented file names."""
        print('self.index_file: %s' % self.index_file)
        df = pd.read_csv(self.index_file)
        label_df = df[['pdb', 'pKd/pKi']]

        all_aug_data = os.listdir(self.src_atomtypes_dir)
        valid_aug_data_dict = defaultdict(list)
        for aug_sample in all_aug_data:
            pdb = aug_sample.split('_')[0]      # Real pdb code
            if pdb in label_df['pdb'].values:
                pdb_dict = {'pdb': pdb,
                            'label': label_df.loc[label_df['pdb'] == pdb]['pKd/pKi'].values[0],  # float
                            'atomtypes': os.path.join(self.src_atomtypes_dir, aug_sample)}
                valid_aug_data_dict[pdb].append(pdb_dict)

        if not self.keep_all_aug:
            for k, v_list in valid_aug_data_dict.items():
                selected_aug_one_pdb = self.random_select(v_list)
                valid_aug_data_dict[k] = selected_aug_one_pdb

        # change key from real pdb to file name for atomtypes and npz, and flatten the dict
        self.src_atomtypes_dict = dict()
        for k, selected_aug_one_pdb in valid_aug_data_dict.items():
            assert len(selected_aug_one_pdb) == self.select_k, \
                'Aug for {} is incorrect, aug: {}'.format(k, len(selected_aug_one_pdb))
            for select_aug_sample in selected_aug_one_pdb:
                file_name = os.path.basename(select_aug_sample['atomtypes']).split('.atomtypes')[0]
                self.src_atomtypes_dict[file_name] = select_aug_sample
        print("All the augmented data: {}".format(len(self.src_atomtypes_dict)))

    def collect_aug_data_without_index(self):
        """For virtual screening (no index / no labels)."""
        all_aug_data = os.listdir(self.src_atomtypes_dir)
        valid_aug_data_dict = defaultdict(list)
        print('-----------------debug2 self.src_atomtypes_dir: %s' % self.src_atomtypes_dir)
        print('-----------------debug2 all_aug_data: %s' % all_aug_data)
        for aug_sample in all_aug_data:
            pdb = aug_sample.split('_')[0]  # real pdb code
            pdb_dict = {'pdb': pdb,
                        'label': np.nan,
                        'atomtypes': os.path.join(self.src_atomtypes_dir, aug_sample)}
            valid_aug_data_dict[pdb].append(pdb_dict)   # Use real pdb code to collect all aug data for one pdb

        if not self.keep_all_aug:
            for k, v_list in valid_aug_data_dict.items():
                selected_aug_one_pdb = self.random_select(v_list)
                valid_aug_data_dict[k] = selected_aug_one_pdb

        print('-----------------debug2 valid_aug_data_dict: %s' % valid_aug_data_dict)
        # change key from real pdb to file name for atomtypes and npz, and flatten the dict
        self.src_atomtypes_dict = dict()
        for k, selected_aug_one_pdb in valid_aug_data_dict.items():
            for select_aug_sample in selected_aug_one_pdb:
                file_name = os.path.basename(select_aug_sample['atomtypes']).split('.atomtypes')[0]
                self.src_atomtypes_dict[file_name] = select_aug_sample
        print('All the augmented data: {}'.format(len(self.src_atomtypes_dict)))

    def random_select(self, augment_list_one_pdb):
        """Randomly select select_k augmented samples from all augmented data of one pdb."""
        return random.sample(augment_list_one_pdb, self.select_k)

    def exclude_exist(self):
        exist_np = os.listdir(self.dest_np_dir)
        processed_pdbs = [x.split('.npz')[0] for x in exist_np]

        for processed_pdb in processed_pdbs:
            del self.src_atomtypes_dict[processed_pdb]

        self.remaining_atomtpyes = len(self.src_atomtypes_dict)
        print("The pdb complexes which need to be generated: {}.".format(self.remaining_atomtpyes))


def make_grid_aug(config, dest_atomtypes_dir, dest_np_dir, counter, total_num, file_info):
    """Build one AUGMENTED voxel grid (random translation) and save it as <file_name>.npz."""
    file_name, pdb_dict = file_info
    data_source = config['data_source']
    grid_length = config['grid_length']
    resolution = config['resolution']
    num_feature = config['num_feature']
    mode = config['mode']
    pdb_name = pdb_dict['pdb']
    label = pdb_dict['label']
    src_atomtypes = pdb_dict['atomtypes']

    with counter.lock:
        counter.val.value += 1
        atomtypes_counter = counter.val.value

    dest_atomtypes_pdb_dir = os.path.join(dest_atomtypes_dir, pdb_name)
    if not os.path.exists(dest_atomtypes_pdb_dir):
        os.makedirs(dest_atomtypes_pdb_dir, exist_ok=True)

    # =======================================================================
    # 0. copy the src atomtype file to dest dir, and change the work dir to it.
    shutil.copy(src_atomtypes, dest_atomtypes_pdb_dir)  # work dir doesn't change
    os.chdir(dest_atomtypes_pdb_dir)

    # (For augmented data the ligand-only PDB is not written; the center is read
    #  straight from the .atomtypes 'y' lines below.)

    # =======================================================================
    # 2. calculate the center of ligand
    lig_center = get_ligand_center_from_src_atomtypes(src_atomtypes)

    # =======================================================================
    # 2.5. perform the (random) translation on the ligand center
    moved_lig_center = translation(lig_center)

    # =======================================================================
    # 3. extract the atoms inside the binding box (the greater box, not the
    #    final grid).  For augmented data we ALWAYS regenerate the binding-box
    #    file, because the old one is based on a previous (random) center.
    bindingBox_atomtypes = '{0}_{1}_bindingBox.atomtypes'.format(file_name, int(grid_length))

    greater_box_max = moved_lig_center + (grid_length / 2.0)
    greater_box_min = moved_lig_center - (grid_length / 2.0)

    extract_binding_box(src_atomtypes, bindingBox_atomtypes, greater_box_max, greater_box_min)

    # =======================================================================
    # 4. generate the grid
    pickled_channels = os.path.join(dest_np_dir, '{0}.npz'.format(file_name))

    grid_max = greater_box_max - resolution / 2.0
    grid_min = greater_box_min + resolution / 2.0

    num_points = int(math.ceil(grid_length / resolution))  # number of points, inclusive
    num_points_j = complex(0, num_points)

    # binding site grid (only one grid, used for all channels)
    bs_grid = np.mgrid[grid_min[0]:grid_max[0]:num_points_j,
                       grid_min[1]:grid_max[1]:num_points_j,
                       grid_min[2]:grid_max[2]:num_points_j]

    x, y, z = bs_grid
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_tree = spatial.cKDTree(points)
    grid_shape = (num_points, num_points, num_points)

    with open(bindingBox_atomtypes, 'r') as inBindingSite:
        if mode == 'binary':
            pocket_cube = np.zeros((num_points, num_points, num_points, num_feature), dtype=np.int8)
            for line in inBindingSite:
                linetuple = parse_line_by_substr(line)
                overlap_indices_flat = find_overlap_grid_for_binary(linetuple, grid_tree)
                mask_vector = generate_mask_channel(num_feature, linetuple, config)
                pocket_cube = update_channel_binary(pocket_cube, mask_vector, overlap_indices_flat, grid_shape)
        elif mode == 'gaussian':
            # TODO
            pocket_cube = np.zeros((num_points, num_points, num_points, num_feature), dtype=np.float32)
            pass
        elif mode == 'pcmax':
            pocket_cube = np.zeros((num_points, num_points, num_points, num_feature), dtype=np.float32)
            for line in inBindingSite:
                linetuple = parse_line_by_substr(line)
                overlap_indices_flat = find_overlap_grid_for_pc(linetuple, grid_tree)
                mask_vector = generate_mask_channel(num_feature, linetuple, config)
                pocket_cube = update_channel_pc(pocket_cube, mask_vector, linetuple,
                                                overlap_indices_flat, points, grid_shape)
        print('pickled_channels: %s' % pickled_channels)
        print('file_name: %s' % file_name)
        print('pdb_name: %s' % pdb_name)
        np.savez(pickled_channels, pdb=pdb_name, label=label, pocket=pocket_cube)


def run_augmented(batch_dir, dataset_name, num_feature=24, grid_length=32,
                  resolution=1.0, mode='pcmax', arpeggio_length=11,
                  keep_all_aug=True, select_k=36, data_source='vs'):
    """Equivalent of make_grid_for_aug_mp.py's main()."""
    starter = time.time()

    config_dict = dict()
    config_dict['data_source'] = data_source        # [Bmoad_general_refined, 2016, tl, vs, ind]
    config_dict['batch_dir'] = str(batch_dir)
    config_dict['dataset_name'] = dataset_name
    config_dict['num_feature'] = num_feature        # [11/60/24]
    config_dict['grid_length'] = grid_length
    config_dict['resolution'] = resolution          # [0.375/0.5/1.0]
    config_dict['mode'] = mode                      # [binary/gaussian/pcmax/pcsum]
    config_dict['ARPEGGIO_LENGTH'] = arpeggio_length
    config_dict['keep_all_aug'] = keep_all_aug      # Whether keep all the augmented data
    config_dict['select_k'] = select_k              # Select k among all augmented data of one pdb code

    mgic = MakeGridInfoCollectAug(**config_dict)
    if config_dict['data_source'] in ['PDBbind_2016', 'Bmoad_general_refined']:
        mgic.collect_aug_data_with_index()
    elif config_dict['data_source'] in ['vs', 'ind']:
        mgic.collect_aug_data_without_index()

    mgic.exclude_exist()

    if mgic.remaining_atomtpyes > 0:
        m = Manager()
        shared_src_atomtypes_dict = m.dict(mgic.src_atomtypes_dict)
        counter = Counter(0)

        p = Pool()
        try:
            p.map(partial(make_grid_aug,
                          config_dict,
                          mgic.dest_atomtypes_dir,
                          mgic.dest_np_dir,
                          counter,
                          mgic.remaining_atomtpyes),
                  list(shared_src_atomtypes_dict.items()))
        except Exception as exc:
            print("ERROR in worker: {}".format(exc))
            raise
        finally:
            p.close()
            p.join()

        end = time.time()
        print("During: {}s".format(end - starter))

    if mgic.index_file:
        verify_npz_aug(mgic.index_file, mgic.dest_np_dir)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate .npz voxel grids for the DeepAtom VS pipeline "
                    "(merges make_grid_mp.py + make_grid_for_aug_mp.py).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("batch_dir", help="Batch directory (was sys.argv[1]).")
    parser.add_argument("dataset_name", help="Dataset name (was sys.argv[2]).")
    parser.add_argument(
        "--stage", choices=["both", "non_augmented", "augmented"], default="both",
        help="Which grids to generate (default: both, i.e. non-augmented then augmented)."
    )
    # Shared grid / feature settings (defaults match the original scripts).
    parser.add_argument("--data-source", default="vs",
                        help="Data source tag (default: vs).")
    parser.add_argument("--num-feature", type=int, default=24)
    parser.add_argument("--grid-length", type=int, default=32)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--mode", default="pcmax",
                        choices=["binary", "gaussian", "pcmax"])
    parser.add_argument("--arpeggio-length", type=int, default=11)
    # Augmented-only settings.
    parser.add_argument("--select-k", type=int, default=36,
                        help="(augmented) number of augmented samples kept per pdb when "
                             "--no-keep-all-aug is set.")
    parser.add_argument("--no-keep-all-aug", dest="keep_all_aug", action="store_false",
                        default=True,
                        help="(augmented) randomly subsample to --select-k instead of "
                             "keeping every augmented sample.")

    args = parser.parse_args()

    batch_dir = os.path.abspath(args.batch_dir)

    if args.stage in ("both", "non_augmented"):
        print("\n########## NON-AUGMENTED GRIDS ##########")
        run_non_augmented(
            batch_dir, args.dataset_name,
            num_feature=args.num_feature, grid_length=args.grid_length,
            resolution=args.resolution, mode=args.mode,
            arpeggio_length=args.arpeggio_length, data_source=args.data_source,
        )

    if args.stage in ("both", "augmented"):
        print("\n########## AUGMENTED GRIDS ##########")
        run_augmented(
            batch_dir, args.dataset_name,
            num_feature=args.num_feature, grid_length=args.grid_length,
            resolution=args.resolution, mode=args.mode,
            arpeggio_length=args.arpeggio_length,
            keep_all_aug=args.keep_all_aug, select_k=args.select_k,
            data_source=args.data_source,
        )


if __name__ == "__main__":
    main()