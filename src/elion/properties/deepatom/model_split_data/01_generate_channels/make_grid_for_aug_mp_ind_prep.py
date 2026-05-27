#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
s4_make_grid_for_aug_mp.py: 
"""
from config import *
import os
import sys
from glob import glob
import shutil
import numpy as np
import math
from collections import namedtuple, defaultdict
from scipy import spatial
import pandas as pd
import time
from multiprocessing import Pool, Manager, Value, Lock, cpu_count
from functools import partial
from scipy.spatial import distance
import re
import random

__author__ = "Yanjun Li"
__license__ = "MIT"


class MakeGridInfoCollect(object):
    def __init__(self, **config_dict):
        """
        mode: 'binary': only recorder whether this kind of atom exist of nor;
              'Gaussian': calculate the continuous impart of the atom for the index"
        """
        data_source = config_dict['data_source']
        dataset_name = config_dict['dataset_name']
        num_feature = config_dict['num_feature']
        grid_length = config_dict['grid_length']
        resolution = config_dict['resolution']
        self.mode = config_dict['mode']
        self.keep_all_aug = config_dict['keep_all_aug']    # Whether keep all the augmented data
        if self.keep_all_aug:
            self.select_k = 36
        else:
            self.select_k = config_dict['select_k']            # Select k number among all the augmented data of one pdb code
        #print 'data source: {}'.format(data_source)

        num_grid = int(np.ceil(grid_length / resolution))  # [48/86]
        self.dataset_name = dataset_name
        self.index_file = None
        # assert data_source == 'aug', 'Invalid data_source: {}'.format(data_source)

        self.base_data_dir = os.path.join(config_dict['DRUG_ROOT_DIR'], data_source, self.dataset_name)
        #self.non_aug_base_data_dir = os.path.join(config_dict['DRUG_ROOT_DIR'], data_source, self.dataset_name)
        self.original_data_dir = os.path.join(self.base_data_dir, 'original_aug')
        # changed from original_dir/src to base_dir/atomtypes_aug !
        self.src_atomtypes_dir = os.path.join(self.base_data_dir, 'atomtypes_aug')
        self.dest_atomtypes_dir = os.path.join(self.original_data_dir, 'dest')
        self.dest_np_dir = os.path.join(self.base_data_dir,
                                        '3d_' + str(num_grid) + '_' + str(num_feature) + '_' + self.mode + '_aug')

        if not os.path.exists(self.src_atomtypes_dir):
            os.makedirs(self.src_atomtypes_dir)
        if not os.path.exists(self.dest_atomtypes_dir):
            os.makedirs(self.dest_atomtypes_dir)

        #for f in glob(os.path.join(self.base_data_dir, '*.atomtypes')):
        #    shutil.move(f, os.path.join(self.src_atomtypes_dir, os.path.split(f)[-1]))

        #print 'Org dir: {}\nNumpy saved: {}'.format(self.src_atomtypes_dir, self.dest_np_dir)
        if not os.path.exists(self.dest_np_dir):
            os.makedirs(self.dest_np_dir, 0777)

    def collect_aug_data_with_index(self):
        """
        Similar to the pdbbind data, based on index, but file name in src_atomtypes_list is different for one pdb code.
        :return:
        """
        # if self.dataset_name == 'general':
        #     # TODO
        #     # Only select the valid complex based on the INDEX_general_2015_final.csv
        #     self.index_file = os.path.join(self.original_data_dir, 'index', 'INDEX_general_2015_final.csv')
        # elif self.dataset_name == 'refined':
        #     # Delete the complex whose ligand length is larger than 27.
        #     self.index_file = os.path.join(self.original_data_dir, 'index', 'refined.csv')
        # else:
        #     # All the core complexes are valid.
        #     self.index_file = os.path.join(self.original_data_dir, 'index', 'core.csv')

        #self.index_file = os.path.join(self.non_aug_base_data_dir, 'original', 'index', self.dataset_name + '.csv')
        df = pd.read_csv(self.index_file)
        label_df = df[['pdb', 'pKd/pKi']]

        all_aug_data = os.listdir(self.src_atomtypes_dir)
        valid_aug_data_dict = defaultdict(list)
        for aug_sample in all_aug_data:
            pdb = aug_sample.split('_')[0]      # Real pdb code
            if pdb in label_df['pdb'].values:
                pdb_dict = {'pdb': pdb,
                            'label': label_df.loc[label_df['pdb'] == pdb]['pKd/pKi'].values[0],     # float
                            'atomtypes': os.path.join(self.src_atomtypes_dir, aug_sample)
                }
                valid_aug_data_dict[pdb].append(pdb_dict)

        if not self.keep_all_aug:
            for k, v_list in valid_aug_data_dict.iteritems():
                selected_aug_one_pdb = self.random_select(v_list)
                valid_aug_data_dict[k] = selected_aug_one_pdb

        # change key from real pdb to file name for atomtypes and npz, and flatten the dict
        self.src_atomtypes_dict = dict()
        for k, selected_aug_one_pdb in valid_aug_data_dict.iteritems():
            assert len(selected_aug_one_pdb) == self.select_k, \
                'Aug for {} is incorrect, aug: {}'.format(k, len(selected_aug_one_pdb))
            for select_aug_sample in selected_aug_one_pdb:
                file_name = os.path.basename(select_aug_sample['atomtypes']).split('.atomtypes')[0]
                self.src_atomtypes_dict[file_name] = select_aug_sample
        print("All the augmented data: {}".format(len(self.src_atomtypes_dict)))

    def collect_aug_data_without_index(self):
        """For the virtual screen"""
        all_aug_data = os.listdir(self.src_atomtypes_dir)
        valid_aug_data_dict = defaultdict(list)
        for aug_sample in all_aug_data:
            pdb = aug_sample.split('_')[0]  # real pdb code
            pdb_dict = {'pdb': pdb,
                        'label': np.NaN,
                        'atomtypes': os.path.join(self.src_atomtypes_dir, aug_sample)
                        }
            valid_aug_data_dict[pdb].append(pdb_dict)   # Use real pdb code to collect all the aug data for one pdb

        if not self.keep_all_aug:
            for k, v_list in valid_aug_data_dict.iteritems():
                selected_aug_one_pdb = self.random_select(v_list)
                valid_aug_data_dict[k] = selected_aug_one_pdb

        # change key from real pdb to file name for atomtypes and npz, and flatten the dict
        self.src_atomtypes_dict = dict()
        for k, selected_aug_one_pdb in valid_aug_data_dict.iteritems():
            for select_aug_sample in selected_aug_one_pdb:
                file_name = os.path.basename(select_aug_sample['atomtypes']).split('.atomtypes')[0]
                self.src_atomtypes_dict[file_name] = select_aug_sample
        #print 'All the augmented data: {}'.format(len(self.src_atomtypes_dict))

    def random_select(self, augment_list_one_pdb):
        """
        Random select the fractional augmented data from all augmented data of one pdb
        :param augment_list_one_pdb:
        :return:
        """
        return random.sample(augment_list_one_pdb, self.select_k)

    def exclude_exist(self):
        #print "Running 'exclude_exist' ..."

        exist_np = os.listdir(self.dest_np_dir)
        processed_pdbs = [x.split('.npz')[0] for x in exist_np]

        for processed_pdb in processed_pdbs:
            del self.src_atomtypes_dict[processed_pdb]

        self.remaining_atomtpyes = len(self.src_atomtypes_dict)
        print("The pdb complexes which need to be generated: {}.".format(self.remaining_atomtpyes))


def print_message(pdb_name, current_num, total_num):
    print("===================================================================")
    print("{}:   complex {} (out of {})".format(pdb_name, current_num, total_num))


def create_ligand_file(src_atomtypes, lig_pdb):
    with open(src_atomtypes, 'rb') as inFile:
        with open(lig_pdb, 'wb') as outFile:
            for line in inFile:
                if line.startswith('y'):
                    outFile.write(line)


def get_ligand_center_from_ligand_pdb(lig_pdb):
    lig_coords = []

    with open(lig_pdb, 'rb') as ligFile:
        for line in ligFile:
            x_coord = float(line[39:49].strip())
            y_coord = float(line[49:59].strip())
            z_coord = float(line[59:69].strip())
            lig_coords.append((x_coord, y_coord, z_coord))

        maxc = np.squeeze(np.max(lig_coords, axis=0))
        minc = np.squeeze(np.min(lig_coords, axis=0))
        lig_center = (maxc + minc) / 2.0

    return lig_center


def get_ligand_center_from_src_atomtypes(src_atomtypes):
    lig_coords = []
    with open(src_atomtypes, 'rb') as inFile:
        for line in inFile:
            if line.startswith('y'):
                x_coord = float(line[39:49].strip())
                y_coord = float(line[49:59].strip())
                z_coord = float(line[59:69].strip())
                lig_coords.append((x_coord, y_coord, z_coord))

        maxc = np.squeeze(np.max(lig_coords, axis=0))
        minc = np.squeeze(np.min(lig_coords, axis=0))
        lig_center = (maxc + minc) / 2.0
    return lig_center


def translation(center):
    """
    translation the ligand center within [-1, 1] for 3 dimension
    :param center:
    :return:
    """
    translation_vector = 2 * np.random.random_sample(3) - 1
    return center + translation_vector


def extract_binding_box(src_atomtypes, bindingBox_atomtypes, greater_box_max, greater_box_min):
    with open(src_atomtypes, 'rb') as inComplex:
        with open(bindingBox_atomtypes, 'wb') as outBindingSite:
            for line in inComplex:
                x_coord = float(line[39:49].strip())
                y_coord = float(line[49:59].strip())
                z_coord = float(line[59:69].strip())
                atom_coord = np.array([x_coord, y_coord, z_coord])

                if np.all(atom_coord < greater_box_max) & \
                        np.all(atom_coord > greater_box_min):
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
    Generate the mask channel for each atom according to the number of feature
    :param num_feature:
    :param _linetuple:
    :param config_dict:
    :return:
    """
    if num_feature == 60:
        anolea_vector = map(int, [ix == int(_linetuple.anolea) - 1 for ix in xrange(ANOLEA_SIZE + LIG_ELEMENT_SIZE)])
        arpeggio_vector = map(int, list(_linetuple.arpeggio))
        mask_vector = np.asarray(anolea_vector + arpeggio_vector)
    elif num_feature == 11:
        arpeggio_vector = map(int, list(_linetuple.arpeggio))
        mask_vector = np.asarray(arpeggio_vector)

    elif num_feature == 24:
        if not _linetuple.atomid.startswith('y'):  # i.e. a protein atom
            arpeggio_vector_pr = map(int, list(_linetuple.arpeggio))
            ExVol_pr = [1]  # Excluded volume for protein
            arpeggio_vector_lig = map(int, list('0' * config_dict['ARPEGGIO_LENGTH']))
            ExVol_lig = [0]  # Excluded volume for ligand

        else:
            arpeggio_vector_pr = map(int, list('0' * config_dict['ARPEGGIO_LENGTH']))
            ExVol_pr = [0]  # Excluded volume for protein
            arpeggio_vector_lig = map(int, list(_linetuple.arpeggio))
            ExVol_lig = [1]  # Excluded volume for ligand

        mask_vector = np.asarray(arpeggio_vector_pr + ExVol_pr +
                                    arpeggio_vector_lig + ExVol_lig)
    return mask_vector


def update_channel_binary(_pocket_cube, _mask_vector, _update_indexes_flat, grid_shape):
    """
    Update the channel using binary way
    :param _pocket_cube: 4D pocket cube
    :param _mask_vector: the mask vector masks the channels which should not be updated as 0, otherwise as 1
    :param _update_indexes_flat: For one atom(line), which indexes in the grid need to be update
    :param grid_shape:
    :return: the updated 4D pocket cube
    """
    x_index, y_index, z_index = np.unravel_index(_update_indexes_flat, grid_shape)
    overlap_indices_on_grid = zip(x_index, y_index, z_index)

    for ix in overlap_indices_on_grid:
        _pocket_cube[ix] = map(int, np.logical_or(_pocket_cube[ix], _mask_vector))
    return _pocket_cube


def update_channel_pc(_pocket_cube, _mask_vector, _linetuple, _update_indexes_flat, points, grid_shape):
    """
    Update the channel using dense way
    :param _pocket_cube: 4D pocket cube
    :param _mask_vector: the mask vector masks the channels which should not be updated as 0, otherwise as 1
    :param _linetuple:  Every line parser in the complex file
    :param _update_indexes_flat: For one atom(line), which indexes in the grid need to be update
    :param points:
    :param grid_shape:
    :return:
    """

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
        _pocket_cube[overlap_indices_on_grid] = np.maximum(update_vect, _pocket_cube[overlap_indices_on_grid])
    return _pocket_cube


def make_grid(config, dest_atomtypes_dir, dest_np_dir, counter, total_num, (file_name, pdb_dict)):
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

    # print_message(pdb_name, atomtypes_counter.value(), total_num)
    #print_message(pdb_name, atomtypes_counter, total_num)

    dest_atomtypes_pdb_dir = os.path.join(dest_atomtypes_dir, pdb_name)
    if not os.path.exists(dest_atomtypes_pdb_dir):
        os.makedirs(dest_atomtypes_pdb_dir, 0777)

    # =======================================================================
    # 0. copy the src atomtype file to dest dir, and change the work dir to the dest dir.
    shutil.copy(src_atomtypes, dest_atomtypes_pdb_dir)  # work dir doesn't change
    os.chdir(dest_atomtypes_pdb_dir)

    # # =======================================================================
    # # 1. Generate the ligand file
    # lig_pdb = '{0}_ligand.pdb'.format(file_name)
    #
    # if os.path.exists(lig_pdb):
    #     print "{} file exists.".format(lig_pdb)
    # else:
    #     create_ligand_file(src_atomtypes, lig_pdb)

    #  ========================================================================
    # 2. calculate the center of ligand
    lig_center = get_ligand_center_from_src_atomtypes(src_atomtypes)

    #  ========================================================================
    # 2.5. perform the translation on center of ligand
    moved_lig_center = translation(lig_center)

    # ========================================================================
    # 3. extract the atoms inside of binding box (i.e. the greater box, not
    #    the final grid)

    # Note:
    # Complexes in the filtered datasets have ligands with length <= 27 Angstroms

    # Note: we define the greater box that encompasses the atoms, just to
    # filter out atoms in the complex that are located outside of the box;
    # but then we will generate the KDTree from the voxel centers of this
    # greater box.
    bindingBox_atomtypes = '{0}_{1}_bindingBox.atomtypes'.format(file_name, int(grid_length))

    greater_box_max = moved_lig_center + (grid_length / 2.0)
    greater_box_min = moved_lig_center - (grid_length / 2.0)

    # For augment data, every time we will generate a new bindingbox file.
    # Because the old bindingbox file is based on the old moved_lig_center which is random, new moved_lig_center is not
    # correct for the old bindingbox file.
    extract_binding_box(src_atomtypes, bindingBox_atomtypes, greater_box_max, greater_box_min)

    # ========================================================================
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
    points = np.asarray(zip(x.ravel(), y.ravel(), z.ravel()))
    grid_tree = spatial.cKDTree(points)
    grid_shape = (num_points, num_points, num_points)

    with open(bindingBox_atomtypes, 'rb') as inBindingSite:

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
        #print "{} is done".format(file_name)
        np.savez(pickled_channels, pdb=pdb_name, label=label, pocket=pocket_cube)
    # os.remove(bindingBox_atomtypes)
    # os.remove(lig_pdb)


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
    """
    Note: comparing with find_overlap_grid_for_binary, kd tree query is within 2 vdw distance
    :param linetuple:
    :param grid_tree:
    :return:
    """
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


def verify_npz_aug(label_csv, src_dir):
    df = pd.read_csv(label_csv)
    for index, row in df.iterrows():
        real_pdb_code = row['pdb']
        real_label = np.float64(row['pKd/pKi'])

        matched_aug_pdb = [x.split('.')[0] for x in os.listdir(src_dir) if re.search(real_pdb_code, x)]

        for aug_sample in matched_aug_pdb:
            npz = np.load(os.path.join(src_dir, aug_sample + '.npz'))
            print("pdb:{}, {}".format(aug_sample, npz['label'] == real_label))


class Counter(object):
    def __init__(self, initval=0, mag=Manager()):
        self.val = mag.Value('i', initval)
        self.lock = mag.Lock()


def main(*args, **kwargs):
    starter = time.time()

    if kwargs:
        #print("************* KWARGS **************")
        config_dict = kwargs
    else:
        #print("************* CONFIG_DICT **************")
        config_dict = dict()
        config_dict['data_source'] = 'ind_prep'       # [Bmoad_general_refined, 2016, tl, vs, ind]
        config_dict['DRUG_ROOT_DIR'] = sys.argv[1]
        config_dict['dataset_name'] = sys.argv[2]
        config_dict['num_feature'] = 24         # [11/60/24]
        config_dict['grid_length'] = 32
        config_dict['resolution'] = 1.0         # [0.375/0.5/1.0]
        config_dict['mode'] = 'pcmax'          # [binary/gaussian/pcmax/pcsum]
        config_dict['ARPEGGIO_LENGTH'] = 11
        config_dict['keep_all_aug'] = True      # Whether keep all the augmented data
        config_dict['select_k'] = 36            # Select k number among all the augmented data of one pdb code

    mgic = MakeGridInfoCollect(**config_dict)
    if config_dict['data_source'] in ['PDBbind_2016', 'Bmoad_general_refined']:
        mgic.collect_aug_data_with_index()

    elif config_dict['data_source'] in ['tl', 'vs', 'ind', 'ind_prep', 'dlg']:
        mgic.collect_aug_data_without_index()
        #print(mgic.src_atomtypes_dict.items()[0])

    mgic.exclude_exist()

    if mgic.remaining_atomtpyes > 0:
        m = Manager()
        shared_src_atomtypes_dict = m.dict(mgic.src_atomtypes_dict)

        counter = Counter(0)

        # Only for testing with one process
        # for (k, v) in mgic.src_atomtypes_dict.items():
        #     make_grid(config_dict,
        #               mgic.dest_atomtypes_dir,
        #               mgic.dest_np_dir,
        #               counter,
        #               mgic.remaining_atomtpyes,
        #               (k, v))
        # input()

        # p = Pool(processes=cpu_count()-3)

        #_______________________________________________________________
        p = Pool()
        p.map_async(partial(make_grid,
                            config_dict,
                            mgic.dest_atomtypes_dir,
                            mgic.dest_np_dir,
                            counter,
                            mgic.remaining_atomtpyes
                            ),
                    shared_src_atomtypes_dict.items())
        #             mgic.src_atomtypes_dict.items())  # Note: not advised

        p.close()
        p.join()
        #_______________________________________________________________

        end = time.time()
        print("During: {}s".format(end-starter))

    if mgic.index_file:
        verify_npz_aug(mgic.index_file, mgic.dest_np_dir)


if __name__ == "__main__":
    main()
