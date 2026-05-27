#!/usr/bin/env
# -*- coding: utf-8 -*-

from config import *
import os
import sys
from glob import glob
import shutil
import numpy as np
import math
from collections import namedtuple
from scipy import spatial
import pandas as pd
import time
from multiprocessing import Pool, Manager, Value, Lock, cpu_count
from scipy.spatial import distance
from functools import partial

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
        #print 'data source: {}'.format(data_source)

        num_grid = int(np.ceil(grid_length / resolution))  # [48/86]
        self.dataset_name = dataset_name

        #self.base_data_dir = os.path.join(config_dict['DRUG_ROOT_DIR'], data_source, self.dataset_name)
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

        for f in glob(os.path.join(self.base_data_dir, '*.atomtypes')):
            shutil.move(f, os.path.join(self.src_atomtypes_dir, os.path.split(f)[-1]))

        #print 'Org dir: {}\nNumpy saved: {}'.format(self.src_atomtypes_dir, self.dest_np_dir)
        os.makedirs(self.dest_np_dir, exist_ok=True)

    def collect_data_with_index(self):
        """
        This process is different with TL data, because for pdbbind we refer to the clean index file,
        for transfer learning data no index
        :return:
        """

        #print "Running 'collect_data_with_index' ..."
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

        index_path = os.path.join(self.original_data_dir, 'index')
        os.makedirs(index_path, exist_ok=True)
        self.index_file = os.path.join(self.original_data_dir, 'index', self.dataset_name + '.csv')
        print('self.index_file: %s' % self.index_file)
        print('index_path: %s' % index_path)
        # shutil.copy('/blue/lic/huangzihang/repos/LP_PDBBind/1a30/index/1a30.csv', index_path)
        shutil.copy(self.index_file, index_path)
        print('Copy complete.')
        df = pd.read_csv(self.index_file)
        label_df = df[['pdb', 'pKd/pKi']]
        print('label_df: %s' % label_df)

        # Format: {'pdb': {pdb: ..., label: .., 'atomtypes': ...}} the key is file name
        self.src_atomtypes_dict = dict()
        for idx, x in label_df.iterrows():
            file_name = x['pdb']                      # Eg. 1w4r
            pdb_dict = {'pdb': x['pdb'],
                        'label': x['pKd/pKi'],
                        'atomtypes': os.path.join(self.src_atomtypes_dir, x['pdb'] + '.atomtypes')}
            self.src_atomtypes_dict[file_name] = pdb_dict

        #print 'Number of complex: {}'.format(len(self.src_atomtypes_dict))

    def collect_data_without_index(self):
        print("Running 'collect_data_without_index' ...")

        self.index_file = None
        self.src_atomtypes_dict = dict()
        print("(self.src_atomtypes_dir) = " + self.src_atomtypes_dir)

        # Format: {'pdb_MOD_...': {pdb: ..., label: .., 'atomtypes': ...}} the key is file name
        for x in os.listdir(self.src_atomtypes_dir):
            file_name = x.split('.atomtypes')[0]        # Eg. 1w4r_PLM_282_B_MOD for TL
            pdb_dict = {'pdb': file_name,
                        'label': np.nan,
                        'atomtypes': os.path.join(self.src_atomtypes_dir, x)}
            self.src_atomtypes_dict[file_name] = pdb_dict

        print("Number of complex: {}".format(len(self.src_atomtypes_dict)))

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
    #print "Running 'create_ligand_file' ..."

    with open(src_atomtypes, 'r') as inFile:
        with open(lig_pdb, 'w') as outFile:
            for line in inFile:
                # chain 'y' is assigned to ligand atoms in 01_preprocess_complexes_VS.py
                if line.startswith('y'):
                    outFile.write(line)


def get_ligand_center(lig_pdb):
    #print "Running 'get_ligand_center' ..."

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


def extract_binding_box(src_atomtypes, bindingBox_atomtypes, greater_box_max, greater_box_min):
    #print "Running 'extract_binding_box' ..."

    # print 'src_atomtypes: %s' % src_atomtypes
    with open(src_atomtypes, 'r') as inComplex:
        with open(bindingBox_atomtypes, 'w') as outBindingSite:
            for line in inComplex:
                x_coord = float(line[39:49].strip())
                y_coord = float(line[49:59].strip())
                z_coord = float(line[59:69].strip())
                atom_coord = np.array([x_coord, y_coord, z_coord])
                # print 'atom_coord: %s' % atom_coord

                if np.all(atom_coord < greater_box_max) & np.all(atom_coord > greater_box_min):
                    # print 'line: %s' % line
                    outBindingSite.write(line)
            print('np version: %s' % np.__version__)


def parse_line_by_substr(line):
    #print "Running 'parse_line_by_substr' ..."

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
    #print "Running 'generate_mask_channel' ..."

    if num_feature == 60:
        anolea_vector = list(map(int, [ix == int(_linetuple.anolea) - 1 for ix in range(ANOLEA_SIZE + LIG_ELEMENT_SIZE)]))
        arpeggio_vector = list(map(int, list(_linetuple.arpeggio)))
        mask_vector = np.asarray(anolea_vector + arpeggio_vector)
    elif num_feature == 11:
        arpeggio_vector = list(map(int, list(_linetuple.arpeggio)))
        mask_vector = np.asarray(arpeggio_vector)

    elif num_feature == 24:
        if not _linetuple.atomid.startswith('y'):  # i.e. a protein atom
            arpeggio_vector_pr = list(map(int, list(_linetuple.arpeggio)))
            ExVol_pr = [1]  # Excluded volume for protein
            arpeggio_vector_lig = list(map(int, list('0' * config_dict['ARPEGGIO_LENGTH'])))
            ExVol_lig = [0]  # Excluded volume for ligand

        else:
            arpeggio_vector_pr = list(map(int, list('0' * config_dict['ARPEGGIO_LENGTH'])))
            ExVol_pr = [0]  # Excluded volume for protein
            arpeggio_vector_lig = list(map(int, list(_linetuple.arpeggio)))
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
    #print "Running 'update_channel_binary' ..."

    x_index, y_index, z_index = np.unravel_index(_update_indexes_flat, grid_shape)
    overlap_indices_on_grid = list(zip(x_index, y_index, z_index))

    for ix in overlap_indices_on_grid:
        _pocket_cube[ix] = list(map(int, np.logical_or(_pocket_cube[ix], _mask_vector)))
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
    #print "Running 'update_channel_pc' ..."

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


def make_grid(config, dest_atomtypes_dir, dest_np_dir, counter, total_num, file_info):
    #print "Running 'make_grid' ..."
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
    # 0. copy the src atomtype file to dest dir, and change the work dir to the dest dir.
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

    #  ========================================================================
    # 2. calculate the center of ligand
    lig_center = get_ligand_center(lig_pdb)

    # ========================================================================
    # 3. extract the atoms inside of binding box (i.e. the greater box, not
    #    the final grid)

    # Note:
    # Complexes in the filtered datasets have ligands with length <= 27 Angstroms

    # Note: we define the greater box that encompasses the atoms, just to
    # filter out atoms in the complex that are located outside of the box;
    # but then we will generate the KDTree from the voxel centers of this
    # greater box.
    greater_box_max = lig_center + (grid_length / 2.0)
    greater_box_min = lig_center - (grid_length / 2.0)

    bindingBox_atomtypes = '{0}_{1}_bindingBox.atomtypes'.format(pdb_name, int(grid_length))

    if os.path.exists(bindingBox_atomtypes):
        print("{} file exists.".format(bindingBox_atomtypes))
    else:
        extract_binding_box(src_atomtypes, bindingBox_atomtypes, greater_box_max, greater_box_min)

    #print "CREATED 'bindingBox_atomtypes' ..."
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
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_tree = spatial.cKDTree(points)
    grid_shape = (num_points, num_points, num_points)

    with open(bindingBox_atomtypes, 'r') as inBindingSite:
        #print "OPENED 'bindingBox_atomtypes' ..."

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
    print('pickled_channels: %s' % pickled_channels)


def find_overlap_grid_for_binary(linetuple, grid_tree):
    #print "Running 'find_overlap_grid_for_binary' ..."

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
    #print "Running 'find_overlap_grid_for_pc' ..."

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


def verify_npz(label_csv, src_dir):
    #print "Running 'verify_npz' ..."

    df = pd.read_csv(label_csv)
    for index, row in df.iterrows():
        real_pdb_code = row['pdb']
        real_label = np.float64(row['pKd/pKi'])

        npz = np.load(os.path.join(src_dir, real_pdb_code + '.npz'))
        #print 'pdb:{}, {}'.format(real_pdb_code, npz['label'] == real_label)


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
        config_dict['data_source'] = 'vs'  # [Bmoad_general_refined, PDBbind_2016, tl, vs, ind]
        # config_dict['data_source'] = 'PDBbind_2016'
        #config_dict['DRUG_ROOT_DIR'] = sys.argv[1]
        config_dict['batch_dir'] = sys.argv[1]
        config_dict['dataset_name'] = sys.argv[2]
        config_dict['num_feature'] = 24  # [ARPEGGIO_LENGTH + excluded volume} * 2
        config_dict['grid_length'] = 32
        config_dict['resolution'] = 1.0  # [0.375/0.5/1.0]
        config_dict['mode'] = 'pcmax'  # [binary/gaussian/pcmax/pc_sum]
        config_dict['ARPEGGIO_LENGTH'] = 11

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

        # Only for testing with one process
        # init()
        #make_grid(config_dict,
        #           mgic.dest_atomtypes_dir,
        #           mgic.dest_np_dir,
        #           counter,
        #           mgic.remaining_atomtpyes,
        #           mgic.src_atomtypes_dict.items()[0])
        #input()

        #p = Pool(processes=cpu_count() - 3)

        #_______________________________________________________________
        p = Pool()
        try:
            p.map(partial(make_grid,
                          config_dict,
                          mgic.dest_atomtypes_dir,
                          mgic.dest_np_dir,
                          counter,
                          mgic.remaining_atomtpyes
                          ),
                  list(shared_src_atomtypes_dict.items()))
        except Exception as exc:
            print("ERROR in worker: {}".format(exc))
            raise
        finally:
            p.close()
            p.join()
        #_______________________________________________________________
        
        end = time.time()
        print("During: {}s".format(end - starter))

    print('mgic.index_file: %s' % mgic.index_file)
    if mgic.index_file:
        verify_npz(mgic.index_file, mgic.dest_np_dir)


if __name__ == "__main__":
    main()