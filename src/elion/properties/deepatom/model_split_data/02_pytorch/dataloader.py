#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
dataloader.py: 
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import *
import gc
import argparse
from test_net_config import DRUG_ROOT_DIR
import sys

__author__ = "Yanjun Li"
__license__ = "MIT"


class PdbBindDataset(Dataset):
    """Pdb binding dataset."""

    def __init__(self, phase, **kwargs):
        self.data_source = kwargs['data_source']
        self.data_kind = kwargs['data_kind']
        self.batch_dir = kwargs['batch_dir']
        self.test_dataset = kwargs['test_dataset']
        self.input_feature = kwargs['input_feature']
        self.num_grid = kwargs['num_grid']
        self.occupancy = kwargs['occupancy']
        self.split_mode = kwargs['split_mode']
        self.mode = kwargs['mode']
        self.aug_train = kwargs['aug_train']
        self.aug_num_train = kwargs['aug_num_train']
        self.avg_valid = kwargs['avg_valid']
        self.avg_test = kwargs['avg_test']
        self.batch_size = kwargs['batch_size']
        self.label_mean = kwargs['mean']
        self.label_std = kwargs['std']
        self.num_gpus = kwargs['num_gpus']
        self.phase = phase
        self.normalize = kwargs['normalize']

        if self.phase not in ['vs', 'ind', 'ind_prep']:
            # Extract data according to the index file
            org_np_dir = os.path.join(DRUG_ROOT_DIR,
                                      self.data_kind,
                                      '3d_{}_{}_{}'.format(self.num_grid, self.input_feature, self.occupancy))

            split_index_dir = os.path.join(DRUG_ROOT_DIR,
                                           self.data_kind,
                                           'split_index',
                                           self.split_mode)

            aug_np_dir = os.path.join(DRUG_ROOT_DIR,
                                      self.data_kind + '_aug',
                                      '3d_{}_{}_{}'.format(self.num_grid, self.input_feature, self.occupancy))
            aug_nps = os.listdir(aug_np_dir)
            # Transfer to string first to avoid iterate over list in re
            aug_nps_str = ','.join(aug_nps)

            if self.phase == 'train':
                train_csv = os.path.join(split_index_dir, 'train.csv')
                train_df = pd.read_csv(train_csv)
                train_pdb = train_df['pdb'].values
                self.filenames = [os.path.join(org_np_dir, x + '.npz') for x in train_pdb]

                if self.aug_train:
                    for pdb in train_pdb:
                        pattern = pdb + '_augmented_' + r'\d+' + '.npz'
                        pdb_aug_nps = re.findall(pattern, aug_nps_str)
                        pdb_aug_nps = self.select_from_aug(pdb_aug_nps)
                        pdb_aug_nps_fullname = [os.path.join(aug_np_dir, x) for x in pdb_aug_nps]
                        self.filenames.extend(pdb_aug_nps_fullname)

                # print len(self.train_filenames)

            elif self.phase == 'valid':
                valid_csv = os.path.join(split_index_dir, 'valid.csv')
                valid_df = pd.read_csv(valid_csv)
                valid_pdb = valid_df['pdb'].values
                self.filenames = [os.path.join(org_np_dir, x + '.npz') for x in valid_pdb]

                if self.avg_valid:
                    for pdb in valid_pdb:
                        pattern = pdb + r'_augmented_\d+\.npz'
                        pdb_aug_nps = re.findall(pattern, aug_nps_str)
                        pdb_aug_nps_fullname = [os.path.join(aug_np_dir, x) for x in pdb_aug_nps]
                        self.filenames.extend(pdb_aug_nps_fullname)

                # print self.train_steps_per_epoch, self.valid_steps_per_epoch

            elif self.phase == 'test':
                test_csv = os.path.join(split_index_dir, 'test.csv')
                test_df = pd.read_csv(test_csv)
                test_pdb = test_df['pdb'].values
                self.filenames = [os.path.join(org_np_dir, x + '.npz') for x in test_pdb]

                if self.avg_test:
                    for pdb in test_pdb:
                        pattern = pdb + r'_augmented_\d+\.npz'
                        pdb_aug_nps = re.findall(pattern, aug_nps_str)
                        pdb_aug_nps_fullname = [os.path.join(aug_np_dir, x) for x in pdb_aug_nps]
                        self.filenames.extend(pdb_aug_nps_fullname)

        else:
            # Extract data according to actual file stored in the folder
            org_np_dir = os.path.join(self.batch_dir,
                                      '3d_{}_{}_{}'.format(self.num_grid, self.input_feature, self.occupancy))

            aug_np_dir = os.path.join(self.batch_dir,
                                      '3d_{}_{}_{}_aug'.format(self.num_grid, self.input_feature, self.occupancy))

            self.filenames = listdir_fullpath(org_np_dir)
            print("Original files from %s: %s" % (org_np_dir, len(self.filenames)))
            if self.avg_test:
                self.filenames.extend(listdir_fullpath(aug_np_dir))
                print("Augmented files from %s: %s" % (aug_np_dir, len(listdir_fullpath(aug_np_dir))))
            print("Total filenames: %s" % (len(self.filenames)))

        # self.steps_per_epoch = 10
        if self.num_gpus != 0:
            self.steps_per_epoch = (len(self.filenames) + self.num_gpus * self.batch_size - 1) // \
                                   (self.num_gpus * self.batch_size)
        gc.collect()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        np_file = self.filenames[idx]
        np_data = np.load(np_file)
        pocket = np_data['pocket']
        label = np_data['label'].reshape(-1)    # It has to be a tensor with at least 1D
        pdb = str(np_data['pdb'])


        if self.normalize:
            label = self.z_score(label)

        sample = {'pdb': pdb, 'label': label, 'pocket': pocket}

        transform = transforms.Compose([ToTensor()])
        sample = transform(sample)
        return sample

    def z_score(self, org_label):
        return (org_label - self.label_mean) / self.label_std

    def select_from_aug(self, aug_samples):
        np.random.shuffle(aug_samples)
        return aug_samples[0: self.aug_num_train]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        pdb, pocket, label = sample['pdb'], sample['pocket'].astype('float32'), sample['label'].astype('float32')

        # swap last channel axis because
        # numpy image: H x W x D x C
        # torch image: C X H X W x D
        pocket = pocket.transpose((3, 0, 1, 2))
        return {'pdb': str(pdb),
                'pocket': torch.from_numpy(pocket),
                'label': torch.from_numpy(label)}


def get_arguments():
    parser = argparse.ArgumentParser(description='Get the pocket input.')
    parser.add_argument('--data_source', default='PDBbind_2016')
    parser.add_argument('--data_kind', default='refined')
    parser.add_argument('--batch_dir')
    parser.add_argument('--input_feature', default=24)
    parser.add_argument('--num_grid', default=32)
    parser.add_argument('--occupancy', default='pcmax')
    parser.add_argument('--mode', '-m', type=str, default='train', help='train/test')
    parser.add_argument('--test_dataset', type=str, default='csar2012')    # csar2012
    parser.add_argument('--aug_train', default=True)
    parser.add_argument('--aug_num_train', default=10)
    parser.add_argument('--avg_valid', default=False)
    parser.add_argument('--avg_test', default=True)
    parser.add_argument('--split_mode', default='refined_core')
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--max_epoches', type=int, default=100)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--mean', type=float, default=6.392)
    parser.add_argument('--std', type=float, default=1.995)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    args_dict = vars(args)
    starter = time.time()
    transformed_dataset = PdbBindDataset('train', **args_dict)
    print(transformed_dataset.__len__())

    # for i in range(len(transformed_dataset)):
    #     sample = transformed_dataset[i]
    #
    #     print(i, sample['pdb'], sample['pocket'].size(), sample['label'].size())
    #
    #     # if i == 3:
    #     #     break

    train_loader = torch.utils.data.DataLoader(transformed_dataset,
                                               batch_size=args.batch_size, shuffle=False,
                                               num_workers=0)

    for batch_idx, examples in enumerate(train_loader):
        print(examples['pocket'].size(), examples['label'])
        # print 'done'
        # break
    end = time.time()
    duration = (end - starter) / 60.0
    print('Duration: {}'.format(duration))