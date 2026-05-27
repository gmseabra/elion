#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
test.py: 
"""
from torch.utils.data import Dataset
from torch.autograd import Variable
from dataloader import PdbBindDataset
from shufflenet_v3 import ShuffleNetV3
from utils import *
from record_config import *
from collections import defaultdict
import gc
import random
from collections import OrderedDict
import sys
import os

os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'   # see issue #152


__author__ = "Yanjun Li"
__license__ = "MIT"


script_dir = os.path.realpath(__file__)
os.chdir(os.path.dirname(script_dir))
print(os.getcwd())

#  ========================================================================
# Basic config
print("Process Id: {}".format(os.getpid()))

batch_dir = sys.argv[1].split('=')[-1]
input_type = sys.argv[2].split('=')[-1]
test_dataset = sys.argv[3].split('=')[-1]
model_dir = sys.argv[4].split('=')[-1]
out_csv_dir = sys.argv[5].split('=')[-1]

args = get_arguments('test', batch_dir=batch_dir, test_type=input_type, 
                     test_dataset=test_dataset, model_dir=model_dir, out_csv_dir=out_csv_dir)

if args.num_gpus == 0:
    os.environ["CUDA_VISIBLE_DEVICES"]='none'
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    assert torch.cuda.device_count() == args.num_gpus, "The visible gpu {} does not match with args.num_gpus {}."
    device = None

'''
root_weight_dir = os.path.join('./weight',
                               args.data_source + '_' + args.data_kind,
                               '{}_{}_{}'.format(args.num_grid, args.input_feature, args.occupancy),
                               args.model,
                               str(args.number))
'''


root_weight_dir = DRUG_ROOT_DIR

#weight_dir_dict = create_weight_dir(root_weight_dir)
args_dict = vars(args)
args_dict.update({"batch_dir": batch_dir})
print_dict_byline(args_dict)

#  =================================== =====================================
# Data related

print('args.test_type: %s' % args.test_type)
print('args_dict: %s' % args_dict)
test_dataset = PdbBindDataset(args.test_type, **args_dict)

# for i in test_dataset:
#     print('-----------------debug4 i: %s' % i)

print('test_dataset len: %s' % len(test_dataset))

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers)

#  ========================================================================
# Model related
if args.model == 'ShuffleNetV3x2':
    model = ShuffleNetV3(input_channel=args.input_feature, dropout_prob=0.0,
                         width_multiplier=2.0)
    print(torch.utils.data.__file__)
    print(torch.nn.__file__)

#  ========================================================================
# Summary related (Only accept the torch.FloatTensor on CPU)

#  ========================================================================
# Model report
parallel_model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus))
num_test_sample = test_dataset.__len__()
print("Test samples: {}".format(num_test_sample))


# #  ========================================================================
# # Model on cuda related
if args.num_gpus != 0:
    parallel_model.cuda()

#  ========================================================================
# Restore the previous weight if necessary
print("==> restoring weights, based on {}".format(args.restore_criteria))
#restore_weight_path = weight_dir_dict.get(args.restore_criteria)
restore_weight_path = model_dir
print(restore_weight_path)

restore_file = os.path.join(restore_weight_path, os.listdir(restore_weight_path)[0])
if restore_file.endswith('.pk'):
    # For the model which only saves the weight
    parallel_model.load_state_dict(torch.load(restore_file))
    print('=> loaded checkpoint from weight.pk')
elif restore_file.endswith('.tar'):
    # For the model which saves the weight and training status, as 'model.pth.tar'
    # {key: value for key, value in dict2.items() if key in dict1}
    checkpoint = torch.load(restore_file, map_location=device)

    # checkpoint = torch.load(restore_file, map_location=device)
    parallel_model.load_state_dict(checkpoint['state_dict'])
    print('=> loaded checkpoint from model.path.tar')


#  ========================================================================
# Test for one epoch
def evaluate(data_loader):
    test_loss = 0
    parallel_model.eval()
    data_dict = defaultdict(list)
    for i_batch, sample_batched in enumerate(data_loader):
        pdb, x, y_true = sample_batched['pdb'], sample_batched['pocket'], sample_batched['label']
        if args.num_gpus != 0:
            x, y_true = x.cuda(), y_true.cuda()
        output = parallel_model(x)
        # print 'Batch %d: x shape %s, x mean %f, x std %f' % (i_batch, x.shape, x.mean(), x.std())
        # print 'y_true: %s, output: %s' % (y_true.data, output.data)
        data_dict['y_pdb'].extend(pdb)
        data_dict['y_true'].extend(y_true.data)
        data_dict['y_pred'].extend(output.data)
        gc.collect()
        del x, y_true
        print('Complete batch: %d' % i_batch)
    for k, v in data_dict.items():
        data_dict[k] = np.asarray(v)
    return data_dict

#  ========================================================================
# Evaluation process
epoch_start = time.time()

# Evaluate on the valid data
test_data_dict = evaluate(test_loader)
print('-----------------------debug2')

if args.avg_test:
    # Use the multiple avg augmented technique to valid and test data set
    test_y_pdb, test_y_true, test_y_pred = multi_avg(test_data_dict['y_pdb'], test_data_dict['y_true'], test_data_dict['y_pred'])

dG = [-1.36 * pk_pred for pk_pred in test_y_pred.tolist()]
result_dict = {'PDB': test_y_pdb, 'deltaG_kcal_mol': dG}

# metric_dict = cal_metrics(test_y_true, test_y_pred, 'test')
print("== Test == ")
# print_metrics(metric_dict)

epoch_end = time.time()
print("Epoch duration: {}".format(epoch_end - epoch_start))

saved_csv = os.path.join(out_csv_dir, '{}_{}.csv'.format(args.test_type, args.test_dataset))

print('result_dict: %s' % result_dict)

# Save the final results
save_dict_to_csv(saved_csv, result_dict)