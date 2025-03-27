#!/usr/bin/env python

import os
import json
import yaml
import pickle
import itertools
import random
import numpy as np

data_root = 'SUSTech1K-Released-voxel.20tmpcomp2'
lidargait = False
#def move_view_fix(data_root, dst):
#    variance = ['00-nm', '01-nm']
#    view = '000'
#    count = 0
#    for ith, data_var in enumerate(variance):
#        data_list = [item[:-4] for item in sorted(os.listdir(os.path.join(data_root, data_var, view))) if not item=='sample_list.npy']
#        sample_list = np.load(os.path.join(data_root, data_var, view, 'sample_list.npy'))
#        if ith == 0:
#            sample_probe = sample_list
#        
#        for name in data_list:
#            data = np.load(os.path.join(data_root, data_var, view, name+'.npy'), allow_pickle=True)
#            if ith == 0:
#                np.save(os.path.join(dst, 'probe', str(count).zfill(6)+'-'+data_var+'.npy'), data, allow_pickle=True)
#            else:
#                if str(data[1]).zfill(4) in sample_probe:
#                    np.save(os.path.join(dst, 'gallery', str(count).zfill(6)+'-'+data_var+'.npy'), data, allow_pickle=True)
#                else:
#                    np.save(os.path.join(dst, 'train', str(count).zfill(6)+'-'+data_var+'.npy'), data, allow_pickle=True)
#            count += 1
#            if count%20==0:
#                print('{} {} move complete'.format(data_var, name))
#    sample_list = set(sample_list) | set(sample_probe)
#    np.save(os.path.join(dst, 'sample_list.npy'), list(sample_list))
#    print(len(sample_probe))
                    
def move_data(root, dst, partition):
    variance = ['bg', 'cl', 'cr', 'oc', 'ub', 'uf', 'nt', '01-nm', '00-nm']
    view = ['000', '045', '090', '135', '180', '225', '270', '315', '000-far', '090-near', '180-far', '270-far']

    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]
    label_list = os.listdir('SUSTech1K-Released-pkl')
    train_list = []
    test_list = []

    train_set = [label for label in train_set if label in label_list]
    test_set = [label for label in test_set if label in label_list]
    miss_pids = [label for label in label_list if label not in (train_set + test_set)]
    #test_set = [name for name in target_list if name in test_set]
    #with open('ignoresamples.yaml', 'r') as f:
    #    ignore = yaml.safe_load(f)
    #    ignore_all = ignore['all']
    #    ignore_part = ignore['part']
    #f.close()
    ignore_all = []
    ignore_part = {}

    #for constrained subset
    """
    train_names = {}
    test_names = {}
    attribute = ['01-bg_', '01-cl_', '01-cr_', '01-oc_', '01-ub_', 'uf', 'nt', '01-nm_']
    data_list = os.listdir(os.path.join(root, '000-far'))

    for var in attribute:
        train_names[var] = [name for name in data_list if (var in name) and (name[-8:-4] in train_set)]
        test_names[var] = [name for name in data_list if (var in name) and (name[-8:-4] in test_set)]
    train_num = min([len(train_names[item]) for item in attribute])
    test_num = min([len(test_names[item]) for item in attribute])
    print('{} samples in train class\n{} samples in test class'.format(train_num, test_num))
    for var in attribute:
        train_names[var] = random.sample(train_names[var], train_num)
        test_names[var] = random.sample(test_names[var], test_num)
    train_sampled = list(dict.fromkeys(list(itertools.chain(*train_names.values()))))
    train_list = list(dict.fromkeys([name[-8:-4] for name in train_sampled]))

    for vp in view:
        print('processing {}'.format(vp))
        for name in train_sampled:
            split, view, sample = name.split('_')
            name = '{}_{}_{}'.format(split, vp, sample)
            normal_name = '{}_{}_{}'.format('00-nm', vp, sample)
            if os.path.exists('{}/{}/{}'.format(root, vp, name)):
                os.system('cp {}/{}/{} {}/train/{}'.format(root, vp, name, dst, name))
            else:
                print('{} not exists'.format(name))
            if os.path.exists('{}/{}/{}'.format(root, vp, normal_name)):
                os.system('cp {}/{}/{} {}/train/{}'.format(root, vp, normal_name, dst, normal_name))
            else:
                print('{} not exists'.format(normal_name))
        data_list = os.listdir(os.path.join(root, vp))
        for name in data_list:
            split, view, sample = name.split('_')
            sample = sample[:-4]
            #test set
            if sample in test_set:
                if os.path.exists('{}/{}/{}'.format(root, vp, name)):
                    os.system('cp {}/{}/{} {}/probe/{}'.format(root, view, name, dst, name))
                    if sample not in test_list:
                        test_list.append(sample)
    """
    #use all samples
    for vp in view:
        print('processing {}'.format(vp))
        data_list = os.listdir(os.path.join(root, vp))
        #print(len(os.listdir(os.path.join(root, 'tmp/gallery'))), len(test_set)/2)
        for name in data_list:
            split, view, sample = name.split('_')
            sample = sample[:-4]
            #train set
            if sample in train_set:
                try:
                    os.system('cp {}/{}/{} {}/train/{}'.format(root, view, name, dst, name))
                    if sample not in train_list:
                        train_list.append(sample)
                except FileNotFoundError:
                    pass
            #test set
            elif sample in test_set:
                if sample in ignore_all:
                    continue
                elif sample in ignore_part.keys():
                    if split == ignore_part[sample]:
                        continue
                try:
                    os.system('cp {}/{}/{} {}/test/{}'.format(root, view, name, dst, name))
                    if sample not in test_list:
                        test_list.append(sample)
                except FileNotFoundError:
                    pass
            else:
                print('Unused sample: {}'.format(name))
    return train_list, test_list

def create_baseline(data_root, dst, ref_root, partition):
    variance = ['bg', 'cl', 'cr', 'oc', 'ub', 'uf', 'nt', '01-nm', '00-nm']
    view = ['000', '045', '090', '135', '180', '225', '270', '315', '000-far', '090-near', '180-far', '270-far']
    frame_num = 10
    train_samples = partition["TRAIN_SET"]
    test_samples = partition["TEST_SET"]
    train_list = []
    test_list = []
    label_list = os.listdir(data_root)
    train_set = [label for label in train_samples if label in label_list]
    test_set = [label for label in test_samples if label in label_list]
    count = 0
    
    for vp in view:
        print('Processing {}'.format(vp))
        for sample in label_list:
            sample_root = os.path.join(data_root, sample)
            attributes = list(os.listdir(sample_root))
            if len(attributes) > 0:
                for item in attributes:
                    if sample in train_set:
                        try:
                            with open(os.path.join(sample_root, item, vp, '01-{}-LiDAR-PCDs_depths.pkl'.format(vp)), 'rb') as PCD:
                                data = pickle.load(PCD)
                            PCD.close()
                            if not sample in train_list:
                                print('add target {}, total num {}'.format(sample, len(train_list)))
                                np.save(os.path.join(dst, 'train', '{}_{}_{}.npy'.format(item, vp, sample)), data)
                                train_list.append(sample)
                            print('{} {} {} complete'.format(item, vp, sample))
                        except FileNotFoundError as e:
                            print('File not found:', e)
                    elif sample in test_set:
                        try:
                            with open(os.path.join(sample_root, item, vp, '01-{}-LiDAR-PCDs_depths.pkl'.format(vp)), 'rb') as PCD:
                                data = pickle.load(PCD)
                            PCD.close()
                            if not sample in test_list:
                                print('add target {}, total num {}'.format(sample, len(test_list)))
                                np.save(os.path.join(dst, 'probe', '{}_{}_{}.npy'.format(item, vp, sample)), data)
                                test_list.append(sample)
                            print('{} {} {} complete'.format(item, vp, sample))
                        except FileNotFoundError as e:
                            print('File not found:', e)
                    else:
                        print('Unused sample:', sample)
    return train_list, test_list

def create_inference(root, std_dev, test_set):
    view = ['000', '045', '090', '135', '180', '225', '270', '315', '000-far', '090-near', '180-far', '270-far']
    label_list = os.listdir('/home/sx-zhang/SUSTech1K/SUSTech1K-Released-pkl')
    test_set = [label for label in test_set if label in label_list]
    for vp in view:
        print('processing {}'.format(vp))
        data_list = os.listdir(os.path.join(root, vp))
        #print(len(os.listdir(os.path.join(root, 'tmp/gallery'))), len(test_set)/2)
        for name in data_list:
            split, view, sample = name.split('_')
            sample = sample[:-4]
            #test set
            if sample in test_set and os.path.exists('{}/{}/{}'.format(root, vp, name)):
                    os.system('cp {}/{}/{} /home/sx-zhang/SUSTech1K/inference/dev_{}/{}'.format(root, view, name, std_dev, name))

if __name__ == '__main__':
    if lidargait:
        data_root = 'SUSTech1K-Released-pkl'
        dst = 'SUSTech1K-Released-baseline'
        ref_root = 'SUSTech1K-Released-voxel.20'
    else:
        dst = data_root+'/tmp'

    os.system('rm -rf {}'.format(dst))
    os.system('mkdir {}'.format(dst))
    os.system('mkdir {}/train'.format(dst))
    os.system('mkdir {}/test'.format(dst))

    with open('SUSTech1K.json', 'rb') as f:
        partition = json.load(f)
    f.close()

    ##training
    if lidargait:
        train_list, test_list = create_baseline(data_root, dst, ref_root, partition)
    else:
        train_list, test_list = move_data(data_root, dst, partition)

    np.save(os.path.join(dst, 'train_list.npy'), sorted(train_list))
    np.save(os.path.join(dst, 'test_list.npy'), sorted(test_list))

    #inference
    #data_root = '/home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel.20'
    #os.system('mkdir /home/sx-zhang/SUSTech1K/inference/dev_0')
    #create_inference(data_root, 0, partition['TEST_SET'])
    #for noise in [0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095]:
    #    print('create data set with noise {}'.format(noise))
    #    data_root = '/home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel.20dev{}'.format(noise)
    #    if os.path.exists('/home/sx-zhang/SUSTech1K/inference/dev_{}'.format(noise)):
    #        os.system('rm -rf /home/sx-zhang/SUSTech1K/inference/dev_{}'.format(noise))
    #    os.system('mkdir /home/sx-zhang/SUSTech1K/inference/dev_{}'.format(noise))
    #    create_inference(data_root, noise, partition['TEST_SET'])
