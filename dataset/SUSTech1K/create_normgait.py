#!/usr/bin/env python

import numpy as np
import os

def create_symb(data_root):
    count = 0
    sample_list = os.listdir(os.path.join(data_root, 'train'))
    for item in sample_list:
        if '00-nm' in item:
            if count == 0:
                gait_symb = np.load(os.path.join(data_root, 'train', item), allow_pickle=True)[0].astype(np.float64)
                assert len(gait_symb.shape) == 3
            else:
                gait_symb += np.load(os.path.join(data_root, 'train', item), allow_pickle=True)[0].astype(np.float64)
            count += 1
            if count%50 == 0:
                print('{} samples merged'.format(count))
    gait_symb = np.round(gait_symb/(count/3)).astype(np.uint8)
    print('<Completed>')
    return gait_symb

if __name__ == '__main__':
    data_root = '/home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel.20/tmp'
    normalized_gait = create_symb(data_root)
    np.save(os.path.join(data_root, 'norm_gait.npy'), normalized_gait)
