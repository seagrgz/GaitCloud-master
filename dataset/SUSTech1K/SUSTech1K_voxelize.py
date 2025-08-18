#!/usr/bin/env python
import numpy as np
import os
import sys
import pickle
import math
import time
import yaml

sys.path.append('../../util/')
from pretreatment import seq_voxelize

data_root = 'SUSTech1K-Released-pkl'
dst = 'SUSTech1K-depthimgdil2'
frame_num = 20
box_size = [1.25,1.25,2]
res = 0.03125
compress = True
tfusion = 'depth' #tmp, spat, tmp+spat, depth
tmp_diff = False
dilution = 2
noise = 0
#for noise in [0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095]:

def SUSTech1K_voxelize(data_root,
        dst,
        frame_num,
        box_size,
        res,
        dilution=1, 
        noise=0, 
        counter=1, 
        compress=0,
        tfusion=False,
        tmp_diff=False):

    if isinstance(res, float):
        res = [res, res, res]
    elif isinstance(res, list):
        pass
    else:
        raise NotImplementedError('res should be list or float')

    if os.path.exists(dst):
        print('Removing old data ...')
        os.system('rm -rf {}/**'.format(dst))
    else:
        os.system('mkdir {}'.format(dst))

    target_list = sorted(os.listdir(data_root)) 
    target_num = len(target_list)
    st = time.time()
    for i, target in enumerate(target_list):
        attr_root = os.path.join(data_root, target)
        attributes = os.listdir(attr_root)
        for attr in attributes:
            vp_root = os.path.join(attr_root, attr)
            views = os.listdir(vp_root)
            for vp in views:
                sample_path = os.path.join(vp_root, vp, '00-{}-LiDAR-PCDs.pkl'.format(vp))
                namespace = '_'.join([target, attr, vp])
                if os.path.exists(sample_path):
                    with open(sample_path, 'rb') as f:
                        data = pickle.load(f)
                    f.close()
                else:
                    print('Warning: {} not found!'.format(sample_path))
                    continue

                flag = seq_voxelize(data, dst, namespace, frame_num, box_size, res, dilution, noise, counter, compress, tfusion, tmp_diff)
                if flag:
                    print('Warning: Target {}~{}~{} discard'.format(target, attr, vp))

        t_remain = (time.time()-st)/(i+1) * (target_num-i-1)
        print('Target {} completed, time remain: {}h-{}m-{}s'.format(target, int(t_remain//3600), int(t_remain%3600//60), int(t_remain%60)))

if __name__ == '__main__':
    #for noise in [0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095]:
    SUSTech1K_voxelize(data_root = data_root,
                dst = dst.format(noise),
                frame_num = frame_num,
                box_size = box_size,
                res = res,
                dilution = dilution,
                noise = noise,
                compress = compress,
                tfusion = tfusion,
                tmp_diff = tmp_diff)
