#!/usr/bin/env python
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

sys.path.append('../../util/')
from gait_voxelize import create_SUSTech, find_minimum_cluster

name = 'SUSTech1K-Released-voxel.20comp2'
frame_num = 20
box_size = [1.25,1.25,2]
res = 0.03125
dilution=1
noise=0
compress=0
tfusion=False

def voxelize(data_root,
        frame_num,
        box_size,
        res,
        dilution=1,
        noise=0,
        counter=1,
        compress=False,
        tfusion=False):
    split_list = ['nt', 'cr', 'bg', 'ub', 'oc', 'cl', 'uf', '01-nm', '00-nm']
    view_list = ['000', '045', '090', '135', '180', '225', '270', '315', '000-far', '090-near', '180-far', '270-far']
    if os.path.exists(data_root):
        print('Removing old data ...')
        os.system('rm -rvf {}'.format(data_root))
        os.system('mkdir {}'.format(data_root))
    else:
        os.system('mkdir {}'.format(data_root))

    for view in view_list:
        os.system('mkdir {}/{}'.format(data_root, view))
        for split in split_list:
            print('Processing {} {}...'.format(split, view))
            create_SUSTech(data_root, frame_num, box_size, res, split, view, dilution, noise, counter, compress, tfusion)

def get_info(frame):
    min_x = np.min(frame[:,0])
    max_x = np.max(frame[:,0])
    min_y = np.min(frame[:,1])
    max_y = np.max(frame[:,1])
    min_z = np.min(frame[:,2])
    max_z = np.max(frame[:,2])
    point_num = len(frame)
    return [max_x-min_x, max_y-min_y, max_z-min_z, point_num]

def briefing():
    sub_list = ['delta_x','delta_y','delta_z','num_p','delta_fx','delta_fy','delta_fz','num_fp']
    src = '/home/sx-zhang/work/GaitCloud-master/dataset/SUSTech1K/SUSTech1K-Released-pkl'
    SUSTech1K_info = [] #[T, 2, 4]
    frame_num = []
    frames = 0
    frame_now = 0
    p_num_ex = 500
    for split in split_list:
        for view in view_list:
            print('processing {} {}...'.format(split, view))
            sample_list = sorted(os.listdir(src))
            for sample in sample_list:
                data_root = os.path.join(src, sample, '{}/{}/00-{}-LiDAR-PCDs.pkl'.format(split, view, view))
                try:
                    with open(data_root, 'rb') as f:
                        data = pickle.load(f)
                        for iframe in range(len(data)):
                            frame = data[iframe]
                            if not isinstance(frame, np.ndarray):
                                frame = np.asarray(frame)
                            try:
                                frame_info = get_info(frame)
                                cluster = DBSCAN(eps=0.1, min_samples=4).fit(frame)
                                labels = cluster.labels_
                                frame = frame[labels!=-1]
                                p_num = len(frame)
                                filtered_info = get_info(frame)
                                if (p_num > 400) or (abs(p_num - p_num_ex) < 200):
                                    SUSTech1K_info.append([frame_info, filtered_info])
                                    frames += 1
                                    p_num_ex = p_num
                                else:
                                    if frame_now == 0:
                                        frame_now = frames
                                        frames = 0
                                        print('{} {} {} frame {} incomplete gait'.format(split, view, sample, iframe))
                            except ValueError as e:
                                if frame_now == 0:
                                    frame_now = frames
                                    frames = 0
                                print('{} {} sample {} frame {}: {}'.format(split, view, sample, iframe, e))
                        if frame_now == 0:
                            frame_now = frames
                        #print(frame_now)
                        frame_num.append(frame_now)
                        frame_now = 0
                        frames = 0
                except FileNotFoundError:
                    print('{} not exist'.format(data_root))

    SUSTech1K_info = np.array(SUSTech1K_info)
    frame_num = np.array(frame_num)
    target = {}
    count = {}
    for i, subject in enumerate(sub_list):
        target[subject], count[subject] = ordering(SUSTech1K_info, i//4, i%4)

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    for i, subject in enumerate(sub_list):
        axes[i%4, i//4].plot(target[subject], count[subject], label=subject)
        axes[i%4, i//4].legend()

    sample_length, counts = np.unique(frame_num, return_counts=True)
    index = np.argsort(sample_length)
    sample_length = sample_length[index]
    counts = counts[index]
    fig = plt.figure(figsize=(10,5))
    plt.plot(sample_length, counts, label='sample_length')
    fig.legend()
    plt.show()


def ordering(info, pos1, pos2):
    subject, count = np.unique(np.round(info[:,pos1,pos2], 1), return_counts=True)
    index = np.argsort(subject)
    subject = subject[index]
    count = count[index]
    return subject, count


if __name__ == '__main__':
    #log_file = open('briefing.log', 'w')
    #sys.stdout = log_file
    #sys.stderr = log_file
    #briefing()
    #log_file.close()
    #for noise in [0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095]:
    #    voxelize('/home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel.20dev{}'.format(noise), frame_num=20, dilution=1, noise=noise)
    voxelize(name=name, frame_num=frame_num, box_size=box_size, res = res, dilution=dilution, noise=noise, compress=compress, tfusion=tfusion)
