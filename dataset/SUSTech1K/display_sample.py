#!/usr/bin/env python

import numpy as np
import rclpy
import pickle
import os
import time
import argparse
import struct
from sensor_msgs_py import point_cloud2
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

import sys
sys.path.append('/home/work/sx-zhang/GaitCloud-master')
sys.path.append('/home/work/sx-zhang/GaitCloud-master/util')
from util.pretreatment import assign_ID, frame_voxelize, frame_rotation, frame_dilution, frame_clean, frame_clean
from util.gait_voxelize import get_dir
from models.module import PCA_image

#sample = 371
display_fullscene = True
trace = True
frame_num = 20
ch_cali = [100, -0.01]
w = 40
l = 40
h = 64

def get_parser():
    parser = argparse.ArgumentParser(description='Path to numpy array')
    parser.add_argument('-p', '--path', type=str)
    args = parser.parse_args()
    return args.path

class VisualArray(Node):
    def __init__(self):
        super().__init__('visual_array')
        self.pub = self.create_publisher(PointCloud2, 'v_array', 10)
        self.field = [PointField(name='x', offset=0, datatype=7, count=1),
                PointField(name='y', offset=4, datatype=7, count=1),
                PointField(name='z', offset=8, datatype=7, count=1),
                PointField(name='rgb', offset=12, datatype=7, count=1)]
        self.points = []
        self.header = Header(frame_id = 'velodyne')

    def publish(self):
        pubdata = point_cloud2.create_cloud(self.header, self.field, self.points)
        self.pub.publish(pubdata)
        print('Published')

class FrameChecker(Node):
    def __init__(self, display_fullscene):
        super().__init__('frame_checker')
        self.pub = self.create_publisher(PointCloud2, 'frame_xyz', 10)
        if display_fullscene:
            self.field = [PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                            PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                            PointField(name = 'z', offset = 8, datatype = 7, count = 1)]
        else:
            self.field = [PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                            PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                            PointField(name = 'z', offset = 8, datatype = 7, count = 1),
                            PointField(name = 'intensity', offset = 12, datatype = 2, count = 1)]
        self.points = []
        self.header = Header(frame_id = 'velodyne')

    def publish(self, name=None):
        pubdata = point_cloud2.create_cloud(self.header, self.field, self.points)
        self.pub.publish(pubdata)
        print('publishing ', name, len(self.points))

def load_raw(data_root):
    with open(data_root, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

def save_repaired(data, data_root):    
    with open(data_root, 'wb') as f:
        pickle.dump(data, f)
    f.close()
                       
#raw sample located     at SUSTech1K-Released-pkl
def display_raw(data_root):
    global h, w, l, frame_num, display_fullscene, trace
    checker = FrameChecker(display_fullscene)
    if display_fullscene:
        data = load_raw(data_root)
        sample_buff = []
        for iframe in range(len(data)):
            #print(iframe)

            #data[iframe] = frame_clean(data[iframe])
            #data[iframe] = data[iframe][data[iframe][:,1]>1]
            #if iframe < 30:
            #    data[iframe] = data[iframe][data[iframe][:,1]<1.3]
            #else:
            #    data[iframe] = data[iframe][data[iframe][:,1]<2]
            #if iframe > 37:
            #    data[iframe] = data[iframe][data[iframe][:,0]>-10.5]
            #print(np.mean(data[iframe], 0))
            #if iframe > 50:
            #sample_buff.append(data[iframe])

            if iframe < 9999:
                points = data[iframe].tolist()
                if trace:
                    checker.points += points
                else:
                    checker.points = points
                checker.publish(iframe)
                time.sleep(0.1)
        if sample_buff != []:
            print('Sample repaired')
            save_repaired(sample_buff, data_root)
        else:
            print('Sample plot complete')
    else:
        with open(data_root, 'rb') as f:
            data = pickle.load(f)
        f.close()
        ch_cal = np.load('elevation_128.npy')
        voxel_sample = np.zeros((w, l, h))
        #alpha = np.deg2rad(225)
        state = 0
        offset = 0
        for iframe in range(len(data)):
            data[iframe] = frame_clean(data[iframe])
        for iframe in range(len(data)):
            st = time.time()
            data_frame, Id = assign_ID(data[iframe], ch_cal)
            #print(time.time()-st)
            #data_frame, Id, ground = frame_dilution(data_frame, Id, stride=1)
            ins = np.expand_dims((np.asarray(Id)*50)%255, axis=1)
            if state == 0:
                alpha = get_dir(data, iframe, min(iframe+20, len(data)-1), '000')
                ground = min([min(frame[:,2]) for frame in data[iframe:]])
                print(np.rad2deg(alpha))
                state = 1
            if len(data_frame) > 0:
                frame_rotated = frame_rotation(data_frame, alpha)
                voxel_data, _ = frame_voxelize(frame_rotated, ground)
                voxel_data = np.transpose(voxel_data[...,-1], (1,2,0))
                voxel_sample += voxel_data

                points = np.transpose(np.nonzero(voxel_data))/32
                points[:,0] = points[:,0]+offset/32
                if trace:
                    checker.points += np.append(points, [[1]]*len(points), axis=1).tolist()
                    offset += 16
                else:
                    checker.points = np.append(points, [[1]]*len(points), axis=1).tolist()
                time.sleep(0.05)
                checker.publish()
                time.sleep(0.05)

        #sample_points = []
        #for z in range(h):
        #    for x in range(w):
        #        for y in range(l):
        #            if not voxel_sample[z,x,y] == 0:
        #                sample_points.append([float(x/32), float(y/32), float(z/32), voxel_sample[z, x, y]])
        #checker.points = sample_points
        #checker.publish()
        #time.sleep(1)

#train sample located at SUSTech1K-Released-voxel
def display_train(data_root):
    checker = FrameChecker(False)
    _data = np.load(data_root, allow_pickle=True)
    if len(_data.shape) == 1:
        _data = _data[0]
    data = np.transpose(_data, (1,2,0))
    points = np.transpose(np.nonzero(data))
    checker.points = np.append(points/32, data[np.nonzero(data)][:,np.newaxis], axis=1).tolist()
    checker.publish(data_root)
    time.sleep(0.25)
    checker.publish(data_root)
    time.sleep(0.25)
    checker.points = []

def display_array(data_root):
    checker = VisualArray()
    _data = np.load(data_root)
    sh = 40
    if len(_data.shape) == 4:
        data = np.transpose(PCA_image(_data), (1,2,0,3))
        bg = data[4,4,4]
        #bg = [-100, -100, -100]
        print(bg)
        coors = np.transpose(np.nonzero(data.sum(axis=-1)))
        for p in coors:
            color = data[tuple(p)]
            if (abs(color[0]-bg[0]) < sh) and (abs(color[1]-bg[1]) < sh) and (abs(color[2]-bg[2]) < sh) or p[0]%36 < 4 or p[1]%36 < 4 or p[2]%60 < 4:
                continue
            else:
                rgb = struct.unpack('I', struct.pack('BBBB', *color.astype('uint'), 0))
                checker.points.append(list(p/32)+list(rgb))
    else:
        data = np.transpose(_data, (1,2,0))
        coors = np.transpose(np.nonzero(data))
        checker.points = np.append(coors/32, data[np.nonzero(data)][:,np.newaxis], axis=1).tolist()

    for i in range(5):
        checker.publish()

if __name__ == '__main__':
    rclpy.init(args=None)
    #data_root = 'SUSTech1K-Released-voxel.20/tmp/train/norm_gait.npy'
    #data_root = 'SUSTech1K-Released-voxel.20/090-near/01-bg_090-near_0035.npy'
    #data_root = 'SUSTech1K-Released-voxel.20/tmp/test/01-nm_135_0408.npy'
    #display_train(data_root)

    #data_root = 'SUSTech1K-Released-pkl/0408/01-nm/135/00-135-LiDAR-PCDs.pkl'
    #display_raw(data_root)

    #if display_fullscene:
    #    data_root = data_source+'SUSTech1K-Released-pkl/0747/01-ub/270-far/00-270-far-LiDAR-PCDs.pkl'
    #    display_raw(data_root)
    #else:
    #    view = '270-far'
    #    split = '01-ub' #'00-nm', '01-cr', '01-bg', '01-ub', '01-nm', '01-oc', '01-cl'
    #    sample_root = data_source+'SUSTech1K-Released-voxel.20/tmp/probe'
    #    data_root = data_source+'SUSTech1K-Released-pkl/{}/{}/{}/00-{}-LiDAR-PCDs.pkl'
    #    sample_list = os.listdir(sample_root)
    #    direction = []
    #    for sample in sample_list:
    #        if '_{}_'.format(view) in sample:
    #            display_train(os.path.join(sample_root, sample))
    #            #if split in sample:
    #            #    print('plotting: ', sample)
    #            #    true_view = display_raw(data_root.format(sample[-10:-6], split, view, view))
    #            #    direction.append([true_view,sample])
    #    for item in direction:
    #        print(item)

    data_root = get_parser()
    display_array(data_root)
