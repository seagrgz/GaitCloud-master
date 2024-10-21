#!/usr/bin/env python

import numpy as np
import open3d as o3d
import random
import struct
import pickle
import time
import sys
import os

#import rclpy
#from sensor_msgs_py import point_cloud2
#from rclpy.node import Node
#from std_msgs.msg import Header
#from sensor_msgs.msg import PointCloud2, PointField
#
#class FrameChecker(Node):
#    def __init__(self):
#        super().__init__('frame_checker')
#        self.pub = self.create_publisher(PointCloud2, 'frame_xyz', 10)
#        self.field = [PointField(name = 'x', offset = 0, datatype = 7, count = 1),
#                        PointField(name = 'y', offset = 4, datatype = 7, count = 1),
#                        PointField(name = 'z', offset = 8, datatype = 7, count = 1),
#                        PointField(name = 'rgb', offset = 12, datatype = 6, count = 1)]
#        self.points = []
#        self.header = Header(frame_id = 'velodyne')
#
#    def publish(self):
#        pubdata = point_cloud2.create_cloud(self.header, self.field, self.points)
#        self.pub.publish(pubdata)
#
#def show_pkl(data_root):
#    rclpy.init(args=None)
#    checker = FrameChecker()
#    with open(data_root, 'rb') as f:
#        data = pickle.load(f)
#
#        #for iframe in range(len(data)):
#        #    checker.points = data[iframe]
#        #    time.sleep(0.5)
#        #    checker.publish(iframe)
#        #    time.sleep(0.5)
    
def show_pcd(data_root):
    pcd = o3d.io.read_point_cloud(data_root)
    data = np.asarray(pcd.points)
    coor_z = data[:,2]
    clusters = [[]]
    clusters_idx = [[]]
    for idx, point in enumerate(coor_z):
        if idx == 0:
            clusters[0].append(point)
            clusters_idx[0].append(idx)
        else:
            #print(clusters)
            clustered = False
            for clu in range(len(clusters)):
                clu_ave = sum(clusters[clu])/len(clusters[clu])
                if abs(point-clu_ave) < 0.027:
                    clusters[clu].append(point)
                    clusters_idx[clu].append(idx)
                    clustered = True
            if not clustered:
                clusters.append([point])
                clusters_idx.append([idx])
    print(len(clusters))
    return data, clusters_idx


if __name__ == '__main__':
    #data_root = '/home/sx-zhang/SUSTech1K/SUSTech1K-Released-pkl/0000/00-nm/000/00-000-LiDAR-PCDs.pkl'
    data_root = '/home/sx-zhang/SUSTech1K/SUSTech1K-Released-2023/0000/00-nm/000/PCDs/1657079013.900059000.pcd'

    #show_pkl(data_root)

    #rclpy.init(args=None)
    #checker = FrameChecker()
    data, cluster_id = show_pcd(data_root)
    np.save('/home/sx-zhang/testdata.npy', np.array([data, cluster_id], dtype=object), allow_pickle=True)
    print('transformation complete')
    #SUM = len(cluster_id)
    #for i, cluster in enumerate(cluster_id):
    #    for pos in cluster:
    #        co = int(i/SUM*255)
    #        rgb = struct.unpack('I', struct.pack('BBBB', co, co, co, 0))[0]
    #        checker.points.append([data[pos[0]]+[rgb]])
    #checker.publish()
