import torch
import random
import os
import numpy as np
from torch.utils.data import Dataset

class GaitDummy(Dataset):
    def __init__(self, split, data_root, args, datalist):
        super().__init__()
        self.args = args
        dummy_num = 1000

        #load data
        self.data_idx = np.arange(dummy_num)
        print("Totally {} samples in {} set.".format(dummy_num, split))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):

        #sample
        data = np.random.randint(0, 20, size=(64,40,40))
        #label
        label = np.random.randint(0, 500)
        #misc
        position = np.random.randint(0, 10)
        name = 'name'

        return data, label, [position, name]
