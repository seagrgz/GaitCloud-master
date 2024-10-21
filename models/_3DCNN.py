import torch
import time
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp


#input size [15, num_of_point, 3]
class _3DCNN(nn.Module):

    def __init__(self, args, in_size=1):
        super().__init__()
        self.out_size, self.use_metric = args.classes, args.use_metric

        self._3dcnn = CNN(in_size, self.out_size).double()

    def forward(self, batch, **kwargs):
        batch = torch.unsqueeze(batch, 1)
        #print(batch.size())

        scores = self._3dcnn(batch, self.use_metric)
        return scores

class CNN(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.trans1 = nn.Sequential(nn.ReLU(), nn.MaxPool3d((2, 2, 1)), nn.Dropout(0.5))
        self.trans2 = nn.Sequential(nn.ReLU(), nn.MaxPool3d((2, 2, 2)), nn.Dropout(0.5))
        self.enc_1 = nn.Sequential(
                nn.Conv3d(in_size, 64, (5, 5, 3)),
                nn.BatchNorm3d(64),
                self.trans2
                )
        self.enc_2 = nn.Sequential(
                nn.Conv3d(64, 128, 3),
                nn.BatchNorm3d(128),
                self.trans2
                )
        self.enc_3 = nn.Sequential(
                nn.Conv3d(128, 256, 3),
                nn.BatchNorm3d(256),
                self.trans2
                )
        self.enc_4 = nn.Sequential(
                nn.Conv3d(256, 512, 3),
                nn.BatchNorm3d(512),
                nn.ReLU()#,
                #nn.MaxPool3d((2, 2, 2))
                )
        self.flat = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(2048)
                )
        #full connect
        #######################
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(4096, 512)
        self.linear2 = nn.Linear(512, 100)
        self.linear3 = nn.Linear(100, out_size)
        self.softmax = nn.Softmax(dim = 1)

        self.dense = nn.Sequential(self.linear1, self.relu, self.drop,
                self.linear2, self.relu, self.drop,
                self.linear3, self.softmax)
        #######################

    def forward(self, x, use_metric):
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        output = self.flat(x)
        ######################
        if not use_metric:
            output = self.dense(output)
        ######################
        return output
