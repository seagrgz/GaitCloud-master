import torch
import time
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp


#input size [15, num_of_point, 3]
class CNN(nn.Module):

    def __init__(self, args, in_size):
        super().__init__()
        self.dtype, self.num_processes, self.batch_size, self.out_size, self.use_metric = args.dtype, args.frame_size, args.batch_size, args.classes, args.use_metric
        if type(args.lines) == int:
            self.feat_size = args.lines*512
        else:
            self.feat_size = len(args.lines)*512

        if args.dtype == 'double':
            self.encoder = CNN(in_size, self.feat_size, self.out_size).double()
            #self.dense = Dense(self.feat_size, self.out_size).double()
        elif args.dtype == 'float32':
            self.encoder = CNN(in_size, self.feat_size, self.out_size).float()
            #self.dense = Dense(self.feat_size, self.out_size).float()
        elif args.dtype == 'float16':
            self.encoder = CNN(in_size, self.feat_size, self.out_size).half()
            #self.dense = Dense(self.feat_size, self.out_size).half()
        else:
            self.encoder = CNN(in_size, self.feat_size, self.out_size).type(torch.int8)
            #self.dense = Dense(self.feat_size, self.out_size).type(torch.int8)

        ###################
        ##multi
        #self.que = mp.Queue()
        #self.pool = []
        ###################

    def forward(self, batch):
        batch = torch.movedim(batch, -1, 2)
        self.batch_size = batch.size(0)
        #print(batch.size())

        if self.dtype == 'double':
            self.feature = torch.empty((self.num_processes, self.batch_size, self.feat_size), dtype=torch.double, device=torch.device('cuda'))
            #self.feature = torch.empty((self.num_processes, self.batch_size, self.feat_size), dtype=torch.double, device=torch.device('cuda'))
        elif self.dtype == 'float32':
            self.feature = torch.empty((self.num_processes, self.batch_size, self.out_size), dtype=torch.float, device=torch.device('cuda'))
            #self.feature = torch.empty((self.num_processes, self.batch_size, self.feat_size), dtype=torch.float, device=torch.device('cuda'))
        elif self.dtype == 'float16':
            self.feature = torch.empty((self.num_processes, self.batch_size, self.out_size), dtype=torch.half, device=torch.device('cuda'))
            #self.feature = torch.empty((self.num_processes, self.batch_size, self.feat_size), dtype=torch.half, device=torch.device('cuda'))
        else:
            self.feature = torch.empty((self.num_processes, self.batch_size, self.out_size), dtype=torch.int8, device=torch.device('cuda'))
            #self.feature = torch.empty((self.num_processes, self.batch_size, self.feat_size), dtype=torch.int8, device=torch.device('cuda'))

        #loop
        for i in range(self.num_processes):
            self.feature[i] = self.encoder(batch[:,i], self.use_metric)
        scores = torch.sum(self.feature, dim=0)/self.num_processes
        #scores = self.dense(torch.sum(self.feature, dim=0)/self.num_processes)
        return scores

#class Dense(nn.Module):
#    def __init__(self, feat_size, out_size):
#        super().__init__()
#        self.drop = nn.Dropout(0.5)
#        self.relu = nn.ReLU()
#        self.linear1 = nn.Linear(feat_size, 512)
#        self.linear2 = nn.Linear(512, 100)
#        self.linear3 = nn.Linear(100, out_size)
#        self.softmax = nn.Softmax(dim = 1)
#
#        self.dense = nn.Sequential(self.linear1, self.relu, self.drop,
#                self.linear2, self.relu, self.drop,
#                self.linear3, self.softmax)
#
#    def forward(self, x):
#        output = self.dense(x)
#        return output

class CNN(nn.Module):
    def __init__(self, in_size, feat_size, out_size):
        super().__init__()
        self.trans = nn.Sequential(nn.ReLU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.5))

        self.enc_1 = nn.Sequential(
                nn.Conv2d(in_size, 16, (3, 5), padding = 'same'),
                nn.BatchNorm2d(16),
                self.trans
                )
        self.enc_2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding = 'same'),
                nn.BatchNorm2d(32),
                self.trans
                )
        self.enc_3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding = 'same'),
                nn.BatchNorm2d(64),
                self.trans)
        self.enc_4 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding = 'same'),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d((1, 2))
                )
        self.flat = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(feat_size)
                )
        #######################
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(feat_size, 512)
        self.linear2 = nn.Linear(512, 100)
        self.linear3 = nn.Linear(100, out_size)
        self.softmax = nn.Softmax(dim = 1)

        self.dense = nn.Sequential(self.linear1, self.relu, self.drop,
                self.linear2, self.relu, self.drop,
                self.linear3, self.softmax)
        #######################

    def forward(self, x, use_metric):
        #print(x.size())
        out_1 = self.enc_1(x)
        out_2 = self.enc_2(out_1)
        out_3 = self.enc_3(out_2)
        out_4 = self.enc_4(out_3)
        output = self.flat(out_4)
        ######################
        if not use_metric:
            output = self.dense(output)
        ######################
        return output

def CNN_cls_repro(args, in_size):
    model = GaitNet(args, in_size)
    return model
