import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module import ResBlock, FlatIn, ScaleFusionCNN, HPP, SeparateFCs, SeparateBNNecks, LossAggregator
import numpy as np

class TestEncoder(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        self.layerencoder = FlatIn(32)
        self.scalencoder = ScaleFusionCNN()
        self.encoder = nn.Sequential(
                ResBlock([64,128], stride=2),
                ResBlock([128,256], stride=2),
                ResBlock([256,512], stride=1))
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.HPP = HPP()
        self.mergeloss = LossAggregator()

    def forward(self, x, positions, labels=None, training=True):
        #input size [n, 64, 40, 40]
        x = self.layerencoder(x)
        x = self.scalencoder(x, positions.round())
        out = self.encoder(x)

        #Horizontal Pyramid Pooling
        feat = self.HPP(out)

        #dense
        embed_tp = self.FCs(feat)

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            loss, mined_triplets = self.mergeloss(embed_tp, embed_ce, labels)
            output = tuple([loss, mined_triplets, embed_tp])
        else:
            output = embed_tp

        return output
