import torch
import torch.nn as nn
from models.module import ResNet9_3D, LayerCNN, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator
import numpy as np

class TestCHPP(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        self.layerencoder = LayerCNN(channels=[64,64])
        self.encoder = ResNet9_3D()
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.CHPP = ConvHPP()
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)

    def forward(self, x, label=None, training=True, positions=None):
        #ResNet9
        out = self.encoder(x)
        #print('encoder out: ', out.shape)

        #Horizontal Pyramid Pooling
        feat = self.CHPP(out)
        #print('HPP out: ', feat.shape)

        #dense
        embed_tp = self.FCs(feat)
        #print('embeddings: ', embed_tp.shape)

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, label)
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp

        return output
