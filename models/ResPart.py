#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.my_resnet import BasicBlock, ResNet
#from pytorch_metric_learning import losses, miners
import numpy as np

class ResPart(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        self.encoder_l1 = ResNet9()
        self.encoder_l2 = ResNet9()
        self.encoder_l3 = ResNet9()
        self.encoder_l4 = ResNet9()
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.HPP = HorizontalPyramidPooling()
        self.mergeloss = LossAggregator()

    def forward(self, x, label=None, training=True, **kwargs):
        #ResNet9
        x = x.unsqueeze(1)
        _, _, h, w, l = x.shape
        #print(h,w,l)
        out_l1 = self.encoder_l1(x[:,:,0:int(h/4),:,:])
        out_l2 = self.encoder_l2(x[:,:,int(h/4):int(h/2),:,:])
        out_l3 = self.encoder_l3(x[:,:,int(h/2):int(h*3/4),:,:])
        out_l4 = self.encoder_l4(x[:,:,int(h*3/4):h,:,:])
        out = torch.stack([out_l1, out_l2, out_l3, out_l4]).mean(dim=0)
        #print('encoder out: ', out.shape)

        #Horizontal Pyramid Pooling
        feat = self.HPP(out)
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

class ResNet9(ResNet):
    def __init__(self, maxpool=False, channels=[64,128,256,512], layers = [1,1,1,1]):
        self.maxpool_flag = maxpool
        block = BasicBlock
        strides = [1,2,2,1]
        in_channels = 1

        super(ResNet9, self).__init__(block, layers)

        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm3d(self.inplanes)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2)

        self.conv1 = nn.Conv3d(in_channels, self.inplanes, 3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], dilate=False)

        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
           x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class HorizontalPyramidPooling():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=[4]):
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w, l]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

class SeparateFCs(nn.Module):
    def __init__(self, parts_num=4, in_channels=512, out_channels=256, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()

class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num=4, in_channels=256, class_num=250, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()

class LossAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.TPloss = losses.TripletMarginLoss()
        self.CEloss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.TP_weight = 1#nn.Parameter(torch.Tensor(1))
        self.CE_weight = 0.1#nn.Parameter(torch.Tensor(1))
        self.miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")

        self.scale = 16

    def forward(self, TP_feat, CE_feat, labels):
        TP_feat = TP_feat.view(TP_feat.shape[0], -1)
        d = CE_feat.shape[-1]
        
        indices_tuple = self.miner(TP_feat, labels)
        TP_loss = self.TPloss(TP_feat, labels, indices_tuple)
        CE_loss = self.CEloss(CE_feat*self.scale, labels.unsqueeze(1).repeat(1, d).long())
        loss_sum = TP_loss.mean()*self.TP_weight + CE_loss.mean()*self.CE_weight

        #print('TP_loss: {}/CE_loss: {}'.format(TP_loss, CE_loss))
        mined_triplets = self.miner.num_triplets
        return [loss_sum,TP_loss,CE_loss], mined_triplets
