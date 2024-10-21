#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.my_resnet import BasicBlock, ResNet
#from pytorch_metric_learning import losses, miners
import numpy as np

class ResTemp(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        self.feat_buff = []
        self.sample_L = args.frame_size

        self.encoder = ResNet9()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.HPP = HorizontalPyramidPooling()
        self.TempMerge = SetLSTM(hidden_size=512, feat_size=512, set_length=3)
        self.TempDense = TemporalPool()
        self.mergeloss = LossAggregator()

    def forward(self, x, label=None, training=True, **kwargs):
        '''
        input   : [n, t, h, w, l]
        output  : embedding[n, c, p]
        '''
        in_shape = x.shape
        #ResNet9
        #feat = self.encoder(x.view(-1, *in_shape[2:])) #(n, t, h, w, l) -> (n*t, c, h, w, l)
        for iframe in range(self.sample_L):
            if iframe == 0:
                self.feat_buff.append(self.encoder(x[:,iframe,...]))
                feat_shape = self.feat_buff[0].shape
            else:
                with torch.no_grad():
                    self.feat_buff.append(self.encoder(x[:,iframe,...]))
        feat = torch.stack(self.feat_buff, dim=1).view(-1, *feat_shape[1:]).contiguous()
        self.feat_buff.clear()
        
        #Horizontal Pyramid Pooling
        feat = self.HPP(feat) #(n*t, c, h, w, l) -> (n*t, c, p)
        
        #Temporal merging
        feat = self.TempMerge(feat, in_shape[0], in_shape[1]) #(n*t, c_in, p) -> (n, t_merge, p, h)

        #dense
        embed_tp = self.TempDense(feat) #(n, t_merge, p, h) -> (n, c_out, p)

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
    def __init__(self, maxpool=False):
        self.maxpool_flag = maxpool
        block = BasicBlock
        channels = [64,128,256,512]
        layers = [1,1,1,1]
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
        '''
            x       : [n*t, h, w, l]
            out_feat: [n*t, c, h, w, l]
        '''
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
           x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out_feat = self.layer4(x)
        return out_feat

class SetLSTM(nn.Module): #get dynamic feature within set_length frames
    def __init__(self, hidden_size, feat_size, set_length=3):
        super().__init__()
        self.lstm = nn.LSTM(feat_size, hidden_size, batch_first=True)
        self.flatten = nn.Flatten()
        self.feat_buff = []
        self.set_length, self.hidden_size = set_length, hidden_size

    def forward(self, x, n, t):
        '''
            x   : [n*t, c, p]
            out : [n, t_merge, p, c]
        '''
        #seq_x = self.flatten(x).contiguous().view(n, t, -1) #(n, t, p)
        #_, (temp_feat, _) = self.lstm(torch.cat([seq_x[:,st:st+self.set_length,:] for st in range(t-self.set_length+1)])) #c_out*(n, set_l, p) -> (1, t_merge*n, p)
        seq_x = x.view(n, t, *x.shape[1:]) #(n, t, c, p)
        batched_seq = torch.cat([seq_x[:,st:st+self.set_length,:,:] for st in range(t-self.set_length+1)]).permute(0, 3, 1, 2).contiguous() #(t_merge*n, p, set_l, c)
        _, parts_num, set_l, feat_size = batched_seq.shape
        _, (temp_feat, _) = self.lstm(batched_seq.view(-1, set_l, feat_size)) #(t_merge*n*, p, set_l, c) -> (1, t_merge*n*p, h)
        return torch.transpose(temp_feat[0].view(t-self.set_length+1, n, parts_num, temp_feat.shape[-1]), 0, 1) #(n, t_merge, p, h)

class TemporalPool(nn.Module): #take means on temporal dimension with fc
    def __init__(self, parts_num=16, in_channels=512, out_channels=256, norm=False):
        super(TemporalPool, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        '''
            x   : [n, t_merge, p, c_in]
            out : [n, c_out, p]
        '''
        x = torch.mean(x, dim=1) #(n, t_merge, p, c_in) -> (n, p, c_in)
        x = x.permute(1, 0, 2).contiguous() #(p, n, c_in)
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin) #(p, n, c_out)
        return out.permute(1, 2, 0).contiguous()

class HorizontalPyramidPooling():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=[16]):
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n*t, c, h, w, l]
            ret: [n*t, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

#2 dim
#class SeparateBNNecks(nn.Module):
#    def __init__(self, in_channels=256, class_num=250):
#        super(SeparateBNNecks, self).__init__()
#        self.class_num = class_num
#        self.bn1d = nn.BatchNorm1d(in_channels)
#        self.dense = nn.Linear(in_channels, class_num)
#
#    def forward(self, x):
#        """
#            x: [n, p]
#        """
#        feature = self.bn1d(x)
#        logits = self.dense(feature)
#        return feature.contiguous(), logits.contiguous()

#3 dim
class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num=16, in_channels=256, class_num=250, norm=True, parallel_BN1d=True):
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
    def __init__(self):
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
        TP_loss = self.TPloss(TP_feat, labels)
        CE_loss = self.CEloss(CE_feat*self.scale, labels.unsqueeze(1).repeat(1, d).long())
        loss_sum = TP_loss.mean()*self.TP_weight + CE_loss.mean()*self.CE_weight

        #print('TP_loss: {}/CE_loss: {}'.format(TP_loss, CE_loss))
        mined_triplets = self.miner.num_triplets
        return [loss_sum,TP_loss,CE_loss], mined_triplets
