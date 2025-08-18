import torch
import torch.nn as nn
import time
from models.module import ResBlock, LayerCNN, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator, HPP, FlatIn

class Gaitcloud(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layerencoder = LayerCNN(in_channel=args.feature, channels=[16,64], resblock=True)
        #self.layerencoder = FlatIn(args.feature,64)
        self.enc_1 = ResBlock([64,64])
        self.enc_2 = ResBlock([64,128], stride=2)
        self.enc_3 = ResBlock([128,256], stride=2)
        self.enc_4 = ResBlock([256,512])
        
        self.HPP = ConvHPP(bin_num=[16], channel=512, win_size=[1, 10, 10])
        #self.HPP = HPP()
        self.FCs = SeparateFCs(in_channels=512)
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)
        self.embeddings = {}

    def forward(self, x, labels=None, training=False, addons=None, visual=False, **kwargs): #input[n, z, x, y] / [n, t, z, x, y]
        if len(x.shape) == 5:
            temp_in = True
            n, t, h, w, l = x.shape
            x = x.view(n*t, h, w, l).contiguous()
        else:
            temp_in = False

        #x = addons.transpose(1,-1) #for temp data only
        x = x.unsqueeze(1) #for spat data only
        x = self.layerencoder(x)

        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        out = self.enc_4(x)
        if visual:
            self.embeddings['encoder_out'] = out.detach().to(torch.float).to('cpu')

        if temp_in:
            _, c, h, w, l = out.shape
            out = torch.max(out.view(n, t, c, h, w, l), dim=1)[0]
        feat = self.HPP(out)

        #dense
        embed_tp = self.FCs(feat)

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, labels)
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp
        #out_t = time.time()

        return output, self.embeddings
