import torch
import torch.nn as nn
import time
from models.module import ResBlock, LE_2D, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator, HPP, FlatIn

class Gaitcloud2D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layerencoder = LE_2D(in_channel=args.feature, channels=[16,64], resblock=True)
        self.enc_1 = ResBlock([64,64], sample_dim=2)
        self.enc_2 = ResBlock([64,128], stride=2, sample_dim=2)
        self.enc_3 = ResBlock([128,256], stride=2, sample_dim=2)
        self.enc_4 = ResBlock([256,512], sample_dim=2)
        
        self.HPP = ConvHPP(bin_num=[16], channel=512, win_size=[1, 10])
        self.FCs = SeparateFCs(in_channels=512)
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)
        self.embeddings = {}

    def forward(self, x, labels=None, training=False, addons=None, visual=False, **kwargs): #input[n, t, z, x]
        n, t, h, w = x.shape
        x = x.view(n*t, h, w).contiguous().unsqueeze(1)
        #x = addons.transpose(1,-1)

        x = self.layerencoder(x)

        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        out = self.enc_4(x)
        if visual:
            self.embeddings['encoder_out'] = out.detach().to(torch.float).to('cpu')
        _, c, h, l = out.shape
        out = torch.max(out.view(n, t, c, h, l), dim=1)[0]

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
