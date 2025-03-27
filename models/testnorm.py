import torch
import torch.nn as nn
import time
from models.module import ResBlock, LayerCNN, GaitNorm, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator

class TestNorm(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        self.layerencoder = LayerCNN(channels=[16,64], resblock=True)
        self.enc_1 = GaitNorm(64)
        self.enc_2 = ResBlock([64,128])
        self.enc_3 = ResBlock([128,256])
        self.enc_4 = ResBlock([256,512], stride=1)
        
        self.HPP = ConvHPP()
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)
        self.embeddings = {}

    def forward(self, x, ref=None, labels=None, training=True, visual=False, **kwargs):
        #st = time.time()
        batch_size = x.shape[0]
        if training:
            x = torch.cat([x,ref]) #[n+ref, c, h, w, l]
        x = self.layerencoder(x)
        if visual:
            self.embeddings['layer_out'] = x.detach().to(torch.float).to('cpu')
        #layer_t = time.time()

        x, penalty = self.enc_1(x, batch_size, training=training)
        if visual:
            self.embeddings['res1_out'] = x.detach().to(torch.float).to('cpu')
        x = self.enc_2(x)
        if visual:
            self.embeddings['res2_out'] = x.detach().to(torch.float).to('cpu')
        x = self.enc_3(x)
        if visual:
            self.embeddings['res3_out'] = x.detach().to(torch.float).to('cpu')
        out = self.enc_4(x)
        if visual:
            self.embeddings['encoder_out'] = out.detach().to(torch.float).to('cpu')

        feat = self.HPP(out)
        #hpp_t = time.time()

        #dense
        embed_tp = self.FCs(feat)
        #embed_t = time.time()

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, labels, penalty)
            #loss_t = time.time()
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp
            #loss_t = time.time()
        out_t = time.time()
        #print('loss:{}, out:{}'.format(round(loss_t-embed_t, 4), round(out_t-loss_t, 4)))

        return output, self.embeddings
