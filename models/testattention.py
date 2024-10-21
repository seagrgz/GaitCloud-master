import torch.nn as nn
import torch
import time
from models.module import ConvBlock, FlatIn, LayerCNN, ScaleFusionCNN, SymbolMHA, SymbolAttention, HPP, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator

class TestAttention(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        self.layerencoder = LayerCNN(channels=[16,64], resblock=True)
        #self.layerencoder = FlatIn(64)
        #self.scalencoder = ScaleFusionCNN(in_channel=64, out_channel=64)
        #self.attention = SymbolMHA(embed_dim=64)
        self.attention = SymbolAttention()
        self.encoder = nn.Sequential(
                ConvBlock([64,64], stride=1),
                ConvBlock([64,128]),
                ConvBlock([128,256]),
                ConvBlock([256,512], stride=1))
        #self.attention = SymbolMHA(embed_dim=512)
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.HPP = HPP()
        #self.HPP = ConvHPP()
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)

        self.embeddings = {}

    def forward(self, x, labels=None, training=True, positions=None, symbol=None):
        #st = time.time()
        self.embeddings['input'] = x.detach().to(torch.float).to('cpu')

        batch = torch.cat((x, symbol.unsqueeze(0))) #[n+1, h, w, l]
        batch = self.layerencoder(batch)
        self.embeddings['layer_out'] = batch[:-1].detach().to(torch.float).to('cpu')

        x = self.attention(batch[:-1], batch[-1:].expand(x.shape[0],-1,-1,-1,-1))
        self.embeddings['attention_out'] = x.detach().to(torch.float).to('cpu')

        #layer_t = time.time()
        #x = self.scalencoder(x, positions.round())
        #scale_t = time.time()
        out = self.encoder(x)
        self.embeddings['encoder_out'] = out.detach().to(torch.float).to('cpu')
        #encode_t = time.time()
        #with torch.no_grad():
        #symbol = self.encoder(symbol)
        #out = self.attention(x, symbol.expand(x.shape[0],-1,-1,-1,-1))

        #Horizontal Pyramid Pooling
        feat = self.HPP(out)
        self.embeddings['hpp_out'] = feat.detach().to(torch.float).to('cpu')
        #hpp_t = time.time()

        #dense
        embed_tp = self.FCs(feat)
        self.embeddings['dense_out'] = embed_tp.detach().to(torch.float).to('cpu')
        #embed_t = time.time()

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, labels)
            #loss_t = time.time()
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp
            #loss_t = time.time()
        out_t = time.time()
        #print('loss:{}, out:{}'.format(round(loss_t-embed_t, 4), round(out_t-loss_t, 4)))

        return output, self.embeddings
