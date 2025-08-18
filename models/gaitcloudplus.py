import torch.nn as nn
import torch
import time
import sys
sys.path.append('/home/work/sx-zhang/GaitCloud-master')
from models.module import ResBlock, FlatIn, LayerCNN, LE_2D, TemporalAttention, HPP, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator
from util import config

class GaitCloudplus(nn.Module):
    def __init__(self, args):
        self.with_temp = args.with_temp
        super().__init__()
        self.inlayer = LayerCNN(in_channel=args.feature, channels=[16,64], resblock=True)
        #self.inlayer = FlatIn(args.feature,16) #baseline
        self.encoder = nn.Sequential(
                #ResBlock([16,64], stride=1), #baseline
                ResBlock([64,64]),
                ResBlock([64,128], stride=2),
                ResBlock([128,256], stride=2),
                ResBlock([256,512]))

        #self.temp_inlayer = LE_2D(in_channel=args.temp_feature, channels=[16,64], resblock=True)
        self.temp_inlayer = LayerCNN(in_channel=args.temp_feature, channels=[16,64], resblock=True)
        #self.temp_inlayer = FlatIn(args.temp_feature, 16) #baseline
        self.temp_encoder = nn.Sequential(
                #ResBlock([16,64], stride=1), #baseline
                ResBlock([64,64], sample_dim=3),
                ResBlock([64,128], stride=2, sample_dim=3),
                ResBlock([128,256], stride=2, sample_dim=3),
                ResBlock([256,512], sample_dim=3))

        #self.attention = TemporalAttention(in_channel=512, out_channel=512, dk=512)
        self.attention = TemporalAttention()
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        if self.with_temp:
            self.HPP = ConvHPP(bin_num=[16], channel=512, win_size=[1, 10, 5])
        else:
            self.HPP = ConvHPP(bin_num=[16], channel=512, win_size=[1, 10, 10])
        #self.HPP = HPP() #baseline
        #self.HPP = self.to_vector #not used
        #self.dim_fc = nn.Sequential(
        #        nn.Linear(800,16),
        #        nn.BatchNorm1d(16),
        #        nn.ReLU(inplace=True),
        #        nn.Dropout())
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)

        self.embeddings = {}

    def spatial_branch(self, x):
        x = self.inlayer(x.unsqueeze(1))
        x = self.encoder(x)
        return x

    def temporal_branch(self, x):
        x = self.temp_inlayer(x.transpose(1,-1))
        x = self.temp_encoder(x)
        return x

    #def temporal_branch(self, x):
    #    n, t, h, w, l = x.shape
    #    x = torch.permute(x, (0,1,4,2,3))
    #    x = self.temp_inlayer(x.view(n*t,l,h,w).contiguous())
    #    x = self.temp_encoder(x)
    #    _, c, h, w = x.shape
    #    return torch.max(x.view(n, t, c, h, w), dim=1)[0]

    def to_vector(self, x):
        #return torch.mean(x, dim=(3,4))
        n, c = x.shape[:2]
        x = torch.flatten(x, start_dim=2).view(n*c,-1)
        x = self.dim_fc(x)
        return x.view(n, c, -1).contiguous()

    def forward(self, x, labels=None, training=False, addons=None, visual=False, **kwargs):

        #extract spatial features
        if visual:
            self.embeddings['spat_in'] = x.detach().to(torch.float).to('cpu')
        feat_spat = self.spatial_branch(x) #[n, c, h, w, l]
        n, c, h, w, l = feat_spat.shape

        if self.with_temp:
            #extract temporal features
            feat_temp = self.temporal_branch(addons) #[n, c, h, w, t]
            n_t, c_t, h_t, w_t, t = feat_temp.shape

            assert n == n_t
            assert c == c_t
            assert h == h_t
            assert w == w_t

            if visual:
                self.embeddings['spat_feat'] = feat_spat.detach().to(torch.float).to('cpu')
                self.embeddings['temp_feat'] = feat_temp.detach().to(torch.float).to('cpu')

            #attenrion fusion
            #print(feat_spat.shape, feat_temp.shape)
            feat_spat = feat_spat.transpose(1,2).reshape(n*h, c, w*l)
            feat_temp = feat_temp.transpose(1,2).reshape(n*h, c, w*t)
            #feat_temp = feat_temp.transpose(1,2).reshape(n*h, c, w)

            feat, weights = self.attention(feat_temp, feat_spat) #[n*h, c, w*t]
            feat = feat.view(n, h, -1, w, t)
            #feat = self.attention(feat_spat, feat_temp).view(n, h, -1, w, l) #[n*h, c, w*l]
            feat = torch.permute(feat, (0,2,1,3,4)).contiguous() #[n, c, h, w, t/l]

            #feat = self.attention(feat_temp, feat_spat).view(n, h, -1, w) #[n*h, c, w]
            #feat = torch.permute(feat, (0,2,1,3)).contiguous() #[n, c, h, w]
            self.embeddings['attention'] = weights.view(n,h,w*t,w,l).detach().to(torch.float).to('cpu')

        else:
            feat = feat_spat

        if visual:
            self.embeddings['attention_out'] = feat.detach().to(torch.float).to('cpu')
        #feat = torch.cat([feat_spat, feat_temp], dim=-1)

        #Horizontal Pyramid Pooling
        feat = self.HPP(feat)
        if visual:
            self.embeddings['hpp_out'] = feat.detach().to(torch.float).to('cpu')

        #dense
        embed_tp = self.FCs(feat)
        if visual:
            self.embeddings['dense_out'] = embed_tp.detach().to(torch.float).to('cpu')

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, labels)
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp

        return output, self.embeddings

if __name__ == '__main__':
    cfg_root = '/home/work/sx-zhang/GaitCloud-master/config/SUSTech1k/SUSTech1k_TestAttention_repro.yaml'
    cfg = config.load_cfg_from_cfg_file(cfg_root)
    cfg.target = [0]*250
    test_model = TestAttention(cfg)
    spat_in = torch.randn(8,64,40,40)
    temp_in = torch.randn(8,20,64,40,2)
    labels = torch.randint(250, (8,))
    (losses, mined_triplets, embeddings), _ = test_model(spat_in, labels=labels, training=True, addons=temp_in, visual=False)
    print('Test complete, output size: {}'.format(embeddings.shape))
