#!/usr/bin/env python
import torch
import time
import sys
from ptflops import get_model_complexity_info
from torch.cuda.amp import autocast

sys.path.append('./')
from util import config
import models

#cfg = 'config/SUSTech1k/SUSTech1k_LidarGait_repro.yaml'
#cfg = 'config/SUSTech1k/SUSTech1k_GaitCloud_repro.yaml'
#cfg = 'config/SUSTech1k/SUSTech1k_TestAttention_repro.yaml'
#cfg = 'config/SUSTech1k/SUSTech1k_GaitCloud_repro.yaml'
cfg = 'config/SUSTech1k/SUSTech1k_LidarGait3D_repro.yaml'

#in_size = (1,20,1,64,64,)
#in_size = (1,20,3,64,64,)
#in_size = (1,64,40,40)
in_size = (1,20,64,40,40)
#add_size = (1,20,64,40,2)
target_num = 250
device = 'cuda'

dum_in = torch.randn(in_size, dtype=torch.float ,device=device)
dum_labels = torch.randn((1,), dtype=torch.float, device=device)
if 'add_size' in globals():
    dum_addons = torch.rand(add_size, dtype=torch.float, device=device)

args = config.load_cfg_from_cfg_file(cfg)
args.target = list(range(target_num))
Model = getattr(models, args.structure)
model = Model(args)
model.eval().to(device)
model.to(device)

#params
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print('Model parameters: ', params)

#forwarding time
st = time.time()
for loops in range(1000):
    with autocast(dtype=torch.bfloat16):
        if 'add_size' in globals():
            _ = model(dum_in, dum_labels, training=False, addons=dum_addons)
        else:
            _ = model(dum_in, dum_labels, training=False)
proc_time = time.time()-st
print('Average forwarding time:', proc_time/1000)


def GCplus_cons(res):
    return {'x':torch.randn(res), 'addons':torch.randn(1,20,64,40,2)}

model.to('cpu')
if 'add_size' in globals():
    result = get_model_complexity_info(model, in_size, as_strings=False, print_per_layer_stat=False, input_constructor=GCplus_cons, verbose=False)
else:
    result = get_model_complexity_info(model, in_size[1:], as_strings=True, print_per_layer_stat=False, verbose=False)
print(result)
