#encoding: utf-8

import torch
from torch import nn

from parallel.base import DataParallelModel

tmod = nn.Sequential(nn.Linear(4,5),nn.Sigmoid(),nn.Linear(5,1,bias=False))
torch.cuda.set_device(0)
tmod.to("cuda:0", non_blocking=True)
cdevs = [0, 7]
tmod_p = DataParallelModel(tmod, cdevs, torch.device("cuda:0"), 0, True)

td = torch.randn(13, 4, requires_grad=True).to("cuda:0", non_blocking=True)

tmod(td).sum().backward()
for p in tmod.parameters():
		print(p.grad)
		p.grad = None

tmod_p(td).sum().backward()
tmod_p.collect_gradients()
for p in tmod.parameters():
		print(p.grad)
