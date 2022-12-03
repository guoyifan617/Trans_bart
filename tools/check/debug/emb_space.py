#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.optim import Adam

from utils.torch.comp import torch_inference_mode
from utils.tqdm import tqdm

vdim = 32
vsize = 65536
bsize = 8192
nacc = 12
nreport = 20 * nacc
nrun = 100000
bind = False
device = torch.device("cuda", 0)
nrun *= nacc

tmod = nn.Sequential(nn.Embedding(vsize, vdim), nn.Linear(vdim, vsize, bias=False))
with torch_inference_mode():
	for p in tmod.parameters():
		_r = (1.0 / sqrt(p.size(0)))
		p.data.uniform_(-_r, _r)
if bind:
	tmod[-1].weight = tmod[0].weight
lossf = nn.CrossEntropyLoss(reduction="sum")

if device is not None:
	tmod.to(device, non_blocking=True)
	lossf.to(device, non_blocking=True)
optm = Adam(tmod.parameters(), lr=1e-5, betas=(0.9, 0.98,), eps=1e-9, weight_decay=0, amsgrad=False)

num_cor = 0
_r_bsize = float(nreport * bsize) / 100.0

for i in tqdm(range(1, nrun + 1)):

	td = torch.randint(0, vsize, (bsize,), dtype=torch.long, device=device)
	out = tmod(td)
	loss_v = lossf(out, td)
	num_cor += out.argmax(-1).eq(td).int().sum().item()
	loss_v.backward()
	if i % nacc == 0:
		optm.step()
		optm.zero_grad(set_to_none=True)
	if i % nreport == 0:
		acc = float(num_cor) / _r_bsize
		print(acc)
		if acc == 100.0:
			print("solved")
			break
		num_cor = 0
