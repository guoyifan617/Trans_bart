#encoding: utf-8

import torch
from tqdm import tqdm

bsize, nquery, seql, nhead, isize = 128, 256, 256, 8, 64
nwarm, ntest = 2, 10

device = torch.device("cuda", 0)

def mm(a, b):

	return a.transpose(1, 2).matmul(b.permute(0, 2, 3, 1))

def einsum_mm(a, b):

	return torch.einsum("abcd,aecd->acbe", a, b)

def mv(a, b):

	return a.mm(b.unsqueeze(-1)).squeeze(-1)

def einsum_mv(a, b):

	return torch.einsum("ab,b->a", a, b)

a = torch.randn(bsize, nquery, nhead, isize, device=device, requires_grad=True)
b = torch.randn(bsize, seql, nhead, isize, device=device, requires_grad=True)

for i in tqdm(range(nwarm)):
	_ = mm(a, b).sum()
	_.backward()
	_ = einsum_mm(a, b).sum()
	_.backward()

for i in tqdm(range(ntest)):
	_ = mm(a, b).sum()
	_.backward()
for i in tqdm(range(ntest)):
	_ = einsum_mm(a, b).sum()
	_.backward()

c = torch.randn(bsize * nhead * isize, nquery, device=device, requires_grad=True)
d = torch.randn(nquery, device=device, requires_grad=True)

for i in tqdm(range(nwarm)):
	_ = mv(c, d).sum()
	_.backward()
	_ = einsum_mv(c, d).sum()
	_.backward()

for i in tqdm(range(ntest)):
	_ = mv(c, d).sum()
	_.backward()
for i in tqdm(range(ntest)):
	_ = einsum_mv(c, d).sum()
	_.backward()
