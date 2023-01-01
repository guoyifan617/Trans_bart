#encoding: utf-8

# usage: python attn_time.py data.h5 $iter_repeat

import sys
import torch
from math import inf, sqrt
from time import time
from torch import nn

from utils.h5serial import h5File
from utils.torch.comp import torch_inference_mode
from utils.tqdm import tqdm

from cnfg.ihyp import tqdm_mininterval
from cnfg.vocab.base import pad_id

device = torch.device("cuda:7")
isize, nheads, adim = 512, 8, 64

num_iter = int(sys.argv[2])

with h5File(sys.argv[1], "r") as td:
	ntest = td["ndata"][()].item()
	src_grp, tgt_grp = td["src"], td["tgt"]
	tl = []
	with torch_inference_mode():
		for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
			seq_batch = torch.from_numpy(src_grp[str(i)][()])
			seq_o = torch.from_numpy(tgt_grp[str(i)][()])
			src_size = list(seq_batch.size())
			src_size.append(isize)
			tgt_size = list(seq_o.size())
			tgt_size[-1] -= 1
			tgt_size.append(isize)
			tl.append((torch.randn(src_size).to(device, non_blocking=True), torch.randn(tgt_size).to(device, non_blocking=True), seq_batch.eq(pad_id).to(device, non_blocking=True),))

query_adaptor = nn.Linear(isize, isize, bias=False)
kv_adaptor = nn.Linear(isize, isize * 2, bias=False)
outer = nn.Linear(isize, isize, bias=False)
normer = nn.Softmax(dim=-1)
drop = nn.Dropout(0.1, inplace=False)
for m in [query_adaptor, kv_adaptor, outer, normer, drop]:
	m.eval()
	m.to(device, non_blocking=True)

# do some warm up
for iK, iQ, mask in tl:
	bsize, nquery = iQ.size()[:2]
	seql = iK.size(1)
	real_iQ = query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
	real_iK, real_iV = kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
	real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
	scores = real_iQ.matmul(real_iK)
	scores = scores / sqrt(adim)
	if mask is not None:
		scores.masked_fill_(mask.unsqueeze(1).unsqueeze(1), -inf)
	scores = normer(scores)
	if drop is not None:
		scores = drop(scores)
	oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()
	output = outer(oMA.view(bsize, nquery, isize))

start = time()
for i in range(num_iter):
	for _, iQ, _ in tl:
		bsize, nquery = iQ.size()[:2]
		real_iQ = query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
cost = time() - start
print("Query transform of %d iter costs %f s" % (num_iter, cost,))

start = time()
for i in range(num_iter):
	for iK, _, _ in tl:
		bsize, seql = iK.size()[:2]
		real_iK, real_iV = kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
		real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
cost = time() - start
print("Key/Value transform of %d iter costs %f s" % (num_iter, cost,))

attn_inputs = []
for iK, iQ, mask in tl:
	bsize, nquery = iQ.size()[:2]
	seql = iK.size(1)
	real_iQ = query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
	real_iK, real_iV = kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
	real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
	attn_inputs.append((real_iQ, real_iK, real_iV, mask.unsqueeze(1)))

start = time()
for i in range(num_iter):
	for real_iQ, real_iK, real_iV, mask in attn_inputs:
		scores = real_iQ.matmul(real_iK)
cost = time() - start
print("Attention score of %d iter costs %f s" % (num_iter, cost,))

scores_input = []
for real_iQ, real_iK, real_iV, mask in attn_inputs:
	scores = real_iQ.matmul(real_iK)
	scores_input.append((scores, mask,))

start = time()
for i in range(num_iter):
	for score, mask in scores_input:
		scores = score / sqrt(adim)
		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf)
		scores = normer(scores)
		if drop is not None:
			scores = drop(scores)
cost = time() - start
print("Normalizing scores of %d iter costs %f s" % (num_iter, cost,))

matmul_inputs = []
for real_iQ, real_iK, real_iV, mask in attn_inputs:
	scores = real_iQ.matmul(real_iK)
	scores = scores / sqrt(adim)
	if mask is not None:
		scores.masked_fill_(mask.unsqueeze(1), -inf)
		scores = normer(scores)
		if drop is not None:
			scores = drop(scores)
	matmul_inputs.append((scores, real_iV,))

start = time()
for i in range(num_iter):
	for scores, real_iV in matmul_inputs:
		output = scores.matmul(real_iV).transpose(1, 2).contiguous()
cost = time() - start
print("Attending values of %d iter costs %f s" % (num_iter, cost,))

outer_inputs = []
for scores, real_iV in matmul_inputs:
	output = scores.matmul(real_iV).transpose(1, 2).contiguous()
	outer_inputs.append((output, scores.size(0), scores.size(-2),))

start = time()
for i in range(num_iter):
	for oMA, bsize, nquery in outer_inputs:
		output = outer(oMA.view(bsize, nquery, isize))
cost = time() - start
print("Output transform of %d iter costs %f s" % (num_iter, cost,))

print("For retrival attention")

start = time()
for i in range(num_iter):
	for score, mask in scores_input:
		score.masked_fill_(mask.unsqueeze(1), -inf)
cost = time() - start
print("Normalizing scores of %d iter costs %f s" % (num_iter, cost,))

retr_attn_matmul_inputs = []
for scores, real_iV in matmul_inputs:
	bsize, _, nquery, seql = scores.size()
	isize = real_iV.size(-1)
	retr_attn_matmul_inputs.append((bsize * _, nquery, seql, isize, scores.contiguous().view(bsize * _, nquery, seql), real_iV.contiguous().view(bsize * _, seql, isize),))

start = time()
for i in range(num_iter):
	for bsize, nquery, seql, isize, scores, real_iV in retr_attn_matmul_inputs:
		_ind = scores.argmax(-1)
		_ind += torch.arange(0, bsize * seql, seql, dtype=_ind.dtype, device=_ind.device).view(bsize, 1)
		rs = real_iV.view(bsize * seql, isize).index_select(0, _ind.view(bsize * nquery)).view(bsize, nquery, isize)
cost = time() - start
print("Attending values of %d iter costs %f s" % (num_iter, cost,))

start = time()
for i in range(num_iter):
	for bsize, nquery, seql, isize, scores, real_iV in retr_attn_matmul_inputs:
		_ind = scores.argmax(-1)
cost = time() - start
print("argmax of %d iter costs %f s" % (num_iter, cost,))

ind_inputs = []
for bsize, nquery, seql, isize, scores, real_iV in retr_attn_matmul_inputs:
	_ind = scores.argmax(-1)
	ind_inputs.append((bsize, seql, nquery, _ind,))

start = time()
for i in range(num_iter):
	for bsize, seql, nquery, ind in ind_inputs:
		ind += torch.arange(0, bsize * seql, seql, dtype=ind.dtype, device=ind.device).view(bsize, 1)
		ind = ind.view(bsize * nquery)
cost = time() - start
print("Update indexing of %d iter costs %f s" % (num_iter, cost,))

retrival_inputs = []
for bsize, nquery, seql, isize, scores, real_iV in retr_attn_matmul_inputs:
	_ind = scores.argmax(-1)
	_ind += torch.arange(0, bsize * seql, seql, dtype=_ind.dtype, device=_ind.device).view(bsize, 1)
	retrival_inputs.append((bsize, seql, nquery, isize, _ind.view(bsize * nquery), real_iV,))

start = time()
for i in range(num_iter):
	for bsize, seql, nquery, isize, ind, real_iV in retrival_inputs:
		rs = real_iV.view(bsize * seql, isize).index_select(0, ind).view(bsize, nquery, isize)
cost = time() - start
print("Retrieving values of %d iter costs %f s" % (num_iter, cost,))
