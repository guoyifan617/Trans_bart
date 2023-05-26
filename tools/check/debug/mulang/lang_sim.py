#encoding: utf-8

import sys
import torch
from math import acos, pi, sqrt

from transformer.MuLang.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_inference_mode

import cnfg.mulang as cnfg
from cnfg.ihyp import *

def lang_distance(lang1, lang2):

	cosin = (lang1 * lang2).norm(p=2) / lang1.norm(p=2) / lang2.norm(p=2)
	angle = acos(cosin.item()) * 180.0 / pi

	return angle

with h5File(cnfg.dev_data, "r") as td:
	ntest = td["ndata"][()].item()
	nword = td["nword"][()].tolist()
	nwordi, ntask, nwordt = nword[0], nword[1], nword[-1]

set_random_seed(cnfg.seed, False)

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, ntask=ntask, ngroup=cnfg.ngroup)
mymodel = load_model_cpu(sys.argv[1], mymodel)

with torch_inference_mode():
	rs = torch.cat([mymodel.enc.task_emb.weight * sqrt(cnfg.isize), mymodel.enc.group_weight.softmax(-2).view(ntask, -1), mymodel.enc.group_weight_flayer.softmax(-2).view(ntask, -1), mymodel.dec.group_weight.softmax(-2).view(ntask, -1), mymodel.dec.group_weight_fl_common.softmax(-2).view(ntask, -1), mymodel.dec.group_weight_layer.softmax(-2).view(ntask, -1)], dim=-1).tolist()

rsd = {}
vcbtask, _ = ldvocab(sys.argv[2], minf=False, omit_vsize=False, vanilla=True)
vcbtask = reverse_dict(vcbtask)

for i, rsu in enumerate(rs):
	rsd[vcbtask[i]] = rsu

ens = "\n".encode("utf-8")
with open(sys.argv[3], "wb") as f:
	f.write(repr(rsd).encode("utf-8"))
	f.write(ens)

rsd = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in rsd.items()}
distance = {}
for k, v in rsd.items():
	for l2k, l2v in rsd.items():
		if k != l2k:
			if k in distance.get(l2k, {}):
				rs = distance[l2k][k]
			else:
				rs = lang_distance(v, l2v)
			if k in distance:
				distance[k][l2k] = rs
			else:
				distance[k] = {l2k: rs}

doc_head = ",,".join([vcbtask[i] for i in range(ntask)] + [""])
doc_cont = []
for i in range(ntask):
	slang = vcbtask[i]
	tmp = {}
	d = distance[slang]
	doc_l = []
	for k, v in d.items():
		if v in tmp:
			tmp[v].append(k)
		else:
			tmp[v] = [k]
	_sv = list(tmp.keys())
	_sv.sort()
	for k in _sv:
		for _vu in sorted(tmp[k]):
			doc_l.append("%s,%f" % (_vu,k,))
	doc_cont.append(doc_l)
doc_cont = list(zip(*doc_cont))
with open(sys.argv[4], "wb") as f:
	f.write(doc_head.encode("utf-8"))
	f.write(ens)
	for doc_l in doc_cont:
		f.write(",".join(doc_l).encode("utf-8"))
		f.write(ens)
