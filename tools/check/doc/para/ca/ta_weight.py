#encoding: utf-8

import sys

from transformer.Doc.Para.Gate.CANMT import NMT
from utils.h5serial import h5File
from utils.io import load_model_cpu

import cnfg.docpara as cnfg
from cnfg.ihyp import *

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

with h5File(sys.argv[2], "r") as td:
	nword = td["nword"][()].tolist()
	nwordi, nwordt = nword[0], nword[-1]

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, cnfg.num_prev_sent, cnfg.num_layer_context)

mymodel = load_model_cpu(sys.argv[1], mymodel)
mymodel.apply(load_fixing)

mymodel.eval()

rs = mymodel.enc.tattn_w.softmax(dim=0).tolist()
print(rs)
acc = 0.0
tmp = []
for rsu in reversed(rs):
	acc += rsu
	tmp.append(acc)
tmp.reverse()
acc = sum(tmp)
print([tmpu / acc for tmpu in tmp])
