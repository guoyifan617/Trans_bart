#encoding: utf-8

import sys
from torch import nn

from modules.group.base import GroupLinear
from transformer.MuLang.NMT import NMT
from utils.h5serial import h5File
from utils.io import load_model_cpu, save_model

import cnfg.mulang as cnfg
from cnfg.ihyp import *

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()
	if isinstance(module, GroupLinear) and (module.bias is not None) and (module.bias.dim() == 2):
		module.bias = nn.Parameter(module.bias.unsqueeze(1))

with h5File(cnfg.dev_data, "r") as td:
	nword = td["nword"][()].tolist()
	nwordi, ntask, nwordt = nword[0], nword[1], nword[-1]

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, ntask=ntask, ngroup=cnfg.ngroup)

mymodel = load_model_cpu(sys.argv[1], mymodel)
mymodel.apply(load_fixing)

save_model(mymodel, sys.argv[-1], False)
