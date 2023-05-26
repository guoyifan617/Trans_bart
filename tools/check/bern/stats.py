#encoding: utf-8

import sys

from transformer.Bern.NMT import NMT
from utils.base import report_parameters
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.prune import remove_maskq, report_prune_ratio

import cnfg.base as cnfg
from cnfg.ihyp import *

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

with h5File(cnfg.dev_data, "r") as td:
	nword = td["nword"][()].tolist()
	nwordi, nwordt = nword[0], nword[-1]

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)
mymodel.useBernMask(False)
mymodel = remove_maskq(mymodel)
mymodel = load_model_cpu(sys.argv[1], mymodel)
mymodel.apply(load_fixing)

print("Total parameter(s): %d" % (report_parameters(mymodel),))
for k, v in report_prune_ratio(mymodel).items():
	print("%s: %.2f" % (k, v * 100.0,))
