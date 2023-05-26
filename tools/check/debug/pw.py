#encoding: utf-8

"""
	example usage:
	ln tools/check/checkpw.py .
	python checkpw.py expm/debug/checkpoint.h5 un-cache/tgt.vcb
"""

import sys

from transformer.NMT import NMT
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File, h5load

import cnfg.base as cnfg
from cnfg.ihyp import *

with h5File(cnfg.test_data, "r") as td:
	nwordi = td["nword"][()].tolist()[0]

vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize)

mymodel.load_state_dict(h5load(sys.argv[1]))

initmodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize)

print(initmodel.enc.pemb.w.data.equal(initmodel.dec.pemb.w.data))
print(initmodel.enc.pemb.w.data.equal(mymodel.enc.pemb.w.data))
print(initmodel.enc.pemb.w.data.equal(mymodel.dec.pemb.w.data))
