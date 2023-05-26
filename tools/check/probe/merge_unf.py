#encoding: utf-8

import sys

from transformer.NMT import NMT
from transformer.Probe.UnfDecoder import Decoder
from utils.fmt.parser import parse_double_value_tuple
from utils.h5serial import h5File
from utils.init.base import init_model_params
from utils.io import load_model_cpu, save_model

import cnfg.unf as cnfg
from cnfg.ihyp import *

def handle(cnfg, srcmtf, unfdecf, rsf):

	with h5File(cnfg.dev_data, "r") as tdf:
		nwordi, nwordt = tdf["nword"][()].tolist()

	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)
	init_model_params(mymodel)
	mymodel = load_model_cpu(srcmtf, mymodel)
	_, dec_layer = parse_double_value_tuple(cnfg.nlayer)
	_tmpm_dec = Decoder(cnfg.isize, nwordt, dec_layer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, None, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes).nets# comment .nets for old code
	init_model_params(_tmpm_dec)
	_tmpm_dec = load_model_cpu(unfdecf, _tmpm_dec)
	mymodel.dec.nets = _tmpm_dec# add .nets for old code
	_tmpm_dec = None

	save_model(mymodel, rsf, False, None, h5args=h5zipargs)

if __name__ == "__main__":
	handle(cnfg, sys.argv[1], sys.argv[2], sys.argv[3])
