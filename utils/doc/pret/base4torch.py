#encoding: utf-8

from math import ceil, floor
from torch.nn import ReLU

from modules.act import GeLU
from modules.base import Linear, PositionwiseFF
from utils.base import clear_pad
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

def set_linear_bias(linear_m, enable_bias=enable_prev_ln_bias_default):

	rsm = Linear(linear_m.in_features, linear_m.out_features, bias=enable_bias)
	rsm.weight = linear_m.weight
	if enable_bias:
		if linear_m.bias is None:
			with torch_no_grad():
				rsm.bias.zero_()
		else:
			rsm.bias = linear_m.bias

	return rsm

def replace_act4ffn(modin, src_act=ReLU, tgt_act=GeLU):

	for _m in modin.modules():
		if isinstance(_m, PositionwiseFF) and isinstance(_m.net[1], src_act):
			_m.net[1] = tgt_act()

	return modin

def set_bias4ffn(modin, enable_bias=enable_prev_ln_bias_default):

	for _m in modin.modules():
		if isinstance(_m, PositionwiseFF):
			_m.net[2] = set_linear_bias(_m.net[2], enable_bias)

	return modin

def patch_pret_model_ffn(modin, src_act=ReLU, tgt_act=GeLU, enable_bias=enable_prev_ln_bias_default):

	for _m in modin.modules():
		if isinstance(_m, PositionwiseFF):
			if isinstance(_m.net[1], src_act):
				_m.net[1] = tgt_act()
			for i in range(len(_m.net) - 1, -1, -1):
				if isinstance(_m.net[i], Linear):
					_m.net[i] = set_linear_bias(_m.net[i], enable_bias)
					break

	return modin

def split_doc(seq_batch, seq_o, max_sent=24):

	nsent = seq_batch.size(1)
	lind = 0
	if nsent > max_sent:
		num_chunk = ceil(nsent / max_sent)
		num_sent_chunk = max(1, floor(nsent/num_chunk))
		while lind < nsent:
			_n_use = min(nsent - lind, num_sent_chunk)
			_seq_batch = clear_pad(seq_batch.narrow(1, lind, _n_use)).contiguous()
			_seqo = clear_pad(seq_o.narrow(1, lind, _n_use))
			lo = _seqo.size(-1) - 1
			yield _seq_batch, _seqo.narrow(-1, 0, lo).contiguous(), _seqo.narrow(-1, 1, lo).contiguous(), lind
			lind += _n_use
	else:
		lo = seq_o.size(-1) - 1
		yield seq_batch, seq_o.narrow(-1, 0, lo).contiguous(), seq_o.narrow(-1, 1, lo).contiguous(), 0
