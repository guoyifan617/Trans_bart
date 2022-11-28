#encoding: utf-8

import torch
#from math import ceil

from cnfg.ihyp import ieps_default
from cnfg.vocab.base import pad_id

# vsize > num_cache_topk + num_cache_topk
def init_cache(vsize, num_cache_topk, p=0.5, num_topk=None):

	#_num_topk = ceil(num_cache_topk / 2.0) if num_topk is None else num_topk
	_ = vsize - num_cache_topk
	_b = torch.arange(num_cache_topk, dtype=torch.int)
	#_num_ext = num_cache_topk - _num_topk
	_p = torch.full((vsize, num_cache_topk,), ieps_default)#(1.0 - p - _num_ext * ieps_default) / (_num_topk - 1)
	#_p.narrow(-1, _num_topk, _num_ext).fill_(ieps_default)
	#_p.select(-1, 0).fill_(p)
	_p[pad_id].zero_()

	return torch.cat((torch.arange(_, dtype=torch.int).unsqueeze(-1) + _b, torch.arange(_, vsize, dtype=torch.int).unsqueeze(-1) - _b,), dim=0), _p
