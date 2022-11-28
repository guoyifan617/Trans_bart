#encoding: utf-8

import torch
from random import randint

# inpute: (bsize, nsent, seql)
def mask_rand_token(inpute, p_rand, p_sent, startid, endid):

	_isize = inpute.size()
	_m = inpute.new_full(_isize, p_rand, dtype=torch.float, device=inpute.device, requires_grad=False).bernoulli().byte()
	_m = _m & inpute.new_full(_isize[:2], p_sent, dtype=torch.float, device=inpute.device, requires_grad=False).bernoulli().byte().unsqueeze(-1)

	return inpute.masked_scatter_(_m, torch.randint(startid, endid, (_m.int().sum().item(),), dtype=inpute.dtype, device=inpute.device, requires_grad=False))
