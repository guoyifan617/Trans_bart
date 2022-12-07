#encoding: utf-8

import torch
from random import randint

from utils.torch.comp import mask_tensor_type

# inpute: (bsize, nsent, seql)
def mask_rand_token(inpute, p_rand, p_sent, start_id, end_id):

	_isize = inpute.size()
	_m = inpute.new_full(_isize, p_rand, dtype=torch.float, device=inpute.device, requires_grad=False).bernoulli().to(mask_tensor_type, non_blocking=True)
	_m = _m & inpute.new_full(_isize[:2], p_sent, dtype=torch.float, device=inpute.device, requires_grad=False).bernoulli().to(mask_tensor_type, non_blocking=True).unsqueeze(-1)

	return inpute.masked_scatter_(_m, torch.randint(start_id, end_id, (_m.int().sum().item(),), dtype=inpute.dtype, device=inpute.device, requires_grad=False))
