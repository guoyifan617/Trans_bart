#encoding: utf-8

import torch
from math import ceil
from random import randint

from utils.torch.comp import mask_tensor_type

def mask_token(inpute, p, mask_id):

	return inpute.masked_fill_(inpute.new_full(inpute.size(), p, dtype=torch.float, device=inpute.device).bernoulli().to(mask_tensor_type, non_blocking=True), mask_id)

def mask_rand_token(inpute, p, start_id, end_id):

	_m = inpute.new_full(inpute.size(), p, dtype=torch.float, device=inpute.device).bernoulli().to(mask_tensor_type, non_blocking=True)

	return inpute.masked_scatter_(_m, torch.randint(start_id, end_id, (_m.int().sum().item(),), dtype=inpute.dtype, device=inpute.device))

def get_sind(seql, p, maxv=None):

	_len = max(2, ceil(seql * p))
	sind = randint(0, max(0, seql - _len))
	if maxv is not None:
		sind = min(maxv, sind)

	return sind, _len

def update_p(p_mask, p_rand):

	return p_mask / (1.0 - p_rand), p_rand
