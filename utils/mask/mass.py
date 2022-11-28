#encoding: utf-8

import torch
from math import ceil
from random import randint

def mask_token(inpute, p, mask_id):

	return inpute.masked_fill_(inpute.new_full(inpute.size(), p, dtype=torch.float, device=inpute.device, requires_grad=False).bernoulli().byte(), mask_id)

def mask_rand_token(inpute, p, startid, endid):

	_m = inpute.new_full(inpute.size(), p, dtype=torch.float, device=inpute.device, requires_grad=False).bernoulli().byte()

	return inpute.masked_scatter_(_m, torch.randint(startid, endid, (_m.int().sum().item(),), dtype=inpute.dtype, device=inpute.device, requires_grad=False))

def get_sind(seql, p, maxv=None):

	_len = max(2, ceil(seql * p))
	sind = randint(0, max(0, seql - _len))
	if maxv is not None:
		sind = min(maxv, sind)

	return sind, _len

def get_batch(batch_in, p_ext, p_mask, p_rand, mask_id, startid, endid):

	seql = batch_in.size(-1)
	_sind, _elen = get_sind(seql, p_ext, max(0, seql - 2 - batch_in.eq(0).sum(-1).max().item()))
	tgt_batch = batch_in.narrow(1, _sind, _elen).clone()
	sel_batch = batch_in.narrow(1, _sind + 1, _elen - 1)
	mask_rand_token(mask_token(sel_batch, p_mask, mask_id), p_rand, startid, endid)

	return batch_in, tgt_batch, _sind

def update_p(p_mask, p_rand):

	return p_mask / (1.0 - p_rand), p_rand
