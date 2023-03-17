#encoding: utf-8

import torch
from math import sqrt
from torch import nn
#from torch.nn.functional import mse_loss as loss_func

from loss.kd import cosim_loss as loss_func
from utils.fmt.parser import parse_none
from utils.kd.base import identity_norm as norm_func#, rms_norm for mse_loss
from utils.torch.comp import torch_no_grad

from cnfg.vocab.base import pad_id

class Cache(nn.Module):

	def __init__(self, vsize, isize, warm_cache_steps=None, mavg_beta=None, warm_mvavg_steps=None, **kwargs):

		super(Cache, self).__init__()
		self.mavg_beta = parse_none(mavg_beta, 0.0)
		self.register_buffer("cache_index", torch.zeros(vsize, dtype=torch.long), persistent=False)
		self.register_buffer("cache_p", torch.zeros(vsize, isize), persistent=False)
		self.cache_update_steps = 0
		self.warm_cache_steps = parse_none(warm_cache_steps, 0)
		self.warm_mvavg_steps = parse_none(warm_mvavg_steps, 0)

	def forward(self, x, p, gold=None, gold_pad_mask=None, update_cache=True, **kwargs):

		_compute_loss = self.cache_update_steps >= self.warm_cache_steps
		if _compute_loss or update_cache:
			_gold_pad_mask = gold.eq(pad_id) if gold_pad_mask is None else gold_pad_mask
		_loss = self.get_loss(x, gold=gold, gold_pad_mask=_gold_pad_mask) if _compute_loss else x.new_zeros(1)
		if update_cache:
			self.update_cache(x, p, gold=gold, gold_pad_mask=_gold_pad_mask)

		return _loss

	def get_loss(self, x, gold=None, gold_pad_mask=None):

		_c_p = norm_func(self.cache_p.index_select(0, gold.view(-1)), dim=-1).view_as(x)
		_x = norm_func(x, dim=-1)
		#if gold_pad_mask is not None:
			#_mask = gold_pad_mask.unsqueeze(-1)
			#_c_p.masked_fill_(_mask, 0.0)
			#_x.masked_fill_(_mask, 0.0)

		return loss_func(_x, _c_p, mask=gold_pad_mask, reduction="sum")

	def update_cache(self, x, p, gold=None, gold_pad_mask=None):

		with torch_no_grad():
			_isize = x.size(-1)
			_f_gold = gold.view(-1)
			# scaling with p leads to convergence issue when using mse_loss with identity_norm
			_p = p.gather(dim=-1, index=gold.unsqueeze(-1))
			_ = x.mul(_p)#self.cache_p.index_select(0, _f_gold).view_as(x).mul_(1.0 - _p).addcmul_(norm_func(x, dim=-1), _p)
			if gold_pad_mask is not None:
				_.masked_fill_(gold_pad_mask.unsqueeze(-1), 0.0)
			_gold_ind, _ind_counts = _f_gold.unique(sorted=False, return_counts=True)
			_num_ind = _gold_ind.size(0)
			_base = x.new_zeros(_num_ind, _isize)
			self.cache_index.index_copy_(0, _gold_ind, torch.arange(_num_ind, dtype=_gold_ind.dtype, device=_gold_ind.device))
			_base.index_add_(0, self.cache_index.index_select(0, _f_gold), _.view(-1, _isize)).div_(_ind_counts.to(_base.dtype).unsqueeze(-1))
			_mavg_beta = self.mavg_beta if self.cache_update_steps >= self.warm_mvavg_steps else (self.mavg_beta * sqrt(float(self.cache_update_steps) / float(self.warm_mvavg_steps)))
			self.cache_p.index_copy_(0, _gold_ind, self.cache_p.index_select(0, _gold_ind).mul_(_mavg_beta).view_as(_base).add_(_base, alpha=1.0 - _mavg_beta) if _mavg_beta > 0.0 else _base)
		self.cache_update_steps += 1
