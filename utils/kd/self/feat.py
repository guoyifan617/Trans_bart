#encoding: utf-8

import torch

#mse_loss may boost the magnitude and lead to convergence issue

#from torch.nn.functional import mse_loss as loss_func

from loss.kd import cosim_loss as loss_func
from utils.kd.base import identity_norm as norm_func

def get_kd_loss(x, mask=None):

	_x = norm_func(x, dim=-1)
	#if mask is not None:
		#_x.masked_fill_(mask, 0.0)
	_n = _x.size(0) - 1
	_t = _x.narrow(0, _n, 1).detach()
	_s = _x.narrow(0, 0, _n)

	return loss_func(_t, _s, mask=mask, reduction="sum")

def get_iter_kd_loss(x, mask=None):

	_x = norm_func(x, dim=-1)
	#if mask is not None:
		#_x.masked_fill_(mask, 0.0)
	_n = _x.size(0) - 1
	_t = _x.narrow(0, 1, _n).detach()
	_s = _x.narrow(0, 0, _n)

	return loss_func(_t, _s, mask=mask, reduction="sum")

class ATAKDLoss:

	def __init__(self, *args, **kwargs):

		self.index_cache = {}
		if args:
			for _ in set(args):
				self.get_index(_)

	def get_index(self, nt, device=None):

		_key = (nt, device,)
		if _key in self.index_cache:
			ind = self.index_cache[_key]
		else:
			ind = (torch.arange(1, nt, dtype=torch.long, device=device).unsqueeze(0) + torch.arange(nt, dtype=torch.long, device=device).unsqueeze(-1)).view(-1) % nt
			self.index_cache[(nt, device,)] = ind

		return ind

	def __call__(self, x, mask=None, **kwargs):

		_x = norm_func(x, dim=-1)
		#if mask is not None:
			#_x.masked_fill_(mask, 0.0)
		_nt = _x.size(0)
		_t = _x.detach().unsqueeze(1)
		_s = _x.index_select(0, self.get_index(_nt, device=_x.device)).view(_nt, _nt - 1, *_x.size()[1:])

		return loss_func(_t, _s, mask=mask, reduction="sum")

get_ata_kd_loss = ATAKDLoss()
