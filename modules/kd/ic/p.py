#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.nn.functional import kl_div

from utils.fmt.parser import parse_none
from utils.kd.base import renorm as norm_func
from utils.kd.ic.p import init_cache
from utils.kd.self.p import correct_index, fix_gold
from utils.torch.comp import torch_any_dim, torch_no_grad
from utils.torch.ext import arcsoftmax

from cnfg.vocab.base import pad_id

class TopCache(nn.Module):

	def __init__(self, vsize, num_topk, T=1.0, warm_cache_steps=None, min_gold_p=None, mavg_beta=None, warm_mvavg_steps=None, num_cache_topk=None, p=None, **kwargs):

		super(TopCache, self).__init__()
		self.num_topk, self.min_gold_p, self.mavg_beta = num_topk, min_gold_p, mavg_beta
		self.T = parse_none(T, 1.0)
		_init_topk, _init_p = init_cache(vsize, (num_topk + num_topk) if num_cache_topk is None else num_cache_topk, p=((min_gold_p + 0.5) if min_gold_p < 0.5 else min_gold_p) if p is None else p, num_topk=num_topk)
		self.register_buffer("cache_index", _init_topk, persistent=False)
		self.register_buffer("cache_p", _init_p, persistent=False)
		self.cache_update_steps = 0
		self.warm_cache_steps = parse_none(warm_cache_steps, 0)
		self.warm_mvavg_steps = parse_none(warm_mvavg_steps, 0)

	def forward(self, x, gold=None, gold_pad_mask=None, update_cache=True, **kwargs):

		_compute_loss = self.cache_update_steps >= self.warm_cache_steps
		if _compute_loss or update_cache:
			_gold_pad_mask = gold.eq(pad_id) if gold_pad_mask is None else gold_pad_mask
		_loss = self.get_loss(x, gold=gold, gold_pad_mask=_gold_pad_mask) if _compute_loss else x.new_zeros(1)
		if update_cache:
			self.update_cache(x, gold=gold, gold_pad_mask=_gold_pad_mask)

		return _loss

	def get_loss(self, x, gold=None, gold_pad_mask=None):

		_f_g = gold.view(-1)
		_e_g_size = [*gold.size(), -1]
		# (bsize, nquery, ntopk)
		_c_i = self.cache_index.index_select(0, _f_g).narrow(-1, 0, self.num_topk).long().view(_e_g_size)
		_c_p = norm_func(self.cache_p.index_select(0, _f_g).narrow(-1, 0, self.num_topk), dim=-1).view(_e_g_size)
		_c_p.masked_fill_(gold_pad_mask.unsqueeze(-1), 0.0)
		_m_s = x.gather(-1, _c_i)

		return kl_div(_m_s.div_(self.T).log_softmax(-1) if self.T != 1.0 else _m_s.log_softmax(-1), _c_p, reduction="sum")

	def update_cache(self, x, gold=None, gold_pad_mask=None):

		with torch_no_grad():
			vsize, num_topk_cache = self.cache_p.size()
			num_topk = self.num_topk
			# (bsize * nquery, ntopkc)
			_new_p, _new_i = x.detach().view(-1, vsize).topk(num_topk, dim=-1)
			_f_gold = gold.view(-1)
			_mask = ~gold_pad_mask.view(-1)
			_f_gold = _f_gold[_mask]
			#_mask = _mask.unsqueeze(-1).expand(-1, num_topk)
			_new_p, _new_i = _new_p[_mask], _new_i[_mask]#.view(-1, num_topk)
			_new_i, _mask = correct_index(_new_i, _f_gold)
			_new_p = _new_p.div_(self.T).softmax(-1) if self.T != 1.0 else _new_p.softmax(-1)
			if self.min_gold_p is not None:
				_new_p = fix_gold(_new_p, _mask, self.min_gold_p)
			_gold_ind, _ind_counts = _f_gold.unique(sorted=False, return_counts=True)
			_mask = _ind_counts.gt(1)
			_df = _gold_ind[_mask]
			if _df.numel() > 0:
				_ind_counts = _ind_counts[_mask]
				_mask = _f_gold.unsqueeze(-1).eq(_df)
				_ = torch_any_dim(_mask, -1, keepdim=False)#.expand(-1, num_topk)
				_new_p[_] = _new_p[_].div_(_ind_counts.to(_new_p.dtype, non_blocking=True).unsqueeze(0).expand(_new_p.size(0), -1)[_mask].unsqueeze(-1)).view(-1)#.view(-1, num_topk)
			_mavg_beta = self.mavg_beta if self.cache_update_steps >= self.warm_mvavg_steps else (self.mavg_beta * sqrt(float(self.cache_update_steps) / float(self.warm_mvavg_steps)))
			_m_s_t = torch.sparse_coo_tensor(torch.stack([_gold_ind.unsqueeze(-1).repeat(1, num_topk_cache).view(-1), self.cache_index.index_select(0, _gold_ind).long().view(-1)], 0), (self.cache_p.index_select(0, _gold_ind) * _mavg_beta).view(-1), size=(vsize, vsize,), device=_new_p.device).add_(torch.sparse_coo_tensor(torch.stack([_f_gold.unsqueeze(-1).repeat(1, num_topk).view(-1), _new_i.view(-1)], 0), _new_p.view(-1), size=(vsize, vsize,), device=_new_p.device), alpha=1.0 - _mavg_beta).coalesce()
			# when sorted is False, _gold_ind is in reverse order with pytorch 1.11.0
			_gold_ind, _ind_counts = _m_s_t.indices()[0].unique(sorted=False, return_counts=True)
			_ind_mlen = _ind_counts.max()
			_ind_pad = _ind_mlen - _ind_counts
			_max_pad = _ind_pad.max().item()
			_n_inds = _gold_ind.unsqueeze(-1).expand(-1, _max_pad)[torch.arange(_max_pad, dtype=_gold_ind.dtype, device=_gold_ind.device).lt(_ind_pad.unsqueeze(-1))]
			_num_pad = _n_inds.numel()
			_esize = vsize + _num_pad
			_m_s_t = _m_s_t.sparse_resize_((vsize, _esize,), 2, 0).add_(torch.sparse_coo_tensor(torch.stack([_n_inds, torch.arange(vsize, _esize, dtype=_n_inds.dtype, device=_n_inds.device)], 0), _new_p.new_zeros(1).expand(_num_pad), size=(vsize, _esize,), device=_new_p.device)).coalesce()
			_ind_mlen = _ind_mlen.item()
			_inds, _p = _m_s_t.indices().view(2, -1, _ind_mlen)[1], _m_s_t.values().view(-1, _ind_mlen)
			if _ind_mlen > num_topk_cache:
				_p, _ = _p.topk(num_topk_cache, dim=-1)
				_inds = _inds.gather(-1, _)
			self.cache_index.index_copy_(0, _gold_ind, _inds.int())
			# is normalization necessary?
			self.cache_p.index_copy_(0, _gold_ind, norm_func(_p, dim=-1))
		self.cache_update_steps += 1

	def update_T(self, T):

		if self.T != T:
			with torch_no_grad():
				self.register_buffer("cache_p", arcsoftmax(self.cache_p).mul_(self.T / T).softmax(-1), persistent=False)
			self.T = T

def shift_caches(topkm_l):

	with torch_no_grad():
		_net = topkm_l[0]
		_base_topk, _base_p = _net.cache_index, _net.cache_p
		for _ in topkm_l[1:]:
			_t, _p = _.cache_index, _.cache_p
			_.register_buffer("cache_index", _base_topk.to(_t.device, non_blocking=True))
			_.register_buffer("cache_p", _base_p.to(_p.device, non_blocking=True))
			_base_topk, _base_p = _t, _p
		_t, _p = _net.cache_index, _net.cache_p
		_net.register_buffer("cache_index", _base_topk.to(_t.device, non_blocking=True))
		_net.register_buffer("cache_p", _base_p.to(_p.device, non_blocking=True))

def swap_caches(netin):

	_ms = {}
	for net in netin.modules():
		if isinstance(net, TopCache):
			_key = (net.num_topk, net.T, net.min_gold_p, tuple(net.cache_index.size()),)
			if _key in _ms:
				_ms[_key].append(net)
			else:
				_ms[_key] = [net]

	if _ms:
		for _v in _ms.values():
			if len(_v) > 1:
				shift_caches(_v)

	return netin
