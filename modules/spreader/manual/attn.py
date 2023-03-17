#encoding: utf-8

import torch
from torch.nn import functional as nnFunc

from modules.spreader.Spreader import SpreaderFunc
from modules.spreader.manual.rnn import Spreader as SpreaderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Spreader(SpreaderBase):

	def __init__(self, isize, hsize=None, start=2, end=8, factor=0.5, dropout=0.0, norm_residual=norm_residual_default, xseql=cache_len_default, **kwargs):

		_hsize = parse_none(hsize, isize)

		super(Spreader, self).__init__(isize, hsize=_hsize, start=start, end=end, factor=factor, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.xseql = xseql
		self.register_buffer("decay_emb", torch.cat((self.decay.new_zeros(1, _hsize), self.decay.new_ones(1, _hsize), self.decay.unsqueeze(0).expand(xseql, _hsize).cumprod(dim=0),), dim=0), persistent=False)
		_rpm = torch.arange(0, xseql, dtype=torch.long)
		self.register_buffer("rel_pos", (_rpm.unsqueeze(0) - _rpm.unsqueeze(1) + 1).triu(), persistent=False)
		self.ref_rel_posm = None
		self.register_buffer("rel_pos_cache", None, persistent=False)
		self.ref_attn_matm = None
		self.register_buffer("attn_mat_cache", None, persistent=False)

	def forward(self, x, states=None, **kwargs):

		_x = self.normer(x)
		out = self.trans(_x).mul_(self.decay_beta)

		cx_out = out.permute(2, 0, 1).bmm(self.get_attn_mat(out.size(1))).permute(1, 2, 0) if (states is None) or (states == "init") else SpreaderFunc(self.decay, out, states, 1, False)

		out = self.outer(self.normer_csum(cx_out))
		_res_add = _x if self.norm_residual else x
		gate = self.gate(torch.cat((_res_add, out,), dim=-1))

		_res_add = (1.0 - gate).mul(_res_add)
		out = _res_add.addcmul_(gate, out) if self.drop is None else _res_add.add_(self.drop(out * gate))

		if states is None:
			return out
		else:
			return out, cx_out.select(1, -1)

	def get_attn_mat_core(self, length):

		_rel_pos = self.get_rel_pos(length)
		_decay_emb = self.get_decay_emb(length)

		# default args: https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
		return nnFunc.embedding(_rel_pos, _decay_emb, 0, None, 2.0, False, False).permute(2, 0, 1).contiguous()

	def get_rel_pos_core(self, length):

		if length <= self.xseql:
			return self.rel_pos.narrow(0, 0, length).narrow(1, 0, length)
		else:
			_rpm = torch.arange(0, length, dtype=self.rel_pos.dtype, device=self.rel_pos.device)
			return (_rpm.unsqueeze(0) - _rpm.unsqueeze(1) + 1).triu()

	def get_decay_emb(self, length):

		if length <= self.xseql:
			return self.decay_emb
		else:
			_tail = self.decay.unsqueeze(0).repeat(length - self.xseql, 1)
			_tail[0].mul_(self.decay_emb[-1])
			return torch.cat((self.decay_emb, _tail.cumprod(dim=0),), dim=0)

	def get_rel_pos(self, length):

		if self.ref_rel_posm is None:
			self.rel_pos_cache = self.get_rel_pos_core(length)
		else:
			self.rel_pos_cache = self.ref_rel_posm.rel_pos_cache

		return self.rel_pos_cache

	def get_attn_mat(self, length):

		if self.ref_attn_matm is None:
			self.attn_mat_cache = self.get_attn_mat_core(length)
		else:
			self.attn_mat_cache = self.ref_attn_matm.attn_mat_cache

		return self.attn_mat_cache

	def reset_buffer(self, rel_pos=None, attn_mat=None):

		self.rel_pos_cache = rel_pos
		self.attn_mat_cache = attn_mat

class BiSpreader(Spreader):

	def __init__(self, isize, hsize=None, start=2, end=8, factor=0.5, dropout=0.0, norm_residual=norm_residual_default, xseql=cache_len_default, **kwargs):

		_hsize = parse_none(hsize, isize)

		super(BiSpreader, self).__init__(isize, hsize=_hsize, start=start, end=end, factor=factor, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.register_buffer("decay_emb", torch.cat((self.decay.new_ones(1, _hsize), self.decay.unsqueeze(0).expand(xseql, _hsize).cumprod(dim=0),), dim=0), persistent=False)
		_rpm = torch.arange(0, xseql, dtype=torch.long)
		self.register_buffer("rel_pos", (_rpm.unsqueeze(0) - _rpm.unsqueeze(1)).abs(), persistent=False)

	def forward(self, x, mask=None, **kwargs):

		_x = self.normer(x)
		out = self.trans(_x).mul_(self.decay_beta)

		if mask is not None:
			out = out.masked_fill(mask, 0.0)

		cx_out = out.permute(2, 0, 1).bmm(self.get_attn_mat(out.size(1))).permute(1, 2, 0)

		out = self.outer(self.normer_csum(cx_out))
		_res_add = _x if self.norm_residual else x
		gate = self.gate(torch.cat((_res_add, out,), dim=-1))

		_res_add = (1.0 - gate).mul(_res_add)

		return _res_add.addcmul_(gate, out) if self.drop is None else _res_add.add_(self.drop(out * gate))

	def get_rel_pos_core(self, length):

		if length <= self.xseql:
			return self.rel_pos.narrow(0, 0, length).narrow(1, 0, length)
		else:
			_rpm = torch.arange(0, length, dtype=self.rel_pos.dtype, device=self.rel_pos.device)
			return (_rpm.unsqueeze(0) - _rpm.unsqueeze(1)).abs()
