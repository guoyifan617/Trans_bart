#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import CrossAttn as CrossAttnBase, Custom_Act, Dropout, Linear, PositionwiseFF as PositionwiseFFBase, SelfAttn as SelfAttnBase
from modules.group.base import GroupLinear
from modules.mulang.base import LayerNorm
from utils.fmt.parser import parse_none
from utils.torch.ext import bmv

from cnfg.ihyp import *

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, ngroup, hsize=None, dropout=0.0, act_drop=None, ntask=None, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=_act_drop, custom_act=custom_act, enable_bias=enable_bias, **kwargs)

		_hsize *= ngroup
		self.ngroup = ngroup

		self.net = nn.Sequential(Linear(isize, _hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), GroupLinear(_hsize, isize * ngroup, ngroup, bias=enable_bias, shuffle=False, trans_input=True, flatten_output=False))
		if dropout > 0.0:
			self.net.append(Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			self.net.insert(2, Dropout(_act_drop, inplace=inplace_after_Custom_Act))
		self.normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	# weight: (bsize, ngroup)
	def forward(self, x, weight=None, taskid=None, **kwargs):

		_out = self.normer(x, taskid=taskid)
		out = self.net(_out)

		_res_add = _out if self.norm_residual else x
		if weight is None:
			_res_add = _res_add.unsqueeze(-2)
		else:
			out = bmv(out.transpose(-1, -2).contiguous().view(out.size(0), -1, self.ngroup), weight).view(x.size())

		out = out + _res_add

		return out

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, ngroup, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize * ngroup, osize, num_head=num_head * ngroup, dropout=dropout, enable_bias=enable_bias, **kwargs)

		self.ngroup, self.osize = ngroup, osize
		self.outer = GroupLinear(self.hsize, osize * ngroup, ngroup, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False)

	def forward(self, iQ, mask=None, states=None, weight=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads, ngroup, adim = self.num_head, self.ngroup, self.attn_dim

		real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)
		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		if states is not None:
			_h_real_iK, _h_real_iV = states
			if _h_real_iK is None:
				seql = nquery
			else:
				seql = nquery + _h_real_iK.size(-1)
				real_iK, real_iV = torch.cat((_h_real_iK, real_iK,), dim=-1), torch.cat((_h_real_iV, real_iV,), dim=2)

		scores = real_iQ.matmul(real_iK)

		if self.rel_pemb is not None:
			if states is None:
				self.rel_pos_cache = self.get_rel_pos(nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, nquery).permute(1, 2, 0, 3)
			else:
				self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3)

		scores = scores / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		out = self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, ngroup, -1))

		if weight is not None:
			out = bmv(out.transpose(-1, -2).contiguous().view(bsize, -1, ngroup), weight).view(iQ.size())

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize, osize, ngroup, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize * ngroup, osize, num_head=num_head * ngroup, dropout=dropout, k_isize=k_isize, enable_bias=enable_bias, **kwargs)

		self.ngroup, self.osize = ngroup, osize
		self.outer = GroupLinear(self.hsize, osize * ngroup, ngroup, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False)

	def forward(self, iQ, iK, mask=None, weight=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads, ngroup, adim = self.num_head, self.ngroup, self.attn_dim

		real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
		if (self.real_iK is not None) and self.iK.is_set_to(iK) and self.is_decoding:
			real_iK, real_iV = self.real_iK, self.real_iV
		else:
			real_iK, real_iV = self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if self.is_decoding:
				self.iK, self.real_iK, self.real_iV = iK, real_iK, real_iV

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		out = self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, ngroup, -1))

		if weight is not None:
			out = bmv(out.transpose(-1, -2).contiguous().view(bsize, -1, ngroup), weight).view(iQ.size())

		return out
