#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import CrossAttn as CrossAttnBase, Custom_Act, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from modules.group.base import GroupLinear

from cnfg.ihyp import *

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, custom_act=use_adv_act_default, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, enable_bias=enable_bias, **kwargs)

		self.trans_gf = GroupLinear(self.hsize + self.hsize, self.hsize + self.hsize, num_head, bias=enable_bias, trans_input=False, shuffle=False, flatten_output=False)
		self.trans_gf_ln = nn.LayerNorm((num_head, 2, self.attn_dim,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.ffn_act = Custom_Act() if custom_act else nn.ReLU(inplace=False)

	def forward(self, iQ, mask=None, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

		_real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)
		real_iQ, real_iK, real_iV = _real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

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

		out = scores.matmul(real_iV).transpose(1, 2).contiguous()
		gate, ffn = self.trans_gf_ln(self.trans_gf(torch.cat((out, _real_iQ,), dim=-1)).view(bsize, nquery, nheads, 2, adim)).unbind(-2)
		out = self.outer(self.ffn_act(ffn).addcmul(out, gate.sigmoid()).view(bsize, nquery, self.hsize))

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, k_isize=k_isize, enable_bias=enable_bias, **kwargs)

		self.trans_gf = GroupLinear(self.hsize + self.hsize, self.hsize + self.hsize, num_head, bias=enable_bias, trans_input=False, shuffle=False, flatten_output=False)
		self.trans_gf_ln = nn.LayerNorm((num_head, 2, self.attn_dim,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.ffn_act = Custom_Act() if custom_act else nn.ReLU(inplace=False)

	def forward(self, iQ, iK, mask=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		_real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim)
		real_iQ = _real_iQ.transpose(1, 2)
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

		out = scores.matmul(real_iV).transpose(1, 2).contiguous()
		gate, ffn = self.trans_gf_ln(self.trans_gf(torch.cat((out, _real_iQ,), dim=-1)).view(bsize, nquery, nheads, 2, adim)).unbind(-2)

		return self.outer(self.ffn_act(ffn).addcmul(out, gate.sigmoid()).view(bsize, nquery, self.hsize))

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
