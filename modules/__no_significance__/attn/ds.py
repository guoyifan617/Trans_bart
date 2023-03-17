#encoding: utf-8

import torch
from math import sqrt

from modules.base import Linear, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from utils.torch.comp import mask_tensor_type

from cnfg.ihyp import *

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, xseql=cache_len_default, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize, osize, xseql=xseql, **kwargs)

		self.hsize *= 2
		self.outer = Linear(self.hsize, osize, bias=(False if self.outer.bias is None else True))

		self.register_buffer("l_mask", torch.ones(xseql, xseql, dtype=mask_tensor_type).tril(-1), persistent=False)
		self.register_buffer("r_mask", torch.ones(xseql, xseql, dtype=mask_tensor_type).triu(1), persistent=False)

		self.xseql = xseql

	# for encoder self-attention, states is always None.
	def forward(self, iQ, mask=None, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)
		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		"""if states is not None:
			_h_real_iK, _h_real_iV = states
			if _h_real_iK is None:
				seql = nquery
			else:
				seql = nquery + _h_real_iK.size(-1)
				real_iK, real_iV = torch.cat((_h_real_iK, real_iK,), dim=-1), torch.cat((_h_real_iV, real_iV,), dim=2)"""

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

		l_mask, r_mask = self._get_directional_mask(nquery)

		l_scores, r_scores = scores.masked_fill(l_mask, 0.0), scores.masked_fill(r_mask, 0.0)

		out = self.outer(torch.cat((l_scores.matmul(real_iV), r_scores.matmul(real_iV),), -1).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

	def _get_directional_mask(self, length):

		if length <= self.xseql:
			return self.l_mask.narrow(0, 0, length).narrow(1, 0, length), self.r_mask.narrow(0, 0, length).narrow(1, 0, length)
		else:
			return self.l_mask.new_ones(length, length).tril(-1), self.r_mask.new_ones(length, length).triu(1)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
