#encoding: utf-8

import torch
from math import sqrt

from modules.mulang.eff.base import LayerNorm
from modules.mulang.o2m import CrossAttn as CrossAttnBase, PositionwiseFF as PositionwiseFFBase, SelfAttn as SelfAttnBase

from cnfg.ihyp import *

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, ngroup, *inputs, ntask=None, **kwargs):

		super(PositionwiseFF, self).__init__(isize, ngroup, *inputs, ntask=ntask, **kwargs)

		self.normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	# weight: (bsize, ngroup)
	def forward(self, x, weight=None, taskid=None, **kwargs):

		_out = self.normer(x, taskid=taskid)
		out = self.net(_out)

		_res_add = _out if self.norm_residual else x
		if weight is None:
			_res_add = _res_add.unsqueeze(-2)
		else:
			out = out.transpose(-1, -2).contiguous().view(-1, self.ngroup).mv(weight).view(x.size())

		out = out + _res_add

		return out

class SelfAttn(SelfAttnBase):

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
			out = out.transpose(-1, -2).contiguous().view(-1, ngroup).mv(weight).view(iQ.size())

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

class CrossAttn(CrossAttnBase):

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
			out = out.transpose(-1, -2).contiguous().view(bsize, -1, ngroup).mv(weight).view(iQ.size())

		return out
