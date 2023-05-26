#encoding: utf-8

import torch
from torch import nn

from modules.attn.rap import CrossAttn as CrossAttnBase, SelfAttn as SelfAttnBase
from modules.base import Linear, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase
from modules.nas.gdart import Cell
from utils.decode.beam import repeat_bsize_for_beam_tensor
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class SelfAttn(SelfAttnBase):

	def forward(self, iQ, mask=None, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

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

		out = self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		if states is None:
			return out, scores
		else:
			return out, (real_iK, real_iV,), scores

class CrossAttn(CrossAttnBase):

	def forward(self, iQ, iK, mask=None, **kwargs):

		bsize, nquery, _ = iQ.size()
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

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

		return self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize)), scores

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class TaughtSelfAttn(nn.Module):

	def __init__(self, isize, hsize, osize, num_head=8, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(SelfAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.adaptor = Linear(isize, self.hsize, bias=enable_proj_bias)

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

	def forward(self, iQ, scores, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

		real_iV = self.adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
		if states is not None:
			_h_real_iV = states[-1]
			if _h_real_iV is None:
				seql = nquery
			else:
				seql = nquery + _h_real_iV.size(-1)
				real_iV = torch.cat((_h_real_iV, real_iV,), dim=2)

		out = self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		if states is None:
			return out
		else:
			return out, real_iV

	def load_base(self, base_module):

		self.attn_dim, self.hsize, self.num_head = base_module.attn_dim, base_module.hsize, base_module.num_head

		self.outer = base_module.outer

		with torch_no_grad():
			self.adaptor.weight.copy_(base_module.adaptor.weight.narrow(0, self.hsize + self.hsize, self.hsize))
			if (base_module.adaptor.bias is not None) and (self.adaptor.bias is not None):
				self.adaptor.bias.copy_(base_module.adaptor.bias.narrow(0, self.hsize + self.hsize, self.hsize))

class TaughtCrossAttn(nn.Module):

	def __init__(self, isize, hsize, osize, num_head=8, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(TaughtCrossAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.v_adaptor = Linear(isize, self.hsize, bias=enable_proj_bias)

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

		self.register_buffer("real_iV", None, persistent=False)
		self.register_buffer("iK", None, persistent=False)

	# context is not used, keep for ResTaughtCrossAttn wrapper
	def forward(self, context, iK, scores, **kwargs):

		bsize, seql = iK.size()[:2]
		nquery = scores.size(2)
		nheads = self.num_head
		adim = self.attn_dim

		real_iV = self.v_adaptor(iK).view(bsize, seql, nheads, adim).transpose(1, 2)
		if (self.real_iV is not None) and self.iK.is_set_to(iK):
			real_iV = self.real_iV
		else:
			real_iV = self.v_adaptor(iK).view(bsize, seql, nheads, adim).transpose(1, 2)
			if not self.training:
				self.iK, self.real_iV = iK, real_iV

		return self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

	def load_base(self, base_module):

		self.attn_dim, self.hsize, self.num_head = base_module.attn_dim, base_module.hsize, base_module.num_head

		self.outer = base_module.outer

		with torch_no_grad():
			self.v_adaptor.weight.copy_(base_module.kv_adaptor.weight.narrow(0, self.hsize, self.hsize))
			if (base_module.kv_adaptor.bias is not None) and (self.v_adaptor.bias is not None):
				self.v_adaptor.bias.copy_(base_module.kv_adaptor.bias.narrow(0, self.hsize, self.hsize))

	def train(self, mode=True):

		super(TaughtCrossAttn, self).train(mode)

		if mode:
			self.reset_buffer()

		return self

	def reset_buffer(self, value=None):

		self.iK = self.real_iV = value

	def repeat_buffer(self, beam_size):

		if self.real_iV is not None:
			self.real_iV = repeat_bsize_for_beam_tensor(self.real_iV, beam_size)

	def index_buffer(self, indices, dim=0):

		if self.real_iV is not None:
			self.real_iV = self.real_iV.index_select(dim, indices)

class ResTaughtSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResTaughtSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = TaughtSelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class ResTaughtCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResTaughtCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual_default, **kwargs)

		self.net = TaughtCrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=_act_drop, **kwargs)

		self.net = Cell(max(1, _hsize // isize), isize, dropout=dropout, act_drop=_act_drop)

	def load_base(self, base_module):

		self.normer, self.norm_residual = base_module.normer, base_module.norm_residual
