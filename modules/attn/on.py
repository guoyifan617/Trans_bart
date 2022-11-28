#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Dropout, Linear, ResSelfAttn as ResSelfAttnBase, SparseNormer
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class SelfAttn(nn.Module):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, num_anchor=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, sparsenorm=False, **kwargs):

		super(SelfAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head
		self.num_anchor = self.attn_dim if num_anchor is None else num_anchor

		self.anchors = nn.Parameter(torch.Tensor(1, self.num_head, self.num_anchor, self.attn_dim).uniform_(- sqrt(1.0 / self.attn_dim), sqrt(1.0 / self.attn_dim)))
		self.adaptor = Linear(isize, self.hsize * 4, bias=enable_proj_bias)# input: iQ, output: a_k, q, a_v1, a_v2

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

	# iQ: (bsize, nquery, isize)
	# mask: (bsize, 1, nquery)

	def forward(self, iQ, mask=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim
		sqrt_adim = sqrt(adim)

		real_iQ, real_aK, real_aV1, real_aV2 = self.adaptor(iQ).view(bsize, nquery, 4, nheads, adim).unbind(2)
		real_iQ, real_aK, real_aV1, real_aV2 = real_iQ.transpose(1, 2), real_aK.permute(0, 2, 3, 1), real_aV1.transpose(1, 2), real_aV2.transpose(1, 2)

		# (1, nheads, nanchor, adim) * (bsize, nheads, adim, nquery) => (bsize, nheads, nanchor, nquery)
		scores = self.anchors.matmul(real_aK)

		scores = scores / sqrt_adim

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# (bsize, nheads, nanchor, nquery) * (bsize, nheads, nquery, adim) => (bsize, nheads, nanchor, adim)
		real_aV1, real_aV2 = scores.matmul(real_aV1), scores.matmul(real_aV2)

		# (bsize, nheads, nquery, adim) * T(bsize, nheads, nanchor, adim) => (bsize, nheads, nquery, nanchor)
		scores = self.normer(real_iQ.matmul(real_aV1.transpose(-1, -2)) / sqrt_adim)

		if self.drop is not None:
			scores = self.drop(scores)

		# (bsize, nheads, nquery, nanchor) * (bsize, nheads, nanchor, adim) => (bsize, nheads, nquery, adim)
		out = self.outer(scores.matmul(real_aV2).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		return out

	def fix_init(self):

		with torch_no_grad():
			self.anchors.data.uniform_(- sqrt(1.0 / self.attn_dim), sqrt(1.0 / self.attn_dim))

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class Summer(nn.Module):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, num_anchor=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, sparsenorm=False, **kwargs):

		super(Summer, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head
		self.num_anchor = self.attn_dim if num_anchor is None else num_anchor

		self.anchors = nn.Parameter(torch.Tensor(1, self.num_head, self.num_anchor, self.attn_dim).uniform_(- sqrt(1.0 / self.attn_dim), sqrt(1.0 / self.attn_dim)))
		self.kv_adaptor = Linear(isize, self.hsize * 2, bias=enable_proj_bias)# input: iQ, output: a_k, q, a_v1, a_v2

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

	# iK: (bsize, seql, isize)
	# mask: (bsize, 1, seql)

	def forward(self, iK, mask=None, **kwargs):

		bsize, seql = iK.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

		real_iK, real_iV = self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
		real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		# (1, nheads, nanchor, adim) * (bsize, nheads, adim, seql) => (bsize, nheads, nanchor, seql)
		scores = self.anchors.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		return self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, self.num_anchor, self.hsize))

	def fix_init(self):

		with torch_no_grad():
			self.anchors.data.uniform_(- sqrt(1.0 / self.attn_dim), sqrt(1.0 / self.attn_dim))
