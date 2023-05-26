#encoding: utf-8

from math import sqrt
from torch import nn

from modules.base import CrossAttn as CrossAttnBase, Dropout, ResCrossAttn as ResCrossAttnBase
from modules.hplstm.wrapper import LSTM4RNMT as HPLSTM
from utils.fmt.parser import parse_none
from utils.timeseries import index_tensors, repeat_bsize_for_beam_tensor

from cnfg.ihyp import *

class PositionwiseFF(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, act_drop=None, **kwargs):

		super(PositionwiseFF, self).__init__()

		self.net = HPLSTM(isize, num_head=num_head, osize=osize, fhsize=fhsize, dropout=parse_none(act_drop, dropout), **kwargs)
		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

	# inpute: (bsize, ngroup, seql, isize)
	def forward(self, inpute, **kwargs):

		_bsize, _ngroup, _seql, _isize = inpute.size()
		_out = self.net(inpute.transpose(1, 2).contiguous().view(_bsize * _seql, _ngroup, _isize)).view(_bsize, _seql, _ngroup, _isize).transpose(1, 2).contiguous()
		if self.drop is not None:
			_out = self.drop(_out)

		return inpute + _out

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, num_steps=None, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, k_isize=k_isize, **kwargs)

		self.num_steps = num_steps

	# iQ: (bsize * ngroup, nquery, isize)
	# iK: (bsize, seql, isize)
	def forward(self, iQ, iK, mask=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
		if (self.real_iK is not None) and self.iK.is_set_to(iK) and self.is_decoding:
			real_iK, real_iV = self.real_iK, self.real_iV
		else:
			_ = iK.size(0)
			real_iK, real_iV = self.kv_adaptor(iK).view(_, seql, 2, nheads, adim).repeat(1, self.num_steps, 1, 1, 1).view(_ * self.num_steps, seql, 2, nheads, adim).unbind(2)
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if self.is_decoding:
				self.iK, self.real_iK, self.real_iV = iK, real_iK, real_iV

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		return self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

	def repeat_buffer(self, beam_size):

		if self.real_iK is not None:
			self.real_iK, self.real_iV = repeat_bsize_for_beam_tensor(self.real_iK, beam_size, self.num_steps), repeat_bsize_for_beam_tensor(self.real_iV, beam_size, self.num_steps)

	def index_buffer(self, indices, dim=0):

		if self.real_iK is not None:
			self.real_iK, self.real_iV = index_tensors(self.real_iK, self.real_iV, indices=indices, dim=dim, ngroup=self.num_steps)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, num_steps=None, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, num_steps=num_steps, **kwargs)
