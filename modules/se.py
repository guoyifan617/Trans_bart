#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Dropout, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *
from cnfg.se import k_se

class Squeezer(nn.Module):

	def __init__(self, isize, k, dropout=0.0, **kwargs):

		super(Squeezer, self).__init__()

		_ = 1.0 / sqrt(float(k))
		self.weight = nn.Parameter(torch.Tensor(isize // k, k, 1).uniform_(-_, _))
		self.drop = Dropout(dropout, inplace=False) if dropout > 0.0 else None
		self.register_buffer("cache", None, persistent=False)

	def forward(self, x, **kwargs):

		if self.training:
			_ = self.weight.softmax(1) if self.drop is None else self.drop(self.weight).softmax(1)
		else:
			if self.cache is None:
				self.cache = self.weight.softmax(1)
			_ = self.cache
		_a, _b = _.size()[:2]

		return x.view(-1, _a, _b).transpose(0, 1).bmm(_).squeeze(-1).transpose(0, 1).contiguous().view(*x.size()[:-1], -1)

	def fix_init(self):

		_ = 1.0 / sqrt(float(self.weight.size(1)))
		with torch_no_grad():
			self.weight.data.uniform_(-_, _)

class Expander(nn.Module):

	def __init__(self, isize, k, dropout=0.0, **kwargs):

		super(Expander, self).__init__()

		_ = 1.0 / sqrt(float(k))
		self.weight = nn.Parameter(torch.Tensor(isize, k).uniform_(-_, _))
		self.drop = Dropout(dropout, inplace=False) if dropout > 0.0 else None
		self.register_buffer("cache", None, persistent=False)

	def forward(self, x, **kwargs):

		if self.training:
			_ = self.weight.softmax(-1) if self.drop is None else self.drop(self.weight).softmax(-1)
		else:
			if self.cache is None:
				self.cache = self.weight.softmax(-1)
			_ = self.cache

		return x.unsqueeze(-1).mul(_).view(*x.size()[:-1], -1)

	def fix_init(self):

		_ = 1.0 / sqrt(float(self.weight.size(-1)))
		with torch_no_grad():
			self.weight.data.uniform_(-_, _)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, k_se=k_se, **kwargs):

		_s_isize, _s_hsize = isize // k_se, hsize // k_se

		super(ResSelfAttn, self).__init__(_s_isize, _s_hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.s_net = Squeezer(isize, k_se, dropout=dropout)
		self.e_net = Expander(_s_isize, k_se, dropout=dropout)

	def forward(self, iQ, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(self.s_net(_iQ), *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return self.e_net(_out) + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return self.e_net(outs) + (_iQ if self.norm_residual else iQ)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, k_isize=None, k_se=k_se, **kwargs):

		_k_isize = parse_none(k_isize, isize)
		_s_isize, _s_hsize, _s_ksize = isize // k_se, hsize // k_se, _k_isize // k_se

		super(ResCrossAttn, self).__init__(_s_isize, _s_hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, k_isize=_s_ksize, **kwargs)

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.qs_net = Squeezer(isize, k_se, dropout=dropout)
		self.ks_net = Squeezer(_k_isize, k_se, dropout=dropout)
		self.e_net = Expander(_s_isize, k_se, dropout=dropout)
		self.register_buffer("s_iK", None, persistent=False)
		self.register_buffer("iK", None, persistent=False)

	def forward(self, iQ, iK, *inputs, **kwargs):

		_iQ = self.normer(iQ)
		if (self.s_iK is not None) and self.iK.is_set_to(iK) and (not self.training):
			_s_iK = self.s_iK
		else:
			_s_iK = self.ks_net(iK)
			if not self.training:
				self.iK, self.s_iK = iK, _s_iK
		outs = self.net(self.qs_net(_iQ), _s_iK, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return self.e_net(_out) + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return self.e_net(outs) + (_iQ if self.norm_residual else iQ)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, k_se=k_se, **kwargs):

		_s_isize = isize // k_se

		super(PositionwiseFF, self).__init__(_s_isize, hsize=hsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, **kwargs)

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.s_net = Squeezer(isize, k_se, dropout=dropout)
		self.e_net = Expander(_s_isize, k_se, dropout=dropout)

	def forward(self, x, **kwargs):

		_out = self.normer(x)

		out = self.e_net(self.net(self.s_net(_out)))

		out = out + (_out if self.norm_residual else x)

		return out
