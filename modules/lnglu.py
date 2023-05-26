#encoding: utf-8

import torch
from math import sqrt
from numbers import Integral
from torch import nn

from modules.base import Dropout, Linear, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class LNGLU(nn.LayerNorm):

	def __init__(self, normalized_shape, eps=ieps_ln_default, elementwise_affine=True, **kwargs):

		if isinstance(normalized_shape, Integral):
			normalized_shape = (normalized_shape, 2,)
		else:
			normalized_shape = tuple([*normalized_shape, 2])

		super(LNGLU, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

	def forward(self, x, **kwargs):

		_std, _mean = torch.std_mean(x, dim=-1, unbiased=False, keepdim=True)# x.std(dim=-1, unbiased=False, keepdim=True), x.mean(dim=-1, keepdim=True)#.detach()
		_xn = (x - _mean) / (_std + self.eps)
		_x1, _x2 = self.bias.addcmul(self.weight, _xn.unsqueeze(-1)).unbind(-1)

		return _x1.sigmoid().mul(_x2)

	def reset_parameters(self):

		with torch_no_grad():
			if self.weight is not None:
				_ = 1.0 / sqrt(self.weight.size(-2))
				self.weight.uniform_(-_, _)
			if self.bias is not None:
				self.bias.zero_()

	def fix_init(self):

		self.reset_parameters()

class ResLNGLU(nn.Module):

	def __init__(self, isize, dropout=0.0, **kwargs):

		super(ResLNGLU, self).__init__()

		self.net = nn.Sequential(LNGLU(isize), Dropout(dropout, inplace=True)) if dropout > 0.0 else LNGLU(isize)

	def forward(self, x, **kwargs):

		return self.net(x) + x

class PositionwiseFF_LNGLU(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		super(PositionwiseFF_LNGLU, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=_act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias, **kwargs)

		_ = [Linear(isize, _hsize, bias=enable_bias), LNGLU(_hsize), Linear(_hsize, isize, bias=enable_bias)]
		if dropout > 0.0:
			_.append(Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			_.insert(2, Dropout(_act_drop, inplace=True))
		self.net = nn.Sequential(*_)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.eff_layer = ResLNGLU(isize, dropout=dropout)

	def forward(self, iQ, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return self.eff_layer(_out + (_iQ if self.norm_residual else iQ)), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return self.eff_layer(outs + (_iQ if self.norm_residual else iQ))

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.eff_layer = ResLNGLU(isize, dropout=dropout)

	def forward(self, iQ, iK, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, iK, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return self.eff_layer(_out + (_iQ if self.norm_residual else iQ)), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return self.eff_layer(outs + (_iQ if self.norm_residual else iQ))

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, **kwargs):

		super(PositionwiseFF, self).__init__(isize, hsize=hsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, **kwargs)

		self.eff_layer = ResLNGLU(isize, dropout=dropout)

	def forward(self, x, **kwargs):

		_out = self.normer(x)

		out = self.net(_out)

		out = out + (_out if self.norm_residual else x)

		return self.eff_layer(out)
