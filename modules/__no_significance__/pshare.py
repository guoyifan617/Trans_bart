#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.nn import functional as nnFunc

from modules.act import Custom_Act
from modules.base import CrossAttn as CrossAttnBase, Dropout, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class PShareLinear(nn.Module):

	def __init__(self, weight_bank, bias=True, **kwargs):

		super(PShareLinear, self).__init__()

		self.wb = weight_bank
		osize, isize, nw = weight_bank.size()

		self.weight = nn.Parameter(torch.Tensor(nw).uniform_(- 1.0 / sqrt(nw), 1.0 / sqrt(nw)))
		if bias:
			self.bias = nn.Parameter(torch.zeros(osize))
		else:
			self.register_parameter("bias", None)

	def forward(self, input, **kwargs):

		_osize, _isize, _nw = self.wb.size()
		_w = self.wb.view(-1, _nw).mv(self.weight).view(_osize, _isize)

		return nnFunc.linear(input, _w, self.bias)

Linear = PShareLinear

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, wb1, wb2, dropout=0.0, act_drop=None, custom_act=use_adv_act_default, **kwargs):

		_act_drop = parse_none(act_drop, dropout)
		super(PositionwiseFF, self).__init__(wb1.size(1), wb1.size(0), dropout=dropout, act_drop=_act_drop, custom_act=custom_act, **kwargs)

		self.net = nn.Sequential(Linear(wb1), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(wb2))
		if dropout > 0.0:
			self.net.append(Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			self.net.insert(2, Dropout(_act_drop, inplace=inplace_after_Custom_Act))

class SelfAttn(SelfAttnBase):

	def __init__(self, wb1, wb2, hsize, osize, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(SelfAttn, self).__init__(wb1.size(1), hsize, osize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.adaptor = Linear(wb1, bias=enable_proj_bias)

		self.outer = Linear(wb2, bias=enable_bias)

class CrossAttn(CrossAttnBase):

	def __init__(self, wb1, wb2, wb3, hsize, osize, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(CrossAttn, self).__init__(wb1.size(1), hsize, osize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.query_adaptor = Linear(wb1, bias=enable_proj_bias)
		self.kv_adaptor = Linear(wb2, bias=enable_proj_bias)

		self.outer = Linear(wb3, bias=enable_bias)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, wb1, wb2, hsize, norm_residual=norm_residual_default, **kwargs):

		isize = wb1.size(1)
		super(ResSelfAttn, self).__init__(isize, hsize, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(wb1, wb2, hsize, isize, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, wb1, wb2, wb3, hsize, norm_residual=norm_residual_default, **kwargs):

		isize = wb1.size(1)
		super(ResCrossAttn, self).__init__(isize, hsize, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(wb1, wb2, wb3, hsize, isize, **kwargs)
