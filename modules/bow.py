#encoding: utf-8

import torch
from math import floor
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear, PositionwiseFF as PositionwiseFFBase
from modules.group.base import GroupLinear

from cnfg.ihyp import *

class CBOWInput(nn.Module):

	def __init__(self, kernel_size, isize, **kwargs):

		super(CBOWInput, self).__init__()

		_swsize = floor((kernel_size - 1) / 2.0)
		_wsize = _swsize * 2 + 1

		self.net = nn.AvgPool1d(_wsize, stride=1, padding=_swsize, ceil_mode=False, count_include_pad=True)
		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.k = float(_wsize)

	def forward(self, x, **kwargs):

		return self.normer(self.net(x.transpose(-1, -2)).mul_(self.k).transpose(-1, -2) - x)#.sub_(x)

class CBOW(nn.Module):

	# different from MultiHeadAttn, we use num_head for window size
	# osize is ignored
	def __init__(self, isize, hsize, osize, num_head=11, dropout=0.0, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(CBOW, self).__init__()

		self.inet = CBOWInput(num_head, isize)

		self.net = nn.Sequential(Linear(isize * 2, hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(hsize, isize, bias=enable_bias), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(Linear(isize * 2, hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(hsize, isize, bias=enable_bias))

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.norm_residual = norm_residual

	def forward(self, x, mask=None, **kwargs):

		if mask is not None:
			x.masked_fill_(mask, 0.0)

		_x = self.normer(x)

		return self.net(torch.cat((_x, self.inet(_x),), dim=-1)) + (_x if self.norm_residual else x)

class DBOWInput(nn.Module):

	def __init__(self, kernel_size, isize, **kwargs):

		super(DBOWInput, self).__init__()

		self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
		self.normer = nn.LayerNorm((4, isize,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.exnet = GroupLinear(isize * 5, isize * 5, 5, bias=False, trans_input=False, shuffle=True, flatten_output=False)
		self.zpad, self.k = kernel_size, float(kernel_size)

	# this function is not correct, rbow_out, lctxsum, rctxsum are wrongly computed, see modules.caffn for reference.
	def forward(self, x, **kwargs):

		_isize = list(x.size())
		_padsize = _isize[:]
		seql = _padsize[-2]
		_padsize[-2] = self.zpad
		_pad = x.new_zeros(_padsize, dtype=x.dtype, device=x.device, requires_grad=False)

		out = torch.cat((_pad, x, _pad,), dim=-2)
		bow_out = self.pool(out.transpose(-1, -2)).mul_(self.k).transpose(-1, -2)# - x .sub_(x)

		lbow_out, rbow_out = bow_out.narrow(-2, 0, seql), bow_out.narrow(-2, self.zpad + 1, seql)

		_lcumsum = out.cumsum(dim=-2)
		_sum = _lcumsum.narrow(-2, -1, 1)
		_rcumsum = _sum - _lcumsum.narrow(-2, self.zpad + 1, seql)
		_padsize[-2] = 1
		lctxsum, rctxsum = _lcumsum.narrow(-2, 0, seql) - lbow_out, _rcumsum - rbow_out

		return self.exnet(torch.cat((self.normer(torch.stack((lctxsum, lbow_out, rbow_out, rctxsum,), dim=-2)), x.unsqueeze(-2),), dim=-2))

class DBOW(nn.Module):

	# different from MultiHeadAttn, we use num_head for window size
	# osize is ignored
	def __init__(self, isize, hsize, osize, num_head=5, dropout=0.0, norm_residual=norm_residual_default, ngroup=4, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(DBOW, self).__init__()

		self.inet = DBOWInput(num_head, isize)

		self.net = nn.Sequential(GroupLinear(isize * 5, hsize, ngroup, trans_input=False, shuffle=False, flatten_output=True), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(hsize, isize, bias=enable_bias), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(GroupLinear(isize * 5, hsize, ngroup, trans_input=False, shuffle=False, flatten_output=True), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(hsize, isize, bias=enable_bias))

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.norm_residual = norm_residual

	def forward(self, x, mask=None, **kwargs):

		if mask is not None:
			x.masked_fill_(mask, 0.0)

		_x = self.normer(x)

		return self.net(self.inet(_x)) + (_x if self.norm_residual else x)

class PositionwiseFF(PositionwiseFFBase):

	# hsize is ignored in this version
	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, kernel_size=5, ngroup=3, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		_hsize = isize * 3# if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=_act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias)

		self.net = nn.Sequential(GroupLinear(isize * 3, _hsize, ngroup, shuffle=False), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias))
		if dropout > 0.0:
			self.net.append(Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			self.net.insert(2, Dropout(_act_drop, inplace=inplace_after_Custom_Act))

		self.trans = GroupLinear(isize * 3, isize * 3, 3, bias=False, shuffle=True)

		self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
		self.zpad, self.k = kernel_size, float(kernel_size)
		self.ctx_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, x, mask=None, **kwargs):

		if mask is not None:
			x.masked_fill_(mask, 0.0)

		_psize = list(x.size())
		nretr = _psize[-2]
		_psize[-2] = self.zpad
		_pad = x.new_zeros(_psize, dtype=x.dtype, device=x.device, requires_grad=False)

		_ctx = self.ctx_normer(self.pool(torch.cat((_pad, x, _pad,), dim=-2).transpose(-1, -2)).mul_(self.k).transpose(-1, -2))
		_out = self.normer(x)

		_inet = self.trans(torch.cat((_ctx.narrow(-2, 0, nretr), _out, _ctx.narrow(-2, 2, nretr),), dim=-1))

		out = self.net(_inet)

		out = out + (_out if self.norm_residual else x)

		return out
